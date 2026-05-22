"""Hermes-style skills system extracted for reuse in other agents.

This module preserves the key Hermes techniques:

1. Progressive disclosure:
   list metadata first, load full instructions only when needed.
2. Skills as procedural memory:
   each skill is a directory with SKILL.md plus optional references/templates.
3. YAML frontmatter:
   metadata is structured and machine-readable.
4. Safe skill creation and patching:
   frontmatter validation, atomic writes, and supporting-file guardrails.
5. Prompt-friendly skill index:
   compact skill summaries are injected into the system prompt; full text stays
   out of context until explicitly loaded.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from planner_agent.skills_text_matching import fuzzy_find_and_replace

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_SKILL_CONTENT_CHARS = 100_000
MAX_SKILL_FILE_BYTES = 1_048_576
VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}
PLATFORM_MAP = {"macos": "darwin", "linux": "linux", "windows": "win32"}
EXCLUDED_SKILL_DIRS = frozenset((".git", ".github", ".hub"))

_yaml_load_fn = None
_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "you are now",
    "disregard your",
    "forget your instructions",
    "system prompt:",
    "<system>",
]


def yaml_load(content: str) -> Any:
    global _yaml_load_fn
    if _yaml_load_fn is None:
        import yaml

        loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader

        def _load(value: str) -> Any:
            return yaml.load(value, Loader=loader)

        _yaml_load_fn = _load
    return _yaml_load_fn(content)


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    if not content.startswith("---"):
        return {}, content

    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    yaml_content = content[3 : end_match.start() + 3]
    body = content[end_match.end() + 3 :]
    try:
        parsed = yaml_load(yaml_content)
        return (parsed if isinstance(parsed, dict) else {}), body
    except Exception:
        fallback: Dict[str, Any] = {}
        for line in yaml_content.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fallback[key.strip()] = value.strip()
        return fallback, body


def skill_matches_platform(frontmatter: Dict[str, Any]) -> bool:
    platforms = frontmatter.get("platforms")
    if not platforms:
        return True
    if not isinstance(platforms, list):
        platforms = [platforms]
    current = os.sys.platform
    for platform_name in platforms:
        normalized = str(platform_name).lower().strip()
        mapped = PLATFORM_MAP.get(normalized, normalized)
        if current.startswith(mapped):
            return True
    return False


def extract_skill_description(frontmatter: Dict[str, Any], body: str) -> str:
    description = str(frontmatter.get("description", "")).strip()
    if description:
        return description[:MAX_DESCRIPTION_LENGTH]
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return stripped[:MAX_DESCRIPTION_LENGTH]
    return ""


def extract_skill_conditions(frontmatter: Dict[str, Any]) -> Dict[str, List[str]]:
    metadata = frontmatter.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    hermes_meta = metadata.get("hermes") or {}
    if not isinstance(hermes_meta, dict):
        hermes_meta = {}
    return {
        "fallback_for_toolsets": list(hermes_meta.get("fallback_for_toolsets", [])),
        "requires_toolsets": list(hermes_meta.get("requires_toolsets", [])),
        "fallback_for_tools": list(hermes_meta.get("fallback_for_tools", [])),
        "requires_tools": list(hermes_meta.get("requires_tools", [])),
    }


def skill_should_show(
    conditions: Dict[str, List[str]],
    available_tools: Optional[Iterable[str]] = None,
    available_toolsets: Optional[Iterable[str]] = None,
) -> bool:
    tools = set(available_tools or [])
    toolsets = set(available_toolsets or [])

    required_tools = set(conditions.get("requires_tools", []))
    if required_tools and not required_tools.issubset(tools):
        return False

    required_toolsets = set(conditions.get("requires_toolsets", []))
    if required_toolsets and not required_toolsets.issubset(toolsets):
        return False

    fallback_for_tools = set(conditions.get("fallback_for_tools", []))
    if fallback_for_tools and fallback_for_tools.isdisjoint(tools):
        return False

    fallback_for_toolsets = set(conditions.get("fallback_for_toolsets", []))
    if fallback_for_toolsets and fallback_for_toolsets.isdisjoint(toolsets):
        return False

    return True


def validate_frontmatter(content: str) -> Optional[str]:
    if not content.strip():
        return "Content cannot be empty."
    if not content.startswith("---"):
        return "SKILL.md must start with YAML frontmatter (---)."
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return "SKILL.md frontmatter is not closed."

    frontmatter, body = parse_frontmatter(content)
    if not isinstance(frontmatter, dict):
        return "Frontmatter must be a YAML mapping."
    if "name" not in frontmatter:
        return "Frontmatter must include 'name'."
    if "description" not in frontmatter:
        return "Frontmatter must include 'description'."
    if len(str(frontmatter["description"])) > MAX_DESCRIPTION_LENGTH:
        return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
    if not body.strip():
        return "SKILL.md must include instructions after the frontmatter."
    return None


def validate_skill_content_security(content: str, *, source: str = "skill") -> Optional[str]:
    lowered = content.lower()
    for needle in _INJECTION_PATTERNS:
        if needle in lowered:
            return f"Blocked: {source} content matched suspicious pattern '{needle}'."
    return None


@dataclass(slots=True)
class SkillMetadata:
    name: str
    description: str
    category: str = "general"
    path: str = ""
    tags: Tuple[str, ...] = ()
    linked_files: Tuple[str, ...] = ()


class SkillsStore:
    """Portable skills store with Hermes-style progressive disclosure."""

    def __init__(self, local_dir: str | Path, external_dirs: Optional[Iterable[str | Path]] = None) -> None:
        self.local_dir = Path(local_dir)
        self.external_dirs = [Path(path) for path in (external_dirs or [])]

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def all_skill_dirs(self) -> List[Path]:
        return [self.local_dir, *self.external_dirs]

    def iter_skill_files(self) -> Iterable[Path]:
        for skills_dir in self.all_skill_dirs():
            if not skills_dir.exists():
                continue
            for skill_md in skills_dir.rglob("SKILL.md"):
                if any(part in EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                    continue
                yield skill_md

    def find_skill(self, name: str) -> Optional[Dict[str, Any]]:
        requested_name = str(name).strip()
        for skill_md in self.iter_skill_files():
            if skill_md.parent.name == requested_name:
                return {"path": skill_md.parent, "external": not self._is_local_skill(skill_md.parent)}
            try:
                content = skill_md.read_text(encoding="utf-8")
            except OSError:
                continue
            frontmatter, _ = parse_frontmatter(content)
            skill_name = str(frontmatter.get("name", "")).strip()
            if skill_name == requested_name:
                return {"path": skill_md.parent, "external": not self._is_local_skill(skill_md.parent)}
        return None

    def list_skills(
        self,
        *,
        category: Optional[str] = None,
        available_tools: Optional[Iterable[str]] = None,
        available_toolsets: Optional[Iterable[str]] = None,
    ) -> List[SkillMetadata]:
        skills: List[SkillMetadata] = []
        for skill_md in self.iter_skill_files():
            try:
                content = skill_md.read_text(encoding="utf-8")
            except OSError:
                continue
            frontmatter, body = parse_frontmatter(content)
            if not skill_matches_platform(frontmatter):
                continue

            skill_name = str(frontmatter.get("name", skill_md.parent.name))[:MAX_NAME_LENGTH]
            skill_category = self._skill_category(skill_md.parent)
            if category and skill_category != category:
                continue

            conditions = extract_skill_conditions(frontmatter)
            if not skill_should_show(conditions, available_tools, available_toolsets):
                continue

            linked_files = tuple(sorted(self._list_supporting_files(skill_md.parent)))
            tags = tuple(self._extract_tags(frontmatter))
            description = extract_skill_description(frontmatter, body)
            skills.append(
                SkillMetadata(
                    name=skill_name,
                    description=description,
                    category=skill_category,
                    path=str(skill_md.parent),
                    tags=tags,
                    linked_files=linked_files,
                )
            )
        return sorted(skills, key=lambda item: (item.category, item.name))

    def view_skill(self, name: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        existing = self.find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}

        skill_dir = existing["path"]
        if file_path:
            target, error = self._resolve_skill_target(skill_dir, file_path)
            if error:
                return {"success": False, "error": error}
            if not target.exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            return {
                "success": True,
                "name": name,
                "file_path": file_path,
                "content": target.read_text(encoding="utf-8"),
                "path": str(target),
            }

        skill_md = skill_dir / "SKILL.md"
        content = skill_md.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(content)
        return {
            "success": True,
            "name": str(frontmatter.get("name", skill_dir.name)),
            "description": extract_skill_description(frontmatter, body),
            "frontmatter": frontmatter,
            "content": content,
            "linked_files": {
                "references": sorted(str(path.relative_to(skill_dir)) for path in (skill_dir / "references").glob("*") if path.is_file())
                if (skill_dir / "references").exists()
                else [],
                "templates": sorted(str(path.relative_to(skill_dir)) for path in (skill_dir / "templates").rglob("*") if path.is_file())
                if (skill_dir / "templates").exists()
                else [],
                "scripts": sorted(str(path.relative_to(skill_dir)) for path in (skill_dir / "scripts").rglob("*") if path.is_file())
                if (skill_dir / "scripts").exists()
                else [],
                "assets": sorted(str(path.relative_to(skill_dir)) for path in (skill_dir / "assets").rglob("*") if path.is_file())
                if (skill_dir / "assets").exists()
                else [],
            },
            "path": str(skill_md),
        }

    def build_system_prompt(
        self,
        *,
        available_tools: Optional[Iterable[str]] = None,
        available_toolsets: Optional[Iterable[str]] = None,
    ) -> str:
        skills = self.list_skills(
            available_tools=available_tools,
            available_toolsets=available_toolsets,
        )
        if not skills:
            return ""

        grouped: Dict[str, List[SkillMetadata]] = {}
        for skill in skills:
            grouped.setdefault(skill.category, []).append(skill)

        lines = []
        for category in sorted(grouped):
            lines.append(f"  {category}:")
            for skill in grouped[category]:
                lines.append(f"    - {skill.name}: {skill.description}")

        return (
            "## Skills (mandatory)\n"
            "Before replying, scan the skills below. If a skill matches or is even partially relevant "
            "to your task, load it and follow its instructions. Err on the side of loading.\n"
            "If a loaded skill is stale or missing steps, patch it immediately.\n"
            "After difficult or iterative tasks, consider saving the approach as a new skill.\n\n"
            "<available_skills>\n"
            + "\n".join(lines)
            + "\n</available_skills>\n\n"
            "Only proceed without loading a skill if genuinely none are relevant."
        )

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def create_skill(self, name: str, content: str, category: Optional[str] = None) -> Dict[str, Any]:
        error = self._validate_name(name) or self._validate_category(category)
        if error:
            return {"success": False, "error": error}
        error = validate_frontmatter(content) or self._validate_content_size(content)
        if error:
            return {"success": False, "error": error}
        error = validate_skill_content_security(content, source="SKILL.md")
        if error:
            return {"success": False, "error": error}
        if self.find_skill(name):
            return {"success": False, "error": f"A skill named '{name}' already exists."}

        skill_dir = self._resolve_skill_dir(name, category)
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        self._atomic_write_text(skill_md, content)
        return {
            "success": True,
            "message": f"Skill '{name}' created.",
            "path": str(skill_dir),
            "hint": "Add references/templates/scripts/assets with write_file().",
        }

    def edit_skill(self, name: str, content: str) -> Dict[str, Any]:
        error = validate_frontmatter(content) or self._validate_content_size(content)
        if error:
            return {"success": False, "error": error}
        error = validate_skill_content_security(content, source="SKILL.md")
        if error:
            return {"success": False, "error": error}

        existing = self.find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}
        if existing["external"]:
            return {"success": False, "error": f"Skill '{name}' is in an external directory and cannot be modified."}

        skill_md = existing["path"] / "SKILL.md"
        self._atomic_write_text(skill_md, content)
        return {"success": True, "message": f"Skill '{name}' updated.", "path": str(skill_md)}

    def patch_skill(
        self,
        name: str,
        old_string: str,
        new_string: str,
        *,
        file_path: Optional[str] = None,
        replace_all: bool = False,
    ) -> Dict[str, Any]:
        if not old_string:
            return {"success": False, "error": "old_string is required for patch."}
        if new_string is None:
            return {"success": False, "error": "new_string is required for patch."}

        existing = self.find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}
        if existing["external"]:
            return {"success": False, "error": f"Skill '{name}' is in an external directory and cannot be modified."}

        skill_dir = existing["path"]
        if file_path:
            error = self._validate_file_path(file_path)
            if error:
                return {"success": False, "error": error}
            target, error = self._resolve_skill_target(skill_dir, file_path)
            if error:
                return {"success": False, "error": error}
        else:
            target = skill_dir / "SKILL.md"

        if not target.exists():
            return {"success": False, "error": f"File not found: {target.relative_to(skill_dir)}"}

        content = target.read_text(encoding="utf-8")
        new_content, match_count, strategy, match_error = fuzzy_find_and_replace(
            content=content,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        if match_error:
            preview = content[:500] + ("..." if len(content) > 500 else "")
            return {
                "success": False,
                "error": match_error,
                "file_preview": preview,
            }

        error = self._validate_content_size(new_content, label=file_path or "SKILL.md")
        if error:
            return {"success": False, "error": error}
        if not file_path:
            error = validate_frontmatter(new_content)
            if error:
                return {"success": False, "error": f"Patch would break SKILL.md structure: {error}"}
        error = validate_skill_content_security(new_content, source=file_path or "SKILL.md")
        if error:
            return {"success": False, "error": error}

        self._atomic_write_text(target, new_content)
        return {
            "success": True,
            "message": (
                f"Patched {file_path or 'SKILL.md'} in skill '{name}' "
                f"({match_count} replacement{'s' if match_count != 1 else ''}, strategy={strategy})."
            ),
        }

    def delete_skill(self, name: str) -> Dict[str, Any]:
        existing = self.find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}
        if existing["external"]:
            return {"success": False, "error": f"Skill '{name}' is in an external directory and cannot be deleted."}

        skill_dir = existing["path"]
        shutil.rmtree(skill_dir)
        parent = skill_dir.parent
        if parent != self.local_dir and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return {"success": True, "message": f"Skill '{name}' deleted."}

    def write_file(self, name: str, file_path: str, file_content: str) -> Dict[str, Any]:
        error = self._validate_file_path(file_path)
        if error:
            return {"success": False, "error": error}
        if file_content is None:
            return {"success": False, "error": "file_content is required."}
        content_bytes = len(file_content.encode("utf-8"))
        if content_bytes > MAX_SKILL_FILE_BYTES:
            return {"success": False, "error": f"File content exceeds {MAX_SKILL_FILE_BYTES:,} bytes."}
        error = self._validate_content_size(file_content, label=file_path)
        if error:
            return {"success": False, "error": error}
        error = validate_skill_content_security(file_content, source=file_path)
        if error:
            return {"success": False, "error": error}

        existing = self.find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}
        if existing["external"]:
            return {"success": False, "error": f"Skill '{name}' is in an external directory and cannot be modified."}

        target, error = self._resolve_skill_target(existing["path"], file_path)
        if error:
            return {"success": False, "error": error}
        target.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(target, file_content)
        return {"success": True, "message": f"File '{file_path}' written to skill '{name}'.", "path": str(target)}

    def remove_file(self, name: str, file_path: str) -> Dict[str, Any]:
        error = self._validate_file_path(file_path)
        if error:
            return {"success": False, "error": error}
        existing = self.find_skill(name)
        if not existing:
            return {"success": False, "error": f"Skill '{name}' not found."}
        if existing["external"]:
            return {"success": False, "error": f"Skill '{name}' is in an external directory and cannot be modified."}

        target, error = self._resolve_skill_target(existing["path"], file_path)
        if error:
            return {"success": False, "error": error}
        if not target.exists():
            return {"success": False, "error": f"File '{file_path}' not found in skill '{name}'."}
        target.unlink()
        parent = target.parent
        if parent != existing["path"] and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return {"success": True, "message": f"File '{file_path}' removed from skill '{name}'."}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_local_skill(self, skill_path: Path) -> bool:
        try:
            skill_path.resolve().relative_to(self.local_dir.resolve())
            return True
        except ValueError:
            return False

    def _resolve_skill_dir(self, name: str, category: Optional[str]) -> Path:
        return self.local_dir / category / name if category else self.local_dir / name

    def _skill_category(self, skill_dir: Path) -> str:
        try:
            rel = skill_dir.relative_to(self.local_dir)
            if len(rel.parts) > 1:
                return rel.parts[0]
        except ValueError:
            pass
        return "general"

    def _list_supporting_files(self, skill_dir: Path) -> List[str]:
        results: List[str] = []
        for subdir in ALLOWED_SUBDIRS:
            base = skill_dir / subdir
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file():
                    results.append(str(path.relative_to(skill_dir)))
        return results

    def _extract_tags(self, frontmatter: Dict[str, Any]) -> List[str]:
        metadata = frontmatter.get("metadata")
        if not isinstance(metadata, dict):
            return []
        hermes_meta = metadata.get("hermes") or {}
        if not isinstance(hermes_meta, dict):
            return []
        raw_tags = hermes_meta.get("tags") or frontmatter.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [tag.strip() for tag in raw_tags.split(",")]
        return [str(tag).strip() for tag in raw_tags if str(tag).strip()]

    @staticmethod
    def _validate_name(name: str) -> Optional[str]:
        if not name:
            return "Skill name is required."
        if len(name) > MAX_NAME_LENGTH:
            return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
        if not VALID_NAME_RE.match(name):
            return "Skill names must be lowercase and filesystem-safe."
        return None

    @staticmethod
    def _validate_category(category: Optional[str]) -> Optional[str]:
        if category is None:
            return None
        category = category.strip()
        if not category:
            return None
        if "/" in category or "\\" in category:
            return "Category must be a single directory name."
        if len(category) > MAX_NAME_LENGTH:
            return f"Category exceeds {MAX_NAME_LENGTH} characters."
        if not VALID_NAME_RE.match(category):
            return "Category names must be lowercase and filesystem-safe."
        return None

    @staticmethod
    def _validate_content_size(content: str, label: str = "SKILL.md") -> Optional[str]:
        if len(content) > MAX_SKILL_CONTENT_CHARS:
            return (
                f"{label} content is {len(content):,} characters "
                f"(limit: {MAX_SKILL_CONTENT_CHARS:,})."
            )
        return None

    @staticmethod
    def _validate_file_path(file_path: str) -> Optional[str]:
        if not file_path:
            return "file_path is required."
        normalized = Path(file_path)
        if ".." in normalized.parts:
            return "Path traversal ('..') is not allowed."
        if not normalized.parts or normalized.parts[0] not in ALLOWED_SUBDIRS:
            allowed = ", ".join(sorted(ALLOWED_SUBDIRS))
            return f"File must be under one of: {allowed}."
        if len(normalized.parts) < 2:
            return "Provide a file path, not just a directory."
        return None

    @staticmethod
    def _resolve_skill_target(skill_dir: Path, file_path: str) -> Tuple[Optional[Path], Optional[str]]:
        target = (skill_dir / file_path).resolve()
        try:
            target.relative_to(skill_dir.resolve())
        except ValueError:
            return None, "Resolved path escapes the skill directory."
        return target, None

    @staticmethod
    def _atomic_write_text(file_path: Path, content: str, encoding: str = "utf-8") -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=str(file_path.parent), prefix=f".{file_path.name}.tmp.", suffix="")
        try:
            with os.fdopen(fd, "w", encoding=encoding) as handle:
                handle.write(content)
            os.replace(temp_path, file_path)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.debug("Failed to remove temporary skill file %s", temp_path, exc_info=True)

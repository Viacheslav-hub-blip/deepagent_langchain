"""Skills service backed by planner_agent SkillsStore."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from planner_agent.schemas.skills import SkillPatchProposal, SkillRecord
from planner_agent.skills_store import SkillsStore

from ._json import append_jsonl, read_jsonl


class SkillsService:
    def __init__(self, skills_dir: str | Path = "skills") -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.store = SkillsStore(self.skills_dir)

    def skills_list(self) -> list[SkillRecord]:
        return [
            SkillRecord.model_validate(asdict(skill))
            for skill in self.store.list_skills()
        ]

    def skill_view(self, name: str, file_path: str | None = None) -> dict[str, Any]:
        return self.store.view_skill(name, file_path)

    def skill_create(self, name: str, content: str, category: str | None = None) -> dict[str, Any]:
        return self.store.create_skill(name, content, category)

    def skill_patch(
        self,
        name: str,
        old_string: str,
        new_string: str,
        *,
        file_path: str | None = None,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        return self.store.patch_skill(
            name,
            old_string,
            new_string,
            file_path=file_path,
            replace_all=replace_all,
        )

    def skill_delete(self, name: str) -> dict[str, Any]:
        return self.store.delete_skill(name)

    def build_skills_index(self) -> str:
        skills = self.skills_list()
        if not skills:
            return "No skills are available."
        lines = ["Available skills:"]
        for skill in skills:
            files = f" linked_files={skill.linked_files}" if skill.linked_files else ""
            lines.append(f"- {skill.name}: {skill.description}{files}")
        return "\n".join(lines)

    def build_skill_previews(self) -> dict[str, str]:
        """Формирует краткие описания всех доступных skills.

        Returns:
            Словарь, где ключом является имя skill из frontmatter, а значением
            является компактное описание с привязанными файлами.
        """

        previews: dict[str, str] = {}
        for skill in self.skills_list():
            files = f" linked_files={skill.linked_files}" if skill.linked_files else ""
            previews[skill.name] = f"{skill.description}{files}".strip()
        return previews

    def skill_propose_patch(
        self,
        *,
        skill_name: str,
        old_string: str,
        new_string: str,
        file_path: str | None = None,
        rationale: str = "",
        risk: str = "medium",
        run_id: str | None = None,
        node_id: str | None = None,
    ) -> SkillPatchProposal:
        proposal = SkillPatchProposal(
            skill_name=skill_name,
            old_string=old_string,
            new_string=new_string,
            file_path=file_path,
            rationale=rationale,
            risk=risk,  # type: ignore[arg-type]
            run_id=run_id,
            node_id=node_id,
        )
        append_jsonl(self.skills_dir / "review_queue.jsonl", proposal)
        return proposal

    def list_patch_proposals(self) -> list[SkillPatchProposal]:
        return [
            SkillPatchProposal.model_validate(row)
            for row in read_jsonl(self.skills_dir / "review_queue.jsonl")
        ]


__all__ = ["SkillsService"]

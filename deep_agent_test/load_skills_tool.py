"""Tool пакетной загрузки skills для supervisor.

Содержит:
- build_load_skills_tool: фабрика tool ``load_skills`` с замыканием на settings.
- _build_skill_lookup: индекс соответствий токен -> файл skill (path/name/folder).
- _resolve_token: разрешение одного токена запроса в запись skill.
- _split_tokens: разбор строкового списка вида ``skill1, skill2``.
- _build_report: сборка финального текстового отчёта tool с контентом skills.

Tool детерминированно конкатенирует контент выбранных ``SKILL.md`` с заголовком-именем
перед каждым skill, дедуплицирует уже загруженные skills (middleware preload, прошлые
вызовы load_skills и явный ``already_loaded``) и не отдаёт их контент в контекст supervisor
повторно.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agent_test.settings import DeepAgentSettings
from deep_agent_test.skills_context_middleware import (
    _parse_skill_index_entry,
    _read_context_file,
    _virtual_skill_path,
    discover_skill_context_files,
)

LOAD_SKILLS_TOOL_NAME = "load_skills"

LOAD_SKILLS_DESCRIPTION = """
Загружает контент выбранных skills (`SKILL.md`) ОДНИМ вызовом.

Когда использовать:
- нужного skill нет среди предзагруженных (раздел Preloaded Skills);
- контент предзагруженного skill неполон и нужен дополнительный источник;
- по Skills Index видно, что для задачи нужен ещё один skill.

Что делает:
- читает каждый указанный `SKILL.md` и возвращает их дословный контент;
- перед контентом каждого skill ставит его имя и виртуальный путь;
- пропускает skills, уже загруженные ранее (middleware preload, прошлые вызовы
  load_skills и список из `already_loaded`), чтобы не дублировать контекст supervisor.

Аргументы:
- `skill_names` — имена или виртуальные пути нужных skills из Skills Index ЧЕРЕЗ ЗАПЯТУЮ
  в одной строке (например: `skill-a, skill-b` или
  `/skills/skill-a/SKILL.md, /skills/skill-b/SKILL.md`);
- `already_loaded` — уже загруженные skills через запятую, которые НЕ нужно грузить
  повторно. Можно оставить пустым.

Для создания, изменения или изучения структуры skill используй стандартные файловые
инструменты (`ls`, `read_file`, `write_file`). `load_skills` нужен только для удобной
пакетной загрузки уже известных skills в контекст.
""".strip()


def _split_tokens(raw: str) -> list[str]:
    """Разбирает строковый список вида ``skill1, skill2`` в список токенов."""

    if not raw:
        return []
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def _build_skill_lookup(settings: DeepAgentSettings) -> dict[str, dict[str, str]]:
    """Строит индекс соответствий токен (lower) -> запись skill.

    Один skill доступен по нескольким токенам: виртуальный путь, относительный путь,
    имя папки и ``name`` из front matter.
    """

    lookup: dict[str, dict[str, str]] = {}
    for local_path in discover_skill_context_files(settings.skills_root):
        virtual_path = _virtual_skill_path(settings.skills_root, local_path, settings.skills_virtual_dir)
        header = _read_context_file(local_path, max_chars=4000) or ""
        parsed = _parse_skill_index_entry(header)
        name = parsed.get("name") or local_path.parent.name
        try:
            relative_path = local_path.relative_to(settings.skills_root).as_posix()
        except ValueError:
            relative_path = local_path.name
        entry = {"virtual_path": virtual_path, "local_path": str(local_path), "name": name}
        for token in {virtual_path, relative_path, local_path.parent.name, name}:
            token = (token or "").strip().lower()
            if token:
                lookup.setdefault(token, entry)
    return lookup


def _resolve_token(token: str, lookup: dict[str, dict[str, str]]) -> dict[str, str] | None:
    """Разрешает один токен запроса в запись skill или ``None``, если не найден."""

    key = token.strip().lower()
    if key in lookup:
        return lookup[key]
    if not key.endswith("/skill.md"):
        with_file = f"{key.rstrip('/')}/skill.md"
        if with_file in lookup:
            return lookup[with_file]
    return None


def _build_report(
    blocks: list[str],
    newly_loaded: list[str],
    skipped: list[str],
    unknown: list[str],
) -> str:
    """Собирает финальный текстовый отчёт tool с контентом и заметками."""

    sections: list[str] = []
    if blocks:
        sections.append("## Загруженные skills\n\n" + "\n\n".join(blocks))
    else:
        sections.append("## Загруженные skills\n\nНовых skills не загружено.")

    notes: list[str] = []
    if skipped:
        notes.append("Пропущены как уже загруженные: " + ", ".join(skipped))
    if unknown:
        notes.append("Не найдены в Skills Index: " + ", ".join(unknown))
    if notes:
        sections.append("\n".join(notes))
    return "\n\n".join(sections)


def build_load_skills_tool(settings: DeepAgentSettings) -> Any:
    """Собирает tool ``load_skills`` для supervisor с замыканием на settings."""

    lookup = _build_skill_lookup(settings)
    max_chars = settings.max_chars_per_skill

    @tool(LOAD_SKILLS_TOOL_NAME, description=LOAD_SKILLS_DESCRIPTION)
    def load_skills(
        skill_names: str,
        already_loaded: str = "",
        state: Annotated[dict, InjectedState] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        """Загружает skills по строковому списку имён/путей и дедуплицирует контекст.

        Args:
            skill_names: Имена или виртуальные пути skills через запятую.
            already_loaded: Уже загруженные skills через запятую, которые нужно пропустить.
        """

        state = state or {}
        requested = _split_tokens(skill_names)

        already_seen: set[str] = set()
        already_seen.update(state.get("preloaded_skill_paths") or [])
        already_seen.update(state.get("materialized_skill_paths") or [])
        for token in _split_tokens(already_loaded):
            entry = _resolve_token(token, lookup)
            already_seen.add(entry["virtual_path"] if entry else token)

        blocks: list[str] = []
        newly_loaded: list[str] = []
        skipped: list[str] = []
        unknown: list[str] = []
        for token in requested:
            entry = _resolve_token(token, lookup)
            if entry is None:
                unknown.append(token)
                continue
            virtual_path = entry["virtual_path"]
            if virtual_path in already_seen:
                skipped.append(virtual_path)
                continue
            content = _read_context_file(Path(entry["local_path"]), max_chars)
            if content is None:
                unknown.append(token)
                continue
            already_seen.add(virtual_path)
            newly_loaded.append(virtual_path)
            blocks.append(f"### {entry['name']} ({virtual_path})\n\n{content}")

        report = _build_report(blocks, newly_loaded, skipped, unknown)
        materialized = [*(state.get("materialized_skill_paths") or []), *newly_loaded]
        return Command(
            update={
                "materialized_skill_paths": materialized,
                "messages": [
                    ToolMessage(report, tool_call_id=tool_call_id, name=LOAD_SKILLS_TOOL_NAME)
                ],
            }
        )

    return load_skills


__all__ = [
    "LOAD_SKILLS_DESCRIPTION",
    "LOAD_SKILLS_TOOL_NAME",
    "build_load_skills_tool",
]

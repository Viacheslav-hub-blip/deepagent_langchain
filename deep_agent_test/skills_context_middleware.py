"""Middleware предзагрузки domain context из локальных skills.

Содержит:
- PreloadedSkillsContextMiddleware: middleware чтения skills и добавления context в prompt.
- PreloadedSkillsContextMiddleware.before_agent: чтение skills и запись context в state.
- PreloadedSkillsContextMiddleware.wrap_model_call: добавление skills context в system prompt.
- build_preloaded_skills_context: автосканирование папки skills и сборка compact context.
- select_relevant_skill_paths_with_llm: выбор релевантных skills по index через LLM.
- discover_skill_context_files: поиск файлов SKILL.md для предзагрузки.
- build_skills_index: построение компактного index skills.
- _read_context_file: чтение одного context-файла с ограничением размера.
- _latest_user_query: получение последнего пользовательского запроса из state.
- _parse_skill_index_entry: извлечение имени и описания skill.
- _virtual_skill_path: построение виртуального пути skills для найденного файла.
- _normalize_virtual_dir: нормализация виртуальной папки skills.
- _truncate_text: ограничение длины текста skill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from deepagents.middleware._utils import append_to_system_message

from deep_agent_test.agent_state import AnalyticsAgentState
from deep_agent_test.prompts import PRELOADED_SKILLS_CONTEXT_PROMPT_TEMPLATE, SKILLS_INDEX_CONTEXT_PROMPT_TEMPLATE


class SelectedSkillPaths(BaseModel):
    """Результат выбора релевантных skills перед запуском агента.

    Attributes:
        paths: Виртуальные пути выбранных файлов ``SKILL.md``.
    """

    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Виртуальные пути выбранных skill-файлов. Для многошаговых задач включай "
            "все потенциально нужные skills, а не один наиболее похожий."
        ),
    )


@dataclass(frozen=True)
class PreloadedSkillsContextMiddleware(AgentMiddleware[AnalyticsAgentState]):
    """Автоматически загружает файлы ``SKILL.md`` до первого рассуждения модели.

    Используется в двух режимах, которые делят один кэш через ``shared_selection``:

    - ``select_skills=True`` (supervisor): выбирает релевантные skills через LLM один раз
      на пользовательский запрос и кладёт результат в общий кэш.
    - ``select_skills=False`` (subagent): не вызывает LLM, а переиспользует выбор
      supervisor-а из общего кэша, чтобы в субагентов попадали те же skills.

    Args:
        skills_root: Локальная папка проекта, которую нужно рекурсивно просканировать.
        skills_virtual_dir: Виртуальная папка skills внутри DeepAgents backend.
        max_chars_per_file: Максимальная длина текста одного ``SKILL.md`` в context.
        model: Chat model для выбора релевантных skills по index. Если ``None``, загружаются все skills.
        select_skills: Режим выбора. ``True`` — выбирать и кэшировать (supervisor),
            ``False`` — только переиспользовать кэш supervisor-а (subagent).
        shared_selection: Общий мутируемый кэш выбора skills. Один и тот же словарь нужно
            передать supervisor- и subagent-экземплярам, чтобы они делили выбор.

    Returns:
        Middleware, который добавляет compact domain context в state и system message.
    """

    skills_root: Path
    skills_virtual_dir: str = "/skills/"
    max_chars_per_file: int = 18000
    model: Any | None = None
    select_skills: bool = True
    shared_selection: dict[str, Any] = field(default_factory=dict, compare=False)

    state_schema = AnalyticsAgentState

    def before_agent(
        self,
        state: AnalyticsAgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Читает skills и сохраняет context в state перед запуском агента.

        Args:
            state: Текущий state агента.
            runtime: Runtime LangGraph текущего запуска.

        Returns:
            Обновление state с context и списком прочитанных skill-файлов.
        """

        user_query = _latest_user_query(state)
        if state.get("skills_context_loaded") and state.get("preloaded_skills_selection_user_key") == user_query:
            return None

        selection = self._resolve_selection(user_query)
        if selection is None:
            return None

        context, loaded_paths, skills_index = selection
        return {
            "skills_context_loaded": True,
            "preloaded_skills_selection_user_key": user_query,
            "preloaded_skill_paths": loaded_paths,
            "preloaded_skills_context": context,
            "preloaded_skills_index": skills_index,
        }

    def _resolve_selection(
        self,
        user_query: str,
    ) -> tuple[str, list[str], list[dict[str, str]]] | None:
        """Возвращает выбор skills из кэша или вычисляет его (только для supervisor).

        Args:
            user_query: Последний пользовательский запрос текущего агента.

        Returns:
            Кортеж ``(context, loaded_paths, skills_index)`` или ``None``, если выбор
            недоступен (субагент без кэша supervisor-а).
        """

        cached = self.shared_selection.get("entry")

        if not self.select_skills:
            if cached is None:
                return None
            return cached["context"], cached["paths"], cached["index"]

        if cached is not None and cached.get("user_query") == user_query:
            return cached["context"], cached["paths"], cached["index"]

        context, loaded_paths = build_preloaded_skills_context(
            skills_root=self.skills_root,
            skills_virtual_dir=self.skills_virtual_dir,
            max_chars_per_file=self.max_chars_per_file,
            model=self.model,
            user_query=user_query,
        )
        skills_index = build_skills_index(
            skill_files=discover_skill_context_files(self.skills_root),
            skills_root=self.skills_root,
            skills_virtual_dir=self.skills_virtual_dir,
        )
        self.shared_selection["entry"] = {
            "user_query": user_query,
            "context": context,
            "paths": loaded_paths,
            "index": skills_index,
        }
        return context, loaded_paths, skills_index

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        """Добавляет предзагруженный domain context в system message.

        Args:
            request: Запрос модели с текущим state.
            handler: Функция реального вызова модели.

        Returns:
            Ответ модели без изменений.
        """

        context = request.state.get("preloaded_skills_context")
        skills_index = request.state.get("preloaded_skills_index") or []
        if not context and not skills_index:
            return handler(request)

        system_message = request.system_message
        if skills_index:
            skills_index_text = json.dumps(skills_index, ensure_ascii=False, indent=2)
            system_message = append_to_system_message(
                system_message,
                SKILLS_INDEX_CONTEXT_PROMPT_TEMPLATE.format(skills_index=skills_index_text),
            )
        if context:
            system_message = append_to_system_message(
                system_message,
                PRELOADED_SKILLS_CONTEXT_PROMPT_TEMPLATE.format(context=context),
            )
        return handler(request.override(system_message=system_message))


def build_preloaded_skills_context(
    skills_root: Path,
    skills_virtual_dir: str,
    max_chars_per_file: int,
    model: Any | None = None,
    user_query: str = "",
) -> tuple[str, list[str]]:
    """Сканирует папку skills и собирает compact context из файлов ``SKILL.md``.

    Args:
        skills_root: Локальная папка проекта ``skills`` или другая переданная папка.
        skills_virtual_dir: Виртуальная папка, через которую DeepAgents видит skills.
        max_chars_per_file: Максимальная длина текста одного ``SKILL.md``.
        model: Chat model для LLM-выбора skills по index.
        user_query: Последний пользовательский запрос для выбора skills.

    Returns:
        Кортеж из общего markdown-context и списка виртуальных путей загруженных skills.
    """

    skill_files = discover_skill_context_files(skills_root)
    selected_paths = select_relevant_skill_paths_with_llm(
        model=model,
        user_query=user_query,
        skill_files=skill_files,
        skills_root=skills_root,
        skills_virtual_dir=skills_virtual_dir,
    )
    selected_path_set = set(selected_paths)
    blocks: list[str] = []
    loaded_paths: list[str] = []
    for skill_path in skill_files:
        virtual_path = _virtual_skill_path(skills_root, skill_path, skills_virtual_dir)
        if selected_path_set and virtual_path not in selected_path_set:
            continue
        content = _read_context_file(skill_path, max_chars_per_file)
        if content is None:
            continue
        loaded_paths.append(virtual_path)
        blocks.append(f"### {virtual_path}\n\n{content}")
    return "\n\n".join(blocks), loaded_paths


def select_relevant_skill_paths_with_llm(
    *,
    model: Any | None,
    user_query: str,
    skill_files: list[Path],
    skills_root: Path,
    skills_virtual_dir: str,
) -> list[str]:
    """Выбирает релевантные skills через LLM по компактному index.

    Args:
        model: Chat model с поддержкой ``with_structured_output``.
        user_query: Последний пользовательский запрос.
        skill_files: Найденные файлы ``SKILL.md``.
        skills_root: Корневая папка локальных skills.
        skills_virtual_dir: Виртуальная папка skills внутри DeepAgents.

    Returns:
        Список виртуальных путей выбранных skills. Если LLM недоступна или ничего не выбрала,
        возвращается полный список путей как безопасный fallback.
    """

    index = build_skills_index(
        skill_files=skill_files,
        skills_root=skills_root,
        skills_virtual_dir=skills_virtual_dir,
    )
    all_paths = [item["path"] for item in index]
    if model is None or not user_query.strip() or not index:
        return all_paths

    structured_model = model.with_structured_output(SelectedSkillPaths)
    try:
        result = structured_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Ты выбираешь domain skills для предварительной загрузки по запросу "
                        "пользователя. Используй только пути из переданного index. "
                        "Для многошаговых запросов выбирай все skills, которые могут понадобиться "
                        "на разных этапах, а не только один наиболее похожий. "
                        "Лучше включить лишний релевантный skill, чем пропустить нужный downstream-источник."
                    )
                ),
                HumanMessage(
                    content=json.dumps(
                        {
                            "user_query": user_query,
                            "skills_index": index,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                ),
            ]
        )
    except Exception:
        return all_paths

    allowed = set(all_paths)
    selected = [path for path in result.paths if path in allowed]
    return selected or all_paths


def build_skills_index(
    *,
    skill_files: list[Path],
    skills_root: Path,
    skills_virtual_dir: str,
) -> list[dict[str, str]]:
    """Строит компактный index skills для LLM-выбора.

    Args:
        skill_files: Найденные файлы ``SKILL.md``.
        skills_root: Корневая папка локальных skills.
        skills_virtual_dir: Виртуальная папка skills внутри DeepAgents.

    Returns:
        Список словарей с путем, именем и описанием skill.
    """

    index: list[dict[str, str]] = []
    for skill_path in skill_files:
        content = _read_context_file(skill_path, max_chars=4000) or ""
        parsed = _parse_skill_index_entry(content)
        index.append(
            {
                "path": _virtual_skill_path(skills_root, skill_path, skills_virtual_dir),
                "name": parsed.get("name") or skill_path.parent.name,
                "description": parsed.get("description") or "",
            }
        )
    return index


def discover_skill_context_files(skills_root: Path) -> list[Path]:
    """Находит файлы ``SKILL.md`` для автоматической предзагрузки.

    Args:
        skills_root: Папка skills, которую нужно просканировать рекурсивно.

    Returns:
        Список файлов ``SKILL.md`` из skill-папок, отсортированный по пути.
    """

    if not skills_root.exists():
        return []
    skill_files = [path for path in skills_root.rglob("SKILL.md") if path.is_file()]
    return sorted(
        skill_files,
        key=lambda path: path.relative_to(skills_root).as_posix().lower(),
    )


def _read_context_file(path: Path, max_chars: int) -> str | None:
    """Читает один context-файл skills и ограничивает его длину.

    Args:
        path: Абсолютный путь к файлу.
        max_chars: Максимальное количество символов для возврата.

    Returns:
        Текст файла или ``None``, если файл не найден.
    """

    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8")
    return _truncate_text(content, max_chars)


def _latest_user_query(state: AnalyticsAgentState) -> str:
    """Извлекает последний пользовательский запрос из state.

    Args:
        state: Текущий state агента.

    Returns:
        Текст последнего HumanMessage или пустая строка.
    """

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", ""))
    return ""


def _parse_skill_index_entry(content: str) -> dict[str, str]:
    """Извлекает имя и описание skill из front matter.

    Args:
        content: Текст ``SKILL.md``.

    Returns:
        Словарь с ключами ``name`` и ``description`` при наличии данных.
    """

    result: dict[str, str] = {}
    for line in content.splitlines()[:20]:
        if line.startswith("name:"):
            result["name"] = line.split(":", 1)[1].strip().strip('"')
        if line.startswith("description:"):
            result["description"] = line.split(":", 1)[1].strip().strip('"')
    return result


def _virtual_skill_path(skills_root: Path, path: Path, skills_virtual_dir: str) -> str:
    """Строит виртуальный путь skills для найденного файла.

    Args:
        skills_root: Локальная папка skills.
        path: Найденный markdown-файл внутри ``skills_root``.
        skills_virtual_dir: Виртуальная папка skills внутри DeepAgents backend.

    Returns:
        Виртуальный путь файла для prompt context и логов.
    """

    try:
        relative_path = path.relative_to(skills_root).as_posix()
    except ValueError:
        relative_path = path.name
    return f"{_normalize_virtual_dir(skills_virtual_dir)}{relative_path}"


def _normalize_virtual_dir(value: str) -> str:
    """Нормализует виртуальную папку skills к виду ``/name/``.

    Args:
        value: Виртуальный путь из настроек.

    Returns:
        Виртуальная папка с ведущим и завершающим слешем.
    """

    stripped = value.strip() or "/skills/"
    if not stripped.startswith("/"):
        stripped = f"/{stripped}"
    if not stripped.endswith("/"):
        stripped = f"{stripped}/"
    return stripped


def _truncate_text(text: str, max_chars: int) -> str:
    """Обрезает текст до заданного количества символов.

    Args:
        text: Исходный текст.
        max_chars: Максимальное количество символов.

    Returns:
        Исходный или обрезанный текст с пометкой о сокращении.
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n\n...[truncated {omitted} chars]"


__all__ = [
    "PreloadedSkillsContextMiddleware",
    "build_preloaded_skills_context",
    "build_skills_index",
    "discover_skill_context_files",
    "select_relevant_skill_paths_with_llm",
]

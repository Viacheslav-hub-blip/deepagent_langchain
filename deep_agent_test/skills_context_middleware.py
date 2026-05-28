"""Middleware предзагрузки domain context из локальных skills.

Содержит:
- PreloadedSkillsContextMiddleware: middleware чтения skills и добавления context в prompt.
- PreloadedSkillsContextMiddleware.before_agent: чтение skills и запись context в state.
- PreloadedSkillsContextMiddleware.wrap_model_call: добавление skills context в system prompt.
- build_preloaded_skills_context: автосканирование папки skills и сборка compact context.
- discover_skill_context_files: поиск файлов SKILL.md для предзагрузки.
- _read_context_file: чтение одного context-файла с ограничением размера.
- _virtual_skill_path: построение виртуального пути skills для найденного файла.
- _normalize_virtual_dir: нормализация виртуальной папки skills.
- _truncate_text: ограничение длины текста skill.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langgraph.runtime import Runtime

from deepagents.middleware._utils import append_to_system_message

from deep_agent_test.agent_logging import DeepAgentEventLogger
from deep_agent_test.plan_approval_middleware import AnalyticsPlanState
from deep_agent_test.prompts import PRELOADED_SKILLS_CONTEXT_PROMPT_TEMPLATE


@dataclass(frozen=True)
class PreloadedSkillsContextMiddleware(AgentMiddleware[AnalyticsPlanState]):
    """Автоматически загружает файлы ``SKILL.md`` до первого рассуждения модели.

    Args:
        skills_root: Локальная папка проекта, которую нужно рекурсивно просканировать.
        skills_virtual_dir: Виртуальная папка skills внутри DeepAgents backend.
        max_chars_per_file: Максимальная длина текста одного ``SKILL.md`` в context.
        event_logger: Файловый логгер для записи загруженных skills.

    Returns:
        Middleware, который добавляет compact domain context в state и system message.
    """

    skills_root: Path
    skills_virtual_dir: str = "/skills/"
    max_chars_per_file: int = 6000
    event_logger: DeepAgentEventLogger | None = None

    state_schema = AnalyticsPlanState

    def before_agent(
        self,
        state: AnalyticsPlanState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Читает skills и сохраняет context в state перед запуском агента.

        Args:
            state: Текущий state агента.
            runtime: Runtime LangGraph текущего запуска.

        Returns:
            Обновление state с context и списком прочитанных skill-файлов.
        """

        if state.get("skills_context_loaded"):
            return None

        context, loaded_paths = build_preloaded_skills_context(
            skills_root=self.skills_root,
            skills_virtual_dir=self.skills_virtual_dir,
            max_chars_per_file=self.max_chars_per_file,
        )
        if self.event_logger is not None:
            self.event_logger.log_loaded_skills(
                {
                    "loaded_paths": loaded_paths,
                    "loaded_count": len(loaded_paths),
                    "context_chars": len(context),
                    "skills_root": str(self.skills_root.resolve()),
                }
            )
        return {
            "skills_context_loaded": True,
            "preloaded_skill_paths": loaded_paths,
            "preloaded_skills_context": context,
        }

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
        if not context:
            return handler(request)

        system_message = append_to_system_message(
            request.system_message,
            PRELOADED_SKILLS_CONTEXT_PROMPT_TEMPLATE.format(context=context),
        )
        return handler(request.override(system_message=system_message))


def build_preloaded_skills_context(
    skills_root: Path,
    skills_virtual_dir: str,
    max_chars_per_file: int,
) -> tuple[str, list[str]]:
    """Сканирует папку skills и собирает compact context из файлов ``SKILL.md``.

    Args:
        skills_root: Локальная папка проекта ``skills`` или другая переданная папка.
        skills_virtual_dir: Виртуальная папка, через которую DeepAgents видит skills.
        max_chars_per_file: Максимальная длина текста одного ``SKILL.md``.

    Returns:
        Кортеж из общего markdown-context и списка виртуальных путей загруженных skills.
    """

    blocks: list[str] = []
    loaded_paths: list[str] = []
    for skill_path in discover_skill_context_files(skills_root):
        content = _read_context_file(skill_path, max_chars_per_file)
        if content is None:
            continue
        virtual_path = _virtual_skill_path(skills_root, skill_path, skills_virtual_dir)
        loaded_paths.append(virtual_path)
        blocks.append(f"### {virtual_path}\n\n{content}")
    return "\n\n".join(blocks), loaded_paths


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
    "discover_skill_context_files",
]

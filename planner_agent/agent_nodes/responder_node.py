"""Узел responder для подготовки финального отчета research-agent.

Содержит:
- Messages: константы служебных сообщений.
- ReportTemplate: шаблон markdown-отчета.
- PromptTemplate: шаблон human prompt для стартового контекста ReAct-агента.
- Formatting: константы форматирования.
- _collect_worker_evidence_blocks: блоки задач с приоритетом ``full_result`` для всех шагов плана.
- _collect_completed_worker_blocks: устаревшая обёртка для тестов совместимости.
- _collect_results: списки текстовых блоков для completed и failed задач (для тестов и интеграций).
- _format_task_for_responder: единая функция форматирования блока задачи; параметр ``include_details`` управляет набором полей.
- _fit_responder_context_budget: подгонка стартового human prompt под общий бюджет.
- _get_user_query: извлечение исходного запроса пользователя.
- _get_current_run_id: выбор актуального run id задач внутри плана.
- _get_latest_planning_error: поиск последней ошибки планирования.
- _format_human_message: сборка стартового HumanMessage для responder ReAct.
- _build_responder_artifact_names_context: список имён/метаданных artifacts по ссылкам из задач.
- _select_responder_artifacts: выбор artifacts по ссылкам из задач плана.
- _should_include_responder_artifact: проверка, нужен ли artifact в каталоге.
- _format_responder_context_artifact: сбор markdown-копии стартового контекста responder.
- _build_responder_react_tools: инструмент ``submit_final_report`` и artifact_* tools.
- _normalize_final_markdown: нормализация итогового markdown с заголовком отчёта.
- _format_responder_tool_calls_for_console: компактный вывод tool calls responder.
- _extract_responder_tool_calls_from_messages: извлечение tool calls responder с preview результатов.
- _format_fallback_message: fallback-отчет при ошибке генерации.
- responder_node: LangGraph-узел финального отчёта (LangGraph ReAct + чтение artifacts).
- _build_final_report_update: подготовка update для state.
- _create_final_report_lineage: запись final_report node и artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..models import AgentState, Task, TaskStatus
from ..schemas.artifacts import Artifact
from ..schemas.lineage import StateNode
from ..services.artifact_service import ArtifactService
from ..services.lineage_service import LineageService
from ..services.prompt_trace_service import write_prompt_trace, write_tool_calls_trace
from ..tools.artifact_read_tools import build_artifact_read_tools


RESPONDER_MAX_CHARS_PER_TASK = 3_000
RESPONDER_PROMPT_BUDGET_CHARS = 60_000
RESPONDER_CONSOLE_BLOCK_MAX_LENGTH = 80_000
RESPONDER_REACT_RECURSION_LIMIT = 40
RESPONDER_MAX_ARTIFACTS_IN_CONTEXT = 20
RESPONDER_ARTIFACT_SUMMARY_MAX_CHARS = 300
RESPONDER_GENERATED_CODE_MAX_CHARS = 4_000
RESPONDER_FALLBACK_COMPLETED_MAX_CHARS = 20_000


class SubmitFinalReportInput(BaseModel):
    """Аргументы инструмента финального отчёта."""

    report: str = Field(
        description="Полный markdown-отчёт пользователю (можно с корневым заголовком).",
    )


def _sort_plan_task_ids(plan: dict[str, Task]) -> list[str]:
    """Возвращает стабильно отсортированные id задач."""

    return sorted(plan.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))


class Messages:
    NO_OUTPUT = "No output"
    UNKNOWN_ERROR = "unknown"
    NO_VALIDATION = "n/a"
    NO_COMPLETED_TASKS = "No completed tasks"
    NO_ARTIFACT_NAMES = "No artifacts linked to completed tasks."
    NO_PLANNING_ERROR = "No planning error"
    PLANNING_FAILED_PREFIX = "Planning failed:"
    REPORT_GENERATION_FAILED = "Analysis completed, but report generation failed."


class ReportTemplate:
    HEADER = "# Analysis Report\n\n"
    ERROR_LABEL = "Error: "
    COMPLETED_TASKS_LABEL = "Completed worker outputs:\n"


class PromptTemplate:
    USER_QUERY = "<user_query>\n{query}\n</user_query>\n\n"
    WORKER_OUTPUTS = (
        "<worker_task_outputs>\n"
        "Приоритет текста: full_result; иначе result_preview; для failed — error_log.\n"
        "{tasks}\n"
        "</worker_task_outputs>\n\n"
    )
    ARTIFACT_NAMES = "<artifact_catalog>\n{artifact_names}\n</artifact_catalog>\n\n"
    PLANNING_ERROR = "<planning_error>\n{error}\n</planning_error>"


class Formatting:
    TASK_SEPARATOR = "\n---\n"
    SECTION_SEPARATOR = "\n\n"


async def _print_content_block(title: str, content: str) -> None:
    """Асинхронно выводит responder prompt context в консоль.

    Args:
        title: Заголовок диагностического блока.
        content: Текст system prompt или human context.

    Returns:
        ``None``. Функция выполняет только консольный вывод.
    """

    text = content or "(empty)"
    if len(text) > RESPONDER_CONSOLE_BLOCK_MAX_LENGTH:
        text = (
            text[:RESPONDER_CONSOLE_BLOCK_MAX_LENGTH]
            + "\n...[truncated for console only; model prompt may contain more]..."
        )
    border = "=" * min(max(len(title), 16), 80)
    print(f"\n{border}\n{title}\n{border}\n{text}\n", flush=True)




def _collect_results(
        plan: dict[str, Task],
        *,
        current_run_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Собирает блоки задач со статусами completed и failed для контекста responder.

    Параметр ``current_run_id`` зарезервирован для обратной совместимости; текущая
    реализация учитывает все записи ``plan`` с соответствующими статусами.

    Args:
        plan: Текущий runtime-план задач.
        current_run_id: Не используется.

    Returns:
        Пара ``(completed_blocks, failed_blocks)`` — списки строк для prompt.
    """

    _ = current_run_id
    completed: list[str] = []
    failed: list[str] = []
    for task_id in _sort_plan_task_ids(plan):
        task = plan[task_id]
        if task.status == TaskStatus.FAILED:
            failed.append(_format_task_for_responder(task_id, task))
        elif task.status == TaskStatus.COMPLETED:
            completed.append(_format_task_for_responder(task_id, task))
    return completed, failed


def _collect_worker_evidence_blocks(plan: dict[str, Task]) -> list[str]:
    """Собирает блоки по всем задачам плана: статус, id артефактов, приоритетно full_result.

    Args:
        plan: Текущий runtime-план задач.

    Returns:
        Список текстовых блоков для стартового контекста ReAct responder.
    """

    blocks: list[str] = []
    for task_id in _sort_plan_task_ids(plan):
        task = plan[task_id]
        lines = [
            f"Task {task_id}: {task.description}",
            f"Status: {task.status.value}",
        ]
        if task.artifact_refs:
            lines.append(f"Artifact ids: {task.artifact_refs}")
        primary = task.full_result or task.result_preview or task.error_log or ""
        if primary.strip():
            lines.append(f"Full worker output:\n{_limit_task_text(primary)}")
        else:
            lines.append("Full worker output: (empty)")
        blocks.append("\n".join(lines))
    return blocks


def _collect_completed_worker_blocks(plan: dict[str, Task]) -> list[str]:
    """Совместимость с тестами: не-failed задачи в формате ``_format_task_for_responder``."""

    return [
        _format_task_for_responder(task_id, plan[task_id], include_details=False)
        for task_id in _sort_plan_task_ids(plan)
        if plan[task_id].status != TaskStatus.FAILED
    ]


def _limit_task_text(text: str, max_chars: int = RESPONDER_MAX_CHARS_PER_TASK) -> str:
    """Ограничивает текст результата задачи для prompt responder.

    Args:
        text: Полный текст результата, ошибки или промежуточного вывода.
        max_chars: Максимальный размер возвращаемого текста.

    Returns:
        Текст в пределах бюджета ``max_chars`` с явной пометкой о truncation,
        если исходный текст был длиннее.
    """

    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + "\n...[task result truncated for responder context]..."
    )


def _append_task_text_field(
        lines: list[str],
        label: str,
        value: str | None,
        max_chars: int = RESPONDER_MAX_CHARS_PER_TASK,
) -> None:
    """Добавляет текстовое поле задачи в responder block, если оно непустое.

    Args:
        lines: Накопленный список строк блока задачи.
        label: Название поля для prompt.
        value: Значение поля задачи.
        max_chars: Максимальный размер значения.

    Returns:
        ``None``. Функция изменяет ``lines``.
    """

    if value:
        lines.append(f"{label}:\n{_limit_task_text(value, max_chars=max_chars)}")


def _format_task_for_responder(
        task_id: str,
        task: Task,
        *,
        include_details: bool = True,
) -> str:
    """Формирует блок задачи для контекста responder.

    Args:
        task_id: Идентификатор задачи в плане.
        task: Runtime-состояние задачи с результатами worker-а.
        include_details: Если ``True``, добавляет метаданные (статус, зависимости,
            retry count, evidence refs, error log). Если ``False`` — только суть
            задачи с результатами worker-а и артефактами (для успешных задач).

    Returns:
        Текстовый блок задачи для responder prompt.
    """

    lines = [f"Task {task_id}: {task.description}"]

    if include_details:
        lines.extend([
            f"Status: {task.status.value}",
            f"Dependencies: {task.dependencies or []}",
            f"Retry count: {task.retry_count}",
        ])

    if task.artifact_refs:
        lines.append(
            f"Artifact refs{' (ids)' if not include_details else ''}: {task.artifact_refs}"
        )
    if include_details and task.evidence_refs:
        lines.append(f"Evidence refs: {task.evidence_refs}")
    if include_details:
        _append_task_text_field(lines, "Error log", task.error_log)
    _append_task_text_field(lines, "Result preview", task.result_preview)
    if task.full_result and task.full_result != task.result_preview:
        _append_task_text_field(lines, "Full worker result", task.full_result)
    _append_task_text_field(
        lines,
        "Generated code",
        task.generated_code,
        max_chars=RESPONDER_GENERATED_CODE_MAX_CHARS,
    )
    return "\n".join(lines)


def _get_user_query(state: AgentState) -> str:
    """Возвращает исходный пользовательский запрос из состояния агента.

    Args:
        state: Финальное состояние агента перед responder.

    Returns:
        Текст исходного запроса или пустая строка.
    """

    if state.initial_user_query:
        return state.initial_user_query

    last_human = next(
        (
            msg for msg in reversed(state.messages)
            if isinstance(msg, HumanMessage)
        ),
        None,
    )
    return str(last_human.content) if last_human else ""


def _get_current_run_id(plan: dict[str, Task], fallback: str) -> str:
    """Определяет актуальный run id задач внутри текущего плана.

    Args:
        plan: Текущий план задач.
        fallback: Значение по умолчанию, если задачи не содержат run_id.

    Returns:
        Строковый идентификатор актуального прохода.
    """

    run_ids = [task.run_id for task in plan.values() if task.run_id]
    if not run_ids:
        return fallback

    numeric_run_ids = [
        int(run_id) for run_id in run_ids if run_id.isdigit()
    ]
    if numeric_run_ids:
        return str(max(numeric_run_ids))

    return run_ids[-1]


def _get_latest_planning_error(messages: list[object]) -> str:
    """Ищет последнюю ошибку планирования в истории сообщений.

    Args:
        messages: История LangChain messages.

    Returns:
        Текст ошибки планирования или пустая строка.
    """

    last_error = next(
        (
            msg for msg in reversed(messages)
            if isinstance(msg, AIMessage)
            and str(msg.content).startswith(Messages.PLANNING_FAILED_PREFIX)
        ),
        None,
    )
    return str(last_error.content) if last_error else ""


def _format_human_message(
        user_query: str,
        completed_text: str,
        artifact_names_text: str,
        planning_error: str,
) -> str:
    """Формирует стартовый human prompt для responder ReAct (до вызовов tools).

    Args:
        user_query: Исходный запрос пользователя.
        completed_text: Сводка выводов worker-ов по задачам (приоритет full_result).
        artifact_names_text: Каталог artifacts (id и метаданные без файлов).
        planning_error: Последняя ошибка планирования, если есть.

    Returns:
        Полный текст HumanMessage для первого шага responder.
    """

    return (
        PromptTemplate.USER_QUERY.format(query=user_query)
        + PromptTemplate.WORKER_OUTPUTS.format(tasks=completed_text)
        + PromptTemplate.ARTIFACT_NAMES.format(
            artifact_names=artifact_names_text or Messages.NO_ARTIFACT_NAMES,
        )
        + PromptTemplate.PLANNING_ERROR.format(
            error=planning_error or Messages.NO_PLANNING_ERROR
        )
    )


def _clip_section(text: str, max_chars: int, marker: str) -> str:
    """Обрезает секцию prompt до заданного размера с явной пометкой.

    Args:
        text: Исходный текст секции.
        max_chars: Максимальное количество символов.
        marker: Текст пометки об усечении.

    Returns:
        Исходный или усеченный текст.
    """

    if max_chars <= 0:
        return f"...[{marker}]..."
    if len(text) <= max_chars:
        return text
    suffix = f"\n...[{marker}]..."
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def _fit_responder_context_budget(
        *,
        user_query: str,
        completed_text: str,
        artifact_names_text: str,
        planning_error: str,
        max_prompt_chars: int = RESPONDER_PROMPT_BUDGET_CHARS,
) -> tuple[str, str]:
    """Подгоняет стартовый responder context под общий бюджет human prompt.

    Args:
        user_query: Исходный запрос пользователя.
        completed_text: Текст сводки worker-выводов по задачам.
        artifact_names_text: Текст каталога artifacts.
        planning_error: Текст ошибки планирования.
        max_prompt_chars: Верхняя граница human prompt в символах.

    Returns:
        Кортеж ``(completed_text, artifact_names_text)`` после мягкого усечения.
        Приоритет — блок worker outputs, затем каталог артефактов.
    """

    static_chars = len(
        _format_human_message(
            user_query=user_query,
            completed_text="",
            artifact_names_text="",
            planning_error=planning_error,
        )
    )
    budget = max(0, max_prompt_chars - static_chars)
    completed_budget = min(len(completed_text), int(budget * 0.78))
    artifact_budget = min(
        len(artifact_names_text),
        max(0, budget - completed_budget),
    )

    return (
        _clip_section(
            completed_text,
            completed_budget,
            "completed tasks truncated to keep responder prompt under budget",
        ),
        _clip_section(
            artifact_names_text,
            artifact_budget,
            "artifact names truncated to keep responder prompt under budget",
        ),
    )


def _format_artifact_names_lines(artifacts: list[Artifact]) -> str:
    """Формирует многострочный список имён и метаданных artifacts без чтения файлов."""

    lines: list[str] = []
    for artifact in artifacts:
        filename = Path(artifact.uri).name if artifact.uri else ""
        summary = _clip_section(
            artifact.summary or "",
            RESPONDER_ARTIFACT_SUMMARY_MAX_CHARS,
            "artifact summary truncated",
        )
        lines.append(
            f"- artifact_id={artifact.artifact_id} | kind={artifact.kind} | "
            f"summary={summary} | file={filename} | "
            f"mime_type={artifact.mime_type or ''}"
        )
    return "\n".join(lines)


def _build_responder_artifact_names_context(
        *,
        state: AgentState,
        artifact_service: ArtifactService | None,
) -> str:
    """Собирает названия/метаданные artifacts по ссылкам из задач плана (без чтения файлов)."""

    if artifact_service is None or not state.run_id:
        return ""

    artifacts = _select_responder_artifacts(
        state=state,
        artifact_service=artifact_service,
    )
    if not artifacts:
        return ""
    shown_artifacts = artifacts[:RESPONDER_MAX_ARTIFACTS_IN_CONTEXT]
    hidden_count = max(0, len(artifacts) - len(shown_artifacts))

    header = (
        "Artifacts linked from plan tasks (names and metadata only; "
        "file contents are not included — use artifact_* tools when needed)."
    )
    if hidden_count:
        header += f"\nHidden artifacts outside prompt budget: {hidden_count}."
    return header + "\n" + _format_artifact_names_lines(shown_artifacts)


def _select_responder_artifacts(
        *,
        state: AgentState,
        artifact_service: ArtifactService,
) -> list[Artifact]:
    """Выбирает artifacts по ссылкам из всех задач плана (порядок — по task id).

    Args:
        state: Состояние агента с планом и artifact refs.
        artifact_service: Сервис чтения artifacts.

    Returns:
        Упорядоченный список artifacts без дублей.
    """

    ordered_ids: list[str] = []
    for task_id in _sort_plan_task_ids(state.plan or {}):
        task = (state.plan or {})[task_id]
        for artifact_id in task.artifact_refs:
            if artifact_id and artifact_id not in ordered_ids:
                ordered_ids.append(artifact_id)

    stored = {
        artifact.artifact_id: artifact
        for artifact in artifact_service.list_artifacts(state.run_id)
    }
    selected = [
        stored[artifact_id]
        for artifact_id in ordered_ids
        if artifact_id in stored and _should_include_responder_artifact(stored[artifact_id])
    ]
    return sorted(selected, key=_artifact_priority)


def _should_include_responder_artifact(artifact: Artifact) -> bool:
    """Проверяет, стоит ли включать artifact в финальный responder context.

    Args:
        artifact: Artifact текущего run.

    Returns:
        ``True`` для данных, выдержек, таблиц и model output; ``False`` для
        технических tool traces, которые обычно не нужны в бизнес-отчете.
    """

    role = artifact.metadata.get("artifact_role")
    if artifact.kind == "tool_trace" or role == "tool_call_trace":
        return False
    return True


def _artifact_priority(artifact: Artifact) -> tuple[int, str]:
    """Возвращает приоритет artifact для загрузки в responder.

    Args:
        artifact: Artifact из текущего run.

    Returns:
        Кортеж сортировки: чем меньше первое число, тем выше приоритет.
    """

    role = artifact.metadata.get("artifact_role")
    if role == "captured_tool_result" or artifact.kind == "dataset":
        return 0, artifact.artifact_id
    if artifact.kind in {"table", "source_excerpt"}:
        return 1, artifact.artifact_id
    if artifact.kind == "model_output":
        return 2, artifact.artifact_id
    return 3, artifact.artifact_id


def _format_responder_context_artifact(
        *,
        user_query: str,
        completed_text: str,
        artifact_names_text: str,
        planning_error: str,
) -> str:
    """Собирает markdown-копию данных, переданных responder для финального отчета.

    Args:
        user_query: Исходный запрос пользователя.
        completed_text: Выводы worker-ов по успешным задачам.
        artifact_names_text: Список имён artifacts из успешных задач.
        planning_error: Последняя ошибка планирования или пустая строка.

    Returns:
        Markdown-документ, который можно сохранить как artifact и открыть при аудите
        финального ответа.
    """

    return (
        "# Responder Context\n\n"
        "## User Query\n\n"
        f"{user_query or ''}\n\n"
        "## Worker task outputs (initial context)\n\n"
        f"{completed_text or Messages.NO_COMPLETED_TASKS}\n\n"
        "## Artifact catalog\n\n"
        f"{artifact_names_text or Messages.NO_ARTIFACT_NAMES}\n\n"
        "## Planning Error\n\n"
        f"{planning_error or Messages.NO_PLANNING_ERROR}\n"
    )


def _normalize_final_markdown(raw: str) -> str:
    """Добавляет стандартный заголовок отчёта, если модель передала тело без него."""

    body = (raw or "").strip()
    if not body:
        return ReportTemplate.HEADER + Messages.NO_OUTPUT
    header_stripped = ReportTemplate.HEADER.strip()
    if body.startswith(header_stripped):
        return body + "\n" if not body.endswith("\n") else body
    return ReportTemplate.HEADER + body


def _build_submit_final_report_tool(submitted: list[str]) -> StructuredTool:
    """Инструмент завершения: одна финальная markdown-строка для пользователя."""

    def submit_final_report(report: str) -> str:
        """Отправь итоговый markdown-отчёт пользователю. Вызови ровно один раз в конце."""
        submitted.append(report)
        return json.dumps({"ok": True, "message": "Final report recorded."}, ensure_ascii=False)

    return StructuredTool.from_function(
        submit_final_report,
        name="submit_final_report",
        description=(
            "Submit the complete final markdown report for the user. Call exactly once "
            "when the analysis is ready, after you have used worker conclusions and any "
            "needed artifact_* tools. The report should answer the user query directly."
        ),
        args_schema=SubmitFinalReportInput,
    )


def _build_responder_react_tools(
        *,
        artifact_service: ArtifactService | None,
        run_id: str,
        submitted: list[str],
) -> list[Any]:
    """Собирает tools ReAct responder: финальный submit и чтение artifacts."""

    tools: list[Any] = [_build_submit_final_report_tool(submitted)]
    if artifact_service is not None and run_id:
        existing = {t.name for t in tools if getattr(t, "name", None)}
        for tool in build_artifact_read_tools(artifact_service=artifact_service, run_id=run_id):
            if tool.name not in existing:
                tools.append(tool)
                existing.add(tool.name)
    return tools


def _append_responder_react_policy(system_prompt: str) -> str:
    """Добавляет к системному промпту правила ReAct и обязательный submit_final_report."""

    appendix = """

<responder_react_execution>
Ты работаешь как ReAct-агент с инструментами.

1. В пользовательском сообщении уже есть сводка по задачам плана: приоритетно полные ответы worker-ов (full_result), статусы и списки artifact id.
2. Каталог artifacts даёт только id и метаданные. Чтобы увидеть данные внутри файла (выгрузки, длинные отчёты), вызывай инструменты artifact_preview, artifact_read_chunk, artifact_profile, artifact_sample, artifact_search, artifact_value_counts или artifact_list.
3. Не придумывай факты, которых нет в выводах worker-ов или в прочитанных через tools artifacts.
4. Когда отчёт готов, один раз вызови инструмент submit_final_report с полным markdown для пользователя (можно со своими заголовками; система добавит стандартный префикс при необходимости).
5. Не завершай работу только общим комментарием в тексте — итог должен быть передан через submit_final_report.
6. Не читай artifacts массово. Читай только те artifacts, которые нужны для
   проверки ключевых выводов, точных чисел, полей, ошибок или доказательств,
   явно важных для запроса пользователя. Если full_result worker-а уже
   достаточен и самодостаточен, формируй отчет вместо лишних tool calls по
   дублирующим техническим traces.
</responder_react_execution>
"""
    return system_prompt.rstrip() + appendix


def _extract_final_markdown_from_react(
        react_messages: list[Any],
        submitted: list[str],
) -> str:
    """Извлекает итоговый markdown из вызова submit_final_report или последнего AI-ответа."""

    if submitted:
        return _normalize_final_markdown(submitted[-1])
    last_ai = next(
        (m for m in reversed(react_messages) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai and str(last_ai.content).strip():
        tool_calls = getattr(last_ai, "tool_calls", None) or []
        if not tool_calls:
            return _normalize_final_markdown(str(last_ai.content))
    raise ValueError("Responder ReAct did not call submit_final_report and left no final text.")


def _format_responder_tool_calls_for_console(react_messages: list[Any]) -> str:
    """Формирует компактный отчёт по вызовам инструментов responder ReAct.

    Args:
        react_messages: Сообщения, возвращённые ``create_react_agent().ainvoke``.

    Returns:
        Многострочный текст для консольного блока.
    """

    lines: list[str] = []
    for msg in react_messages:
        if not isinstance(msg, AIMessage):
            continue
        for raw_call in getattr(msg, "tool_calls", None) or []:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "unknown")
            args = raw_call.get("args")
            args_text = (
                json.dumps(args, ensure_ascii=False, indent=2)
                if isinstance(args, dict)
                else str(args)
            )
            lines.append(f"- tool: {name}\n  args: {args_text}")

    if not lines:
        return "Responder did not call any tools."
    return "\n".join(lines)


def _extract_responder_tool_calls_from_messages(
        react_messages: list[Any],
) -> list[dict[str, Any]]:
    """Извлекает фактические вызовы responder tools из ReAct-сообщений.

    Args:
        react_messages: Список сообщений, возвращенных ``create_react_agent``.

    Returns:
        Список словарей с именем инструмента, аргументами, id вызова и preview
        результата инструмента.
    """

    pending_meta: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for msg in react_messages:
        if isinstance(msg, AIMessage):
            for raw_call in getattr(msg, "tool_calls", None) or []:
                if not isinstance(raw_call, dict):
                    continue
                call_id = raw_call.get("id") or raw_call.get("tool_call_id")
                if not call_id:
                    continue
                pending_meta[str(call_id)] = {
                    "tool_name": raw_call.get("name"),
                    "arguments": raw_call.get("args"),
                }
            continue

        if isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None)
            call_key = str(call_id) if call_id is not None else ""
            meta = pending_meta.pop(call_key, {}) if call_key else {}
            records.append(
                {
                    "tool_call_id": call_id,
                    "tool_name": getattr(msg, "name", None) or meta.get("tool_name"),
                    "arguments": meta.get("arguments"),
                    "tool_result_preview": _clip_section(
                        str(msg.content),
                        RESPONDER_MAX_CHARS_PER_TASK,
                        "responder tool result truncated",
                    ),
                }
            )
    return records


def _format_fallback_message(
        exc: Exception,
        completed_text: str,
) -> str:
    """Формирует fallback-отчет, если responder ReAct не смог сформировать отчёт.

    Args:
        exc: Ошибка генерации или парсинга финального отчета.
        completed_text: Текст сводки worker-результатов.

    Returns:
        Markdown-строка fallback-отчета.
    """

    return (
        Messages.REPORT_GENERATION_FAILED
        + Formatting.SECTION_SEPARATOR
        + ReportTemplate.ERROR_LABEL
        + str(exc)
        + Formatting.SECTION_SEPARATOR
        + ReportTemplate.COMPLETED_TASKS_LABEL
        + _clip_section(
            completed_text,
            RESPONDER_FALLBACK_COMPLETED_MAX_CHARS,
            "fallback completed tasks truncated",
        )
    )


async def responder_node(
        state: AgentState,
        llm: BaseChatModel,
        prompt: str,
        lineage_service: LineageService | None = None,
        artifact_service: ArtifactService | None = None,
) -> Command:
    """Генерирует финальный отчёт через ReAct-агента (tools: artifacts + submit_final_report).

    Args:
        state: Текущее состояние AgentState.
        llm: Chat model для подготовки отчета.
        prompt: Системный prompt responder.
        lineage_service: Опциональный сервис lineage.
        artifact_service: Опциональный сервис artifacts.

    Returns:
        Command с финальным сообщением, ``state.final_report`` и lineage node.
    """

    plan = state.plan
    messages = state.messages

    user_query = _get_user_query(state)
    planning_error = _get_latest_planning_error(messages)

    evidence_blocks = _collect_worker_evidence_blocks(plan)
    completed_text = (
        Formatting.TASK_SEPARATOR.join(evidence_blocks)
        if evidence_blocks
        else Messages.NO_COMPLETED_TASKS
    )
    artifact_names_text = _build_responder_artifact_names_context(
        state=state,
        artifact_service=artifact_service,
    )
    completed_text, artifact_names_text = _fit_responder_context_budget(
        user_query=user_query,
        completed_text=completed_text,
        artifact_names_text=artifact_names_text,
        planning_error=planning_error,
    )

    human_message = _format_human_message(
        user_query=user_query,
        completed_text=completed_text,
        artifact_names_text=artifact_names_text,
        planning_error=planning_error,
    )
    responder_context = _format_responder_context_artifact(
        user_query=user_query,
        completed_text=completed_text,
        artifact_names_text=artifact_names_text,
        planning_error=planning_error,
    )
    system_prompt = _append_responder_react_policy(prompt)
    submitted_reports: list[str] = []
    tools = _build_responder_react_tools(
        artifact_service=artifact_service,
        run_id=state.run_id,
        submitted=submitted_reports,
    )

    await _print_content_block("RESPONDER SYSTEM PROMPT", system_prompt)
    await _print_content_block("RESPONDER HUMAN CONTEXT", human_message)
    prompt_trace_artifacts = write_prompt_trace(
        artifact_service=artifact_service,
        run_id=state.run_id,
        node_id=state.current_node_id,
        stage="responder",
        system_prompt=system_prompt,
        human_prompt=human_message,
        payload={
            "run_id": state.run_id,
            "user_query": user_query,
            "planning_error": planning_error,
            "artifact_catalog": artifact_names_text,
            "task_count": len(plan or {}),
        },
    )

    try:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )
        react_state = await agent.ainvoke(
            {"messages": [HumanMessage(content=human_message)]},
            config={"recursion_limit": RESPONDER_REACT_RECURSION_LIMIT},
        )
        react_messages = react_state.get("messages", [])
        await _print_content_block(
            "RESPONDER TOOL CALLS",
            _format_responder_tool_calls_for_console(react_messages),
        )
        tool_call_artifacts = write_tool_calls_trace(
            artifact_service=artifact_service,
            run_id=state.run_id,
            node_id=state.current_node_id,
            stage="responder",
            tool_calls=_extract_responder_tool_calls_from_messages(react_messages),
        )
        if tool_call_artifacts:
            prompt_trace_artifacts.update(tool_call_artifacts)
        final_md = _extract_final_markdown_from_react(react_messages, submitted_reports)
        return Command(
            update=_build_final_report_update(
                state,
                final_md,
                lineage_service,
                artifact_service,
                responder_context=responder_context,
                prompt_trace_artifacts=prompt_trace_artifacts,
            )
        )

    except Exception as exc:
        fallback = _format_fallback_message(
            exc=exc,
            completed_text=completed_text,
        )
        return Command(
            update=_build_final_report_update(
                state,
                fallback,
                lineage_service,
                artifact_service,
                responder_context=responder_context,
                prompt_trace_artifacts=prompt_trace_artifacts,
            )
        )


def _build_final_report_update(
        state: AgentState,
        final_report: str,
        lineage_service: LineageService | None,
        artifact_service: ArtifactService | None,
        responder_context: str = "",
        prompt_trace_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собирает update payload после генерации финального отчета.

    Args:
        state: Текущее состояние агента.
        final_report: Финальный отчет в markdown.
        lineage_service: Опциональный сервис lineage.
        artifact_service: Опциональный сервис artifacts.
        responder_context: Контекст, который был передан responder при создании отчета.

    Returns:
        Словарь update для LangGraph state.
    """

    update: dict[str, Any] = {
        "messages": [AIMessage(content=final_report)],
        "final_report": final_report,
    }
    if prompt_trace_artifacts:
        update["artifact_index"] = prompt_trace_artifacts
    lineage_event = _create_final_report_lineage(
        state=state,
        final_report=final_report,
        lineage_service=lineage_service,
        artifact_service=artifact_service,
        responder_context=responder_context,
    )
    if lineage_event:
        update["current_node_id"] = lineage_event["node_id"]
        update["parent_node_ids"] = [lineage_event["node_id"]]
        update["lineage_events"] = [lineage_event]
    return update


def _create_final_report_lineage(
        *,
        state: AgentState,
        final_report: str,
        lineage_service: LineageService | None,
        artifact_service: ArtifactService | None,
        responder_context: str = "",
) -> dict[str, Any] | None:
    """Создает lineage node и report artifact для финального отчета.

    Args:
        state: Текущее состояние агента.
        final_report: Финальный отчет в markdown.
        lineage_service: Сервис lineage или ``None``.
        artifact_service: Сервис artifacts или ``None``.
        responder_context: Markdown-копия входного контекста responder.

    Returns:
        JSON-представление StateNode или ``None``.
    """

    if lineage_service is None or not state.run_id:
        return None

    parent_ids = state.parent_node_ids or (
        [state.current_node_id] if state.current_node_id else []
    )
    node = StateNode(
        run_id=state.run_id,
        node_type="final_report",
        title="Final report",
        parent_ids=parent_ids,
        status="succeeded",
        summary=final_report[:500],
        created_by="agent",
        metadata={
            "report_chars": len(final_report),
            "completed_tasks": [
                task_id
                for task_id, task in state.plan.items()
                if task.status == TaskStatus.COMPLETED
            ],
            "failed_tasks": [
                task_id
                for task_id, task in state.plan.items()
                if task.status == TaskStatus.FAILED
            ],
        },
    )
    artifact_index: dict[str, Any] = dict(state.artifact_index or {})
    artifact_refs: list[str] = []
    if artifact_service is not None:
        artifact = artifact_service.write_artifact(
            run_id=state.run_id,
            node_id=node.node_id,
            kind="report",
            filename="final_report.md",
            content=final_report,
            mime_type="text/markdown",
            summary=final_report[:500],
            metadata={"node_type": "final_report"},
        )
        artifact_refs.append(artifact.artifact_id)
        artifact_index[artifact.artifact_id] = artifact.model_dump(mode="json")
        if responder_context:
            context_artifact = artifact_service.write_artifact(
                run_id=state.run_id,
                node_id=node.node_id,
                kind="model_output",
                filename="final_report_context.md",
                content=responder_context,
                mime_type="text/markdown",
                summary="Context package used by responder to build the final report.",
                metadata={
                    "node_type": "final_report",
                    "artifact_role": "responder_context",
                },
            )
            artifact_refs.append(context_artifact.artifact_id)
            artifact_index[context_artifact.artifact_id] = context_artifact.model_dump(mode="json")

    node.artifact_refs = artifact_refs
    snapshot = state.model_copy(
        update={
            "final_report": final_report,
            "artifact_index": artifact_index,
            "current_node_id": node.node_id,
            "parent_node_ids": [node.node_id],
        },
        deep=True,
    )
    lineage_service.append_node(node, state=snapshot)
    return node.model_dump(mode="json")

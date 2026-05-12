"""
Модуль содержит узел критика результата worker-а для исследовательского графа.

Содержит:
- critic_node: анализирует результат одной worker-задачи перед validator.
- _build_worker_critic_human_prompt: формирует входной prompt для LLM-критика.
- _format_available_tools_for_critic: подготавливает список доступных tools.
- _format_tool_traces_for_critic: подготавливает имя tool и аргументы без вывода.
- _format_react_message_tool_calls_for_critic: вызовы из AIMessage/ToolMessage.
- _format_artifacts_for_critic: data-artifacts без служебных prompt/tool traces.
- _create_worker_critic_lineage: сохраняет critic node в lineage.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command, Send

from ..models import (
    CriticPayload,
    Task,
    TaskStatus,
    ValidatorPayload,
    WorkerCriticReview,
)
from ..services.artifact_service import ArtifactService
from ..services.lineage_service import LineageService
from ..services.prompt_trace_service import write_prompt_trace
from ..structured_output import invoke_structured_output

GOTO_VALIDATOR: Final[str] = "validator"
GOTO_WORKER: Final[str] = "worker"
CRITIC_NODE_TYPE: Final[str] = "worker_critic"
MAX_CRITIC_RETRIES: Final[int] = 1
MAX_TEXT_PREVIEW: Final[int] = 10_700
MAX_WORKER_FULL_RESULT_CHARS: Final[int] = 300_000
MAX_TOOL_TRACES: Final[int] = 8
MAX_REACT_MESSAGE_TOOL_CALLS: Final[int] = 50
MAX_ARTIFACTS: Final[int] = 12
# Артефакты prompt/tool trace пишутся в тот же artifact_index, что и данные worker-а;
# их summary содержит копию system prompt — в critic их не показываем.
_PROMPT_DIAGNOSTIC_ARTIFACT_ROLES: Final[frozenset[str]] = frozenset(
    {"prompt_trace", "prompt_payload", "tool_calls_trace"},
)


async def critic_node(
        payload: CriticPayload,
        llm: BaseChatModel,
        prompt: str,
        tools: list[BaseTool] | None = None,
        artifact_service: ArtifactService | None = None,
        lineage_service: LineageService | None = None,
) -> Command:
    """Критикует результат одной worker-задачи перед validator.

    Args:
        payload: Контекст завершенной worker-задачи и ее результата.
        llm: Языковая модель для структурированной критики результата.
        prompt: Системный prompt critic-узла.
        tools: Список реально доступных worker-инструментов.
        lineage_service: Опциональный сервис записи lineage.

    Returns:
        Command с переходом к повторному worker-запуску или validator.
    """

    task = payload.worker_payload.task
    task_id = task.task_id or "unknown_task"
    schema_str = json.dumps(
        WorkerCriticReview.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    system_prompt = prompt.format(schema_str=schema_str)
    available_tools = tools or []
    human_prompt = _build_worker_critic_human_prompt(
        payload=payload,
        tools=available_tools,
    )

    try:
        review = await invoke_structured_output(
            llm=llm,
            schema=WorkerCriticReview,
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ],
        )
    except Exception as exc:
        review = WorkerCriticReview(
            approved=True,
            reasoning=(
                "Critic не смог получить структурированный ответ; "
                f"результат передается validator-у. Ошибка: {str(exc)[:MAX_TEXT_PREVIEW]}"
            ),
            issues=[],
            improvement_instructions="",
        )
    review = _sanitize_review_tool_references(review, available_tools)

    feedback = review.model_dump(mode="json")
    feedback["task_id"] = task_id
    feedback["critic_retry_count"] = task.retry_count
    lineage_event = _create_worker_critic_lineage(
        payload=payload,
        task=task,
        feedback=feedback,
        lineage_service=lineage_service,
    )

    update: dict[str, Any] = {"plan": {task_id: task}}
    prompt_trace_artifacts = write_prompt_trace(
        artifact_service=artifact_service,
        run_id=payload.run_id or payload.worker_payload.run_id,
        node_id=(lineage_event or {}).get("node_id"),
        stage="critic",
        system_prompt=system_prompt,
        human_prompt=human_prompt,
        payload={
            "task": task.model_dump(mode="json"),
            "initial_user_query": payload.worker_payload.initial_user_query,
            "tool_traces": payload.tool_traces,
            "artifact_index": payload.artifact_index,
            "react_message_tool_calls": payload.react_message_tool_calls,
        },
    )
    if prompt_trace_artifacts:
        update["artifact_index"] = prompt_trace_artifacts
    if not review.approved or review.issues:
        update["feedback_context"] = [feedback]
    if lineage_event:
        # Параллельные critic после scheduler не должны писать current_node_id/parent_node_ids
        # в общее состояние (LangGraph InvalidUpdateError); связь через Send payloads и lineage_events.
        update["lineage_events"] = [lineage_event]

    should_retry = (
        not review.approved
        and task.retry_count < MAX_CRITIC_RETRIES
        and bool(review.improvement_instructions.strip())
    )
    if should_retry:
        previous_result = task.full_result or task.result_preview or ""
        task.retry_count += 1
        task.status = TaskStatus.READY
        task.validation_passed = None
        task.validation_reason = None
        task.validation_score = None
        task.error_log = _build_worker_retry_message(review)
        task.result_preview = None
        task.full_result = None
        task.generated_code = None
        task.output_variable_name = None
        task.config = {
            **(task.config or {}),
            "critic_feedback": task.error_log,
            "critic_retry_count": task.retry_count,
            "previous_worker_result_before_critic_retry": previous_result,
        }
        _merge_retry_artifact_context(payload)
        payload.worker_payload.parent_node_ids = (
            [lineage_event["node_id"]] if lineage_event else payload.parent_node_ids
        )
        return Command(
            update={
                "plan": {task_id: task},
                **{k: v for k, v in update.items() if k != "plan"},
            },
            goto=[Send(GOTO_WORKER, payload.worker_payload)],
        )

    return Command(
        update=update,
        goto=[
            Send(
                GOTO_VALIDATOR,
                ValidatorPayload(
                    task=task,
                    run_id=payload.run_id,
                    parent_node_ids=(
                        [lineage_event["node_id"]]
                        if lineage_event
                        else payload.parent_node_ids
                    ),
                ),
            )
        ],
    )


def _build_worker_retry_message(review: WorkerCriticReview) -> str:
    """Формирует краткую инструкцию для повторного запуска worker-а.

    Args:
        review: Структурированный вердикт critic-а.

    Returns:
        Текст, который будет передан worker-у как critic feedback.
    """

    lines = ["Critic requested retry before validation."]
    if review.reasoning:
        lines.append(f"Reasoning: {review.reasoning}")
    if review.issues:
        lines.append("Issues:")
        lines.extend(f"- {item}" for item in review.issues)
    if review.improvement_instructions:
        lines.append(f"Instructions: {review.improvement_instructions}")
    return "\n".join(lines)


def _sanitize_review_tool_references(
        review: WorkerCriticReview,
        tools: list[BaseTool],
) -> WorkerCriticReview:
    """Удаляет из critic feedback имена недоступных инструментов.

    Args:
        review: Структурированный результат critic-а до нормализации.
        tools: Список инструментов, реально доступных worker-у.

    Returns:
        Новый ``WorkerCriticReview`` без ссылок на инструменты, которых нет в
        текущем списке ``tools``. Если доступные инструменты есть, подозрительные
        tool-like имена заменяются первым доступным именем, чтобы retry не
        планировал несуществующий tool.
    """

    available_names = [tool.name for tool in tools if getattr(tool, "name", "")]
    if not available_names:
        return review

    replacement = available_names[0]

    def sanitize_text(text: str) -> str:
        result = str(text or "")
        for token in set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", result)):
            looks_like_tool = token.startswith("spark_") or token.endswith("_tool")
            if looks_like_tool and token not in available_names:
                result = result.replace(token, replacement)
        return result

    return review.model_copy(
        update={
            "reasoning": sanitize_text(review.reasoning),
            "issues": [sanitize_text(issue) for issue in review.issues],
            "improvement_instructions": sanitize_text(
                review.improvement_instructions
            ),
        },
        deep=True,
    )


def _limit_critic_text(text: str, max_chars: int) -> str:
    """Ограничивает текст для critic prompt с явной пометкой.

    Args:
        text: Исходный текст.
        max_chars: Максимальное количество символов.

    Returns:
        Текст в заданном бюджете с пометкой, если он был обрезан.
    """

    if len(text) <= max_chars:
        return text

    truncated_marker = (
        "\n\n[SYSTEM TRUNCATION: The text above was cut off by the critic system "
        "because the full worker result exceeded the prompt budget "
        f"({len(text):,} chars total, only {max_chars:,} shown). "
        "This is NOT a worker error — the original worker response was complete. "
        "Do NOT mark as incomplete based on this truncation alone. "
        "Evaluate the visible content on its merits.]"
    )
    return text[:max_chars] + truncated_marker


def _merge_retry_artifact_context(payload: CriticPayload) -> None:
    """Добавляет artifacts последнего worker-запуска в payload для retry.

    Args:
        payload: Нагрузка critic-а с исходным WorkerPayload и новыми artifacts.

    Returns:
        ``None``. Функция изменяет ``payload.worker_payload.artifact_context``.
    """

    if not payload.artifact_index:
        return

    artifact_context = dict(payload.worker_payload.artifact_context or {})
    artifacts = dict(artifact_context.get("artifacts") or {})
    artifacts.update(payload.artifact_index)
    artifact_context["artifacts"] = artifacts
    artifact_context["total_available_artifacts"] = len(artifacts)
    artifact_context["shown_artifacts"] = len(artifacts)
    payload.worker_payload.artifact_context = artifact_context


def _build_worker_critic_human_prompt(
        *,
        payload: CriticPayload,
        tools: list[BaseTool],
) -> str:
    """Формирует human prompt с результатом worker-а и доступным контекстом.

    Args:
        payload: Данные последнего worker-запуска.
        tools: Список реально доступных worker-инструментов.

    Returns:
        Текстовый prompt для critic LLM.
    """

    worker_payload = payload.worker_payload
    task = worker_payload.task
    worker_answer = task.full_result or task.result_preview or ""
    worker_answer_source = "full_result" if task.full_result else "result_preview_fallback"
    worker_answer_limited = _limit_critic_text(
        worker_answer,
        MAX_WORKER_FULL_RESULT_CHARS,
    )
    return "\n\n".join(
        [
            
            f"Task ID: {task.task_id}",
            f"Исходный запрос пользователя:\n{worker_payload.initial_user_query or '(empty)'}",
            f"Описание задачи:\n{task.description}",
            f"Expected output:\n{task.expected_output or '(empty)'}",
            f"Dependency context:\n{json.dumps(worker_payload.dependency_context, ensure_ascii=False)[:MAX_TEXT_PREVIEW]}",
            f"Worker status: {task.status.value}",
            f"Worker answer source: {worker_answer_source}",
            f"Worker answer:\n{worker_answer_limited}",
            f"Worker error log:\n{(task.error_log or '')[:MAX_TEXT_PREVIEW]}",
            f"Доступные инструменты:\n{_format_available_tools_for_critic(tools)}",
            (
                "Фактические вызовы инструментов из истории сообщений ReAct "
                "(AIMessage.tool_calls + ToolMessage):\n"
                f"{_format_react_message_tool_calls_for_critic(payload.react_message_tool_calls)}"
            ),
            f"Tool traces (artifact wrapper / состояние):\n{_format_tool_traces_for_critic(payload.tool_traces)}",
            f"Artifacts последнего запуска:\n{_format_artifacts_for_critic(payload.artifact_index)}",
        ]
    )


def _format_available_tools_for_critic(tools: list[BaseTool]) -> str:
    """Форматирует список доступных инструментов для critic prompt.

    Args:
        tools: Список реально зарегистрированных worker-инструментов.

    Returns:
        Многострочная строка с именами и краткими описаниями инструментов.
    """

    if not tools:
        return "(нет доступных инструментов)"
    return "\n".join(
        f"- {tool.name}: {str(tool.description or '')}"
        for tool in tools
    )


def _format_plan_for_critic(plan: dict[str, Task]) -> str:
    """Форматирует задачи для critic prompt.

    Args:
        plan: Словарь задач.

    Returns:
        Многострочное компактное описание задач.
    """

    if not plan:
        return "(empty)"

    lines: list[str] = []
    for task_id, task in plan.items():
        content = (
            f"task_id={task_id}; status={task.status.value}; "
            f"deps={task.dependencies}; tools={task.suggested_tools}; "
            f"skills={task.suggested_skills}; description={task.description}; "
            f"result_preview={(task.result_preview or '')[:MAX_TEXT_PREVIEW]}; "
            f"full_result={(task.full_result or '')[:MAX_TEXT_PREVIEW]}; "
            f"validation={task.validation_passed}; "
            f"validation_reason={(task.validation_reason or '')[:MAX_TEXT_PREVIEW]}; "
            f"error={(task.error_log or '')[:MAX_TEXT_PREVIEW]}"
        )
        lines.append(_wrap_critic_prompt_block("TASK", task_id, content))
    return "\n".join(lines)


def _wrap_critic_prompt_block(kind: str, block_id: str, content: str) -> str:
    """РћР±РѕСЂР°С‡РёРІР°РµС‚ Р±Р»РѕРє critic prompt РІ РїР°СЂРЅС‹Рµ С‚РµРіРё.

    Args:
        kind: РўРёРї Р±Р»РѕРєР° РґР»СЏ С‚РµРіР°, РЅР°РїСЂРёРјРµСЂ ``TASK`` РёР»Рё ``ARTIFACT``.
        block_id: РРґРµРЅС‚РёС„РёРєР°С‚РѕСЂ Р±Р»РѕРєР°.
        content: РўРµРєСЃС‚РѕРІРѕРµ СЃРѕРґРµСЂР¶РёРјРѕРµ Р±Р»РѕРєР°.

    Returns:
        РњРЅРѕРіРѕСЃС‚СЂРѕС‡РЅР°СЏ СЃС‚СЂРѕРєР° СЃ РѕС‚РєСЂС‹РІР°СЋС‰РёРј Рё Р·Р°РєСЂС‹РІР°СЋС‰РёРј С‚РµРіРѕРј.
    """

    tag_kind = str(kind).strip().upper() or "BLOCK"
    tag_id = str(block_id).strip() or "unknown"
    return f"<{tag_kind} {tag_id}>\n{content.strip()}\n</{tag_kind} {tag_id}>"


def _format_react_message_tool_calls_for_critic(
        records: list[dict[str, Any]],
) -> str:
    """Форматирует вызовы инструментов, извлечённые из сообщений ReAct-агента.

    Args:
        records: Элементы с ``tool_name``, ``arguments`` (без вывода инструмента).

    Returns:
        Многострочное описание для critic LLM.
    """

    if not records:
        return "(пусто — инструменты не вызывались или история недоступна)"

    shown = records[-MAX_REACT_MESSAGE_TOOL_CALLS:]
    start_idx = len(records) - len(shown) + 1
    lines: list[str] = []
    for i, rec in enumerate(shown, start=start_idx):
        args = rec.get("arguments")
        try:
            args_str = json.dumps(args, ensure_ascii=False)
        except TypeError:
            args_str = str(args)
        if len(args_str) > MAX_TEXT_PREVIEW:
            args_str = f"{args_str[:MAX_TEXT_PREVIEW]}..."
        result_text = str(rec.get("tool_result_preview") or "")
        lines.append(
            (
                f"{i}. tool={rec.get('tool_name')}; "
                f"tool_call_id={rec.get('tool_call_id')}; "
                f"arguments={args_str}; "
                f"result={result_text}"
            )
        )
    if len(records) > MAX_REACT_MESSAGE_TOOL_CALLS:
        lines.append(
            f"... [показаны последние {MAX_REACT_MESSAGE_TOOL_CALLS} из {len(records)} вызовов]"
        )
    return "\n".join(lines)


def _format_tool_traces_for_critic(tool_traces: list[dict[str, Any]]) -> str:
    """Форматирует последние tool traces для critic prompt (имя + параметры, без вывода).

    Args:
        tool_traces: Список trace-записей из состояния агента.

    Returns:
        Компактные строки с ``tool_name`` и аргументами вызова.
    """

    if not tool_traces:
        return "(empty)"

    lines: list[str] = []
    for trace in tool_traces[-MAX_TOOL_TRACES:]:
        if not isinstance(trace, dict):
            continue
        slim: dict[str, Any] = {}
        for key in ("trace_id", "tool_name", "args_preview"):
            if trace.get(key) is not None:
                slim[key] = trace[key]
        lines.append(json.dumps(slim, ensure_ascii=False)[:MAX_TEXT_PREVIEW])
    return "\n".join(lines)


def _is_internal_prompt_trace_artifact(artifact: dict[str, Any]) -> bool:
    """True, если artifact — служебный prompt/tool trace (не данные задачи).

    ``write_prompt_trace`` / ``write_tool_calls_trace`` кладут в индекс записи
    с ``metadata.artifact_role``; их ``summary`` дублирует system/human prompt
    worker-а, поэтому блок «Artifacts последнего запуска» в critic их не включает.
    """

    metadata = artifact.get("metadata")
    if isinstance(metadata, dict):
        role = metadata.get("artifact_role")
        if role in _PROMPT_DIAGNOSTIC_ARTIFACT_ROLES:
            return True
    uri = str(artifact.get("uri") or "").replace("\\", "/").lower()
    filename = str(artifact.get("filename") or "").replace("\\", "/").lower()
    return "prompt_traces/" in uri or "prompt_traces/" in filename


def _format_artifacts_for_critic(artifact_index: dict[str, Any]) -> str:
    """Форматирует список artifacts для critic prompt.

    Args:
        artifact_index: Индекс artifacts из состояния агента.

    Returns:
        Многострочное описание ключевых artifacts.
    """

    if not artifact_index:
        return "(empty)"

    selected: list[tuple[str, dict[str, Any]]] = []
    for artifact_id, artifact in reversed(list(artifact_index.items())):
        if not isinstance(artifact, dict):
            continue
        if _is_internal_prompt_trace_artifact(artifact):
            continue
        selected.append((artifact_id, artifact))
        if len(selected) >= MAX_ARTIFACTS:
            break
    selected.reverse()

    if not selected:
        return "(empty — только служебные prompt/tool traces, без data artifacts)"

    lines: list[str] = []
    for artifact_id, artifact in selected:
        metadata = artifact.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        content = (
            f"artifact_id={artifact_id}; kind={artifact.get('kind')}; "
            f"uri={artifact.get('uri')}; task_id={metadata.get('task_id')}; "
            f"tool_name={metadata.get('tool_name')}; "
            f"summary={str(artifact.get('summary') or '')[:MAX_TEXT_PREVIEW]}"
        )
        lines.append(_wrap_critic_prompt_block("ARTIFACT", artifact_id, content))
    return "\n".join(lines)


def _create_worker_critic_lineage(
        *,
        payload: CriticPayload,
        task: Task,
        feedback: dict[str, Any],
        lineage_service: LineageService | None,
) -> dict[str, Any] | None:
    """Создает lineage node для результата critic.

    Args:
        payload: Контекст завершенной worker-задачи.
        task: Проверяемая worker-задача.
        feedback: JSON-совместимый результат critic.
        lineage_service: Сервис lineage или ``None``.

    Returns:
        JSON-совместимое представление StateNode или ``None``.
    """

    run_id = payload.run_id or payload.worker_payload.run_id
    if lineage_service is None or not run_id:
        return None

    parent_ids = payload.parent_node_ids
    task_id = task.task_id or "unknown_task"
    node = lineage_service.create_state_node(
        run_id=run_id,
        node_type=CRITIC_NODE_TYPE,
        title=f"Worker critic: task {task_id}",
        parent_ids=parent_ids,
        status="succeeded",
        summary=str(feedback.get("reasoning") or "")[:500],
        state={
            "run_id": run_id,
            "task": task.model_dump(mode="json"),
            "initial_user_query": payload.worker_payload.initial_user_query,
            "critic_review": feedback,
            "artifact_index": payload.artifact_index,
            "tool_traces": payload.tool_traces,
            "react_message_tool_calls": payload.react_message_tool_calls,
        },
        created_by="agent",
        metadata={
            "task_id": task_id,
            "approved": feedback.get("approved"),
            "critic_retry_count": task.retry_count,
            "issue_count": len(feedback.get("issues") or []),
        },
    )
    return node.model_dump(mode="json")


__all__ = ["critic_node"]

"""

Модуль содержит узел валидатора (validator_node) для LangGraph-агента.

Валидатор проверяет результат выполнения задачи (Task) воркером:
- при статусе FAILED/SKIPPED обрабатывает задачу без вызова LLM;
- в остальных случаях вызывает LLM для структурированной оценки результата;
 - возвращает команду Command с обновлённым планом и переходом к реплановщику.
"""

from typing import Any, Final

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from ..models import StepValidation, Task, TaskStatus, ValidatorPayload
from ..services.artifact_service import ArtifactService
from ..services.lineage_service import LineageService
from ..services.prompt_trace_service import write_prompt_trace
from ..structured_output import invoke_structured_output

_FALLBACK_TASK_ID: Final[str] = "unknown_task"
_FALLBACK_REASON: Final[str] = "(empty)"
_FALLBACK_CONFIG: Final[str] = "{}"

_SCORE_ZERO: Final[float] = 0.0
_SCORE_FULL: Final[float] = 1.0
_REJECTION_CONFIDENCE_THRESHOLD: Final[float] = 0.75

_PREVIEW_MAX_LEN: Final[int] = 6000

_GOTO_REPLANNER: Final[str] = "replanner"

_REASON_FAILED: Final[str] = "Ошибка выполнения Worker"
_REASON_SKIPPED: Final[str] = "Задача была намеренно пропущена"
_REASON_CRASHED: Final[str] = "Сбой валидатора: {exc}"
_REASON_VALIDATION_FAILED: Final[str] = "Ошибка валидации: {reasoning}"
_REASON_SOFT_ACCEPTED: Final[str] = (
    "Мягкая валидация: validator не нашел достаточно уверенного основания "
    "для блокировки результата. Исходное замечание: {reasoning}"
)
_REASON_VALIDATOR_UNAVAILABLE: Final[str] = (
    "Мягкая валидация: validator не смог получить структурированный ответ; "
    "результат передан дальше без переписывания задачи. Ошибка: {exc}"
)
_CRITIC_CONFIG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "critic_feedback",
        "critic_retry_count",
        "previous_worker_result_before_critic_retry",
    }
)


def _build_human_prompt(task: Task) -> str:
    """
    Формирует текст пользовательского сообщения для LLM-валидатора.

    Args:
        task: задача с описанием, конфигурацией и результатами выполнения воркером.

    Returns:
        Строка с промптом для HumanMessage.
    """
    worker_preview = task.result_preview or ""
    worker_full_result = task.full_result or ""
    worker_error_log = task.error_log or ""
    sanitized_config = {
        key: value
        for key, value in (task.config or {}).items()
        if key not in _CRITIC_CONFIG_KEYS
    }

    return (
        f"Описание задачи: {task.description}\n"
        f"Конфигурация задачи: {sanitized_config}\n\n"
        "Предварительный вывод инструмента worker (сырая сводка инструмента/выполнения):\n"
        f"{worker_preview}\n\n"
        "Финальный ответ worker:\n"
        f"{worker_full_result}\n\n"
        "Журнал ошибок worker (если есть):\n"
        f"{worker_error_log}\n"
    )


async def validator_node(
        payload: ValidatorPayload,
        llm: BaseChatModel,
        prompt: str,
        artifact_service: ArtifactService | None = None,
        lineage_service: LineageService | None = None,
) -> Command:
    """
    Асинхронный узел валидации результата задачи в LangGraph-агенте.

    Логика:
    - Если задача завершилась с FAILED — выставляет провал без вызова LLM.
    - Если задача была SKIPPED — выставляет успех без вызова LLM.
    - В остальных случаях вызывает LLM (structured output) для оценки результата.

    Args:
        payload: объект ValidatorPayload, содержащий текущую задачу.
        llm:     языковая модел
        prompt:  системный промпт для LLM-валидатора.
        lineage_service: Опциональный сервис записи validation_completed node.

    Returns:
        Command с обновлённым планом и переходом к узлу replanner.
    """
    task: Task = payload.task

    if task.status == TaskStatus.FAILED:
        task.validation_passed = False
        task.validation_score = _SCORE_ZERO
        task.validation_reason = task.error_log or _REASON_FAILED
        update = _build_validator_update(payload, task, lineage_service)
        prompt_trace_artifacts = write_prompt_trace(
            artifact_service=artifact_service,
            run_id=payload.run_id,
            node_id=((update.get("lineage_events") or [{}])[0]).get("node_id"),
            stage="validator",
            system_prompt=prompt,
            human_prompt="(validator skipped LLM call because task status is FAILED)",
            payload={"task": task.model_dump(mode="json")},
        )
        if prompt_trace_artifacts:
            update["artifact_index"] = prompt_trace_artifacts
        return Command(
            update=update,
            goto=_GOTO_REPLANNER,
        )

    if task.status == TaskStatus.SKIPPED:
        task.validation_passed = True
        task.validation_score = _SCORE_FULL
        task.validation_reason = _REASON_SKIPPED
        update = _build_validator_update(payload, task, lineage_service)
        prompt_trace_artifacts = write_prompt_trace(
            artifact_service=artifact_service,
            run_id=payload.run_id,
            node_id=((update.get("lineage_events") or [{}])[0]).get("node_id"),
            stage="validator",
            system_prompt=prompt,
            human_prompt="(validator skipped LLM call because task status is SKIPPED)",
            payload={"task": task.model_dump(mode="json")},
        )
        if prompt_trace_artifacts:
            update["artifact_index"] = prompt_trace_artifacts
        return Command(
            update=update,
            goto=_GOTO_REPLANNER,
        )

    # --- вызов LLM для оценки результата ---
    human_prompt = _build_human_prompt(task)

    try:
        verdict: StepValidation = await invoke_structured_output(
            llm=llm,
            schema=StepValidation,
            messages=[
                SystemMessage(content=prompt),
                HumanMessage(content=human_prompt),
            ],
        )

        task.validation_passed = verdict.is_valid
        task.validation_reason = verdict.reasoning
        task.validation_score = verdict.confidence if verdict.is_valid else _SCORE_ZERO

        if verdict.is_valid:
            task.status = TaskStatus.COMPLETED
            task.error_log = None
        elif verdict.confidence < _REJECTION_CONFIDENCE_THRESHOLD:
            task.status = TaskStatus.COMPLETED
            task.validation_passed = True
            task.validation_score = verdict.confidence
            task.validation_reason = _REASON_SOFT_ACCEPTED.format(
                reasoning=verdict.reasoning
            )
            task.error_log = None
        else:
            task.status = TaskStatus.FAILED
            task.error_log = _REASON_VALIDATION_FAILED.format(
                reasoning=verdict.reasoning
            )

    except Exception as exc:
        task.status = TaskStatus.COMPLETED
        task.validation_passed = True
        task.validation_reason = _REASON_VALIDATOR_UNAVAILABLE.format(exc=exc)
        task.validation_score = _SCORE_ZERO
        task.error_log = None

    update = _build_validator_update(payload, task, lineage_service)
    prompt_trace_artifacts = write_prompt_trace(
        artifact_service=artifact_service,
        run_id=payload.run_id,
        node_id=((update.get("lineage_events") or [{}])[0]).get("node_id"),
        stage="validator",
        system_prompt=prompt,
        human_prompt=human_prompt,
        payload={"task": task.model_dump(mode="json")},
    )
    if prompt_trace_artifacts:
        update["artifact_index"] = prompt_trace_artifacts
    return Command(update=update, goto=_GOTO_REPLANNER)


def _build_validator_update(
        payload: ValidatorPayload,
        task: Task,
        lineage_service: LineageService | None,
) -> dict[str, Any]:
    task_id = task.task_id or _FALLBACK_TASK_ID
    update: dict[str, Any] = {"plan": {task_id: task}}
    lineage_event = _create_validation_lineage(
        payload=payload,
        task=task,
        lineage_service=lineage_service,
    )
    if lineage_event:
        update["lineage_events"] = [lineage_event]
        update["validation_results"] = {
            task_id: {
                "validation_passed": task.validation_passed,
                "validation_score": task.validation_score,
                "validation_reason": task.validation_reason,
                "status": task.status.value,
                "node_id": lineage_event["node_id"],
            }
        }
    return update


def _create_validation_lineage(
        *,
        payload: ValidatorPayload,
        task: Task,
        lineage_service: LineageService | None,
) -> dict[str, Any] | None:
    if lineage_service is None or not payload.run_id:
        return None

    task_id = task.task_id or _FALLBACK_TASK_ID
    validation_passed = bool(task.validation_passed)
    node = lineage_service.create_state_node(
        run_id=payload.run_id,
        node_type="validation_completed",
        title=f"Validation completed: task {task_id}",
        parent_ids=payload.parent_node_ids,
        status="succeeded" if validation_passed else "failed",
        summary=(task.validation_reason or _FALLBACK_REASON)[:500],
        state={
            "run_id": payload.run_id,
            "task": task.model_dump(mode="json"),
            "validation": {
                "passed": task.validation_passed,
                "score": task.validation_score,
                "reason": task.validation_reason,
            },
        },
        created_by="agent",
        metadata={
            "task_id": task_id,
            "validation_passed": task.validation_passed,
            "validation_score": task.validation_score,
            "task_status": task.status.value,
        },
    )
    return node.model_dump(mode="json")

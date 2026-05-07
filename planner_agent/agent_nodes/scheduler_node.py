"""
Модуль планировщика задач для агента на LangChain.

Содержит логику для управления расписанием выполнения задач в графе зависимостей,
включая определение готовых к выполнению задач, сбор контекста от зависимостей
и отправку задач в worker nodes.
"""

# Стандартные библиотеки
import ast
import json
from typing import Any, Optional

# Сторонние библиотеки
from langchain_core.messages import AIMessage
from langgraph.types import Command, Send

# Локальные импорты
from ..models import AgentState, Task, TaskStatus, WorkerPayload
from ..services.lineage_service import LineageService


# Константы для статусов задач
TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED}
WAITING_STATUSES = {TaskStatus.RUNNING, TaskStatus.NEEDS_VALIDATION}

# Сообщения об ошибках и состояниях
MSG_EMPTY_PLAN = "Planner returned an empty plan."
MSG_NO_EXECUTABLE_TASKS = "No executable tasks remain."
MAX_ARTIFACTS_IN_WORKER_CONTEXT = 12
MAX_DEPENDENCY_RESULT_CHARS = 3_000
MAX_DEPENDENCY_ERROR_CHARS = 2_000
MAX_DEPENDENCY_VALIDATION_CHARS = 1_000
MAX_ARTIFACT_SUMMARY_CHARS = 500
EXCLUDED_WORKER_CONTEXT_ARTIFACT_ROLES = {
    "prompt_trace",
    "prompt_payload",
    "tool_call_trace",
}


def _collect_ancestor_data(
    plan: dict[str, Task],
    task_id: str,
    visited: Optional[set[str]] = None
) -> tuple[set[str], list[str]]:
    """
    Рекурсивно собирает данные от всех предков задачи в графе зависимостей.

    Args:
        plan: Словарь всех задач в плане, где ключ - ID задачи.
        task_id: Идентификатор задачи, для которой собираются данные предков.
        visited: Множество уже посещённых ID задач для предотвращения циклов.

    Returns:
        Кортеж из двух элементов:
        - Множество имён выходных переменных от всех предков.
        - Список строк с превью результатов от всех предков.
    """
    if visited is None:
        visited = set()

    if task_id in visited:
        return set(), []

    visited.add(task_id)
    node = plan.get(task_id)
    if node is None:
        return set(), []

    output_vars: set[str] = set()
    previews: list[str] = []

    if node.output_variable_name:
        output_vars.add(node.output_variable_name)

    result_text = node.full_result or node.result_preview
    if result_text:
        previews.append(
            f"Task {task_id} result: {result_text[:MAX_DEPENDENCY_RESULT_CHARS]}"
        )

    for parent_id in node.dependencies:
        parent_vars, parent_previews = _collect_ancestor_data(plan, parent_id, visited)
        output_vars |= parent_vars
        previews.extend(parent_previews)

    return output_vars, previews


def _parse_structured_result(text: str) -> Any | None:
    """Извлекает JSON/Python-структуру из текстового результата задачи.

    Args:
        text: Текст результата worker-а или вывода инструмента.

    Returns:
        Распарсенный объект ``dict``/``list`` либо ``None``, если текст не похож
        на структурированные данные.
    """

    stripped = (text or "").strip()
    if not stripped:
        return None

    candidates = [stripped]
    first_brace = min(
        [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0],
        default=-1,
    )
    if first_brace > 0:
        candidates.append(stripped[first_brace:])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
        try:
            return ast.literal_eval(candidate)
        except Exception:
            pass
    return None


def _extract_scalar_inputs(value: Any, prefix: str = "") -> dict[str, Any]:
    """Собирает скалярные параметры из структурированного результата.

    Args:
        value: Результат задачи в виде словаря, списка или скалярного значения.
        prefix: Префикс для вложенных ключей при обходе словарей.

    Returns:
        Словарь плоских ключей и скалярных значений, пригодных для подстановки
        в параметры инструментов.
    """

    scalars: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested in value.items():
            key_text = str(key)
            full_key = f"{prefix}.{key_text}" if prefix else key_text
            if isinstance(nested, (str, int, float, bool)) or nested is None:
                scalars.setdefault(key_text, nested)
                scalars[full_key] = nested
            elif isinstance(nested, dict):
                scalars.update(_extract_scalar_inputs(nested, full_key))
        return scalars

    if isinstance(value, list) and value and isinstance(value[0], dict):
        scalars.update(_extract_scalar_inputs(value[0], prefix))
    return scalars


def _build_dependency_context(
    *,
    plan: dict[str, Task],
    task: Task,
) -> dict[str, Any]:
    """Формирует структурированный context package для worker-а.

    Args:
        plan: Полный план текущего запуска.
        task: Задача, для которой собирается контекст.

    Returns:
        Словарь с транзитивными зависимостями, их результатами, artifacts,
        ошибками и извлеченными скалярными входами.
    """

    dependency_ids = _collect_ancestor_task_ids(plan, task.task_id or "")
    dependencies: list[dict[str, Any]] = []
    resolved_inputs: dict[str, Any] = {}

    for dependency_id in dependency_ids:
        dependency = plan.get(dependency_id)
        if dependency is None:
            continue
        result_text = dependency.full_result or dependency.result_preview or ""
        structured_result = _parse_structured_result(result_text)
        if structured_result is not None:
            resolved_inputs.update(_extract_scalar_inputs(structured_result))
        dependencies.append(
            {
                "task_id": dependency_id,
                "status": dependency.status.value,
                "description": dependency.description,
                "output_variable_name": dependency.output_variable_name,
                "artifact_refs": dependency.artifact_refs,
                "evidence_refs": dependency.evidence_refs,
                "result_preview": result_text[:MAX_DEPENDENCY_RESULT_CHARS],
                "validation_passed": dependency.validation_passed,
                "validation_reason": _limit_text(
                    dependency.validation_reason,
                    max_chars=MAX_DEPENDENCY_VALIDATION_CHARS,
                ),
                "error_log": _limit_text(
                    dependency.error_log,
                    max_chars=MAX_DEPENDENCY_ERROR_CHARS,
                ),
            }
        )

    for key, value in task.config.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            resolved_inputs[str(key)] = value

    return {
        "dependency_ids": dependency_ids,
        "dependencies": dependencies,
        "resolved_inputs": resolved_inputs,
    }


def _collect_ancestor_task_ids(
    plan: dict[str, Task],
    task_id: str,
    visited: Optional[set[str]] = None,
) -> list[str]:
    """Собирает все транзитивные зависимости задачи в порядке от ближних к дальним.

    Args:
        plan: Полный план задач.
        task_id: Идентификатор задачи, для которой нужно собрать предков.
        visited: Уже посещенные задачи для защиты от циклов.

    Returns:
        Список идентификаторов задач-предков без дублей.
    """

    if visited is None:
        visited = set()
    if task_id in visited:
        return []

    visited.add(task_id)
    task = plan.get(task_id)
    if task is None:
        return []

    ordered: list[str] = []
    for parent_id in task.dependencies:
        if parent_id not in ordered:
            ordered.append(parent_id)
        for ancestor_id in _collect_ancestor_task_ids(plan, parent_id, visited):
            if ancestor_id not in ordered:
                ordered.append(ancestor_id)
    return ordered


async def scheduler_node(
    state: AgentState,
    lineage_service: LineageService | None = None,
) -> Command:
    """
    Асинхронная нода-планировщик для управления выполнением задач в графе.

    Анализирует текущее состояние плана, определяет задачи готовые к выполнению
    (все зависимости которых завершены), собирает необходимый контекст и отправляет
    их в worker nodes для выполнения.

    Args:
        state: Текущее состояние агента, содержащее план задач, схемы данных
               и глобальные переменные.
        lineage_service: Опциональный сервис записи task_scheduled node.

    Returns:
        Command объект с инструкциями для следующего шага в графе:
        - переход к responder, если план пуст или все задачи завершены
        - отправка задач в worker nodes, если есть готовые к выполнению
        - ожидание, если есть задачи в процессе выполнения
    """
    plan = state.plan or {}
    schemas = state.data_schemas

    # Проверка на пустой план
    if not plan:
        return Command(
            goto="responder",
            update={"messages": [AIMessage(content=MSG_EMPTY_PLAN)]},
        )

    # Проверка завершения всех задач
    if all(task.status in TERMINAL_STATUSES for task in plan.values()):
        return Command(goto="responder")

    worker_payloads: list[WorkerPayload] = []
    running_patch: dict[str, Task] = {}

    global_vars_on_init = set(state.global_vars)

    # Поиск задач готовых к выполнению
    for task_id, task in plan.items():
        if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
            continue

        # Проверка завершения всех зависимостей
        parents_ok = all(
            parent_id in plan and plan[parent_id].status == TaskStatus.COMPLETED
            for parent_id in task.dependencies
        )
        if not parents_ok:
            continue

        # Сбор данных от всех предков
        all_output_vars: set[str] = set()
        all_results: list[str] = []
        visited_ancestors: set[str] = set()
        for parent_id in task.dependencies:
            parent_vars, parent_results = _collect_ancestor_data(
                plan,
                parent_id,
                visited_ancestors,
            )
            all_output_vars |= parent_vars
            all_results.extend(parent_results)

        # Формирование контекста для задачи
        visible_var_names = global_vars_on_init.union(all_output_vars)
        task_context_schemas = {
            name: schema
            for name, schema in schemas.items()
            if name in visible_var_names
        }

        dependency_context = _build_dependency_context(plan=plan, task=task)

        # Создание payload для worker
        payload = WorkerPayload(
            task=task,
            context_schemas=task_context_schemas,
            previous_results="",
            resolved_inputs=dependency_context.get("resolved_inputs", {}),
            dependency_context=dependency_context,
            filesystem_context=state.filesystem_context,
            skill_previews=state.skill_previews,
            artifact_context=_build_artifact_context(state, task),
        )

        # Обновление статуса задачи на RUNNING
        updated_task = task.model_copy(deep=True)
        updated_task.status = TaskStatus.RUNNING
        running_patch[task_id] = updated_task
        worker_payloads.append(payload)

    # Отправка задач на выполнение
    if worker_payloads:
        update_payload = {"plan": running_patch}
        lineage_update = _create_task_scheduled_lineage(
            state=state,
            running_patch=running_patch,
            lineage_service=lineage_service,
        )
        update_payload.update(lineage_update)

        worker_parent_ids = _resolve_worker_parent_ids(
            state=state,
            lineage_update=lineage_update,
        )
        tasks_to_schedule = [
            Send(
                "worker",
                payload.model_copy(
                    update={
                        "run_id": state.run_id,
                        "parent_node_ids": worker_parent_ids,
                    },
                    deep=True,
                ),
            )
            for payload in worker_payloads
        ]
        return Command(update=update_payload, goto=tasks_to_schedule)

    # Ожидание выполнения других задач
    if any(task.status in WAITING_STATUSES for task in plan.values()):
        return Command(update={})

    # Нет доступных для выполнения задач
    return Command(
        goto="responder",
        update={"messages": [AIMessage(content=MSG_NO_EXECUTABLE_TASKS)]},
    )


def _create_task_scheduled_lineage(
    *,
    state: AgentState,
    running_patch: dict[str, Task],
    lineage_service: LineageService | None,
) -> dict[str, object]:
    if lineage_service is None or not state.run_id or not running_patch:
        return {}

    next_plan = dict(state.plan or {})
    next_plan.update(running_patch)
    scheduled_task_ids = list(running_patch.keys())
    parent_ids = state.parent_node_ids or (
        [state.current_node_id] if state.current_node_id else []
    )
    snapshot = state.model_copy(
        update={
            "plan": next_plan,
            "current_node_id": state.current_node_id,
            "parent_node_ids": parent_ids,
        },
        deep=True,
    )
    node = lineage_service.create_state_node(
        run_id=state.run_id,
        node_type="task_scheduled",
        title="Task batch scheduled",
        parent_ids=parent_ids,
        status="succeeded",
        summary=f"Scheduled task(s): {', '.join(scheduled_task_ids)}",
        state=snapshot,
        created_by="system",
        metadata={
            "scheduled_task_ids": scheduled_task_ids,
            "scheduled_count": len(scheduled_task_ids),
        },
    )

    return {
        "current_node_id": node.node_id,
        "parent_node_ids": [node.node_id],
        "lineage_events": [node.model_dump(mode="json")],
    }


def _build_artifact_context(state: AgentState, task: Task) -> dict[str, Any]:
    """Формирует компактный artifact context для worker.

    Args:
        state: Текущее состояние агента с artifact_index и plan.
        task: Задача, для которой собирается контекст.

    Returns:
        Словарь с выбранными artifacts, общим количеством и количеством скрытых
        artifacts за пределами лимита контекста.
    """

    artifact_index = state.artifact_index or {}
    if not artifact_index:
        return {}

    selected_ids = _select_artifact_ids(state=state, task=task)
    shown_ids = selected_ids[:MAX_ARTIFACTS_IN_WORKER_CONTEXT]
    artifacts: dict[str, Any] = {}
    for artifact_id in shown_ids:
        payload = artifact_index.get(artifact_id)
        if isinstance(payload, dict):
            artifacts[artifact_id] = _compact_artifact_payload(payload)

    return {
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_available_artifacts": len(artifact_index),
        "selected_artifact_ids": list(artifacts.keys()),
        "hidden_artifact_count": max(0, len(selected_ids) - len(artifacts)),
        "max_artifacts_in_context": MAX_ARTIFACTS_IN_WORKER_CONTEXT,
    }


def _select_artifact_ids(state: AgentState, task: Task) -> list[str]:
    """Выбирает artifact ids для передачи в worker.

    Приоритет:
    1. artifacts, уже привязанные к самой задаче;
    2. evidence refs самой задачи;
    3. artifacts/evidence завершенных транзитивных зависимостей;
    4. остальные artifacts из state.artifact_index как общий reusable context.

    Args:
        state: Текущее состояние агента.
        task: Задача, для которой выбираются artifacts.

    Returns:
        Упорядоченный список artifact ids без дублей.
    """

    ordered: list[str] = []

    def add_many(ids: list[str]) -> None:
        for artifact_id in ids:
            payload = (state.artifact_index or {}).get(artifact_id)
            if _is_excluded_worker_context_artifact(payload):
                continue
            if artifact_id and artifact_id not in ordered:
                ordered.append(artifact_id)

    add_many(task.artifact_refs)
    add_many(task.evidence_refs)

    for parent_id in _collect_ancestor_task_ids(state.plan or {}, task.task_id or ""):
        parent = state.plan.get(parent_id) if state.plan else None
        if parent:
            add_many(parent.artifact_refs)
            add_many(parent.evidence_refs)

    add_many(list((state.artifact_index or {}).keys()))
    return ordered


def _is_excluded_worker_context_artifact(payload: Any) -> bool:
    """Проверяет, нужно ли скрыть служебный artifact из prompt worker-а.

    Args:
        payload: JSON-совместимое описание artifact из ``state.artifact_index``.

    Returns:
        ``True``, если artifact является служебным trace/payload и его не нужно
        автоматически добавлять в контекст worker-а.
    """

    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return str(metadata.get("artifact_role") or "") in EXCLUDED_WORKER_CONTEXT_ARTIFACT_ROLES


def _compact_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Обрезает payload artifact до безопасной карточки для worker prompt.

    Args:
        payload: JSON-совместимое представление Artifact.

    Returns:
        Компактный словарь без тяжелых/внутренних metadata.
    """

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    summary = str(payload.get("summary") or "")
    return {
        "artifact_id": payload.get("artifact_id"),
        "kind": payload.get("kind"),
        "uri": payload.get("uri"),
        "mime_type": payload.get("mime_type"),
        "summary": _limit_text(summary, max_chars=MAX_ARTIFACT_SUMMARY_CHARS),
        "checksum": payload.get("checksum"),
        "metadata": {
            key: metadata[key]
            for key in (
                "task_id",
                "tool_name",
                "artifact_role",
                "reusable",
                "editable",
                "branched_from_artifact_id",
                "edited_override",
                "capture_reason",
                "original_size_estimate",
            )
            if key in metadata
        },
    }


def _limit_text(value: str | None, *, max_chars: int) -> str | None:
    """Обрезает текстовое значение для передачи в prompt worker-а.

    Args:
        value: Исходный текст или ``None``.
        max_chars: Максимальное количество символов, которое можно оставить.

    Returns:
        Исходный текст в пределах лимита, помеченный как обрезанный при
        превышении бюджета, или ``None`` для пустого значения.
    """

    if value is None:
        return None
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _resolve_worker_parent_ids(
    *,
    state: AgentState,
    lineage_update: dict[str, object],
) -> list[str]:
    lineage_node_id = lineage_update.get("current_node_id")
    if isinstance(lineage_node_id, str) and lineage_node_id:
        return [lineage_node_id]
    return state.parent_node_ids or (
        [state.current_node_id] if state.current_node_id else []
    )

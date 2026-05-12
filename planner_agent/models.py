"""Модели состояния и схемы обмена между узлами аналитического агента.

Содержит:
- TaskStatus: перечисление статусов задач.
- ActionType: перечисление типов правок плана.
- TaskBase: базовая схема задачи.
- PlanEditAction: схема правки плана.
- Task: runtime-схема задачи.
- PlanUpdate: схема набора правок плана.
- PlannedTask: схема задачи, возвращаемой планировщиком.
- FullPlan: схема полного плана.
- PlanReview: схема критики плана до запуска задач.
- merge_schemas: объединение словарей схем.
- merge_dicts: объединение произвольных словарей состояния.
- merge_plans: объединение планов задач.
- AgentState: общее состояние графа агента.
- WorkerPayload: входной пакет для worker-узла.
- CriticPayload: входной пакет для critic-узла после выполнения worker.
- ValidatorPayload: входной пакет для validator-узла.
- StepValidation: результат проверки шага.
- WorkerCriticReview: результат критики выполнения worker-задачи.
- FinalReport: итоговый отчет responder-узла.
"""

import operator
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field


class TaskStatus(str, Enum):
    """Статусы задачи внутри runtime-плана агента."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    NEEDS_VALIDATION = "needs_validation"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionType(str, Enum):
    """Типы операций для изменения плана."""

    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    RETRY = "retry"
    SKIPPED = "skipped"


class TaskBase(BaseModel):
    """Базовая схема задачи, общая для плана и runtime-состояния."""

    model_config = ConfigDict(extra="ignore")

    task_id: Optional[str] = Field(description="Task ID. Example: 1, 2, 3")
    description: str = Field(description="Detailed task description")
    dependencies: List[str] = Field(default_factory=list)
    expected_output: Optional[str] = Field(
        default=None,
        description="Expected concrete output for the task.",
    )
    suggested_tools: List[str] = Field(
        default_factory=list,
        description="Tool names that are likely useful for this task.",
    )
    suggested_skills: List[str] = Field(
        default_factory=list,
        description="Skill names that are likely useful for this task.",
    )
    required_artifacts: List[str] = Field(
        default_factory=list,
        description="Artifacts that should be produced or referenced.",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task-specific settings. Do not put tool schemas, executable code, "
            "or service-owned status fields here."
        ),
    )


class PlanEditAction(TaskBase):
    """Одна операция над планом в ответе replanner (structured output).

    Используется как элемент списка правок при перепланировании: создание,
    изменение, удаление или retry задачи с обоснованием ``reasoning``.
    """

    action: ActionType
    reasoning: str = Field(..., description="Reasoning behind this action")


class Task(TaskBase):
    """Задача в runtime-плане: статусы, вывод worker-а, валидация, артефакты.

    Planner/replanner отдают в плане упрощённые ``PlannedTask``; после scheduler/worker
    поля результата и валидации заполняются исполнением графа. Модель сериализуется
    в snapshots lineage и в prompt validator/responder.
    """

    generated_code: Optional[str] = None
    output_variable_name: Optional[str] = None
    result_preview: Optional[str] = None
    full_result: Optional[str] = None
    artifact_refs: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    error_log: Optional[str] = None
    run_id: str = Field(default="", description="Research run identifier")

    # Validation fields (worker -> validator -> replanner pipeline)
    validation_passed: Optional[bool] = None
    validation_reason: Optional[str] = None
    validation_score: Optional[float] = None


class PlanUpdate(BaseModel):
    """Набор правок плана, возвращаемый replanner-узлом."""

    edits: List[PlanEditAction] = Field(default_factory=list)


class PlannedTask(TaskBase):
    """Задача в полном плане, возвращаемом planner/replanner."""

    task_id: int = Field(description="Stable numeric task ID. Example: 1, 2, 3")
    dependencies: List[int] = Field(
        default_factory=list,
        description="Numeric task IDs that must be completed before this task.",
    )


class FullPlan(BaseModel):
    """Полный актуальный план выполнения пользовательского запроса."""

    objective: Optional[str] = Field(
        default=None,
        description="Short objective that the full plan is intended to satisfy.",
    )
    tasks: List[PlannedTask] = Field(
        ...,
        min_length=1,
        description="Complete plan with all steps that should remain in execution graph",
    )


class PlanReview(BaseModel):
    """Критика сгенерированного плана перед запуском задач.

    Используется планировщиком для мягкой самопроверки: не слишком ли шаги
    общие, не слишком ли они мелкие, не пропущены ли источники, расчеты,
    проверки, зависимости или условия остановки.
    """

    needs_revision: bool = Field(
        default=False,
        description="True, если план нужно пересоставить перед выполнением.",
    )
    summary: str = Field(
        default="",
        description="Краткое резюме качества плана.",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Проблемы плана: слишком общие/мелкие шаги, дубли, слабые зависимости.",
    )
    missing_steps: List[str] = Field(
        default_factory=list,
        description="Пропущенные шаги, которые могут быть нужны для ответа на запрос.",
    )
    revision_guidance: str = Field(
        default="",
        description="Рекомендация planner/replanner по исправлению плана.",
    )


def merge_schemas(left: Dict[str, str], right: Dict[str, str]) -> Dict[str, str]:
    """Объединяет словари текстовых схем состояния.

    Args:
        left: Текущее значение в состоянии графа.
        right: Новое значение, добавленное узлом.

    Returns:
        Объединенный словарь схем, где значения из ``right`` перекрывают ``left``.
    """

    if not left:
        return right or {}
    return {**left, **(right or {})}


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Объединяет два словаря произвольных данных состояния.

    Args:
        left: Текущее значение в состоянии графа.
        right: Новое значение, добавленное узлом.

    Returns:
        Объединенный словарь, где значения из ``right`` перекрывают ``left``.
    """

    if not left:
        return right or {}
    return {**left, **(right or {})}


def merge_plans(left: dict[str, Task], right: dict[str, Task | None]) -> dict[str, Task]:
    """Объединяет текущий план с обновлениями задач.

    Args:
        left: Текущий план задач.
        right: Обновления плана. Значение ``None`` удаляет задачу из результата.

    Returns:
        Новый словарь задач после применения обновлений.
    """

    left = left or {}
    right = right or {}
    out = dict(left)
    for tid, task in right.items():
        if task is None:
            out.pop(tid, None)
        else:
            out[tid] = task
    return out


class AgentState(BaseModel):
    """Сводное состояние одного прохода LangGraph (план, артефакты, сообщения).

    Узлы графа читают и обновляют это состояние; отдельные поля попадают в prompt
    planner/worker/replanner/responder. Часть полей агрегируется через reducers
    (``operator.add``, ``merge_*``).
    """

    model_config = ConfigDict(extra="ignore")

    run_id: str = Field(default="")
    session_id: str = Field(default="")
    user_id: Optional[str] = Field(default=None)
    current_node_id: Optional[str] = Field(default=None)
    parent_node_ids: List[str] = Field(default_factory=list)

    global_vars: List[str] = Field(default_factory=list)
    messages: Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    plan: Annotated[Dict[str, Task], merge_plans] = Field(default_factory=dict)
    data_schemas: Annotated[Dict[str, str], merge_schemas] = Field(default_factory=dict)
    filesystem_context: Annotated[Dict[str, str], merge_schemas] = Field(default_factory=dict)
    skill_previews: Annotated[Dict[str, str], merge_schemas] = Field(default_factory=dict)
    initial_user_query: str = Field(default="")
    initial_plan: Dict[str, Task] = Field(default_factory=dict)

    memory_snapshot: str = Field(default="")
    skills_index: str = Field(default="")
    loaded_skills: Annotated[Dict[str, str], merge_schemas] = Field(default_factory=dict)
    ephemeral_recalls: Annotated[Dict[str, str], merge_schemas] = Field(default_factory=dict)

    artifact_index: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    task_results: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    validation_results: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    evidence_map: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)

    policy_decisions: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    tool_traces: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    lineage_events: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)

    feedback_context: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)

    final_report: Optional[str] = Field(default=None)


class WorkerPayload(BaseModel):
    """Полезная нагрузка worker-узла с задачей и доступным контекстом.

    Используется scheduler-узлом для передачи worker-у текста задачи,
    транзитивных результатов зависимостей, извлеченных фактических входов,
    схем sandbox-переменных, файлового контекста и artifact-контекста.
    """

    task: Task
    context_schemas: Dict[str, str]
    previous_results: str
    resolved_inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Скалярные значения, извлеченные из транзитивных зависимостей и "
            "config задачи: client_id, event_date, artifact_id, периоды, "
            "фильтры и другие параметры, которые worker должен использовать "
            "вместо строковых имен полей."
        ),
    )
    dependency_context: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Компактный структурированный контекст по транзитивным "
            "зависимостям: статусы задач, результаты, artifacts, ошибки "
            "и критерии валидации."
        ),
    )
    filesystem_context: Dict[str, str] = Field(default_factory=dict)
    skill_previews: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Краткие preview доступных skills. Используются worker-узлом, чтобы "
            "загрузить релевантный полный skill даже если planner не указал "
            "suggested_skills явно."
        ),
    )
    artifact_context: Dict[str, Any] = Field(default_factory=dict)
    initial_user_query: str = Field(
        default="",
        description="Исходный пользовательский запрос, ради которого выполняется задача.",
    )
    run_id: str = ""
    parent_node_ids: List[str] = Field(default_factory=list)


class CriticPayload(BaseModel):
    """Вход critic-узла: задача worker-а, артефакты и tool traces этого шага."""

    worker_payload: WorkerPayload
    run_id: str = ""
    parent_node_ids: List[str] = Field(default_factory=list)
    artifact_index: Dict[str, Any] = Field(default_factory=dict)
    tool_traces: List[Dict[str, Any]] = Field(default_factory=list)
    react_message_tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Фактические вызовы инструментов из истории сообщений ReAct worker "
            "(AIMessage.tool_calls + соответствующие ToolMessage)."
        ),
    )


class ValidatorPayload(BaseModel):
    """Полезная нагрузка validator-узла для проверки результата worker-задачи."""

    task: Task
    run_id: str = ""
    parent_node_ids: List[str] = Field(default_factory=list)


class StepValidation(BaseModel):
    """Structured output validator-узла: валидность результата worker по задаче."""

    is_valid: bool = Field(description="True if worker output satisfies the step task")
    confidence: float = Field(default=0.0, description="Confidence score from 0 to 1")
    reasoning: str = Field(description="Why output is valid or invalid")


class WorkerCriticReview(BaseModel):
    """Structured output critic-узла: допуск к validator или доработка worker-ом."""

    approved: bool = Field(
        description="True, если результат worker-а можно передать validator-у.",
    )
    reasoning: str = Field(
        description="Краткое объяснение решения critic-а по worker-результату.",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Существенные проблемы результата worker-а.",
    )
    improvement_instructions: str = Field(
        default="",
        description="Что именно worker должен проверить или расширить при retry.",
    )


class FinalReport(BaseModel):
    """Structured output responder-узла: финальный markdown-отчёт пользователю."""

    report: str = Field(
        description="Полный markdown-отчёт для пользователя.",
    )

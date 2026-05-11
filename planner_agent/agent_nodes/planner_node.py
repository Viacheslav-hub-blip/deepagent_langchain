"""
Модуль реализует узел планировщика (planner_node) для графа LangGraph.

Отвечает за:
- формирование системного промпта на основе текущего состояния агента;
- вызов языковой модели для получения структурированного плана (FullPlan);
- построение и обновление словаря задач (план) с учётом текущих статусов;
- возврат команды перехода к следующему узлу графа (scheduler / responder).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..models import AgentState, FullPlan, PlanReview, Task, TaskStatus
from ..services.artifact_service import ArtifactService
from ..services.lineage_service import LineageService
from ..services.prompt_trace_service import write_prompt_trace
from ..services.skills_service import SkillsService
from ..structured_output import invoke_structured_output

#: Имя узла-планировщика следующего шага.
GOTO_SCHEDULER: str = "scheduler"

#: Имя узла формирования ответа при ошибке планирования.
GOTO_RESPONDER: str = "responder"

#: Максимальная длина превью результата / обоснования / лога ошибки.
PREVIEW_MAX_LENGTH: int = 800

#: Максимальная длина блока, который выводится в терминал.
CONSOLE_BLOCK_MAX_LENGTH: int = 8_000

#: Максимальное количество попыток пересоставить план после review.
PLAN_REVIEW_REVISION_ATTEMPTS: int = 1

#: Максимальное число полных skills, которые planner/replanner загружает
#: перед составлением плана.
MAX_PLANNER_LOADED_SKILLS: int = 10

#: Сообщение пользователя при принудительном перепланировании.
REPLAN_USER_PROMPT: str = "Обновите план с учетом последних результатов. Если ответ на запрос пользователя был получен, заверши выполнения досрочно"

#: Префикс пользовательского сообщения при первичном планировании.
USER_REQUEST_PREFIX: str = "Запрос пользователя: "

#: Статусы задач, которые нельзя перезаписывать во время выполнения планировщика.
#: Задача в одном из этих статусов считается «занятой» или уже завершённой.
IMMUTABLE_RUNTIME_STATUSES: frozenset[TaskStatus] = frozenset({
    TaskStatus.RUNNING,
    TaskStatus.NEEDS_VALIDATION,
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.SKIPPED,
})

#: Статусы задач, которые можно безопасно удалить при перепланировании.
REMOVABLE_STATUSES: frozenset[TaskStatus] = frozenset({
    TaskStatus.PENDING,
    TaskStatus.READY,
})

#: Статусы задач, которые включаются в сводку результатов выполнения.
FINISHED_STATUSES: frozenset[TaskStatus] = frozenset({
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.SKIPPED,
})

#: Служебные поля задачи, запрещённые в пользовательском config.
#: Берётся из модели Task.
RESERVED_CONFIG_KEYS: frozenset[str] = frozenset(
    str(field).strip().lower()
    for field in Task.model_fields
)


class PlannerSkillSelection(BaseModel):
    """Выбор skills, которые нужно загрузить перед планированием.

    Attributes:
        skill_names: Имена skills из доступного индекса, которые нужны planner-у
            для составления или исправления плана.
        rationale: Краткое объяснение, почему выбраны именно эти skills.
    """

    skill_names: list[str] = Field(
        default_factory=list,
        description="Skill names from the available skills index to load.",
    )
    rationale: str = Field(
        default="",
        description="Short rationale for selected skills.",
    )

async def _print_content_block(title: str, content: str) -> None:
    """Асинхронно выводит читаемый блок текста в стандартный вывод.

    Args:
        title: Заголовок блока, который будет показан в терминале.
        content: Содержимое блока. При превышении лимита текст обрезается.

    Returns:
        ``None``. Функция выполняет только диагностический вывод.
    """

    text = (content or "").strip() or "(empty)"
    if len(text) > CONSOLE_BLOCK_MAX_LENGTH:
        text = f"{text[:CONSOLE_BLOCK_MAX_LENGTH]}\n...[truncated for console]..."
    border = "=" * min(max(len(title), 16), 80)
    await asyncio.to_thread(
        print,
        f"\n{border}\n{title}\n{border}\n{text}\n",
        flush=True,
    )


def _format_full_plan_response(full_plan: FullPlan) -> str:
    """Форматирует сырой ответ модели-планировщика в читаемый вид.

    Args:
        full_plan: Структурированный план, полученный от LLM.

    Returns:
        Многострочная строка с objective и задачами из ответа модели.
    """

    lines = [f"Objective: {full_plan.objective or '(empty)'}", "Tasks:"]
    if not full_plan.tasks:
        lines.append("- (empty)")
        return "\n".join(lines)

    for task in full_plan.tasks:
        details = [
            f"- {task.task_id}: {task.description}",
            f"  dependencies: {task.dependencies or []}",
        ]
        if task.expected_output:
            details.append(f"  expected_output: {task.expected_output}")
        if task.validation_criteria:
            details.append(f"  validation_criteria: {task.validation_criteria}")
        if task.suggested_tools:
            details.append(f"  suggested_tools: {task.suggested_tools}")
        if task.suggested_skills:
            details.append(f"  suggested_skills: {task.suggested_skills}")
        if task.config:
            details.append(f"  config: {task.config}")
        lines.extend(details)
    return "\n".join(lines)


def _format_plan_review(review: PlanReview) -> str:
    """Форматирует критику плана в читаемую строку для терминала.

    Args:
        review: Структурированная критика плана.

    Returns:
        Многострочная строка с признаком revision, проблемами и рекомендацией.
    """

    lines = [
        f"Needs revision: {review.needs_revision}",
        f"Summary: {review.summary or '(empty)'}",
    ]
    if review.issues:
        lines.append("Issues:")
        lines.extend(f"- {item}" for item in review.issues)
    if review.missing_steps:
        lines.append("Missing steps:")
        lines.extend(f"- {item}" for item in review.missing_steps)
    if review.revision_guidance:
        lines.append(f"Revision guidance: {review.revision_guidance}")
    return "\n".join(lines)


def _format_plan_review_feedback(review: PlanReview) -> str:
    """Формирует компактную инструкцию для пересоставления плана.

    Args:
        review: Структурированная критика плана.

    Returns:
        Текстовое замечание, которое добавляется к повторному запросу planner-а.
    """

    return "\n".join(
        [
            "Plan reviewer found issues in the candidate plan.",
            f"Summary: {review.summary}",
            f"Issues: {review.issues}",
            f"Missing steps: {review.missing_steps}",
            f"Revision guidance: {review.revision_guidance}",
            "Return a revised FullPlan JSON. Keep successful/necessary tasks, "
            "split overly broad tasks, merge useless micro-steps, add missing "
            "data/metric/evidence steps, and keep the plan minimal.",
        ]
    )


def _format_plan_content(plan: dict[str, Task]) -> str:
    """Форматирует план в читаемую строку для вывода в консоль.

    Каждая строка содержит ID задачи, её статус, описание,
    зависимости и конфигурацию (если есть).

    Args:
        plan: Словарь задач планировщика.

    Returns:
        Многострочное текстовое представление плана.
    """
    if not plan:
        return "(пустой план)"

    lines: list[str] = []
    for task_id, task in plan.items():
        deps = f" <- {', '.join(task.dependencies)}" if task.dependencies else ""
        config_part = f" | config={task.config}" if task.config else ""
        lines.append(
            f"{task_id}. [{task.status.value}] {task.description}{deps}{config_part}"
        )
    return "\n".join(lines)


def _task_to_summary_line(task_id: str, task: Task) -> str:
    """Формирует однострочное резюме задачи для сводки плана.

    Включает ID, статус, описание, зависимости, счётчик повторных попыток,
    а также опциональные поля: параметры, превью результата, статус валидации,
    обоснование и лог ошибки.

    Args:
        task_id: Идентификатор задачи.
        task: Объект задачи.

    Returns:
        Строка с полями, разделёнными «|».
    """
    parts = [
        f"ID: {task_id}",
        f"Статус: {task.status.value}",
        f"Описание: {task.description}",
        f"Зависимости: {task.dependencies or []}",
        f"Повторные попытки: {task.retry_count}",
    ]

    # Добавляем опциональные поля только при наличии значений
    if task.config:
        parts.append(f"Параметры: {task.config}")
    if task.validation_passed is not None:
        parts.append(f"Статус проверки правильности ответа: {task.validation_passed}")
    if task.validation_reason:
        parts.append(
            f"Обоснование: {task.validation_reason[:PREVIEW_MAX_LENGTH]}"
        )
    if task.error_log:
        parts.append(f"Ошибка: {task.error_log[:PREVIEW_MAX_LENGTH]}")

    return " | ".join(parts)


def _wrap_task_prompt_block(task_id: str, content: str) -> str:
    """Оборачивает текст задачи в парный тег для prompt-контекста.

    Args:
        task_id: Идентификатор задачи, который будет добавлен в тег.
        content: Текстовое описание блока задачи.

    Returns:
        Многострочная строка вида ``<TASK task_id>... </TASK task_id>``.
    """

    tag_id = str(task_id).strip() or "unknown"
    return f"<TASK {tag_id}>\n{content.strip()}\n</TASK {tag_id}>"


def _create_plan_summary(plan: dict[str, Task]) -> str:
    """Создаёт полную текстовую сводку плана из всех задач.

    Args:
        plan: Словарь задач планировщика.

    Returns:
        Многострочная строка с резюме каждой задачи,
        либо сообщение об отсутствии задач.
    """
    if not plan:
        return "Задач еще нет (план пустой)."

    return "\n\n".join(
        _wrap_task_prompt_block(task_id, _task_to_summary_line(task_id, task))
        for task_id, task in plan.items()
    )


def _format_context(state: AgentState) -> str:
    """Извлекает и форматирует предыдущий контекст диалога из истории сообщений.

    Находит последнее сообщение пользователя и последний ответ агента
    (не считая текущего сообщения).

    Args:
        state: Текущее состояние агента.

    Returns:
        Строка с предыдущим контекстом или сообщение о его отсутствии.
    """
    # Ищем предыдущие сообщения, исключая последнее (текущее)
    previous_human = next(
        (msg for msg in reversed(state.messages[:-1]) if isinstance(msg, HumanMessage)),
        None,
    )
    previous_ai = next(
        (msg for msg in reversed(state.messages[:-1]) if isinstance(msg, AIMessage)),
        None,
    )

    parts: list[str] = []
    if previous_human:
        parts.append(f"Последнее сообщение пользователя: {previous_human.content}")
    if previous_ai:
        parts.append(f"Последний ответ агента: {previous_ai.content}")
    dialog_context = state.ephemeral_recalls.get("dialog_context")
    if dialog_context:
        parts.append(str(dialog_context))

    return "\n".join(parts) if parts else "Предыдущего контекста нет."


def _format_critic_feedback(state: AgentState) -> str:
    """Форматирует замечания critic для prompt-а планировщика.

    Args:
        state: Текущее состояние агента с накопленным ``feedback_context``.

    Returns:
        Многострочная строка с последними замечаниями critic или сообщение об
        отсутствии критики.
    """

    if not state.feedback_context:
        return "Замечаний critic пока нет."

    lines: list[str] = []
    for index, item in enumerate(state.feedback_context[-5:], start=1):
        if not isinstance(item, dict):
            lines.append(f"{index}. {item}")
            continue
        lines.append(
            (
                f"{index}. summary={item.get('summary', '')}; "
                f"failed_task_diagnosis={item.get('failed_task_diagnosis', [])}; "
                f"suggested_investigations={item.get('suggested_investigations', [])}; "
                f"replan_guidance={item.get('replan_guidance', '')}"
            )
        )
    return "\n".join(lines)


def _format_data_context(state: AgentState) -> str:
    """Форматирует контекст данных: схемы переменных, файловую систему и навыки.

    Используется для передачи актуального окружения в системный промпт.

    Args:
        state: Текущее состояние агента.

    Returns:
        Многострочная строка с описанием доступного контекста данных,
        либо сообщение об отсутствии данных.
    """
    blocks: list[str] = []

    if state.data_schemas:
        blocks.append("Описание переменных, загруженных в виртуальное окружение:")
        blocks.extend(
            f"- {name}: {preview}" for name, preview in state.data_schemas.items()
        )

    if state.filesystem_context:
        blocks.append("Рабочие файлы и директории:")
        blocks.extend(
            f"- {name}: {value}" for name, value in state.filesystem_context.items()
        )

    if state.skills_index:
        blocks.append("Индекс доступных skills:")
        blocks.append(state.skills_index)

    if state.skill_previews:
        blocks.append(
            "Preview доступных skills"
        )
        blocks.extend(
            f"- {name}: {preview}"
            for name, preview in state.skill_previews.items()
        )

    if not blocks:
        return "Переменные, контекст файловой системы и навыки пока недоступны."

    return "\n".join(blocks)


def _format_tools(tools: list[BaseTool]) -> str:
    """Форматирует список инструментов агента в текстовое описание.

    Args:
        tools: Список доступных инструментов.

    Returns:
        Строка с перечнем инструментов в формате «- name: description».
    """
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


def _format_planner_loaded_skills(loaded_skills: dict[str, str]) -> str:
    """Форматирует загруженные planner skills для системного prompt.

    Args:
        loaded_skills: Словарь ``{skill_name: full_skill_content}``.

    Returns:
        XML-подобный блок с полным содержимым выбранных skills или пустую строку.
    """

    if not loaded_skills:
        return ""

    blocks = [
        "<planner_loaded_skills>",
        "Используй эти skills при выборе источников, зависимостей, recovery-шагов, "
        "validation criteria и suggested_skills. Skill не заменяет фактические "
        "данные: если нужны значения, запланируй получение данных через tools, "
        "artifacts или sandbox.",
    ]
    for skill_name, content in loaded_skills.items():
        blocks.append(f"<skill name=\"{skill_name}\">\n{content.strip()}\n</skill>")
    blocks.append("</planner_loaded_skills>")
    return "\n".join(blocks)


async def _load_planner_skills(
        *,
        llm: BaseChatModel,
        state: AgentState,
        skills_service: SkillsService | None,
        initial_user_query: str,
        plan_str: str,
        execution_results: str,
        tools_desc: str,
        previous_context: str,
        critic_feedback: str,
) -> dict[str, str]:
    """Выбирает и загружает skills перед вызовом planner/replanner.

    Args:
        llm: Языковая модель, которая выбирает релевантные skills.
        state: Текущее состояние агента с индексом и preview skills.
        skills_service: Сервис чтения skills или ``None``.
        initial_user_query: Исходный пользовательский запрос.
        plan_str: Текущий план задач.
        execution_results: Результаты завершенных задач.
        tools_desc: Описание доступных инструментов.
        previous_context: Предыдущий диалоговый контекст.
        critic_feedback: Последняя критика/feedback.

    Returns:
        Словарь ``{skill_name: full_skill_content}`` для включения в prompt.
    """

    if skills_service is None:
        return {}
    if not (state.skills_index or state.skill_previews):
        return {}

    available_names = {
        skill.name
        for skill in skills_service.skills_list()
    }
    if not available_names:
        return {}

    schema_str = json.dumps(
        PlannerSkillSelection.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    selection_prompt = f"""
<role>
Ты — skill selection step для planner/replanner аналитического агента.
</role>

<task>
Выбери только те skills, полный текст которых нужно загрузить перед составлением
или исправлением плана. Выбирай только минимальный набор skills которые нужны только на данный момент

</task>

<available_skills>
{state.skills_index}
</available_skills>

<skill_previews>
{json.dumps(state.skill_previews, ensure_ascii=False, indent=2)}
</skill_previews>

<already_loaded_skill_names>
{sorted(state.loaded_skills)}
</already_loaded_skill_names>

<planning_context>
initial_user_query:
{initial_user_query}

current_plan:
{plan_str}

execution_results:
{execution_results}

available_workers:
{tools_desc}

previous_context:
{previous_context}

critic_feedback:
{critic_feedback}
</planning_context>

<rules>
1. Верни только имена skills из <available_skills>.
2. Не выбирай skill, если он не помогает составить или исправить план в текущий момент
3. Если ни один skill не нужен, верни пустой список.
4. Максимум выбери {MAX_PLANNER_LOADED_SKILLS} skills.
5. Не выдумывай имена skills.
6. Если skill уже есть в <already_loaded_skill_names> и все еще нужен для
   текущего планирования, можешь указать его повторно: runtime переиспользует
   cached content и не загрузит его второй раз.
</rules>

<output_format>
{schema_str}
</output_format>
"""

    try:
        selection = await invoke_structured_output(
            llm=llm,
            schema=PlannerSkillSelection,
            messages=[SystemMessage(content=selection_prompt)],
        )
    except Exception as exc:
        await _print_content_block(
            "PLANNER SKILL SELECTION SKIPPED",
            f"Skill selection failed and planner will use index/previews only.\nError: {exc}",
        )
        return {}

    loaded: dict[str, str] = {}
    for skill_name in dict.fromkeys(selection.skill_names):
        normalized_name = str(skill_name).strip()
        if normalized_name not in available_names:
            continue
        if len(loaded) >= MAX_PLANNER_LOADED_SKILLS:
            break
        cached_content = state.loaded_skills.get(normalized_name)
        if isinstance(cached_content, str) and cached_content.strip():
            loaded[normalized_name] = cached_content
            continue
        result = skills_service.skill_view(normalized_name)
        content = result.get("content")
        if result.get("success") and isinstance(content, str) and content.strip():
            loaded[normalized_name] = content

    await _print_content_block(
        "PLANNER LOADED SKILLS",
        "\n".join(loaded) if loaded else "(none)",
    )
    return loaded


def _format_candidate_plan(full_plan: FullPlan) -> str:
    """Форматирует план-кандидат для prompt-а plan reviewer.

    Args:
        full_plan: План, который нужно проверить до исполнения.

    Returns:
        JSON-строка с objective и задачами плана.
    """

    lines = [f"objective: {full_plan.objective or ''}"]
    if not full_plan.tasks:
        lines.append("tasks: []")
        return "\n".join(lines)

    for task in full_plan.tasks:
        task_id = str(task.task_id).strip() or "unknown"
        task_json = json.dumps(task.model_dump(mode="json"), ensure_ascii=False, indent=2)
        lines.append(_wrap_task_prompt_block(task_id, task_json))
    return "\n\n".join(lines)


def _get_initial_user_query(state: AgentState) -> str:
    """Возвращает исходный запрос пользователя.

    Сначала проверяет поле ``initial_user_query`` в состоянии агента,
    при его отсутствии ищет первое сообщение пользователя в истории.

    Args:
        state: Текущее состояние агента.

    Returns:
        Строка с исходным запросом пользователя или пустая строка.
    """
    if state.initial_user_query:
        return state.initial_user_query

    # Fallback: берём первое HumanMessage из истории
    first_human = next(
        (msg for msg in state.messages if isinstance(msg, HumanMessage)), None
    )
    return str(first_human.content) if first_human else ""


def _format_execution_results(plan: dict[str, Task]) -> str:
    """Форматирует итоги выполнения завершённых задач плана.

    Включает только задачи в статусах COMPLETED, FAILED или SKIPPED.
    Для каждой задачи показывает конфигурацию, превью результата,
    результат валидации и лог ошибки (при наличии).

    Превью для планировщика/replanner берётся как ``full_result`` (финальный
    текст ответа модели worker), иначе ``result_preview`` (например сырой вывод
    инструмента, если модель не вернула отдельного текста).

    Args:
        plan: Словарь задач планировщика.

    Returns:
        Многострочная строка с результатами выполнения,
        либо сообщение об отсутствии завершённых задач.
    """
    if not plan:
        return "Ни одна задача еще не выполнена"

    lines: list[str] = []
    for task_id, task in plan.items():
        if task.status not in FINISHED_STATUSES:
            continue

        line = (
            f"Задача {task_id}"
            f" | Статус выполнения={task.status.value}"
            f" | Описание={task.description}"
        )

        display_preview = (task.full_result or task.result_preview or "").strip()
        if display_preview:
            # Ограничиваем превью константой для единообразия
            line += (
                f" | Превью результата (первые {PREVIEW_MAX_LENGTH} символов)"
                f"={display_preview[:PREVIEW_MAX_LENGTH]}"
            )

        lines.append(_wrap_task_prompt_block(task_id, line))

    if not lines:
        return "Завершённых задач с результатами пока нет."

    return "\n".join(lines)


def _sanitize_task_config(config: dict[str, Any]) -> dict[str, Any]:
    """Очищает пользовательский конфиг задачи от служебных и невалидных ключей.

    Удаляет ключи, входящие в ``RESERVED_CONFIG_KEYS`` (служебные поля модели Task),
    а также пустые ключи. Нормализует ключи к нижнему регистру только для проверки,
    сохраняя оригинальный регистр в итоговом словаре.

    Args:
        config: Исходный словарь конфигурации задачи.

    Returns:
        Очищенный словарь конфигурации без служебных ключей.
    """
    if not config:
        return {}

    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        normalized_key = str(key).strip().lower()

        # Пропускаем пустые ключи
        if not normalized_key:
            continue

        # Пропускаем ключи, совпадающие со служебными полями модели Task
        if normalized_key in RESERVED_CONFIG_KEYS:
            continue

        cleaned[str(key).strip()] = value

    return cleaned


def _canonical_task_payload(
        description: str,
        dependencies: list[str],
        config: dict[str, Any],
) -> str:
    """Формирует каноническое JSON-представление задачи для сравнения.

    Используется для проверки идентичности двух определений задачи
    без учёта порядка ключей.

    Args:
        description: Описание задачи.
        dependencies: Список зависимостей задачи.
        config: Конфигурация задачи (уже очищенная).

    Returns:
        Детерминированная JSON-строка для сравнения задач.
    """
    payload = {
        "description": description.strip(),
        "dependencies": dependencies,
        "config": _sanitize_task_config(config),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _sanitize_dependencies(
        task_id: str,
        dependencies: list[str],
        allowed_dependency_ids: set[str] | None = None,
) -> list[str]:
    """Очищает список зависимостей задачи от дублей и самоссылок.

    Убирает пустые значения, дубликаты и зависимость задачи на саму себя.
    Если передан ``allowed_dependency_ids``, оставляет только зависимости,
    которые уже были объявлены ранее в плане. Это сохраняет топологический
    порядок и не дает задаче зависеть от будущих задач.

    Args:
        task_id: Идентификатор текущей задачи.
        dependencies: Исходный список зависимостей.
        allowed_dependency_ids: Набор задач, на которые можно ссылаться.

    Returns:
        Очищенный список уникальных зависимостей.
    """
    seen: set[str] = set()
    clean: list[str] = []

    for dep in dependencies or []:
        dep_id = str(dep).strip()

        # Пропускаем пустые, самоссылающиеся и дублирующиеся зависимости
        if not dep_id or dep_id == task_id or dep_id in seen:
            continue
        if allowed_dependency_ids is not None and dep_id not in allowed_dependency_ids:
            continue

        seen.add(dep_id)
        clean.append(dep_id)

    return clean


def _is_same_task_definition(
        existing: Task,
        description: str,
        dependencies: list[str],
        config: dict[str, Any],
) -> bool:
    """Проверяет, совпадает ли существующая задача с новым определением.

    Сравнивает каноническое JSON-представление двух задач.

    Args:
        existing: Существующий объект задачи из текущего плана.
        description: Описание новой задачи.
        dependencies: Зависимости новой задачи.
        config: Конфигурация новой задачи.

    Returns:
        ``True``, если определения задач идентичны; иначе ``False``.
    """
    existing_key = _canonical_task_payload(
        description=existing.description,
        dependencies=existing.dependencies,
        config=existing.config,
    )
    candidate_key = _canonical_task_payload(
        description=description,
        dependencies=dependencies,
        config=config,
    )
    return existing_key == candidate_key


def build_full_plan(
        current_plan: dict[str, Task],
        full_plan: FullPlan,
        current_run_id: str,
) -> dict[str, Task]:
    """Строит новый план на основе предложенного LLM и текущего состояния плана.

    Сохраняет существующие задачи, если их определение не изменилось.
    Создаёт новые задачи для изменённых или отсутствующих элементов.
    После построения удаляет зависимости, ссылающиеся на несуществующие задачи.

    Args:
        current_plan: Текущий словарь задач планировщика.
        full_plan: Новый план, предложенный языковой моделью.
        current_run_id: Идентификатор текущего шага выполнения.

    Returns:
        Обновлённый словарь задач планировщика.
    """
    next_plan: dict[str, Task] = {}
    used_ids: set[str] = set()

    for index, planned_task in enumerate(full_plan.tasks, start=1):
        task_id = str(planned_task.task_id).strip() or str(index)

        # Пропускаем дублирующиеся ID в рамках одного прохода
        if task_id in used_ids:
            continue
        allowed_dependency_ids = set(used_ids)
        used_ids.add(task_id)

        description = planned_task.description.strip()
        dependencies = _sanitize_dependencies(
            task_id,
            planned_task.dependencies,
            allowed_dependency_ids=allowed_dependency_ids,
        )
        config = _sanitize_task_config(planned_task.config or {})

        existing_task = current_plan.get(task_id)
        if existing_task and _is_same_task_definition(
                existing_task, description, dependencies, config
        ):
            # Задача не изменилась — сохраняем её текущее состояние
            task = existing_task.model_copy(deep=True)
            task.run_id = current_run_id
            next_plan[task_id] = task
            continue

        # Задача новая или изменилась — создаём с нуля
        next_plan[task_id] = Task(
            task_id=task_id,
            description=description,
            dependencies=dependencies,
            expected_output=planned_task.expected_output,
            validation_criteria=planned_task.validation_criteria,
            suggested_tools=planned_task.suggested_tools,
            suggested_skills=planned_task.suggested_skills,
            required_artifacts=planned_task.required_artifacts,
            config=config,
            status=TaskStatus.PENDING,
            run_id=current_run_id,
        )

    return next_plan


async def _review_and_revise_plan(
        *,
        llm: BaseChatModel,
        plan_review_prompt: str | None,
        full_plan: FullPlan,
        base_messages: list[SystemMessage | HumanMessage],
        initial_user_query: str,
        plan_str: str,
        execution_results: str,
        df_info: str,
        tools_desc: str,
        previous_context: str,
        critic_feedback: str,
) -> FullPlan:
    """Проверяет план до исполнения и при необходимости пересоставляет его один раз.

    Args:
        llm: Языковая модель для plan review и возможной revision-попытки.
        plan_review_prompt: Prompt критика плана. Если ``None`` или пустой,
            проверка пропускается.
        full_plan: План-кандидат, полученный от planner/replanner.
        base_messages: Исходные сообщения, по которым был создан план.
        initial_user_query: Исходный пользовательский запрос.
        plan_str: Текущий runtime-план до применения кандидата.
        execution_results: Сводка завершенных задач.
        df_info: Описание доступных данных и контекста.
        tools_desc: Описание доступных tools.
        previous_context: Предыдущий диалоговый/branch контекст.
        critic_feedback: Накопленная критика предыдущих шагов.

    Returns:
        Исходный или один раз пересоставленный ``FullPlan``. Если plan review
        недоступен или revision не удалась, возвращается исходный план.
    """

    if not plan_review_prompt:
        return full_plan

    schema_str = json.dumps(PlanReview.model_json_schema(), ensure_ascii=False, indent=2)
    review_prompt = plan_review_prompt.format(
        initial_user_query=initial_user_query,
        plan_str=plan_str,
        candidate_plan=_format_candidate_plan(full_plan),
        execution_results=execution_results,
        df_info=df_info,
        tools_desc=tools_desc,
        previous_context=previous_context,
        critic_feedback=critic_feedback,
        schema_str=schema_str,
    )

    try:
        review = await invoke_structured_output(
            llm=llm,
            schema=PlanReview,
            messages=[SystemMessage(content=review_prompt)],
        )
    except Exception as exc:
        await _print_content_block(
            "PLAN REVIEW SKIPPED",
            f"Plan review failed and candidate plan will be used as is.\nError: {exc}",
        )
        return full_plan

    await _print_content_block("PLAN REVIEW", _format_plan_review(review))
    if not review.needs_revision:
        return full_plan

    revised_plan = full_plan
    for _ in range(PLAN_REVIEW_REVISION_ATTEMPTS):
        try:
            revised_plan = await invoke_structured_output(
                llm=llm,
                schema=FullPlan,
                messages=[
                    *base_messages,
                    HumanMessage(content=_format_plan_review_feedback(review)),
                ],
            )
            await _print_content_block(
                "REVISED PLAN AFTER REVIEW",
                _format_full_plan_response(revised_plan),
            )
            return revised_plan
        except Exception as exc:
            await _print_content_block(
                "PLAN REVISION FAILED",
                f"Revision failed and candidate plan will be used as is.\nError: {exc}",
            )
            return full_plan

    return revised_plan


def _build_overwrite_patch(
        current_plan: dict[str, Task],
        next_plan: dict[str, Task],
) -> dict[str, Task | None]:
    """Формирует патч для обновления плана в состоянии агента (LangGraph).

    Логика:
    - Задачи, отсутствующие в новом плане, удаляются (``None``) только если они
      в статусе PENDING или READY.
    - Задачи в «неизменяемых» статусах (RUNNING, NEEDS_VALIDATION и др.)
      сохраняются как есть, даже если новый план предлагает другой вариант.
    - Остальные задачи берутся из нового плана.

    Args:
        current_plan: Текущий словарь задач.
        next_plan: Новый словарь задач после перепланирования.

    Returns:
        Словарь-патч для применения к состоянию агента.
        Значение ``None`` означает удаление задачи.
    """
    patch: dict[str, Task | None] = {}

    # Помечаем для удаления задачи, которых нет в новом плане
    for task_id, current_task in current_plan.items():
        if task_id in next_plan:
            continue
        if current_task.status in REMOVABLE_STATUSES:
            patch[task_id] = None

    # Применяем новые задачи, защищая активные от перезаписи
    for task_id, next_task in next_plan.items():
        existing = current_plan.get(task_id)
        if (
                existing
                and existing.status in IMMUTABLE_RUNTIME_STATUSES
                and next_task.status == TaskStatus.PENDING
        ):
            # Задача уже выполняется или завершена — не трогаем её
            patch[task_id] = existing.model_copy(deep=True)
        else:
            patch[task_id] = next_task

    return patch


def _apply_plan_patch(
        current_plan: dict[str, Task],
        plan_patch: dict[str, Task | None],
) -> dict[str, Task]:
    """Применяет patch плана для получения фактического runtime-плана.

    Args:
        current_plan: Текущий план из состояния агента.
        plan_patch: Patch, который будет передан в LangGraph state update.

    Returns:
        План после применения patch: именно его нужно показывать в терминале
        и сохранять в lineage snapshot.
    """

    effective_plan = dict(current_plan or {})
    for task_id, task in plan_patch.items():
        if task is None:
            effective_plan.pop(task_id, None)
        else:
            effective_plan[task_id] = task
    return effective_plan


async def planner_node(
        state: AgentState,
        llm: BaseChatModel,
        tools: list[BaseTool],
        prompt: str,
        plan_review_prompt: str | None = None,
        force_replan: bool = False,
        lineage_service: LineageService | None = None,
        artifact_service: ArtifactService | None = None,
        skills_service: SkillsService | None = None,
) -> Command:
    """Основной узел планировщика в графе LangGraph.

    Формирует системный промпт на основе текущего состояния агента,
    вызывает языковую модель для получения структурированного плана,
    строит обновлённый план задач и возвращает команду перехода к следующему узлу.

    При ``force_replan=True`` планировщик перестраивает план на основе
    результатов последнего выполнения, не обрабатывая новое сообщение.

    Args:
        state: Текущее состояние агента (план, сообщения, контекст данных).
        llm: Языковая модель для генерации плана.
        tools: Список доступных инструментов агента.
        prompt: Шаблон системного промпта с плейсхолдерами.
        plan_review_prompt: Опциональный prompt для критики плана до запуска.
        force_replan: Если ``True`` — перепланирование по итогам выполнения;
                      если ``False`` — планирование по новому запросу пользователя.
        lineage_service: Опциональный сервис записи plan/replan nodes.
        skills_service: Опциональный сервис чтения skills для предварительной
            загрузки релевантных методик planner-ом.

    Returns:
        :class:`Command` с обновлённым планом и переходом к узлу ``scheduler``,
        либо :class:`Command` с сообщением об ошибке и переходом к узлу ``responder``.
    """
    current_plan: dict[str, Task] = state.plan or {}

    # Используем длину истории как монотонно возрастающий идентификатор шага
    current_run_id = str(len(state.messages))

    # Подготавливаем все части системного промпта
    plan_str = _create_plan_summary(current_plan)
    initial_user_query = _get_initial_user_query(state)
    execution_results = _format_execution_results(current_plan)
    df_info = _format_data_context(state)
    tools_desc = _format_tools(tools)
    previous_context = _format_context(state)
    critic_feedback = _format_critic_feedback(state)

    loaded_planner_skills = await _load_planner_skills(
        llm=llm,
        state=state,
        skills_service=skills_service,
        initial_user_query=initial_user_query,
        plan_str=plan_str,
        execution_results=execution_results,
        tools_desc=tools_desc,
        previous_context=previous_context,
        critic_feedback=critic_feedback,
    )
    planner_skill_context = _format_planner_loaded_skills(loaded_planner_skills)
    if planner_skill_context:
        df_info = f"{df_info}\n\n{planner_skill_context}"

    schema_str = json.dumps(FullPlan.model_json_schema(), ensure_ascii=False, indent=2)
    system_prompt = prompt.format(
        plan_str=plan_str,
        initial_user_query=initial_user_query,
        execution_results=execution_results,
        df_info=df_info,
        tools_desc=tools_desc,
        previous_context=previous_context,
        critic_feedback=critic_feedback,
        schema_str=schema_str,
    )

    # Формируем пользовательский промпт в зависимости от режима
    user_prompt = (
        REPLAN_USER_PROMPT
        if force_replan
        else f"{USER_REQUEST_PREFIX}{state.messages[-1].content}"
    )

    try:
        full_plan: FullPlan = await invoke_structured_output(
            llm=llm,
            schema=FullPlan,
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
        )
        await _print_content_block(
            "REPLANNER MODEL PLAN" if force_replan else "PLANNER MODEL PLAN",
            _format_full_plan_response(full_plan),
        )
        full_plan = await _review_and_revise_plan(
            llm=llm,
            plan_review_prompt=plan_review_prompt,
            full_plan=full_plan,
            base_messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ],
            initial_user_query=initial_user_query,
            plan_str=plan_str,
            execution_results=execution_results,
            df_info=df_info,
            tools_desc=tools_desc,
            previous_context=previous_context,
            critic_feedback=critic_feedback,
        )

        next_plan = build_full_plan(
            current_plan=current_plan,
            full_plan=full_plan,
            current_run_id=current_run_id,
        )
        plan_patch = _build_overwrite_patch(
            current_plan=current_plan,
            next_plan=next_plan,
        )
        effective_plan = _apply_plan_patch(
            current_plan=current_plan,
            plan_patch=plan_patch,
        )

        await _print_content_block(
            "UPDATED EXECUTION PLAN" if force_replan else "EXECUTION PLAN",
            _format_plan_content(effective_plan),
        )

        update_payload: dict[str, Any] = {"plan": plan_patch}
        if loaded_planner_skills:
            update_payload["loaded_skills"] = loaded_planner_skills

        # Сохраняем исходный запрос пользователя при первом планировании
        if not force_replan and not state.initial_user_query:
            update_payload["initial_user_query"] = str(state.messages[-1].content)

        # Сохраняем исходный план при первом планировании
        if not force_replan and not state.initial_plan:
            update_payload["initial_plan"] = {
                task_id: task.model_copy(deep=True)
                for task_id, task in effective_plan.items()
            }

        lineage_update = _create_plan_lineage(
            state=state,
            next_plan=effective_plan,
            full_plan=full_plan,
            force_replan=force_replan,
            lineage_service=lineage_service,
            update_payload=update_payload,
        )
        update_payload.update(lineage_update)
        prompt_trace_artifacts = write_prompt_trace(
            artifact_service=artifact_service,
            run_id=state.run_id,
            node_id=update_payload.get("current_node_id") or state.current_node_id,
            stage="replanner" if force_replan else "planner",
            system_prompt=system_prompt,
            human_prompt=user_prompt,
            payload={
                "initial_user_query": initial_user_query,
                "current_plan_summary": plan_str,
                "execution_results": execution_results,
                "tools_desc": tools_desc,
                "previous_context": previous_context,
                "critic_feedback": critic_feedback,
                "loaded_skill_names": list(loaded_planner_skills.keys()),
                "full_plan": full_plan.model_dump(mode="json"),
            },
        )
        if prompt_trace_artifacts:
            update_payload["artifact_index"] = prompt_trace_artifacts

        return Command(update=update_payload, goto=GOTO_SCHEDULER)

    except Exception as exc:
        return Command(
            goto=GOTO_RESPONDER,
            update={
                "messages": [AIMessage(content=f"Planning failed: {exc}")],
            },
        )


def _create_plan_lineage(
        *,
        state: AgentState,
        next_plan: dict[str, Task],
        full_plan: FullPlan,
        force_replan: bool,
        lineage_service: LineageService | None,
        update_payload: dict[str, Any],
) -> dict[str, Any]:
    if lineage_service is None or not state.run_id:
        return {}

    node_type = "replan_created" if force_replan else "plan_created"
    title = "Replan created" if force_replan else "Plan created"
    parent_ids = state.parent_node_ids or (
        [state.current_node_id] if state.current_node_id else []
    )
    snapshot = state.model_copy(
        update={
            **update_payload,
            "plan": next_plan,
            "current_node_id": state.current_node_id,
            "parent_node_ids": parent_ids,
        },
        deep=True,
    )
    node = lineage_service.create_state_node(
        run_id=state.run_id,
        node_type=node_type,
        title=title,
        parent_ids=parent_ids,
        status="succeeded",
        summary=f"{len(next_plan)} task(s). Objective: {full_plan.objective or ''}".strip(),
        state=snapshot,
        created_by="agent",
        metadata={
            "objective": full_plan.objective,
            "task_ids": list(next_plan.keys()),
            "force_replan": force_replan,
        },
    )

    return {
        "current_node_id": node.node_id,
        "parent_node_ids": [node.node_id],
        "lineage_events": [node.model_dump(mode="json")],
    }

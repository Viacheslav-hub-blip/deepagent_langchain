"""Middleware подтверждения только первого плана DeepAgents в рамках сессии.

Содержит:
- AnalyticsPlanState: расширенный state агента с флагом подтвержденного плана.
- FirstPlanApprovalMiddleware: middleware для interrupt только первого write_todos за сессию.
- FirstPlanApprovalMiddleware.after_model: обработка первого write_todos после ответа модели.
- _context_is_loaded: проверка, что domain context из skills загружен.
- _find_last_ai_message: поиск последнего AIMessage в истории.
- _find_plan_tool_call: поиск tool call встроенного write_todos.
- _is_internal_runner_message: проверка служебного сообщения терминального runner-а.
- _last_user_key: вычисление ключа последнего пользовательского сообщения.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from typing_extensions import NotRequired

PLAN_TOOL_NAME = "write_todos"
INTERNAL_RUNNER_MESSAGE_PREFIX = "Служебная инструкция runner-а:"


class AnalyticsPlanState(AgentState):
    """Расширенный state аналитического агента.

    Args:
        approved_plan_user_key: Ключ пользовательского сообщения, на котором
            план был впервые подтвержден в текущей сессии.
        skills_context_loaded: Признак, что компактный domain context из skills
            загружен в state и добавляется в system message.
        preloaded_skill_paths: Список skill-файлов, которые были прочитаны при
            подготовке domain context.
        preloaded_skills_context: Компактное содержимое прочитанных skill-файлов.
        few_shot_examples_user_key: Ключ последнего пользовательского запроса, для которого подобраны few-shot примеры.
        few_shot_example_names: Названия few-shot примеров, выбранных для текущего запроса.
        few_shot_examples: Полный markdown-текст выбранных few-shot примеров для system prompt.

    Returns:
        State LangChain agent с дополнительным флагом подтверждения плана.
    """

    approved_plan_user_key: NotRequired[str]
    skills_context_loaded: NotRequired[bool]
    preloaded_skill_paths: NotRequired[list[str]]
    preloaded_skills_context: NotRequired[str]
    few_shot_examples_user_key: NotRequired[str]
    few_shot_example_names: NotRequired[list[str]]
    few_shot_examples: NotRequired[str]


class FirstPlanApprovalMiddleware(AgentMiddleware[AnalyticsPlanState]):
    """Запрашивает подтверждение только для первого ``write_todos`` в текущей сессии.

    Args:
        Отсутствуют.

    Returns:
        Middleware, который можно передать в ``create_deep_agent`` через ``middleware``.
    """

    state_schema = AnalyticsPlanState

    def after_model(self, state: AnalyticsPlanState, runtime: Runtime) -> dict[str, Any] | None:
        """Прерывает выполнение перед первым планом и пропускает последующие планы.

        Args:
            state: Текущий state агента с историей сообщений и флагом подтверждения.
            runtime: Runtime LangGraph текущего запуска.

        Returns:
            Обновление state после решения пользователя или ``None``, если interrupt не нужен.
        """

        messages = state.get("messages", [])
        user_key = _last_user_key(messages)
        if state.get("approved_plan_user_key"):
            return None

        last_ai_message = _find_last_ai_message(messages)
        if last_ai_message is None:
            return None

        plan_tool_call = _find_plan_tool_call(last_ai_message)
        if plan_tool_call is None:
            return None

        if not _context_is_loaded(state):
            return {
                "messages": [
                    ToolMessage(
                        content=(
                            "Domain context из skills еще не загружен. "
                            "Не составляй и не выполняй план, пока middleware не добавит "
                            "таблицы, поля, ключи и бизнес-правила из skills."
                        ),
                        name=PLAN_TOOL_NAME,
                        tool_call_id=plan_tool_call["id"],
                        status="error",
                    )
                ]
            }

        decision = interrupt(
            {
                "action_requests": [
                    {
                        "name": PLAN_TOOL_NAME,
                        "args": plan_tool_call.get("args", {}),
                        "description": "Проверьте первоначальный план анализа перед выполнением.",
                    }
                ],
                "review_configs": [
                    {
                        "action_name": PLAN_TOOL_NAME,
                        "allowed_decisions": ["approve", "edit", "reject"],
                    }
                ],
            }
        )["decisions"][0]

        if decision["type"] == "approve":
            return {
                "approved_plan_user_key": user_key,
            }

        if decision["type"] == "edit":
            message = decision.get("message") or "Пользователь просит изменить первоначальный план."
            return {
                "messages": [
                    ToolMessage(
                        content=(
                            "Пользователь просит изменить первоначальный план анализа. "
                            "Не выполняй текущий план. Составь новый write_todos с учетом правки: "
                            f"{message}"
                        ),
                        name=PLAN_TOOL_NAME,
                        tool_call_id=plan_tool_call["id"],
                        status="error",
                    )
                ]
            }

        message = decision.get("message") or "Пользователь отклонил первоначальный план анализа."
        return {
            "messages": [
                ToolMessage(
                    content=message,
                    name=PLAN_TOOL_NAME,
                    tool_call_id=plan_tool_call["id"],
                    status="error",
                )
            ]
        }


def _context_is_loaded(state: AnalyticsPlanState) -> bool:
    """Проверяет, что попытка загрузки domain context из skills уже выполнена.

    Args:
        state: Текущий state аналитического агента.

    Returns:
        ``True``, если middleware preload отметило skills context как загруженный.
        Содержимое context может быть пустым, если папка skills отсутствует или пока пуста.
    """

    return bool(state.get("skills_context_loaded"))


def _find_last_ai_message(messages: list[Any]) -> AIMessage | None:
    """Находит последнее AIMessage в истории сообщений.

    Args:
        messages: История сообщений LangChain agent state.

    Returns:
        Последнее AIMessage или ``None``, если его нет.
    """

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _find_plan_tool_call(message: AIMessage) -> dict[str, Any] | None:
    """Находит вызов встроенного ``write_todos`` в AIMessage.

    Args:
        message: Сообщение модели с возможными tool calls.

    Returns:
        Tool call ``write_todos`` или ``None``, если его нет.
    """

    for tool_call in message.tool_calls:
        if tool_call.get("name") == PLAN_TOOL_NAME:
            return tool_call
    return None


def _is_internal_runner_message(message: HumanMessage) -> bool:
    """Определяет служебное сообщение терминального runner-а.

    Args:
        message: Сообщение пользователя из истории LangGraph.

    Returns:
        ``True``, если сообщение добавлено runner-ом для автоматического
        продолжения текущей задачи и не должно считаться новым запросом
        пользователя.
    """

    content = message.content
    if isinstance(content, str):
        return content.startswith(INTERNAL_RUNNER_MESSAGE_PREFIX)
    return False


def _last_user_key(messages: list[Any]) -> str:
    """Вычисляет стабильный ключ последнего пользовательского сообщения.

    Args:
        messages: История сообщений LangChain agent state.

    Returns:
        Строковый ключ последнего HumanMessage или fallback ``no-user-message``.
    """

    for index, message in reversed(list(enumerate(messages))):
        if isinstance(message, HumanMessage):
            if _is_internal_runner_message(message):
                continue
            message_id = getattr(message, "id", None)
            if message_id:
                return str(message_id)
            return f"{index}:{message.content}"
    return "no-user-message"


__all__ = [
    "AnalyticsPlanState",
    "FirstPlanApprovalMiddleware",
]

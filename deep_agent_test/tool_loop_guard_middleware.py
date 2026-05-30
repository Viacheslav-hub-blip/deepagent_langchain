"""Middleware защиты от зацикливания на подряд идущих вызовах одного и того же tool.

Назначение: не дать агенту бесконечно дёргать один и тот же инструмент, лишь слегка меняя
аргументы (другие фильтры, поля, `max_rows`, `include_schema` и т.п.). Такой «варьирующий»
цикл не ловится проверкой на идентичные аргументы, но хорошо виден по серии подряд идущих
вызовов одного и того же tool без других действий.

Логика: перед выполнением tool middleware считает, сколько раз ПОДРЯД (без иных tool и без
текстового ответа модели между ними) вызывался этот же tool. Если длина серии достигла
``max_consecutive_tool_calls``, инструмент НЕ выполняется — вместо результата возвращается
ToolMessage со статусом error, который просит сменить подход (другой инструмент/шаг плана)
или завершить шаг по уже полученным данным. Это разрывает цикл, но не убивает прогон.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deep_agent_test.agent_state import extract_state_messages


@dataclass(frozen=True)
class ToolLoopGuardMiddleware(AgentMiddleware):
    """Блокирует вызов tool после N подряд идущих вызовов того же tool.

    Args:
        max_consecutive_tool_calls: Сколько подряд идущих вызовов одного tool разрешено.
            На следующем вызове того же tool (без других действий между ними) инструмент
            не выполняется, а модель получает инструкцию сменить подход или завершить шаг.
        exclude_tools: Имена инструментов, к которым guard не применяется.
    """

    max_consecutive_tool_calls: int = 4
    exclude_tools: frozenset[str] = frozenset()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Синхронно проверяет длину серии повторов перед выполнением tool."""

        blocked = self._loop_block_message(request)
        if blocked is not None:
            return blocked
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Асинхронно проверяет длину серии повторов перед выполнением tool."""

        blocked = self._loop_block_message(request)
        if blocked is not None:
            return blocked
        return await handler(request)

    def _loop_block_message(self, request: ToolCallRequest) -> ToolMessage | None:
        """Возвращает блокирующий ToolMessage, если серия повторов исчерпана, иначе None."""

        tool_call = request.tool_call or {}
        tool_name = str(tool_call.get("name") or "")
        if not tool_name or tool_name in self.exclude_tools:
            return None

        # Серия включает текущий вызов: его AIMessage уже в истории к моменту wrap_tool_call.
        consecutive_count = _count_trailing_same_tool_calls(request.state, tool_name)
        if consecutive_count <= self.max_consecutive_tool_calls:
            return None

        return ToolMessage(
            content=(
                f"Вызов инструмента `{tool_name}` заблокирован: он вызван "
                f"{self.max_consecutive_tool_calls} раз(а) подряд без других действий "
                "(обнаружен цикл повторных вызовов). Менять аргументы (фильтры, поля, "
                "`max_rows`, `include_schema`) того же инструмента и повторять — НЕ прогресс. "
                "Либо смени подход (другой инструмент или следующий шаг плана), либо заверши "
                "шаг и верни результат по уже полученным данным, честно отметив ограничение."
            ),
            tool_call_id=str(tool_call.get("id") or ""),
            name=tool_name,
            status="error",
        )


def _count_trailing_same_tool_calls(state: Any, tool_name: str) -> int:
    """Считает длину серии подряд идущих вызовов одного tool в конце истории.

    Серия прерывается, как только встречается AIMessage, который вызывает другой tool или
    отвечает текстом без tool_calls. Параллельные вызовы в одном AIMessage учитываются
    поштучно, только если ВСЕ они адресованы одному и тому же ``tool_name``.
    """

    messages = extract_state_messages(state)
    count = 0
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            break
        names = [str(tool_call.get("name") or "") for tool_call in tool_calls]
        if all(name == tool_name for name in names):
            count += len(names)
        else:
            break
    return count


__all__ = ["ToolLoopGuardMiddleware"]

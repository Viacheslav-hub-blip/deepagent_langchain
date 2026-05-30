"""Middleware жёсткого ограничения числа циклов critic внутри data-retrieval-agent.

Назначение: не дать data-retrieval-agent бесконечно гонять внутренний цикл с
`data-retrieval-critic`. Критик вызывается через `task(subagent_type="data-retrieval-critic")`.
Каждый такой вызов — одна итерация проверки. После ``max_critic_iterations`` итераций
следующий вызов критика НЕ выполняется: вместо запуска критика возвращается ToolMessage,
который просит агента завершить шаг и отдать текущий результат supervisor-у.

В отличие от ``ToolLoopGuardMiddleware`` (ловит идентичные аргументы), этот guard считает
вызовы критика независимо от текста задания — лимит именно на число итераций critic↔subagent.
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
class CriticLoopCapMiddleware(AgentMiddleware):
    """Ограничивает число вызовов внутреннего critic в одном data-retrieval шаге.

    Args:
        critic_subagent_type: Значение ``subagent_type`` критика в вызове ``task``.
        max_critic_iterations: Сколько вызовов критика разрешено. Следующий блокируется.
        task_tool_name: Имя tool делегирования (по умолчанию ``task``).
    """

    critic_subagent_type: str
    max_critic_iterations: int = 3
    task_tool_name: str = "task"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Синхронно блокирует вызов критика, если лимит итераций исчерпан."""

        blocked = self._cap_message(request)
        if blocked is not None:
            return blocked
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Асинхронно блокирует вызов критика, если лимит итераций исчерпан."""

        blocked = self._cap_message(request)
        if blocked is not None:
            return blocked
        return await handler(request)

    def _cap_message(self, request: ToolCallRequest) -> ToolMessage | None:
        """Возвращает блокирующий ToolMessage, если лимит итераций критика исчерпан."""

        tool_call = request.tool_call or {}
        if str(tool_call.get("name") or "") != self.task_tool_name:
            return None
        if not _is_critic_call(tool_call.get("args"), self.critic_subagent_type):
            return None

        # Счётчик включает текущий вызов (AIMessage уже в истории к моменту wrap_tool_call).
        critic_calls = _count_critic_calls(
            request.state,
            task_tool_name=self.task_tool_name,
            critic_subagent_type=self.critic_subagent_type,
        )
        if critic_calls <= self.max_critic_iterations:
            return None

        return ToolMessage(
            content=(
                f"Цикл с `{self.critic_subagent_type}` остановлен: лимит "
                f"{self.max_critic_iterations} проверок исчерпан. Больше не вызывай critic. "
                "Заверши шаг и верни supervisor-у финальный отчёт по фактическим tool results, "
                "которые уже есть в контексте. Если остаётся неустранённое сомнение, честно "
                "отметь его в `limitations`/`summary`, но не уходи в новый цикл проверок."
            ),
            tool_call_id=str(tool_call.get("id") or ""),
            name=self.task_tool_name,
            status="error",
        )


def _is_critic_call(args: Any, critic_subagent_type: str) -> bool:
    """Проверяет, что это вызов ``task`` именно критика по ``subagent_type``."""

    if not isinstance(args, dict):
        return False
    return str(args.get("subagent_type") or "") == critic_subagent_type


def _count_critic_calls(state: Any, *, task_tool_name: str, critic_subagent_type: str) -> int:
    """Считает, сколько раз критик уже вызывался через ``task`` в истории сообщений."""

    messages = extract_state_messages(state)
    count = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
            args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            if str(name or "") == task_tool_name and _is_critic_call(args, critic_subagent_type):
                count += 1
    return count


__all__ = ["CriticLoopCapMiddleware"]

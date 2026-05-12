"""Backend helpers for using the research agent without a UI."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .models import AgentState
from .services.branch_resume_service import BranchResumeService
from .services.lineage_service import LineageService


class ChatRunResult(BaseModel):
    """Результат одного прогона графа из вспомогательных chat/runner функций."""

    state: AgentState
    run_id: str = ""
    final_report: str | None = None


def build_chat_initial_state(
        user_query: str,
        *,
        session_id: str = "",
        user_id: str | None = None,
        filesystem_context: dict[str, str] | None = None,
) -> AgentState:
    """Build a clean state for a normal chat-style research run."""

    return AgentState(
        session_id=session_id,
        user_id=user_id,
        initial_user_query=user_query,
        messages=[HumanMessage(content=user_query)],
        filesystem_context=filesystem_context or {},
    )


def build_branch_chat_state(
        *,
        lineage_service: LineageService,
        branch_run_id: str,
        branch_node_id: str | None = None,
) -> AgentState:
    """Build state for optional branch/resume execution."""

    return BranchResumeService(lineage_service).build_initial_state(
        branch_run_id=branch_run_id,
        branch_node_id=branch_node_id,
    )


async def run_chat_agent(
        graph: Any,
        user_query: str,
        *,
        session_id: str = "",
        user_id: str | None = None,
        filesystem_context: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
) -> ChatRunResult:
    """Run a graph from a plain chat request and return the final state."""

    initial_state = build_chat_initial_state(
        user_query,
        session_id=session_id,
        user_id=user_id,
        filesystem_context=filesystem_context,
    )
    return await run_agent_from_state(graph, initial_state, config=config)


async def run_agent_from_state(
        graph: Any,
        state: AgentState,
        *,
        config: dict[str, Any] | None = None,
        stream_console: bool = False,
) -> ChatRunResult:
    """Run a compiled graph from an explicit AgentState."""

    if stream_console:
        return await _run_agent_from_state_streaming(graph, state, config=config)

    if config is None:
        raw_result = await graph.ainvoke(state)
    else:
        raw_result = await graph.ainvoke(state, config=config)

    final_state = _coerce_agent_state(raw_result, fallback=state)
    return ChatRunResult(
        state=final_state,
        run_id=final_state.run_id,
        final_report=final_state.final_report,
    )


async def _run_agent_from_state_streaming(
        graph: Any,
        state: AgentState,
        *,
        config: dict[str, Any] | None = None,
) -> ChatRunResult:
    """Запускает graph через ``astream`` и печатает читаемые обновления в консоль.

    Args:
        graph: Скомпилированный LangGraph workflow с методом ``astream``.
        state: Начальное состояние агента.
        config: Опциональный LangGraph config.

    Returns:
        ChatRunResult с последним состоянием graph после потокового выполнения.
    """

    if not callable(getattr(graph, "astream", None)):
        return await run_agent_from_state(graph, state, config=config)

    last_state = state
    update_number = 0
    try:
        stream_kwargs: dict[str, Any] = {"stream_mode": "values"}
        if config is not None:
            stream_kwargs["config"] = config
        async for chunk in graph.astream(state, **stream_kwargs):
            update_number += 1
            last_state = _coerce_agent_state(chunk, fallback=last_state)
            await _print_agent_stream_update(update_number, last_state)
    except TypeError:
        return await run_agent_from_state(graph, state, config=config)

    return ChatRunResult(
        state=last_state,
        run_id=last_state.run_id,
        final_report=last_state.final_report,
    )


async def _print_agent_stream_update(update_number: int, state: AgentState) -> None:
    """Пропускает служебные обновления stream-режима без вывода в консоль.

    Args:
        update_number: Порядковый номер обновления из ``astream``.
        state: Текущее полное состояние AgentState.

    Returns:
        ``None``. Функция оставлена как extension point для stream-режима.
    """

    _ = update_number, state


def _coerce_agent_state(raw_result: Any, *, fallback: AgentState) -> AgentState:
    if isinstance(raw_result, AgentState):
        return raw_result
    if isinstance(raw_result, dict):
        merged = fallback.model_dump()
        merged.update(raw_result)
        return AgentState.model_validate(merged)
    return fallback


__all__ = [
    "ChatRunResult",
    "build_branch_chat_state",
    "build_chat_initial_state",
    "run_agent_from_state",
    "run_chat_agent",
]

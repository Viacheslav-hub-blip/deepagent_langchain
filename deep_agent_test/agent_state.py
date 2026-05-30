"""Расширенный state аналитического DeepAgent для skills middleware."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentState
from typing_extensions import NotRequired


class AnalyticsAgentState(AgentState):
    """State агента с полями предзагрузки skills."""

    skills_context_loaded: NotRequired[bool]
    preloaded_skill_paths: NotRequired[list[str]]
    preloaded_skills_index: NotRequired[list[dict[str, str]]]
    preloaded_skills_context: NotRequired[str]
    preloaded_skills_selection_user_key: NotRequired[str]
    materialized_skill_paths: NotRequired[list[str]]


def extract_state_messages(state: Any) -> list[Any]:
    """Достаёт список сообщений из state (dict-подобный AgentState или объект).

    Возвращает пустой список, если поле ``messages`` отсутствует или не список.
    Используется middleware, которым нужна история сообщений из ``ToolCallRequest.state``.
    """

    if isinstance(state, dict):
        messages = state.get("messages")
    else:
        messages = getattr(state, "messages", None)
    return messages if isinstance(messages, list) else []


__all__ = ["AnalyticsAgentState", "extract_state_messages"]

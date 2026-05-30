"""Тесты CriticLoopCapMiddleware: жёсткий лимит циклов task(critic)."""

from __future__ import annotations

import unittest
from typing import Any

from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_agent_test.critic_loop_cap_middleware import CriticLoopCapMiddleware

CRITIC = "data-retrieval-critic"


def _critic_ai_message(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "task", "args": {"subagent_type": CRITIC, "description": "x"}, "id": call_id}],
    )


def _request(state_messages: list[Any], current_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": "task", "args": {"subagent_type": CRITIC, "description": "x"}, "id": current_id},
        tool=None,
        state={"messages": state_messages},
        runtime=None,
    )


def _handler(_request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="critic verdict", tool_call_id="ran", name="task")


class CriticLoopCapMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mw = CriticLoopCapMiddleware(critic_subagent_type=CRITIC, max_critic_iterations=3)

    def test_allows_calls_up_to_limit(self) -> None:
        # Третий вызов критика (в истории 3 критик-вызова, включая текущий) ещё разрешён.
        messages = [HumanMessage(content="task"), _critic_ai_message("c1"), _critic_ai_message("c2"), _critic_ai_message("c3")]
        result = self.mw.wrap_tool_call(_request(messages, "c3"), _handler)
        self.assertEqual(result.content, "critic verdict")

    def test_blocks_call_beyond_limit(self) -> None:
        messages = [
            HumanMessage(content="task"),
            _critic_ai_message("c1"),
            _critic_ai_message("c2"),
            _critic_ai_message("c3"),
            _critic_ai_message("c4"),
        ]
        result = self.mw.wrap_tool_call(_request(messages, "c4"), _handler)
        self.assertEqual(result.status, "error")
        self.assertIn("лимит", result.content.lower())
        self.assertEqual(result.tool_call_id, "c4")

    def test_ignores_non_critic_task_calls(self) -> None:
        retrieval_call = AIMessage(
            content="",
            tool_calls=[{"name": "task", "args": {"subagent_type": "data-retrieval-agent"}, "id": "r1"}],
        )
        request = ToolCallRequest(
            tool_call={"name": "task", "args": {"subagent_type": "data-retrieval-agent"}, "id": "r1"},
            tool=None,
            state={"messages": [retrieval_call] * 10},
            runtime=None,
        )
        result = self.mw.wrap_tool_call(request, _handler)
        self.assertEqual(result.content, "critic verdict")


if __name__ == "__main__":
    unittest.main()

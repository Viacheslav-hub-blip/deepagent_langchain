"""Тесты терминального runner-а native Analytics DeepAgent.

Содержит:
- FakeAgentSnapshot: снимок state для тестового агента.
- FakeAgent: тестовый агент с ``get_state`` и ``invoke`` без вызова LLM.
- NativeAnalyticsChatRunnerTests: проверки автопродолжения до финального ответа.
"""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, ToolMessage

from deep_agent_test.run_native_analytics_chat import (
    build_continue_instruction,
    continue_until_agent_boundary,
    has_pending_tool_calls,
    has_only_final_response_todo_in_progress,
    last_agent_response_text,
    needs_final_response_after_completed_todos,
    should_continue_agent_loop,
)


class FakeAgentSnapshot:
    """Хранит значения checkpoint-state тестового агента.

    Args:
        values: Словарь state, который должен вернуть ``agent.get_state``.

    Returns:
        Объект со свойством ``values`` как у LangGraph snapshot.
    """

    def __init__(self, values: dict) -> None:
        """Инициализирует снимок state.

        Args:
            values: Значения state для возврата runner-у.

        Returns:
            None.
        """

        self.values = values


class FakeAgent:
    """Имитирует LangGraph agent для проверки terminal runner-а без LLM.

    Args:
        state_values: Последовательность state, которую должен возвращать ``get_state``.

    Returns:
        Тестовый агент с подсчетом служебных ``invoke``.
    """

    def __init__(self, state_values: list[dict]) -> None:
        """Инициализирует тестовый агент.

        Args:
            state_values: State-значения для последовательных вызовов ``get_state``.

        Returns:
            None.
        """

        self.state_values = state_values
        self.get_state_calls = 0
        self.invoke_calls: list[dict] = []

    def get_state(self, config: dict) -> FakeAgentSnapshot:
        """Возвращает следующий checkpoint-state.

        Args:
            config: Config LangGraph, который в тесте не используется.

        Returns:
            Снимок state для текущего шага.
        """

        index = min(self.get_state_calls, len(self.state_values) - 1)
        self.get_state_calls += 1
        return FakeAgentSnapshot(self.state_values[index])

    def invoke(self, payload: dict, config: dict) -> dict:
        """Фиксирует служебное продолжение runner-а.

        Args:
            payload: Сообщение, которое runner отправляет агенту.
            config: Config LangGraph, который в тесте не используется.

        Returns:
            Минимальный сырой результат ``agent.invoke``.
        """

        self.invoke_calls.append(payload)
        return {"messages": []}


class NativeAnalyticsChatRunnerTests(unittest.TestCase):
    """Проверяет правила автоматического продолжения терминального runner-а.

    Args:
        Отсутствуют.

    Returns:
        None.
    """

    def test_final_response_todo_requests_final_answer_instruction(self) -> None:
        """Проверяет служебную инструкцию для оставшегося финального ответа.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        result = {
            "todos": [
                {"content": "Получить данные.", "status": "completed"},
                {"content": "Сформировать финальный ответ.", "status": "in_progress"},
            ],
            "messages": [ToolMessage(content="Updated todo list", tool_call_id="todo-1", name="write_todos")],
        }

        self.assertTrue(has_only_final_response_todo_in_progress(result))
        self.assertTrue(should_continue_agent_loop(result, require_progress=False))
        self.assertIn("сформируй финальный ответ", build_continue_instruction(result, require_progress=False))

    def test_completed_todos_with_tool_message_still_need_final_response(self) -> None:
        """Проверяет продолжение после закрытия todo без AI-ответа.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        result = {
            "todos": [
                {"content": "Получить данные.", "status": "completed"},
                {"content": "Сформировать финальный ответ.", "status": "completed"},
            ],
            "messages": [ToolMessage(content="Updated todo list", tool_call_id="todo-1", name="write_todos")],
        }

        self.assertTrue(needs_final_response_after_completed_todos(result))
        self.assertTrue(should_continue_agent_loop(result, require_progress=False))

    def test_completed_todos_with_ai_message_do_not_continue(self) -> None:
        """Проверяет остановку runner-а после финального AI-ответа.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        result = {
            "todos": [
                {"content": "Получить данные.", "status": "completed"},
                {"content": "Сформировать финальный ответ.", "status": "completed"},
            ],
            "messages": [AIMessage(content="Город пользователя: Moscow.")],
        }

        self.assertFalse(needs_final_response_after_completed_todos(result))
        self.assertFalse(should_continue_agent_loop(result, require_progress=False))

    def test_checkpoint_state_allows_continue_after_raw_tool_call_payload(self) -> None:
        """Проверяет продолжение по полному checkpoint-state после tool call.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool_call = {"name": "task", "args": {"description": "Получить данные."}, "id": "task-1"}
        raw_payload = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
        checkpoint_after_tool = {
            "todos": [{"content": "Получить данные.", "status": "in_progress"}],
            "messages": [
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content="Данные получены.", tool_call_id="task-1", name="task"),
            ],
        }
        checkpoint_after_final = {
            "todos": [{"content": "Получить данные.", "status": "completed"}],
            "messages": [AIMessage(content="Город пользователя: Moscow.")],
        }
        agent = FakeAgent([checkpoint_after_tool, checkpoint_after_final])

        result = continue_until_agent_boundary(agent, {"configurable": {"thread_id": "test"}}, raw_payload)

        self.assertEqual(len(agent.invoke_calls), 1)
        self.assertEqual(last_agent_response_text(result), "Город пользователя: Moscow.")

    def test_pending_tool_call_requires_missing_tool_message(self) -> None:
        """Проверяет, что отвеченный tool call не считается pending.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool_call = {"name": "task", "args": {"description": "Получить данные."}, "id": "task-1"}
        raw_payload = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
        completed_state = {
            "todos": [{"content": "Получить данные.", "status": "in_progress"}],
            "messages": [
                AIMessage(content="", tool_calls=[tool_call]),
                ToolMessage(content="Данные получены.", tool_call_id="task-1", name="task"),
            ],
        }

        self.assertTrue(has_pending_tool_calls(raw_payload))
        self.assertFalse(has_pending_tool_calls(completed_state))
        self.assertTrue(should_continue_agent_loop(completed_state, require_progress=False))

    def test_last_agent_response_skips_trailing_empty_ai_message(self) -> None:
        """Проверяет печать финального ответа перед пустым завершающим AIMessage.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        result = {"messages": [AIMessage(content="Город пользователя: Moscow."), AIMessage(content="")]}

        self.assertEqual(last_agent_response_text(result), "Город пользователя: Moscow.")

    def test_continue_until_agent_boundary_stops_on_stagnant_state(self) -> None:
        """Проверяет защиту от зацикливания при повторении одного state."""

        stagnant_state = {
            "todos": [{"content": "Получить данные.", "status": "in_progress"}],
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "task", "args": {"description": "Получить данные."}, "id": "task-1"}],
                ),
                ToolMessage(content="Не хватает входных данных.", tool_call_id="task-1", name="task"),
            ],
        }
        agent = FakeAgent([stagnant_state, stagnant_state, stagnant_state, stagnant_state])

        result = continue_until_agent_boundary(
            agent,
            {"configurable": {"thread_id": "test"}},
            {"messages": []},
            max_auto_continue_steps=10,
            max_stagnant_steps=2,
        )

        self.assertGreaterEqual(len(agent.invoke_calls), 1)
        self.assertEqual(result.get("todos"), stagnant_state["todos"])


if __name__ == "__main__":
    unittest.main()

"""Тесты терминального runner-а native Analytics DeepAgent.

Содержит:
- NativeAnalyticsChatRunnerTests: проверки автопродолжения до финального ответа.
"""

from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, ToolMessage

from deep_agent_test.run_native_analytics_chat import (
    build_continue_instruction,
    has_only_final_response_todo_in_progress,
    needs_final_response_after_completed_todos,
    should_continue_agent_loop,
)


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


if __name__ == "__main__":
    unittest.main()

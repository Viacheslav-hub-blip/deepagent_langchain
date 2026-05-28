"""Тесты middleware трассировки аналитического DeepAgent.

Содержит:
- TraceLoggingMiddlewareTests: проверки очистки служебного state из результата ``task``.
"""

from __future__ import annotations

import unittest

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deep_agent_test.trace_logging_middleware import _strip_service_state_from_task_command


class TraceLoggingMiddlewareTests(unittest.TestCase):
    """Проверяет точечные преобразования результата tool-вызовов.

    Args:
        Отсутствуют.

    Returns:
        None.
    """

    def test_task_command_drops_service_state_keys(self) -> None:
        """Проверяет удаление больших служебных полей subagent-а.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        command = Command(
            update={
                "messages": [ToolMessage(content="Готово.", tool_call_id="task-1", name="task")],
                "files": {"result.csv": "content"},
                "preloaded_skills_context": "очень большой context",
                "preloaded_skill_paths": ["/skills/example/SKILL.md"],
                "skills_context_loaded": True,
                "approved_plan_user_key": "user-1",
                "domain_result": {"city": "Moscow"},
            }
        )

        cleaned = _strip_service_state_from_task_command(command, tool_name="task")

        self.assertIsInstance(cleaned, Command)
        self.assertIn("messages", cleaned.update)
        self.assertIn("files", cleaned.update)
        self.assertIn("domain_result", cleaned.update)
        self.assertNotIn("preloaded_skills_context", cleaned.update)
        self.assertNotIn("preloaded_skill_paths", cleaned.update)
        self.assertNotIn("skills_context_loaded", cleaned.update)
        self.assertNotIn("approved_plan_user_key", cleaned.update)

    def test_non_task_command_is_not_changed(self) -> None:
        """Проверяет, что результат других tools не очищается.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        command = Command(update={"preloaded_skills_context": "context"})

        cleaned = _strip_service_state_from_task_command(command, tool_name="write_todos")

        self.assertIs(cleaned, command)


if __name__ == "__main__":
    unittest.main()

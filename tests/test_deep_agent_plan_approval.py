"""Тесты одноразового подтверждения плана DeepAgents в рамках сессии.

Содержит:
- PlanApprovalMiddlewareTests: проверки фильтрации служебных сообщений runner-а.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

from langchain_core.messages import AIMessage, HumanMessage


def load_plan_approval_module() -> ModuleType:
    """Загружает модуль middleware без импорта пакета deep_agent_test.

    Args:
        Отсутствуют.

    Returns:
        Загруженный Python-модуль ``plan_approval_middleware``.
    """

    module_path = Path(__file__).resolve().parents[1] / "deep_agent_test" / "plan_approval_middleware.py"
    spec = importlib.util.spec_from_file_location("plan_approval_middleware_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить spec для {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN_APPROVAL_MODULE = load_plan_approval_module()
_last_user_key = PLAN_APPROVAL_MODULE._last_user_key
FirstPlanApprovalMiddleware = PLAN_APPROVAL_MODULE.FirstPlanApprovalMiddleware


class PlanApprovalMiddlewareTests(unittest.TestCase):
    """Проверяет логику определения нового пользовательского запроса.

    Args:
        Отсутствуют.

    Returns:
        None.
    """

    def test_service_continue_message_does_not_change_user_key(self) -> None:
        """Проверяет, что служебное продолжение runner-а не считается новым запросом.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        messages = [
            HumanMessage(content="Сколько сработок связано с образовательными услугами?", id="user-1"),
            HumanMessage(content="Служебная инструкция runner-а: продолжай выполнение текущей задачи.", id="runner-1"),
        ]

        self.assertEqual(_last_user_key(messages), "user-1")

    def test_real_user_message_after_approval_does_not_reset_session_approval(self) -> None:
        """Проверяет, что новый запрос не сбрасывает подтверждение текущей сессии.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        messages = [
            HumanMessage(content="Сколько сработок связано с образовательными услугами?", id="user-1"),
            HumanMessage(content="Служебная инструкция runner-а: продолжай выполнение текущей задачи.", id="runner-1"),
            HumanMessage(content="Теперь проверь переводы за тот же период.", id="user-2"),
        ]

        self.assertEqual(_last_user_key(messages), "user-2")

    def test_approved_session_skips_new_write_todos_interrupt(self) -> None:
        """Проверяет, что подтвержденная сессия пропускает новые планы без interrupt.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        middleware = FirstPlanApprovalMiddleware()
        state = {
            "approved_plan_user_key": "user-1",
            "messages": [
                HumanMessage(content="Первый запрос.", id="user-1"),
                HumanMessage(content="Новый запрос в той же сессии.", id="user-2"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_todos",
                            "args": {"todos": [{"content": "Новый план", "status": "in_progress"}]},
                            "id": "tool-write-todos-1",
                        }
                    ],
                ),
            ],
        }

        self.assertIsNone(middleware.after_model(state, runtime=None))


if __name__ == "__main__":
    unittest.main()

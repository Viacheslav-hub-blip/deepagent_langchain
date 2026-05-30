"""Тесты терминального runner-а native Analytics DeepAgent."""

from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from deep_agent_test.run_native_analytics_chat import (
    format_tool_call,
    format_tool_result,
    format_todos_for_user,
    last_agent_response_text,
    print_loaded_skills_once,
    print_messages,
)


class NativeAnalyticsChatRunnerTests(unittest.TestCase):
    def test_last_agent_response_skips_trailing_empty_ai_message(self) -> None:
        result = {"messages": [AIMessage(content="Город пользователя: Moscow."), AIMessage(content="")]}
        self.assertEqual(last_agent_response_text(result), "Город пользователя: Moscow.")

    def test_last_agent_response_skips_ai_message_with_tool_calls(self) -> None:
        result = {
            "messages": [
                AIMessage(content="", tool_calls=[{"name": "task", "args": {}, "id": "task-1"}]),
            ]
        }
        self.assertEqual(last_agent_response_text(result), "")

    def test_last_agent_response_finds_text_before_tool_message(self) -> None:
        result = {
            "messages": [
                AIMessage(content="Город пользователя: Moscow."),
                ToolMessage(content="tool result", tool_call_id="1", name="read_table"),
            ]
        }
        self.assertEqual(last_agent_response_text(result), "Город пользователя: Moscow.")

    def test_format_todos_for_user(self) -> None:
        text = format_todos_for_user(
            [
                {"content": "Получить данные.", "status": "pending"},
                {"content": "Сформировать ответ.", "status": "completed"},
            ]
        )
        self.assertIn("1. Получить данные.", text)
        self.assertIn("2. Сформировать ответ.", text)

    def test_print_loaded_skills_once(self) -> None:
        result = {"preloaded_skill_paths": ["/skills/hit-table/SKILL.md"]}
        self.assertTrue(print_loaded_skills_once(result, already_printed=False))
        self.assertTrue(print_loaded_skills_once(result, already_printed=True))

    def test_format_tool_call_read_table(self) -> None:
        text = format_tool_call(
            "read_table",
            {"table_name": "hits", "filters": [{"column": "event_id", "operator": "eq", "value": "abc"}]},
        )
        self.assertIn("[Tool call] read_table", text)
        self.assertIn("hits", text)

    def test_format_tool_result_read_table(self) -> None:
        payload = {
            "status": "success",
            "table_name": "hits",
            "returned_rows": 2,
            "columns": ["event_id", "city"],
            "rows": [{"event_id": "abc", "city": "Moscow"}],
        }
        text = format_tool_result("read_table", json.dumps(payload, ensure_ascii=False))
        self.assertIn("[Tool result] read_table", text)
        self.assertIn("returned_rows: 2", text)
        self.assertIn("preview:", text)

    def test_print_messages_returns_new_cursor(self) -> None:
        state = {
            "messages": [
                AIMessage(content="Город: Moscow."),
                AIMessage(content="", tool_calls=[{"name": "read_file", "args": {}, "id": "1"}]),
                ToolMessage(content="ok", tool_call_id="1", name="read_file"),
            ]
        }
        cursor = print_messages(state, start_index=0)
        self.assertEqual(cursor, 3)
        self.assertEqual(print_messages(state, start_index=cursor), cursor)


if __name__ == "__main__":
    unittest.main()

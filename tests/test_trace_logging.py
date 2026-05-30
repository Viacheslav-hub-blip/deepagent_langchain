"""Тесты txt-логгера хода агента (FileTraceCallbackHandler)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from deep_agent_test.trace_logging import FileTraceCallbackHandler


class TraceLoggingTests(unittest.TestCase):
    def _run_basic_flow(self, handler: FileTraceCallbackHandler) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_table",
                    "description": "выборка из таблицы",
                    "parameters": {"type": "object", "properties": {"table_name": {"type": "string"}}},
                },
            }
        ]
        handler.on_chat_model_start(
            {"name": "model"},
            [[SystemMessage(content="Ты supervisor."), HumanMessage(content="найди сработки")]],
            run_id="r1",
            invocation_params={"tools": tools},
            metadata={"langgraph_node": "agent"},
        )
        ai = AIMessage(
            content="Делаю выборку.",
            tool_calls=[{"name": "read_table", "args": {"table_name": "hits"}, "id": "c1", "type": "tool_call"}],
            response_metadata={"token_usage": {"total_tokens": 123}},
        )
        handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=ai)]]), run_id="r1")
        handler.on_tool_start(
            {"name": "read_table"},
            "{}",
            inputs={"table_name": "hits", "select_columns": "event_description"},
            run_id="t1",
        )
        handler.on_tool_end(
            ToolMessage(content="event_description=Оплата обучения", tool_call_id="c1", name="read_table"),
            run_id="t1",
        )

    def test_logs_sections_in_order_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = FileTraceCallbackHandler(Path(temp_dir) / "trace.txt")
            self._run_basic_flow(handler)
            text = handler.file_path.read_text(encoding="utf-8")

        # Все нужные секции присутствуют.
        for marker in ("TOOLS (1)", "SYSTEM PROMPT", "USER MESSAGE", "AGENT RESPONSE", "TOOL CALL", "TOOL RESULT"):
            self.assertIn(marker, text)

        # Правильная последовательность.
        order = [text.index(m) for m in ("TOOLS (1)", "SYSTEM PROMPT", "AGENT RESPONSE", "TOOL CALL", "TOOL RESULT")]
        self.assertEqual(order, sorted(order))

        # Содержимое: system prompt, аргументы, content инструмента — полностью.
        self.assertIn("Ты supervisor.", text)
        self.assertIn("найди сработки", text)
        self.assertIn("выборку", text)
        self.assertIn('"table_name": "hits"', text)
        self.assertIn("select_columns", text)
        self.assertIn("Оплата обучения", text)
        self.assertIn("схема аргументов", text)

        # Метаданные не должны попадать в лог.
        self.assertNotIn("token_usage", text)
        self.assertNotIn("response_metadata", text)
        self.assertNotIn("run_id", text)
        self.assertNotIn("c1", text)  # id вызова инструмента не логируется

    def test_unchanged_system_prompt_and_tools_collapsed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            handler = FileTraceCallbackHandler(Path(temp_dir) / "trace.txt")
            self._run_basic_flow(handler)
            # повторный идентичный вызов модели
            handler.on_chat_model_start(
                {"name": "model"},
                [[SystemMessage(content="Ты supervisor."), HumanMessage(content="найди сработки")]],
                run_id="r2",
                invocation_params={
                    "tools": [
                        {"type": "function", "function": {"name": "read_table", "description": "x", "parameters": {}}}
                    ]
                },
                metadata={"langgraph_node": "agent"},
            )
            text = handler.file_path.read_text(encoding="utf-8")

        self.assertIn("(без изменений, см. выше)", text)
        # system prompt полностью записан только один раз
        self.assertEqual(text.count("Ты supervisor."), 1)


if __name__ == "__main__":
    unittest.main()

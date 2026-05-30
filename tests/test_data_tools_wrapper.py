"""Тесты обёртки data-tools, добавляющей код запроса и счётчики строк."""

from __future__ import annotations

import asyncio
import unittest

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from deep_agent_test.data_tools_wrapper import wrap_data_tools_with_query_code


class _Args(BaseModel):
    table_name: str = Field(default="t")
    select_columns: str = Field(default="")
    filters: list[dict] = Field(default_factory=list)
    group_by: str | None = Field(default=None)
    aggregations: list[dict] = Field(default_factory=list)
    max_rows: int | None = Field(default=None)


def _build_tool(frame: pd.DataFrame) -> StructuredTool:
    def read_table(**kwargs) -> pd.DataFrame:
        return frame

    return StructuredTool.from_function(
        func=read_table,
        name="read_table",
        description="base read_table",
        args_schema=_Args,
    )


class DataToolsWrapperTests(unittest.TestCase):
    def test_wraps_dataframe_with_query_code_and_full_result_note(self) -> None:
        frame = pd.DataFrame([{"event_description": "Оплата обучения"}])
        frame.attrs["spark_total_rows"] = 87
        frame.attrs["spark_matched_rows"] = 17
        tool = wrap_data_tools_with_query_code([_build_tool(frame)])[0]

        self.assertEqual(tool.response_format, "content_and_artifact")
        tool_call = {
            "name": "read_table",
            "args": {
                "table_name": "hits",
                "select_columns": "event_description",
                "filters": [{"column": "event_dt", "operator": "between", "values": ["20260101", "20260131"]}],
                "max_rows": 1,
            },
            "id": "c1",
            "type": "tool_call",
        }
        message = tool.invoke(tool_call)

        self.assertIn("SELECT event_description", message.content)
        self.assertIn("BETWEEN '20260101' AND '20260131'", message.content)
        self.assertIn("ПОЛНЫЙ результат запроса в контексте", message.content)
        # Счётчики исходной таблицы больше не публикуются модели.
        self.assertNotIn("всего в таблице", message.content)
        self.assertNotIn("подошло под фильтры", message.content)
        self.assertNotIn("total_rows", message.artifact)
        self.assertNotIn("matched_rows", message.artifact)
        self.assertEqual(message.artifact["returned_rows"], 1)
        self.assertEqual(len(message.artifact["rows"]), 1)

    def test_aggregation_query_code_uses_group_by(self) -> None:
        frame = pd.DataFrame([{"event_description": "x", "count_event_description": 3}])
        tool = wrap_data_tools_with_query_code([_build_tool(frame)])[0]
        tool_call = {
            "name": "read_table",
            "args": {
                "table_name": "hits",
                "group_by": "event_description",
                "aggregations": [{"function": "count", "column": "event_description"}],
            },
            "id": "c2",
            "type": "tool_call",
        }
        message = tool.invoke(tool_call)
        self.assertIn("GROUP BY event_description", message.content)
        self.assertIn("count(event_description)", message.content)

    def test_aggregation_reports_full_result_in_context(self) -> None:
        # group_by + aggregations: модель должна видеть, что это ПОЛНЫЙ результат в контексте
        # (число групп), без счётчиков исходной таблицы и без ложных предупреждений.
        frame = pd.DataFrame([{"event_description": f"v{i}", "count_event_description": 1} for i in range(6)])
        frame.attrs["spark_total_rows"] = 87
        frame.attrs["spark_matched_rows"] = 17
        tool = wrap_data_tools_with_query_code([_build_tool(frame)])[0]
        tool_call = {
            "name": "read_table",
            "args": {
                "table_name": "hits",
                "group_by": "event_description",
                "aggregations": [{"function": "count", "column": "event_description"}],
            },
            "id": "c3",
            "type": "tool_call",
        }
        message = tool.invoke(tool_call)
        self.assertNotIn("всего в таблице", message.content)
        self.assertNotIn("подошло под фильтры", message.content)
        self.assertIn("ПОЛНЫЙ результат запроса в контексте", message.content)
        self.assertIn("групп", message.content)
        self.assertTrue(message.artifact["is_aggregation"])

    def test_plain_select_reports_full_result_in_context(self) -> None:
        frame = pd.DataFrame([{"event_description": f"v{i}"} for i in range(3)])
        frame.attrs["spark_total_rows"] = 87
        frame.attrs["spark_matched_rows"] = 17
        tool = wrap_data_tools_with_query_code([_build_tool(frame)])[0]
        tool_call = {
            "name": "read_table",
            "args": {"table_name": "hits", "select_columns": "event_description", "max_rows": 3},
            "id": "c4",
            "type": "tool_call",
        }
        message = tool.invoke(tool_call)
        self.assertIn("ПОЛНЫЙ результат запроса в контексте", message.content)
        self.assertNotIn("всего в таблице", message.content)
        self.assertFalse(message.artifact["is_aggregation"])

    def test_error_string_keeps_query_code(self) -> None:
        def read_table(**kwargs) -> str:
            return "Ошибка инструмента read_table: таблица не найдена."

        base = StructuredTool.from_function(
            func=read_table,
            name="read_table",
            description="base",
            args_schema=_Args,
        )
        tool = wrap_data_tools_with_query_code([base])[0]
        message = tool.invoke({"name": "read_table", "args": {"table_name": "bad"}, "id": "c3", "type": "tool_call"})
        self.assertIn("FROM bad", message.content)
        self.assertIn("таблица не найдена", message.content)
        self.assertIsNone(message.artifact)


if __name__ == "__main__":
    unittest.main()

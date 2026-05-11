"""Тесты универсального Spark-like LangChain tool для чтения CSV из examples/data.

Содержит:
- FakeSparkToolsTests: набор проверок spark_query_table.
"""

from __future__ import annotations

import asyncio
import unittest

import pandas as pd

from examples.fake_spark_tools import SparkTableFilter, SparkTableQueryInput, build_fake_spark_tools


class FakeSparkToolsTests(unittest.TestCase):
    """Проверяет универсальный Spark-like инструмент.

    Args:
        Отсутствуют.

    Returns:
        None.
    """

    def test_fake_spark_tools_have_single_query_tool(self) -> None:
        """Проверяет, что фабрика возвращает только один универсальный инструмент.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tools = {tool.name: tool for tool in build_fake_spark_tools(delay_seconds=0.0)}

        self.assertEqual(set(tools), {"spark_query_table"})

    def test_spark_query_table_selects_columns(self) -> None:
        """Проверяет выбор конкретных полей из таблицы.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(
            tool.ainvoke(
                {
                    "table_name": "hits",
                    "select_columns": ["event_id", "epk_id"],
                    "max_rows": 3,
                }
            )
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["event_id", "epk_id"])
        self.assertLessEqual(len(result), 3)
        self.assertGreater(result.attrs["spark_matched_rows"], 0)

    def test_spark_query_table_applies_filters(self) -> None:
        """Проверяет фильтрацию строк по переданным ограничениям.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(
            tool.ainvoke(
                {
                    "table_name": "hits_extra_info",
                    "select_columns": ["event_id", "epk_id"],
                    "filters": [
                        {
                            "column": "epk_id",
                            "operator": "eq",
                            "value": "2099007770421986000001",
                        }
                    ],
                    "max_rows": 10,
                }
            )
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        for value in result["epk_id"]:
            self.assertEqual(str(value), "2099007770421986000001")

    def test_spark_query_table_returns_schema_for_missing_column(self) -> None:
        """Проверяет возврат схемы таблицы при несуществующем поле.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(
            tool.ainvoke(
                {
                    "table_name": "hits",
                    "select_columns": ["missing_field"],
                    "max_rows": 1,
                }
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "unknown_columns")
        self.assertEqual(result["error"]["missing_columns"], ["missing_field"])
        self.assertIn("schema", result)
        self.assertGreater(result["schema"]["columns_count"], 0)

    def test_spark_query_table_returns_available_tables_for_unknown_table(self) -> None:
        """Проверяет ошибку и список таблиц при неизвестной таблице.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(tool.ainvoke({"table_name": "unknown_table"}))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "unknown_table")
        self.assertIn("hits", result["error"]["available_tables"])

    def test_spark_query_table_requires_explicit_columns(self) -> None:
        """Проверяет запрет выгрузки всех колонок без явного списка.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(tool.ainvoke({"table_name": "hits", "select_columns": []}))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "select_columns_required")
        self.assertIn("schema", result)

    def test_spark_query_table_rejects_select_all_marker(self) -> None:
        """Проверяет запрет маркеров выбора всех колонок.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(tool.ainvoke({"table_name": "hits", "select_columns": ["*"]}))

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "select_all_forbidden")
        self.assertIn("schema", result)

    def test_spark_query_table_schema_models(self) -> None:
        """Проверяет Pydantic-схемы универсального Spark-like инструмента.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        filter_item = SparkTableFilter(column="epk_id", operator="eq", value="2099007770421986000001")
        schema = SparkTableQueryInput(
            table_name="hits",
            select_columns=["event_id"],
            filters=[filter_item],
            max_rows=5,
        )

        self.assertEqual(schema.table_name, "hits")
        self.assertEqual(schema.filters[0].column, "epk_id")
        with self.assertRaises(Exception):
            SparkTableFilter(column="epk_id", operator="between", values=["20250101"])


if __name__ == "__main__":
    unittest.main()

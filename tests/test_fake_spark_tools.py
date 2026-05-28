"""Тесты универсального Spark-like LangChain tool для чтения CSV из examples/data.

Содержит:
- FakeSparkToolsTests: набор проверок read_table.
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path

import pandas as pd

from examples.fake_spark_tools import SparkTableFilter, SparkTableQueryInput, build_fake_spark_tools
from planner_agent.runtime.tool_text import is_tool_error_result


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

        self.assertEqual(set(tools), {"read_table"})

    def test_read_table_selects_columns(self) -> None:
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
                    "select_columns": "event_id, epk_id",
                    "max_rows": 3,
                }
            )
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), ["event_id", "epk_id"])
        self.assertLessEqual(len(result), 3)
        self.assertGreater(result.attrs["spark_matched_rows"], 0)

    def test_read_table_supports_raw_table_aliases(self) -> None:
        """Проверяет алиасы raw-таблиц cards_event и uko_event.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]

        cards_results = [
            asyncio.run(
                tool.ainvoke(
                    {
                        "table_name": table_name,
                        "select_columns": "event_id, user_id",
                        "max_rows": 1,
                    }
                )
            )
            for table_name in ("cards_event", "csp_afpc_sss_inc.cards_event", "cspfs_repo_features3.cards_event")
        ]
        uko_results = [
            asyncio.run(
                tool.ainvoke(
                    {
                        "table_name": table_name,
                        "select_columns": "event_id, user_id",
                        "max_rows": 1,
                    }
                )
            )
            for table_name in ("uko_event", "csp_afpc_sss_inc.uko_event", "cspfs_repo_features3.uko_event")
        ]

        for result in cards_results + uko_results:
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 1)

        self.assertEqual({result.attrs["spark_source_file"] for result in cards_results}, {"csp_afpc_sss_inc.cards_event.csv"})
        self.assertEqual({result.attrs["spark_source_file"] for result in uko_results}, {"csp_afpc_sss_inc.uko_event.csv"})
        self.assertEqual({result.iloc[0]["event_id"] for result in cards_results}, {cards_results[0].iloc[0]["event_id"]})
        self.assertEqual({result.iloc[0]["event_id"] for result in uko_results}, {uko_results[0].iloc[0]["event_id"]})

    def test_read_table_cards_event_client_fields(self) -> None:
        """Проверяет клиентские поля и фильтрацию в cards_event.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(
            tool.ainvoke(
                {
                    "table_name": "cspfs_repo_features3.cards_event",
                    "select_columns": "event_id, epk_id, event_dt, transaction_amount_in_rub, type_operation",
                    "filters": [
                        {
                            "column": "epk_id",
                            "operator": "eq",
                            "value": "2099007770421986000001",
                        },
                        {
                            "column": "event_dt",
                            "operator": "gte",
                            "value": 20250728,
                        },
                    ],
                    "max_rows": 1000,
                }
            )
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)
        self.assertEqual(
            list(result.columns),
            ["event_id", "epk_id", "event_dt", "transaction_amount_in_rub", "type_operation"],
        )
        self.assertTrue((result["epk_id"].astype(str) == "2099007770421986000001").all())
        self.assertTrue((pd.to_numeric(result["event_dt"]) >= 20250728).all())

    def test_read_table_applies_filters(self) -> None:
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
                    "select_columns": "event_id, epk_id",
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

    def test_read_table_filters_large_identifier_without_float_precision_loss(self) -> None:
        """Проверяет фильтрацию больших идентификаторов без float-приведения.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0, data_dir=Path("deep_agent_test/data"))[0]
        result = asyncio.run(
            tool.ainvoke(
                {
                    "table_name": "csp_afpc_sss_inc.uko_event",
                    "select_columns": "event_id, epk_id, event_dt, user_ip_location_city, ip_device",
                    "filters": [
                        {
                            "column": "epk_id",
                            "operator": "eq",
                            "value": "2099007770421989000001",
                        },
                        {
                            "column": "event_dt",
                            "operator": "eq",
                            "value": "20260124",
                        },
                    ],
                    "max_rows": 10,
                }
            )
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["event_id"], "3486d84b-4eba-4ba4-b044-94764fc9e7a4")
        self.assertEqual(result.iloc[0]["user_ip_location_city"], "Moscow")
        self.assertEqual(result.iloc[0]["ip_device"], "95.31.146.230")

    def test_read_table_returns_schema_for_missing_column(self) -> None:
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
                    "select_columns": "missing_field",
                    "max_rows": 1,
                }
            )
        )

        self.assertTrue(is_tool_error_result(result))
        self.assertIn("missing_field", result)
        self.assertIn("Доступные поля", result)

    def test_read_table_returns_available_tables_for_unknown_table(self) -> None:
        """Проверяет ошибку и список таблиц при неизвестной таблице.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(tool.ainvoke({"table_name": "unknown_table"}))

        self.assertTrue(is_tool_error_result(result))
        self.assertIn("hits", result)

    def test_read_table_requires_explicit_columns(self) -> None:
        """Проверяет запрет выгрузки всех колонок без явного списка.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(tool.ainvoke({"table_name": "hits", "select_columns": ""}))

        self.assertTrue(is_tool_error_result(result))
        self.assertIn("Доступные поля", result)

    def test_read_table_rejects_select_all_marker(self) -> None:
        """Проверяет запрет маркеров выбора всех колонок.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        tool = build_fake_spark_tools(delay_seconds=0.0)[0]
        result = asyncio.run(tool.ainvoke({"table_name": "hits", "select_columns": "*"}))

        self.assertTrue(is_tool_error_result(result))
        self.assertIn("Доступные поля", result)

    def test_read_table_schema_models(self) -> None:
        """Проверяет Pydantic-схемы универсального Spark-like инструмента.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        filter_item = SparkTableFilter(column="epk_id", operator="eq", value="2099007770421986000001")
        schema = SparkTableQueryInput(
            table_name="hits",
            select_columns="event_id",
            filters=[filter_item],
            max_rows=5,
        )

        self.assertEqual(schema.table_name, "hits")
        self.assertEqual(schema.filters[0].column, "epk_id")
        with self.assertRaises(Exception):
            SparkTableFilter(column="epk_id", operator="between", values=["20250101"])


if __name__ == "__main__":
    unittest.main()

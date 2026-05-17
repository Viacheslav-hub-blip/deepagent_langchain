from __future__ import annotations

import tempfile
import unittest
from asyncio import run
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from examples.fake_spark_tools import build_fake_spark_tools
from planner_agent.models import Task
from planner_agent.services.artifact_service import ArtifactService
from planner_agent.tools.artifact_wrappers import wrap_tools_for_artifacts
from planner_agent.runtime.tool_text import is_tool_error_result


class FakeSandbox:
    """Минимальная песочница для проверки добавления DataFrame-переменных.

    Args:
        Отсутствуют.

    Returns:
        Объект с API, нужным ArtifactToolWrapper.
    """

    def __init__(self) -> None:
        """Создает пустое хранилище переменных.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        self.globals: dict = {}
        self.last_dataframe_variable: str | None = None

    async def add_variable(self, name: str, value: object) -> None:
        """Добавляет переменную в тестовую песочницу.

        Args:
            name: Имя переменной.
            value: Значение переменной.

        Returns:
            None.
        """

        self.globals[name] = value


class ToolArtifactTests(unittest.TestCase):
    def test_dataframe_result_is_captured_with_metadata_only_for_model(self) -> None:
        """Проверяет, что DataFrame сохраняется в artifact, а модели возвращается только структура."""

        @tool("load_client_events")
        async def load_client_events(client_id: str) -> pd.DataFrame:
            """Load client events as a DataFrame."""
            return pd.DataFrame(
                [
                    {"client_id": client_id, "event_id": "evt-secret-1", "amount": 100.0},
                    {"client_id": client_id, "event_id": None, "amount": 250.0},
                ]
            )

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="df-load", description="Load dataframe")
            artifact_index: dict = {}
            tool_traces: list[dict] = []
            sandbox = FakeSandbox()

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[load_client_events],
                artifact_service=artifacts,
                run_id="run-df",
                node_id="node-df",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
                sandbox=sandbox,
            )[0]

            result = run(wrapped_tool.ainvoke({"client_id": "client-1"}))

            self.assertIsInstance(result, str)
            self.assertIn("<tool_result>", result)
            self.assertIn("<status>success</status>", result)
            self.assertIn("<dataframe_preview>", result)
            self.assertIn("<data_description>", result)
            self.assertIn("<row_count>2</row_count>", result)
            self.assertIn("<columns>", result)
            self.assertIn("<column_types>", result)
            self.assertIn("<has_empty_values>true</has_empty_values>", result)
            self.assertIn("<preview_row>", result)
            self.assertIn("evt-secret-1", result)
            self.assertNotIn("<artifact>", result)
            self.assertNotIn("<sandbox>", result)
            self.assertNotIn("sandbox_variable_name", result)
            self.assertIn("tdf_load_load_client_events_1", sandbox.globals)
            self.assertIs(sandbox.globals["tdf_load_load_client_events_1"], sandbox.globals[sandbox.last_dataframe_variable])

            stored = artifacts.list_artifacts("run-df")
            captured_artifact = next(
                artifact
                for artifact in stored
                if artifact.metadata.get("artifact_role") == "captured_tool_result"
            )
            self.assertEqual(captured_artifact.kind, "dataset")
            self.assertEqual(captured_artifact.mime_type, "text/csv")
            self.assertTrue(captured_artifact.uri.endswith(f"{captured_artifact.artifact_id}.csv"))
            self.assertEqual(captured_artifact.metadata["row_count"], 2)
            self.assertIn("evt-secret-1", captured_artifact.metadata["preview_row"])
            self.assertEqual(
                captured_artifact.metadata["variable_name"],
                "tdf_load_load_client_events_1",
            )
            self.assertEqual(
                captured_artifact.metadata["columns"],
                ["client_id", "event_id", "amount"],
            )
            self.assertTrue(captured_artifact.metadata["has_empty_values"])
            self.assertTrue(
                captured_artifact.metadata["has_empty_values_by_column"]["event_id"]
            )
            self.assertEqual(task.artifact_refs, [captured_artifact.artifact_id])

            trace_artifact = next(
                artifact
                for artifact in stored
                if artifact.metadata.get("artifact_role") == "tool_call_trace"
            )
            trace_content = Path(trace_artifact.uri).read_text(encoding="utf-8")
            self.assertIn('"row_count": 2', trace_content)
            self.assertIn("evt-secret-1", trace_content)

            captured_content = Path(captured_artifact.uri).read_text(encoding="utf-8")
            self.assertIn("evt-secret-1", captured_content)

    def test_spark_dataframe_result_uses_minimal_xml_contract(self) -> None:
        """Проверяет, что Spark DataFrame возвращает модели только статус, preview и описание."""

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="spark-load", description="Load Spark data")
            artifact_index: dict = {}
            tool_traces: list[dict] = []
            sandbox = FakeSandbox()
            spark_tool = build_fake_spark_tools(delay_seconds=0.0)[0]

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[spark_tool],
                artifact_service=artifacts,
                run_id="run-spark",
                node_id="node-spark",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
                sandbox=sandbox,
            )[0]

            result = run(
                wrapped_tool.ainvoke(
                    {
                        "table_name": "hits",
                        "select_columns": "event_id, epk_id",
                        "max_rows": 1,
                    }
                )
            )

            self.assertIn("<status>success</status>", result)
            self.assertIn("<dataframe_preview>", result)
            self.assertIn("<data_description>", result)
            self.assertIn("<row_count>1</row_count>", result)
            self.assertNotIn("<source>", result)
            self.assertNotIn("<query>", result)
            self.assertNotIn("<artifact>", result)
            self.assertNotIn("<sandbox>", result)

    def test_spark_error_is_returned_as_plain_text(self) -> None:
        """Проверяет, что ошибка Spark-like инструмента возвращается текстом без JSON-обертки."""

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="spark-error", description="Load Spark schema")
            artifact_index: dict = {}
            tool_traces: list[dict] = []
            spark_tool = build_fake_spark_tools(delay_seconds=0.0)[0]

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[spark_tool],
                artifact_service=artifacts,
                run_id="run-spark-error",
                node_id="node-spark-error",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            result = run(
                wrapped_tool.ainvoke(
                    {
                        "table_name": "hits",
                        "select_columns": "",
                    }
                )
            )

            self.assertTrue(is_tool_error_result(result))
            self.assertIn("Доступные поля", result)
            self.assertNotIn('"ok"', result)

    def test_tool_result_is_saved_as_reusable_artifact(self) -> None:
        @tool("download_transactions")
        async def download_transactions(depth_days: int, amount: float) -> str:
            """Download transactions from an approved internal source."""
            return f"transactions depth={depth_days} amount={amount}"

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="case/1", description="Load source data")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[download_transactions],
                artifact_service=artifacts,
                run_id="run-1",
                node_id="node-1",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            result = run(
                wrapped_tool.ainvoke(
                    {"depth_days": 30, "amount": 1000.0}
                )
            )

            self.assertIn("Инструмент download_transactions выполнил запрос.", result)
            self.assertIn("transactions depth=30 amount=1000.0", result)
            self.assertNotIn('"ok"', result)
            # Маленький скалярный результат остается inline и не попадает в
            # task.artifact_refs (только большие/файловые артефакты значимы).
            self.assertEqual(task.artifact_refs, [])
            self.assertEqual(len(tool_traces), 1)
            self.assertEqual(len(artifact_index), 1)

            stored = artifacts.list_artifacts("run-1")
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].kind, "tool_trace")
            self.assertEqual(stored[0].node_id, "node-1")
            self.assertEqual(stored[0].metadata["tool_name"], "download_transactions")
            self.assertTrue(stored[0].metadata["reusable"])
            # Tool-trace artifact получает человеко-читаемый id вида t{task}_{tool}_{seq}_trace.
            self.assertEqual(stored[0].artifact_id, "tcase_1_download_transactions_1_trace")

            content = Path(stored[0].uri).read_text(encoding="utf-8")
            self.assertIn("Tool: download_transactions", content)
            self.assertIn('"depth_days": 30', content)
            self.assertIn("transactions depth=30 amount=1000.0", content)

    def test_large_text_result_is_captured_and_replaced_with_reference(self) -> None:
        @tool("export_long_notes")
        async def export_long_notes(client_id: str) -> str:
            """Return a large text export."""
            return f"client={client_id}\n" + ("event\n" * 12_000)

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="large-text", description="Load long notes")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[export_long_notes],
                artifact_service=artifacts,
                run_id="run-large-text",
                node_id="node-large-text",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            result = run(wrapped_tool.ainvoke({"client_id": "client-1"}))

            self.assertIsInstance(result, str)
            self.assertIn("Tool result was saved as an artifact", result)
            self.assertIn("artifact_id:", result)
            self.assertIn("uri:", result)
            self.assertIn("data_scope: partial_preview", result)
            self.assertIn("full_result_available_in_artifact: true", result)
            self.assertIn("preview_is_truncated: true", result)
            self.assertIn("worker_disclosure_required: true", result)
            self.assertIn("The preview below is not the full tool result", result)

            stored = artifacts.list_artifacts("run-large-text")
            self.assertEqual(len(stored), 2)
            captured_artifact = next(
                artifact
                for artifact in stored
                if artifact.metadata.get("artifact_role") == "captured_tool_result"
            )
            trace_artifact = next(
                artifact
                for artifact in stored
                if artifact.metadata.get("artifact_role") == "tool_call_trace"
            )
            self.assertEqual(captured_artifact.kind, "source_excerpt")
            self.assertTrue(captured_artifact.metadata["reusable"])
            self.assertTrue(captured_artifact.metadata["editable"])
            self.assertTrue(trace_artifact.metadata["captured"])
            self.assertEqual(
                trace_artifact.metadata["captured_artifact_refs"],
                [captured_artifact.artifact_id],
            )
            trace_content = Path(trace_artifact.uri).read_text(encoding="utf-8")
            self.assertIn("Captured: True", trace_content)
            self.assertLess(len(trace_content), 10_000)
            # В task.artifact_refs попадает только большой захваченный результат,
            # tool_trace остается доступен через state.artifact_index.
            self.assertEqual(task.artifact_refs, [captured_artifact.artifact_id])
            self.assertEqual(set(artifact_index), {captured_artifact.artifact_id, trace_artifact.artifact_id})
            self.assertEqual(captured_artifact.artifact_id, "tlarge-text_export_long_notes_1")
            self.assertEqual(trace_artifact.artifact_id, "tlarge-text_export_long_notes_1_trace")

    def test_large_list_result_is_captured_as_dataset_without_tool_contract(self) -> None:
        @tool("load_events")
        async def load_events(client_id: str) -> list[dict]:
            """Return many events as a normal Python list."""
            return [
                {
                    "client_id": client_id,
                    "event_id": index,
                    "amount": index * 10,
                    "description": "regular payment event",
                }
                for index in range(2_000)
            ]

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="large-list", description="Load events")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[load_events],
                artifact_service=artifacts,
                run_id="run-large-list",
                node_id="node-large-list",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            result = run(wrapped_tool.ainvoke({"client_id": "client-2"}))

            self.assertIsInstance(result, str)
            self.assertIn("artifact_id:", result)
            self.assertIn("original_size_estimate_chars:", result)
            self.assertIn("data_scope: partial_preview", result)
            self.assertIn("worker_disclosure_required: true", result)

            stored = artifacts.list_artifacts("run-large-list")
            captured_artifact = next(
                artifact
                for artifact in stored
                if artifact.metadata.get("artifact_role") == "captured_tool_result"
            )
            self.assertEqual(captured_artifact.kind, "dataset")
            self.assertEqual(captured_artifact.mime_type, "application/json")
            captured_content = Path(captured_artifact.uri).read_text(encoding="utf-8")
            self.assertIn('"event_id": 1999', captured_content)

    def test_small_structured_result_is_returned_inline_and_saved_as_artifact(self) -> None:
        """Проверяет, что маленький list/dict остается inline, но сохраняется для responder."""

        @tool("load_day_events")
        async def load_day_events(client_id: str) -> list[dict]:
            """Return a small structured events export."""
            return [
                {"client_id": client_id, "event_id": "evt-1", "amount": 100},
                {"client_id": client_id, "event_id": "evt-2", "amount": 200},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="small-list", description="Load day events")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[load_day_events],
                artifact_service=artifacts,
                run_id="run-small-list",
                node_id="node-small-list",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            result = run(wrapped_tool.ainvoke({"client_id": "client-2"}))

            self.assertIn("Инструмент load_day_events выполнил запрос.", result)
            self.assertIn("evt-1", result)
            self.assertIn("Artifact:", result)
            self.assertNotIn('"ok"', result)

            stored = artifacts.list_artifacts("run-small-list")
            dataset_artifact = next(
                artifact
                for artifact in stored
                if artifact.metadata.get("artifact_role") == "captured_tool_result"
            )
            self.assertEqual(dataset_artifact.kind, "dataset")
            self.assertEqual(
                dataset_artifact.metadata["capture_reason"],
                "inline_structured_result",
            )
            # Маленький inline-структурированный результат сохраняется как artifact
            # для responder/UI, но в task.artifact_refs не дублируется (данные уже
            # в контексте worker-а).
            self.assertNotIn(dataset_artifact.artifact_id, task.artifact_refs)
            self.assertIn(dataset_artifact.artifact_id, artifact_index)
            self.assertEqual(dataset_artifact.artifact_id, "tsmall-list_load_day_events_1")

    def test_records_envelope_result_is_saved_as_csv_artifact(self) -> None:
        """Проверяет, что dict с records сохраняется как CSV dataset artifact."""

        @tool("spark_query")
        async def spark_query(client_id: str) -> dict:
            """Возвращает тестовую Spark-like выборку в envelope-формате."""
            return {
                "ok": True,
                "table_name": "events",
                "records": [
                    {"client_id": client_id, "event_id": "evt-1", "amount": 100},
                    {"client_id": client_id, "event_id": "evt-2", "amount": 200},
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="records-envelope", description="Load records")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[spark_query],
                artifact_service=artifacts,
                run_id="run-records-envelope",
                node_id="node-records-envelope",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            result = run(wrapped_tool.ainvoke({"client_id": "client-2"}))

            self.assertIn("Инструмент spark_query выполнил запрос.", result)
            self.assertIn("Artifact:", result)
            self.assertIn("evt-1", result)
            self.assertNotIn('"ok"', result)

            stored = artifacts.list_artifacts("run-records-envelope")
            dataset_artifact = next(
                artifact
                for artifact in stored
                if artifact.metadata.get("artifact_role") == "captured_tool_result"
            )
            self.assertEqual(dataset_artifact.kind, "dataset")
            self.assertEqual(dataset_artifact.mime_type, "text/csv")
            self.assertEqual(dataset_artifact.metadata["records_payload_key"], "records")
            self.assertEqual(
                dataset_artifact.metadata["records_envelope_metadata"]["table_name"],
                "events",
            )
            csv_content = Path(dataset_artifact.uri).read_text(encoding="utf-8")
            self.assertIn("amount,client_id,event_id", csv_content.splitlines()[0])
            self.assertIn("100,client-2,evt-1", csv_content)

    def test_tool_exception_returns_actionable_error_for_model(self) -> None:
        """Проверяет, что исключение инструмента возвращается как понятный текст."""

        @tool("load_missing_source")
        async def load_missing_source(source_file: str) -> str:
            """Raise a source loading error."""
            raise FileNotFoundError(f"Source not found: {source_file}")

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="err", description="Load missing source")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[load_missing_source],
                artifact_service=artifacts,
                run_id="run-error",
                node_id="node-error",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            raw = run(wrapped_tool.ainvoke({"source_file": "missing.csv"}))

            self.assertTrue(is_tool_error_result(raw))
            self.assertIn("Source not found", raw)
            self.assertIn("Как исправить:", raw)
            self.assertIn("Повтор:", raw)
            self.assertIn("Trace artifact:", raw)
            self.assertNotIn('"ok"', raw)
            self.assertEqual(len(tool_traces), 1)
            self.assertTrue(tool_traces[0]["tool_error"])
            self.assertIn(tool_traces[0]["artifact_id"], artifact_index)

    def test_load_skill_capture_is_not_added_to_task_refs(self) -> None:
        """load_skill мета-инструмент не должен загромождать task.artifact_refs."""

        @tool("load_skill")
        async def load_skill(name: str) -> dict:
            """Read existing skill content."""
            return {
                "name": name,
                "content": "X" * 12_000,
            }

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="3", description="Inspect previous artifact")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped = wrap_tools_for_artifacts(
                tools=[load_skill],
                artifact_service=artifacts,
                run_id="run-meta",
                node_id="node-meta",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            run(wrapped.ainvoke({"name": "foo"}))

            # Мета-инструменты чтения файлов читают существующие данные и не должны
            # порождать новые ссылки в task.artifact_refs.
            self.assertEqual(task.artifact_refs, [])
            # Однако сами artifact-записи (capture + trace) сохраняются для аудита.
            self.assertEqual(len(artifact_index), 2)
            self.assertIn("t3_load_skill_1", artifact_index)
            self.assertIn("t3_load_skill_1_trace", artifact_index)

    def test_artifact_labels_use_retry_count_suffix(self) -> None:
        """При повторе задачи labels включают суффикс ``_r{n}`` для уникальности."""

        @tool("spark_lookup")
        async def spark_lookup(event_id: str) -> list[dict]:
            """Lookup event."""
            return [{"event_id": event_id, "amount": 100}]

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            task = Task(task_id="2", description="Retry test", retry_count=3)
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped = wrap_tools_for_artifacts(
                tools=[spark_lookup],
                artifact_service=artifacts,
                run_id="run-retry",
                node_id="node-retry",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            run(wrapped.ainvoke({"event_id": "evt-1"}))
            run(wrapped.ainvoke({"event_id": "evt-2"}))

            self.assertIn("t2_r3_spark_lookup_1", artifact_index)
            self.assertIn("t2_r3_spark_lookup_1_trace", artifact_index)
            self.assertIn("t2_r3_spark_lookup_2", artifact_index)
            self.assertIn("t2_r3_spark_lookup_2_trace", artifact_index)

    def test_label_collision_gets_unique_suffix(self) -> None:
        """Если запрошенный artifact_id уже занят, добавляется числовой суффикс."""

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            first = artifacts.write_artifact(
                run_id="run-collide",
                node_id="node-1",
                kind="model_output",
                filename="tasks/1/result_v1.md",
                content="first",
                mime_type="text/markdown",
                summary="first",
                metadata={"task_id": "1", "artifact_role": "worker_result"},
                artifact_id="t1_result",
            )
            second = artifacts.write_artifact(
                run_id="run-collide",
                node_id="node-1",
                kind="model_output",
                filename="tasks/1/result_v2.md",
                content="second",
                mime_type="text/markdown",
                summary="second",
                metadata={"task_id": "1", "artifact_role": "worker_result"},
                artifact_id="t1_result",
            )

            self.assertEqual(first.artifact_id, "t1_result")
            self.assertEqual(second.artifact_id, "t1_result_2")

    def test_direct_file_path_result_is_registered_as_editable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_path = Path(tmp) / "transactions.csv"
            export_path.write_text("event_id,amount\n1,100\n", encoding="utf-8")

            @tool("export_transactions")
            async def export_transactions(client_id: str) -> str:
                """Export transactions to a reusable file and return its path."""
                return str(export_path)

            artifacts = ArtifactService(tmp)
            task = Task(task_id="2", description="Export transactions")
            artifact_index: dict = {}
            tool_traces: list[dict] = []

            wrapped_tool = wrap_tools_for_artifacts(
                tools=[export_transactions],
                artifact_service=artifacts,
                run_id="run-2",
                node_id="node-2",
                task=task,
                artifact_index=artifact_index,
                tool_traces=tool_traces,
            )[0]

            result = run(wrapped_tool.ainvoke({"client_id": "client-1"}))

            stored = artifacts.list_artifacts("run-2")
            self.assertIn("artifact_id:", result)
            self.assertEqual([artifact.kind for artifact in stored], ["dataset", "tool_trace"])
            self.assertEqual(stored[0].uri, str(export_path.resolve()))
            self.assertTrue(stored[0].metadata["reusable"])
            self.assertTrue(stored[0].metadata["editable"])
            # Существующий файл (existing_file_reference) считается значимым для task,
            # tool_trace в task.artifact_refs не попадает.
            self.assertEqual(task.artifact_refs, [stored[0].artifact_id])
            self.assertEqual(set(artifact_index), {stored[0].artifact_id, stored[1].artifact_id})


if __name__ == "__main__":
    unittest.main()

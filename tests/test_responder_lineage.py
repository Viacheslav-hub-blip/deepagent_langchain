from __future__ import annotations

import tempfile
import unittest
from asyncio import run
from pathlib import Path

from typing import Any, cast

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from planner_agent.agent_nodes.responder_node import (
    _build_responder_artifact_names_context,
    _collect_results,
    _fit_responder_context_budget,
    _format_human_message,
    _format_task_for_responder,
    responder_node,
)
from planner_agent.models import AgentState, Task, TaskStatus
from planner_agent.services.artifact_service import ArtifactService
from planner_agent.services.lineage_service import LineageService


class BindableFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Поддержка create_react_agent (bind_tools) в тестах без внешнего API."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "BindableFakeMessagesListChatModel":
        return cast(BindableFakeMessagesListChatModel, self)


class ResponderLineageTests(unittest.TestCase):
    """Проверяет lineage и artifact-aware контекст финального responder."""

    def test_responder_creates_final_report_node_and_state_value(self) -> None:
        """Проверяет создание final_report node, state.final_report и report artifact."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            run_record = lineage.create_run(initial_user_query="Analyze case")
            validation_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="validation_completed",
                title="Validation completed",
            )
            state = AgentState(
                run_id=run_record.run_id,
                current_node_id=validation_node.node_id,
                parent_node_ids=[validation_node.node_id],
                initial_user_query="Analyze case",
                artifact_index={"existing-artifact": {"kind": "dataset"}},
                plan={
                    "1": Task(
                        task_id="1",
                        description="Inspect data",
                        full_result="The data looks consistent.",
                        status=TaskStatus.COMPLETED,
                        run_id="1",
                    )
                },
            )
            llm = BindableFakeMessagesListChatModel(
                responses=[
                    AIMessage(content="# Analysis Report\n\nThe data looks consistent."),
                ]
            )

            command = run(
                responder_node(
                    state=state,
                    llm=llm,
                    prompt="Return final report JSON.",
                    lineage_service=lineage,
                    artifact_service=artifacts,
                )
            )

            update = command.update
            self.assertIn("# Analysis Report", update["final_report"])
            self.assertEqual(update["messages"][-1].content, update["final_report"])
            self.assertEqual(len(update["lineage_events"]), 1)
            self.assertEqual(update["parent_node_ids"], [update["current_node_id"]])

            nodes = lineage.get_nodes(run_record.run_id)
            self.assertEqual(nodes[-1].node_type, "final_report")
            self.assertEqual(nodes[-1].parent_ids, [validation_node.node_id])
            self.assertEqual(nodes[-1].metadata["completed_tasks"], ["1"])

            final_report_node = lineage.get_node(run_record.run_id, update["current_node_id"])
            self.assertIsNotNone(final_report_node)
            self.assertEqual(len(final_report_node.artifact_refs), 2)

            stored_artifacts = artifacts.list_artifacts(run_record.run_id)
            report_artifact = next(
                artifact for artifact in stored_artifacts if artifact.kind == "report"
            )
            context_artifact = next(
                artifact
                for artifact in stored_artifacts
                if artifact.metadata.get("artifact_role") == "responder_context"
            )
            self.assertTrue(report_artifact.uri.endswith("final_report.md"))
            self.assertTrue(context_artifact.uri.endswith("final_report_context.md"))
            context_text = Path(context_artifact.uri).read_text(encoding="utf-8")
            self.assertIn("The data looks consistent.", context_text)

            snapshot = lineage.load_snapshot(run_record.run_id, final_report_node.node_id)
            self.assertIn("# Analysis Report", snapshot["final_report"])
            self.assertIn("existing-artifact", snapshot["artifact_index"])
            self.assertIn(context_artifact.artifact_id, snapshot["artifact_index"])

    def test_responder_lists_artifact_names_from_completed_tasks(self) -> None:
        """Проверяет, что в контекст responder попадают имена artifacts без содержимого файлов."""

        with tempfile.TemporaryDirectory() as tmp:
            artifacts = ArtifactService(tmp)
            run_id = "run-artifacts"
            dataset = artifacts.write_artifact(
                run_id=run_id,
                node_id="node-1",
                kind="dataset",
                filename="tasks/1/tool_results/events.json",
                content='[{"event_id":"evt-1","amount":100},{"event_id":"evt-2","amount":200}]',
                mime_type="application/json",
                summary="Two test events",
                metadata={
                    "task_id": "1",
                    "tool_name": "load_events",
                    "artifact_role": "captured_tool_result",
                },
            )
            trace = artifacts.write_artifact(
                run_id=run_id,
                node_id="node-1",
                kind="tool_trace",
                filename="tasks/1/tool_calls/load_events.txt",
                content="technical trace should not be included",
                mime_type="text/plain",
                summary="Trace",
                metadata={"artifact_role": "tool_call_trace"},
            )
            state = AgentState(
                run_id=run_id,
                plan={
                    "1": Task(
                        task_id="1",
                        description="Load events",
                        status=TaskStatus.COMPLETED,
                        artifact_refs=[dataset.artifact_id, trace.artifact_id],
                    )
                },
                artifact_index={
                    dataset.artifact_id: dataset.model_dump(mode="json"),
                    trace.artifact_id: trace.model_dump(mode="json"),
                },
            )

            context = _build_responder_artifact_names_context(
                state=state,
                artifact_service=artifacts,
            )

            self.assertIn(dataset.artifact_id, context)
            self.assertIn("Two test events", context)
            self.assertNotIn('"event_id":"evt-1"', context)
            self.assertNotIn("technical trace should not be included", context)

    def test_responder_collects_completed_tasks_from_all_plan_runs(self) -> None:
        """Проверяет, что responder видит результаты всех шагов после replanning."""

        plan = {
            "1": Task(
                task_id="1",
                description="Load trigger",
                full_result="Trigger details from first plan.",
                status=TaskStatus.COMPLETED,
                run_id="1",
            ),
            "2": Task(
                task_id="2",
                description="Analyze final evidence",
                full_result="Final evidence from second plan.",
                status=TaskStatus.COMPLETED,
                run_id="2",
            ),
        }

        completed, failed = _collect_results(plan, current_run_id="2")
        completed_text = "\n".join(completed)

        self.assertEqual(failed, [])
        self.assertIn("Trigger details from first plan.", completed_text)
        self.assertIn("Final evidence from second plan.", completed_text)

    def test_responder_task_context_keeps_preview_and_full_result(self) -> None:
        """Проверяет, что responder видит preview и полный worker-ответ."""

        task = Task(
            task_id="1",
            description="Analyze events",
            result_preview="Preview: checked +/-3 days.",
            full_result="Full analysis: checked wider period and found deposits.",
            status=TaskStatus.COMPLETED,
            validation_reason="Valid evidence.",
        )

        text = _format_task_for_responder("1", task)

        self.assertIn("Result preview", text)
        self.assertIn("Preview: checked +/-3 days.", text)
        self.assertIn("Full worker result", text)
        self.assertIn("Full analysis: checked wider period and found deposits.", text)

    def test_responder_context_budget_keeps_prompt_bounded(self) -> None:
        """Проверяет общий бюджет prompt responder."""

        completed, artifact_names = _fit_responder_context_budget(
            user_query="Analyze case",
            completed_text="C" * 10_000,
            artifact_names_text="A" * 10_000,
            planning_error="",
            max_prompt_chars=12_000,
        )
        human = _format_human_message(
            user_query="Analyze case",
            completed_text=completed,
            artifact_names_text=artifact_names,
            planning_error="",
        )

        self.assertLess(len(human), 12_000)
        self.assertIn("completed tasks truncated", completed)


if __name__ == "__main__":
    unittest.main()

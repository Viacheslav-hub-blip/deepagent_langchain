"""Тесты сервиса инспекции сохраненных запусков.

Содержит:
- RunInspectionServiceTests: проверки чтения runs, nodes, snapshots и artifacts.
"""

from __future__ import annotations

import tempfile
import unittest

from planner_agent.schemas.lineage import StateNode
from planner_agent.services.artifact_service import ArtifactService
from planner_agent.services.lineage_service import LineageService
from planner_agent.services.run_inspection_service import RunInspectionService


class RunInspectionServiceTests(unittest.TestCase):
    """Проверяет программный доступ к сохраненным результатам ResearchRun."""

    def test_reads_run_graph_snapshot_artifacts_and_final_report(self) -> None:
        """Проверяет чтение финального отчета из snapshot и artifact index."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            inspector = RunInspectionService(lineage, artifacts)

            run_record = lineage.create_run(initial_user_query="Analyze data")
            context_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="context_snapshot",
                title="Context snapshot",
                state={"run_id": run_record.run_id, "memory_snapshot": "Frozen"},
            )
            final_node = StateNode(
                run_id=run_record.run_id,
                node_type="final_report",
                title="Final report",
                parent_ids=[context_node.node_id],
                status="succeeded",
            )
            report_artifact = artifacts.write_artifact(
                run_id=run_record.run_id,
                node_id=final_node.node_id,
                kind="report",
                filename="final_report.md",
                content="# Report\n\nDone",
                mime_type="text/markdown",
                summary="Done",
                metadata={"domain_label": "manual-test"},
            )
            final_node.artifact_refs = [report_artifact.artifact_id]
            lineage.append_node(
                final_node,
                state={
                    "run_id": run_record.run_id,
                    "final_report": "# Report\n\nDone",
                    "messages": [
                        {"type": "human", "content": "Analyze data"},
                        {"type": "ai", "content": "# Report\n\nDone"},
                    ],
                },
            )

            self.assertEqual(len(inspector.list_runs()), 1)
            loaded_run = inspector.get_run(run_record.run_id)
            self.assertIsNotNone(loaded_run)
            assert loaded_run is not None
            self.assertEqual(loaded_run.run_id, run_record.run_id)
            self.assertEqual(loaded_run.root_node_id, context_node.node_id)
            self.assertEqual(len(inspector.list_nodes(run_record.run_id)), 2)
            self.assertEqual(
                inspector.load_node_snapshot(run_record.run_id, context_node.node_id)[
                    "memory_snapshot"
                ],
                "Frozen",
            )

            graph = inspector.get_run_graph(run_record.run_id)
            self.assertIsNotNone(graph)
            assert graph is not None
            self.assertEqual(graph.run.run_id, run_record.run_id)
            self.assertEqual(len(graph.nodes), 2)

            node_details = inspector.get_node_details(
                run_record.run_id,
                final_node.node_id,
            )
            self.assertIsNotNone(node_details)
            assert node_details is not None
            self.assertEqual(node_details.node.node_id, final_node.node_id)
            self.assertEqual(node_details.snapshot["final_report"], "# Report\n\nDone")
            self.assertEqual(len(node_details.artifacts), 1)

            self.assertEqual(len(inspector.list_artifacts(run_record.run_id)), 1)
            self.assertEqual(report_artifact.metadata["schema_version"], "artifact_metadata.v1")
            self.assertEqual(report_artifact.metadata["artifact_role"], "unspecified")
            self.assertEqual(report_artifact.metadata["producer"], "agent")
            self.assertEqual(report_artifact.metadata["content_kind"], "report")
            self.assertEqual(report_artifact.metadata["domain_label"], "manual-test")
            self.assertEqual(
                inspector.read_artifact_text(
                    run_record.run_id,
                    report_artifact.artifact_id,
                    max_chars=8,
                ),
                "# Report",
            )
            artifact_details = inspector.get_artifact_details(
                run_record.run_id,
                report_artifact.artifact_id,
                preview_chars=8,
            )
            self.assertIsNotNone(artifact_details)
            assert artifact_details is not None
            self.assertEqual(artifact_details.artifact.artifact_id, report_artifact.artifact_id)
            self.assertEqual(artifact_details.node.node_id, final_node.node_id)
            self.assertEqual(artifact_details.preview.preview, "# Report")
            self.assertTrue(artifact_details.preview.truncated)
            self.assertIsNone(artifact_details.preview.error)

            artifact_preview = inspector.preview_artifact(
                run_record.run_id,
                report_artifact.artifact_id,
                preview_chars=3,
            )
            self.assertIsNotNone(artifact_preview)
            assert artifact_preview is not None
            self.assertEqual(artifact_preview.preview, "# R")
            self.assertEqual(inspector.get_final_report(run_record.run_id), "# Report\n\nDone")

            summary = inspector.get_run_summary(run_record.run_id)
            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary.node_count, 2)
            self.assertEqual(summary.artifact_count, 1)
            self.assertEqual(summary.final_report_node_id, final_node.node_id)
            self.assertEqual(
                summary.final_report_artifact_id,
                report_artifact.artifact_id,
            )

            result = inspector.get_run_result(run_record.run_id)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.run.run_id, run_record.run_id)
            self.assertEqual(result.final_report, "# Report\n\nDone")
            self.assertEqual(result.final_state["final_report"], "# Report\n\nDone")
            self.assertEqual(result.messages[-1]["content"], "# Report\n\nDone")
            self.assertEqual(len(result.nodes), 2)
            self.assertEqual(len(result.artifacts), 1)

            summaries = inspector.list_run_summaries()
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0].run.run_id, run_record.run_id)

    def test_builds_node_inspector_view_for_ui(self) -> None:
        """Проверяет модель Node Inspector: связи графа, snapshot diff, artifacts и tool traces."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            inspector = RunInspectionService(lineage, artifacts)

            run_record = lineage.create_run(initial_user_query="Inspect node")
            parent_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="worker_started",
                title="Worker started",
                state={
                    "run_id": run_record.run_id,
                    "memory_snapshot": "Frozen",
                    "plan": {"task-1": {"status": "running"}},
                },
            )
            child_node = StateNode(
                run_id=run_record.run_id,
                node_type="task_completed",
                title="Task completed",
                parent_ids=[parent_node.node_id],
                status="succeeded",
            )
            result_artifact = artifacts.write_artifact(
                run_id=run_record.run_id,
                node_id=child_node.node_id,
                kind="model_output",
                filename="tasks/1/result.md",
                content="Detailed worker result",
                mime_type="text/markdown",
                summary="Worker result",
            )
            trace_artifact = artifacts.write_artifact(
                run_id=run_record.run_id,
                node_id=child_node.node_id,
                kind="tool_trace",
                filename="tasks/1/tool_calls/legacy_tool.txt",
                content="legacy_tool(path='demo.json')",
                mime_type="text/plain",
                summary="Tool trace",
            )
            child_node.artifact_refs = [result_artifact.artifact_id]
            child_node.tool_trace_refs = [trace_artifact.artifact_id]
            lineage.append_node(
                child_node,
                state={
                    "run_id": run_record.run_id,
                    "memory_snapshot": "Frozen",
                    "task_results": {"task-1": "Detailed worker result"},
                    "messages": [{"type": "ai", "content": "Detailed worker result"}],
                },
            )

            child_view = inspector.get_node_inspector_view(
                run_record.run_id,
                child_node.node_id,
                preview_chars=12,
                snapshot_preview_chars=20,
            )
            self.assertIsNotNone(child_view)
            assert child_view is not None
            self.assertEqual(child_view.run.run_id, run_record.run_id)
            self.assertEqual(child_view.node.node_id, child_node.node_id)
            self.assertEqual([node.node_id for node in child_view.parent_nodes], [parent_node.node_id])
            self.assertEqual(child_view.child_nodes, [])
            self.assertEqual(child_view.snapshot["task_results"]["task-1"], "Detailed worker result")
            self.assertEqual({section.name for section in child_view.snapshot_sections}, {
                "run_id",
                "memory_snapshot",
                "task_results",
                "messages",
            })
            self.assertTrue(
                any(section.name == "task_results" and section.item_count == 1 for section in child_view.snapshot_sections)
            )
            self.assertIsNotNone(child_view.diff_with_parent)
            assert child_view.diff_with_parent is not None
            self.assertEqual(child_view.diff_with_parent.parent_node_id, parent_node.node_id)
            self.assertIn("task_results", child_view.diff_with_parent.added_keys)
            self.assertIn("messages", child_view.diff_with_parent.added_keys)
            self.assertIn("plan", child_view.diff_with_parent.removed_keys)
            self.assertIn("memory_snapshot", child_view.diff_with_parent.unchanged_keys)
            self.assertEqual(len(child_view.artifacts), 1)
            self.assertEqual(child_view.artifacts[0].artifact.artifact_id, result_artifact.artifact_id)
            self.assertEqual(child_view.artifacts[0].preview.preview, "Detailed wor")
            self.assertTrue(child_view.artifacts[0].preview.truncated)
            self.assertEqual(len(child_view.tool_traces), 1)
            self.assertEqual(child_view.tool_traces[0].artifact.artifact_id, trace_artifact.artifact_id)
            self.assertEqual(child_view.warnings, [])

            parent_view = inspector.get_node_inspector_view(
                run_record.run_id,
                parent_node.node_id,
            )
            self.assertIsNotNone(parent_view)
            assert parent_view is not None
            self.assertEqual([node.node_id for node in parent_view.child_nodes], [child_node.node_id])
            self.assertIsNone(parent_view.diff_with_parent)


if __name__ == "__main__":
    unittest.main()

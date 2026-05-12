from __future__ import annotations

import tempfile
import unittest
from asyncio import run

from planner_agent.agent_nodes.scheduler_node import (
    _build_artifact_context,
    _collect_ancestor_data,
    _select_artifact_ids,
    _select_task_skill_previews,
    scheduler_node,
)
from planner_agent.models import AgentState, Task, TaskStatus
from planner_agent.services.lineage_service import LineageService


class SchedulerLineageTests(unittest.TestCase):
    def test_scheduler_selects_only_task_skill_previews(self) -> None:
        """Проверяет, что worker payload получает только явно назначенные skill previews."""

        task = Task(
            task_id="1",
            description="Analyze transactions",
            suggested_skills=["case-analysis"],
        )

        selected = _select_task_skill_previews(
            {
                "case-analysis": "Analyze cases.",
                "chart-design": "Build charts.",
            },
            task,
        )

        self.assertEqual(selected, {"case-analysis": "Analyze cases."})
        self.assertEqual(
            _select_task_skill_previews(
                {"case-analysis": "Analyze cases."},
                Task(task_id="2", description="No skill"),
            ),
            {},
        )

    def test_scheduler_creates_task_scheduled_node_for_ready_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            run_record = lineage.create_run(initial_user_query="Analyze case")
            plan_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="plan_created",
                title="Plan created",
            )
            state = AgentState(
                run_id=run_record.run_id,
                current_node_id=plan_node.node_id,
                parent_node_ids=[plan_node.node_id],
                plan={
                    "1": Task(task_id="1", description="Inspect data"),
                    "2": Task(task_id="2", description="Summarize context"),
                },
            )

            command = run(scheduler_node(state, lineage_service=lineage))

            update = command.update
            self.assertEqual(set(update["plan"].keys()), {"1", "2"})
            self.assertEqual(update["plan"]["1"].status.value, "running")
            self.assertEqual(update["plan"]["2"].status.value, "running")
            self.assertEqual(len(update["lineage_events"]), 1)
            self.assertTrue(update["current_node_id"])

            nodes = lineage.get_nodes(run_record.run_id)
            self.assertEqual(nodes[-1].node_type, "task_scheduled")
            self.assertEqual(nodes[-1].parent_ids, [plan_node.node_id])
            self.assertEqual(
                nodes[-1].metadata["scheduled_task_ids"],
                ["1", "2"],
            )

            snapshot = lineage.load_snapshot(run_record.run_id, nodes[-1].node_id)
            self.assertEqual(snapshot["plan"]["1"]["status"], "running")
            self.assertEqual(snapshot["plan"]["2"]["status"], "running")

    def test_scheduler_builds_compact_artifact_context_for_worker(self) -> None:
        artifact_payload = {
            "artifact_id": "artifact-1",
            "kind": "dataset",
            "uri": "C:/workspace/transactions.csv",
            "mime_type": "text/csv",
            "summary": "Transaction export",
            "checksum": "abc",
            "metadata": {
                "task_id": "load",
                "tool_name": "export_transactions",
                "reusable": True,
                "editable": True,
                "variable_name": "df_transactions",
                "columns": ["event_id", "amount"],
                "column_types": {"event_id": "str", "amount": "float"},
                "large_internal_field": "should not be copied",
            },
        }
        task = Task(task_id="2", description="Analyze export")
        state = AgentState(
            artifact_index={"artifact-1": artifact_payload},
            plan={"2": task},
        )

        context = _build_artifact_context(state, task)

        self.assertEqual(context["artifact_count"], 1)
        artifact = context["artifacts"]["artifact-1"]
        self.assertEqual(artifact["artifact_name"], "df_transactions")
        self.assertEqual(artifact["schema"], "event_id:str, amount:float")
        self.assertEqual(context["selected_artifact_ids"], ["artifact-1"])
        self.assertEqual(context["hidden_artifact_count"], 0)

    def test_scheduler_worker_context_skips_tool_trace_and_unknown_refs(self) -> None:
        state = AgentState(
            artifact_index={
                "df1": {
                    "artifact_id": "df1",
                    "kind": "dataset",
                    "metadata": {
                        "variable_name": "df1",
                        "columns": ["a"],
                        "column_types": {"a": "int"},
                    },
                },
                "tr1": {
                    "artifact_id": "tr1",
                    "kind": "tool_trace",
                    "metadata": {"artifact_role": "tool_call_trace", "task_id": "1"},
                },
            },
            plan={
                "1": Task(
                    task_id="1",
                    description="Analyze",
                    artifact_refs=["missing-ref"],
                ),
            },
        )
        task = state.plan["1"]
        self.assertEqual(_select_artifact_ids(state, task), ["df1"])

    def test_scheduler_ingests_upstream_artifacts_into_dependent_worker_payload(self) -> None:
        upstream_artifact = {
            "artifact_id": "artifact-events",
            "kind": "dataset",
            "uri": "C:/workspace/runs/run-1/artifacts/events.json",
            "mime_type": "application/json",
            "summary": "Captured client events",
            "checksum": "checksum-events",
            "metadata": {
                "task_id": "1",
                "tool_name": "load_events",
                "artifact_role": "captured_tool_result",
                "reusable": True,
                "editable": True,
                "capture_reason": "context_budget_exceeded",
                "original_size_estimate": 120000,
                "variable_name": "df_events",
                "columns": ["event_id", "event_dt"],
                "column_types": {"event_id": "str", "event_dt": "datetime"},
                "large_internal_field": "should not be copied",
            },
        }
        state = AgentState(
            run_id="run-1",
            artifact_index={"artifact-events": upstream_artifact},
            plan={
                "1": Task(
                    task_id="1",
                    description="Load events",
                    status=TaskStatus.COMPLETED,
                    artifact_refs=["artifact-events"],
                    result_preview="Events saved as artifact.",
                ),
                "2": Task(
                    task_id="2",
                    description="Analyze events",
                    dependencies=["1"],
                    status=TaskStatus.PENDING,
                ),
            },
        )

        command = run(scheduler_node(state))
        sends = command.goto

        self.assertEqual(len(sends), 1)
        payload = sends[0].arg
        artifact_context = payload.artifact_context

        self.assertEqual(artifact_context["artifact_count"], 1)
        self.assertEqual(
            artifact_context["selected_artifact_ids"],
            ["artifact-events"],
        )
        artifact = artifact_context["artifacts"]["artifact-events"]
        self.assertEqual(artifact["artifact_name"], "df_events")
        self.assertEqual(artifact["schema"], "event_id:str, event_dt:datetime")

    def test_scheduler_passes_transitive_dependency_results_to_worker(self) -> None:
        """Проверяет сквозную передачу результатов по цепочке зависимостей."""

        state = AgentState(
            run_id="run-1",
            plan={
                "1": Task(
                    task_id="1",
                    description="Load trigger",
                    status=TaskStatus.COMPLETED,
                    full_result="Trigger details: amount=751, rule=recipient_velocity",
                    artifact_refs=["artifact-trigger"],
                ),
                "2": Task(
                    task_id="2",
                    description="Load transactions",
                    dependencies=["1"],
                    status=TaskStatus.COMPLETED,
                    result_preview="Transactions artifact is ready.",
                    artifact_refs=["artifact-transactions"],
                ),
                "3": Task(
                    task_id="3",
                    description="Analyze transactions against trigger",
                    dependencies=["2"],
                    status=TaskStatus.PENDING,
                ),
            },
            artifact_index={
                "artifact-trigger": {
                    "artifact_id": "artifact-trigger",
                    "kind": "dataset",
                    "uri": "C:/workspace/trigger.json",
                    "metadata": {
                        "task_id": "1",
                        "variable_name": "df_trigger",
                        "columns": ["event_id", "amount"],
                        "column_types": {"event_id": "str", "amount": "float"},
                    },
                },
                "artifact-transactions": {
                    "artifact_id": "artifact-transactions",
                    "kind": "dataset",
                    "uri": "C:/workspace/transactions.json",
                    "metadata": {
                        "task_id": "2",
                        "variable_name": "df_transactions",
                        "columns": ["event_id", "event_dt"],
                        "column_types": {"event_id": "str", "event_dt": "datetime"},
                    },
                },
            },
        )

        command = run(scheduler_node(state))
        payload = command.goto[0].arg

        deps = payload.dependency_context.get("dependencies") or []
        previews = " ".join(str(d.get("result_preview") or "") for d in deps)
        self.assertIn("Trigger details", previews)
        self.assertIn("Transactions artifact is ready", previews)
        self.assertIn("Trigger details", payload.previous_results)
        self.assertIn("Transactions artifact is ready", payload.previous_results)
        self.assertEqual(
            payload.artifact_context["selected_artifact_ids"][:2],
            ["artifact-transactions", "artifact-trigger"],
        )

    def test_scheduler_passes_skill_previews_to_worker_payload(self) -> None:
        """Проверяет передачу preview skills из state в WorkerPayload."""

        state = AgentState(
            skill_previews={"case-analysis": "Domain analysis skill preview."},
            plan={
                "1": Task(
                    task_id="1",
                    description="Analyze case",
                    status=TaskStatus.PENDING,
                    suggested_skills=["case-analysis"],
                )
            },
        )

        command = run(scheduler_node(state))
        sends = command.goto

        self.assertEqual(len(sends), 1)
        payload = sends[0].arg
        self.assertEqual(
            payload.skill_previews,
            {"case-analysis": "Domain analysis skill preview."},
        )

    def test_scheduler_redirects_blocked_plan_to_replanner(self) -> None:
        """Проверяет переход к replanner при блокировке failed-зависимостью."""

        state = AgentState(
            plan={
                "1": Task(
                    task_id="1",
                    description="Load trigger case",
                    status=TaskStatus.FAILED,
                    error_log="Tool failed",
                ),
                "2": Task(
                    task_id="2",
                    description="Extract trigger parameters",
                    dependencies=["1"],
                    status=TaskStatus.PENDING,
                ),
            },
        )

        command = run(scheduler_node(state))

        self.assertEqual(command.goto, "replanner")
        self.assertEqual(len(command.update["feedback_context"]), 1)
        feedback = command.update["feedback_context"][0]
        self.assertEqual(feedback["failed_task_diagnosis"][0]["task_id"], "2")
        self.assertEqual(
            feedback["failed_task_diagnosis"][0]["blocked_by"],
            [{"task_id": "1", "status": "failed"}],
        )

    def test_collect_ancestor_data_uses_full_result_when_available(self) -> None:
        """Проверяет, что в контекст предков попадает полный результат задачи."""

        plan = {
            "1": Task(
                task_id="1",
                description="Load source",
                full_result="Full source result",
                result_preview="Preview only",
                status=TaskStatus.COMPLETED,
            ),
            "2": Task(
                task_id="2",
                description="Analyze source",
                dependencies=["1"],
                result_preview="Analysis preview",
                status=TaskStatus.COMPLETED,
            ),
        }

        _, previews = _collect_ancestor_data(plan, "2")

        self.assertIn("Task 2 result: Analysis preview", previews)
        self.assertIn("Task 1 result: Full source result", previews)
        self.assertNotIn("Task 1 result: Preview only", previews)


if __name__ == "__main__":
    unittest.main()

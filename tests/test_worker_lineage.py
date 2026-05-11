from __future__ import annotations

import tempfile
import unittest

from planner_agent.agent_nodes.worker_node import (
    _create_worker_finished_lineage,
    _create_worker_started_lineage,
)
from planner_agent.models import Task, TaskStatus, WorkerPayload
from planner_agent.services.artifact_service import ArtifactService
from planner_agent.services.lineage_service import LineageService


class WorkerLineageTests(unittest.TestCase):
    def test_worker_started_and_task_completed_nodes_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            run_record = lineage.create_run(initial_user_query="Analyze case")
            parent = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="task_scheduled",
                title="Task scheduled",
            )
            task = Task(
                task_id="1",
                description="Inspect data",
                status=TaskStatus.RUNNING,
            )
            payload = WorkerPayload(
                task=task,
                context_schemas={"df_current": "shape=(1, 3)"},
                previous_results="",
                run_id=run_record.run_id,
                parent_node_ids=[parent.node_id],
            )
            events: list[dict] = []

            started_id = _create_worker_started_lineage(
                payload=payload,
                task=task,
                lineage_service=lineage,
                lineage_events=events,
            )

            task.status = TaskStatus.NEEDS_VALIDATION
            task.result_preview = "Found candidate pattern."
            task.generated_code = "print('trace only')"
            artifact_index: dict = {}
            finished_id = _create_worker_finished_lineage(
                payload=payload,
                task=task,
                data_schemas={"df_current": "shape=(2, 3)"},
                lineage_service=lineage,
                artifact_service=artifacts,
                parent_node_id=started_id,
                lineage_events=events,
                artifact_index=artifact_index,
            )

            self.assertTrue(started_id)
            self.assertTrue(finished_id)
            self.assertEqual(len(events), 2)

            nodes = lineage.get_nodes(run_record.run_id)
            self.assertEqual(nodes[-2].node_type, "worker_started")
            self.assertEqual(nodes[-2].parent_ids, [parent.node_id])
            self.assertEqual(nodes[-1].node_type, "task_completed")
            self.assertEqual(nodes[-1].parent_ids, [started_id])
            self.assertEqual(nodes[-1].metadata["task_id"], "1")
            # task.artifact_refs хранит только worker_result; code_trace
            # доступен через state.artifact_index, но не дублируется в refs.
            self.assertEqual(len(task.artifact_refs), 1)
            self.assertEqual(nodes[-1].artifact_refs, task.artifact_refs)
            self.assertEqual(len(artifact_index), 2)

            stored_artifacts = artifacts.list_artifacts(run_record.run_id)
            self.assertEqual(
                [artifact.kind for artifact in stored_artifacts],
                ["model_output", "code_trace"],
            )
            self.assertTrue(stored_artifacts[0].uri.endswith("result.md"))
            self.assertTrue(stored_artifacts[1].uri.endswith("code_trace.txt"))
            # Артефакты получают человеко-читаемые id вида ``t{task}_result``/``t{task}_code``.
            self.assertEqual(stored_artifacts[0].artifact_id, "t1_result")
            self.assertEqual(stored_artifacts[1].artifact_id, "t1_code")

            snapshot = lineage.load_snapshot(run_record.run_id, nodes[-1].node_id)
            self.assertEqual(snapshot["task"]["status"], "needs_validation")
            self.assertEqual(snapshot["task"]["artifact_refs"], task.artifact_refs)
            self.assertEqual(snapshot["data_schemas"], {"df_current": "shape=(2, 3)"})
            self.assertEqual(set(snapshot["artifact_index"]), set(artifact_index))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import planner_agent
from planner_agent.models import AgentState, Task
from planner_agent.schemas.feedback import UserFeedback
from planner_agent.schemas.lineage import BranchRequest
from planner_agent.services import (
    ArtifactService,
    FeedbackService,
    LineageService,
    MemoryService,
    PolicyEngine,
    SkillsService,
)


class Phase1ServiceTests(unittest.TestCase):
    def test_package_import_does_not_require_sandbox(self) -> None:
        self.assertTrue(callable(planner_agent.planner_agent))
        state = AgentState(run_id="run-1", plan={"1": Task(task_id="1", description="Inspect data")})
        self.assertEqual(state.plan["1"].status.value, "pending")

    def test_lineage_snapshot_and_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = LineageService(tmp)
            run = service.create_run(initial_user_query="Analyze quarterly sales", session_id="s1")
            node = service.create_state_node(
                run_id=run.run_id,
                node_type="plan_created",
                title="Plan created",
                state=AgentState(run_id=run.run_id, initial_user_query=run.initial_user_query),
            )

            snapshot = service.load_snapshot(run.run_id, node.node_id)
            self.assertEqual(snapshot["run_id"], run.run_id)

            branch = service.branch_from(
                BranchRequest(
                    source_run_id=run.run_id,
                    source_node_id=node.node_id,
                    new_task="Revise with margin assumptions",
                    branch_mode="revise",
                )
            )
            self.assertEqual(branch.parent_run_id, run.run_id)
            self.assertEqual(len(service.get_nodes(branch.run_id)), 1)

    def test_artifact_memory_feedback_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact_service = ArtifactService(tmp)
            artifact = artifact_service.write_artifact(
                run_id="run-1",
                node_id="node-1",
                kind="report",
                filename="summary.md",
                content="# Summary",
                mime_type="text/markdown",
            )
            self.assertEqual(len(artifact.checksum), 64)
            self.assertEqual(len(artifact_service.list_artifacts("run-1")), 1)

            memory = MemoryService(Path(tmp) / "memory")
            snapshot = memory.load_snapshot(run_id="run-1")
            self.assertEqual(snapshot.run_id, "run-1")
            proposal = memory.propose_write(target="project", content="Use artifact IDs in final reports.")
            memory.apply_proposal(proposal)
            self.assertIn("artifact IDs", memory.read("project"))

            feedback = FeedbackService(tmp)
            feedback.record_feedback(UserFeedback(run_id="run-1", rating="like"))
            self.assertEqual(len(feedback.list_feedback("run-1")), 1)

            policy = PolicyEngine(runs_dir=tmp)
            allowed = policy.evaluate_tool_call("read_table", {"table_name": "hits", "select_columns": "event_id"})
            denied = policy.evaluate_tool_call("forbidden_tool", {"file_path": "run.ps1"})
            review = policy.evaluate_tool_call("unknown_tool", {})
            self.assertEqual(allowed.decision, "allow")
            self.assertEqual(denied.decision, "deny")
            self.assertEqual(review.decision, "review")

    def test_skills_service_index_and_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "dataset-profiling"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: dataset-profiling\n"
                "description: Profile tabular datasets.\n"
                "---\n"
                "# Procedure\n\nInspect schema.\n",
                encoding="utf-8",
            )

            service = SkillsService(tmp)
            skills = service.skills_list()
            self.assertEqual(skills[0].name, "dataset-profiling")
            self.assertIn("dataset-profiling", service.build_skills_index())
            self.assertTrue(service.skill_view("dataset-profiling")["success"])


if __name__ == "__main__":
    unittest.main()

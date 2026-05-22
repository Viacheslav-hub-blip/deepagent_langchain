"""Тесты узла critic.

Содержит:
- CriticNodeTests: проверки критики результата worker-а, retry и lineage.
"""

from __future__ import annotations

import tempfile
import unittest
from asyncio import run

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langgraph.types import Send

from planner_agent.agent_nodes.critic_node import critic_node
from planner_agent.models import CriticPayload, Task, TaskStatus, WorkerPayload
from planner_agent.prompts import AnalysisAgentPrompts
from planner_agent.services.lineage_service import LineageService


class CriticNodeTests(unittest.TestCase):
    """Проверяет работу critic node без запуска реального LLM-провайдера."""

    def test_critic_sends_task_back_to_worker_when_result_is_incomplete(self) -> None:
        """Проверяет, что critic возвращает worker-задачу на доработку."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            run_record = lineage.create_run(initial_user_query="Analyze trigger")
            parent = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="task_completed",
                title="Task finished",
            )
            task = Task(
                task_id="2",
                description="Analyze events around trigger",
                status=TaskStatus.NEEDS_VALIDATION,
                result_preview="Checked +/-3 days, found nothing.",
                full_result="No events found in +/-3 days.",
            )
            payload = CriticPayload(
                worker_payload=WorkerPayload(
                    task=task,
                    context_schemas={},
                    previous_results="Trigger date: 2026-03-09",
                    run_id=run_record.run_id,
                ),
                run_id=run_record.run_id,
                parent_node_ids=[parent.node_id],
            )
            llm = FakeListChatModel(
                responses=[
                    (
                        '{"approved":false,'
                        '"reasoning":"The time window is too narrow for a negative result.",'
                        '"issues":["Only +/-3 days were checked."],'
                        '"improvement_instructions":"Expand the event search window to at least +/-14 days and report the exact period."}'
                    )
                ]
            )

            command = run(
                critic_node(
                    payload=payload,
                    llm=llm,
                    prompt=AnalysisAgentPrompts().critic_system,
                    lineage_service=lineage,
                )
            )

            self.assertIsInstance(command.goto[0], Send)
            self.assertEqual(command.goto[0].node, "worker")
            updated_task = command.update["plan"]["2"]
            self.assertEqual(updated_task.retry_count, 1)
            self.assertEqual(updated_task.status, TaskStatus.READY)
            self.assertIn("Expand the event search window", updated_task.error_log)
            self.assertEqual(len(command.update["feedback_context"]), 1)

            nodes = lineage.get_nodes(run_record.run_id)
            self.assertEqual(nodes[-1].node_type, "worker_critic")
            self.assertEqual(nodes[-1].metadata["task_id"], "2")
            self.assertFalse(nodes[-1].metadata["approved"])

    def test_critic_sends_approved_result_to_validator(self) -> None:
        """Проверяет, что хороший worker-result передается validator-у."""

        task = Task(
            task_id="1",
            description="Load trigger",
            status=TaskStatus.NEEDS_VALIDATION,
            result_preview="Trigger loaded with event_id and epk_id.",
        )
        payload = CriticPayload(
            worker_payload=WorkerPayload(
                task=task,
                context_schemas={},
                previous_results="",
                run_id="run-1",
            ),
            run_id="run-1",
        )
        llm = FakeListChatModel(
            responses=[
                (
                    '{"approved":true,'
                    '"reasoning":"The worker returned concrete identifiers.",'
                    '"issues":[],'
                    '"improvement_instructions":""}'
                )
            ]
        )

        command = run(
            critic_node(
                payload=payload,
                llm=llm,
                prompt=AnalysisAgentPrompts().critic_system,
            )
        )

        self.assertIsInstance(command.goto[0], Send)
        self.assertEqual(command.goto[0].node, "validator")
        self.assertNotIn("feedback_context", command.update)
        self.assertEqual(command.update["plan"]["1"].retry_count, 0)

    def test_critic_retry_limit_sends_to_validator(self) -> None:
        """Проверяет, что critic не отправляет worker-а больше двух раз."""

        task = Task(
            task_id="1",
            description="Analyze wider period",
            status=TaskStatus.NEEDS_VALIDATION,
            retry_count=2,
            result_preview="Still partial.",
        )
        payload = CriticPayload(
            worker_payload=WorkerPayload(
                task=task,
                context_schemas={},
                previous_results="",
            )
        )
        llm = FakeListChatModel(
            responses=[
                (
                    '{"approved":false,'
                    '"reasoning":"Still incomplete.",'
                    '"issues":["No wider period."],'
                    '"improvement_instructions":"Expand period again."}'
                )
            ]
        )

        command = run(
            critic_node(
                payload=payload,
                llm=llm,
                prompt=AnalysisAgentPrompts().critic_system,
            )
        )

        self.assertEqual(command.goto[0].node, "validator")
        self.assertEqual(command.update["plan"]["1"].retry_count, 2)


if __name__ == "__main__":
    unittest.main()

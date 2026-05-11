from __future__ import annotations

import unittest

from planner_agent.prompts import AnalysisAgentPrompts


class WorkerPromptContractTests(unittest.TestCase):
    def test_replanner_prompt_requires_recovery_for_failed_dependency(self) -> None:
        """Replanner must recover from failed dependencies, not pretend the goal is met."""

        prompt = AnalysisAgentPrompts().replanner_system

        self.assertIn("failed dependency", prompt)
        self.assertIn("replacement/recovery", prompt)
        self.assertIn("новым task_id", prompt)
        self.assertIn("Не считай цель достигнутой", prompt)

    def test_critic_prompt_reviews_worker_before_validator(self) -> None:
        """Critic gates worker output before validator."""

        prompt = AnalysisAgentPrompts().critic_system

        self.assertIn("результат одной worker-задачи перед validator", prompt)
        self.assertIn("approved=false", prompt)
        self.assertIn("±3 дня", prompt)
        self.assertIn("Система сама ограничивает число повторов critic-а до 2", prompt)

    def test_worker_prompt_requires_full_result_not_status_only(self) -> None:
        prompt = AnalysisAgentPrompts().worker_system

        self.assertIn("содержательный результат", prompt)
        self.assertIn("не плейсхолдеры", prompt)
        self.assertIn("демонстрационными/примерными", prompt)
        self.assertIn("artifact_id", prompt)

    def test_validator_prompt_rejects_unsupported_code_calculations(self) -> None:
        """Validator must reject calculations not grounded in real task data."""

        prompt = AnalysisAgentPrompts().validator_system

        self.assertIn("генерации/исполнении кода", prompt)
        self.assertIn("без видимого источника", prompt)
        self.assertIn("демонстрационные/примерные входные записи", prompt)


if __name__ == "__main__":
    unittest.main()

"""Тесты внутреннего data-retrieval-critic и inspect_artifact_path."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

from deep_agent_test.agent_specs import DataRetrievalCriticVerdict
from deep_agent_test.inspect_artifact_tool import inspect_artifact_path
from deep_agent_test.retrieval_subagents import (
    build_critic_filesystem_permissions,
    build_data_retrieval_critic_tools,
)
from deep_agent_test.settings import DeepAgentSettings, PROJECT_ROOT


class RetrievalCriticTests(unittest.TestCase):
    def _settings(self, tmp_dir: Path) -> DeepAgentSettings:
        return DeepAgentSettings(
            thread_id="test",
            skills_virtual_dir="/skills/",
            skills_root=PROJECT_ROOT / "deep_agent_test" / "skills",
            data_tools_factory=None,
            data_tools_factory_kwargs={},
            tool_outputs_dir=tmp_dir,
            max_chars_per_skill=1000,
            tool_output_min_rows_to_save=10,
            tool_output_min_content_chars_to_save=60000,
            tool_output_preview_rows=3,
            tool_output_inline_original_chars=1000,
            context_edit_trigger_tokens=100000,
            context_edit_keep_tool_results=3,
            file_search_use_ripgrep=False,
            max_consecutive_tool_calls=4,
            max_subagent_model_calls=19,
            max_critic_iterations=3,
            graph_recursion_limit=50,
            trace_log_dir=Path("runs/deep_agent_traces"),
        )

    def test_inspect_artifact_path_reports_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            payload = inspect_artifact_path(str(Path(tmp) / "missing.pkl"), settings=settings)
            self.assertIn('"exists": false', payload)

    def test_inspect_artifact_path_reads_pickle_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            file_path = Path(tmp) / "sample.pkl"
            rows = [{"event_id": "a", "value": 1}, {"event_id": "b", "value": 2}]
            with file_path.open("wb") as handle:
                pickle.dump(rows, handle)

            payload = inspect_artifact_path(str(file_path), pickle_preview_rows=1, settings=settings)
            self.assertIn('"exists": true', payload)
            self.assertIn('"pickle_row_count": 2', payload)
            self.assertIn("event_id", payload)

    def test_critic_tools_and_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            tools = build_data_retrieval_critic_tools(settings)
            self.assertEqual(tools[0].name, "inspect_artifact_path")
            permissions = build_critic_filesystem_permissions(settings)
            self.assertTrue(any("/tool_outputs/" in permission.paths[0] for permission in permissions))

    def test_critic_verdict_schema_is_flexible(self) -> None:
        verdict = DataRetrievalCriticVerdict(
            approved=False,
            reasoning="Нет подтверждения файла",
            issues=["claimed pickle missing"],
            revision_instructions="Повтори read_table и сохрани spill",
            checks_performed=["inspect_artifact_path"],
        )
        self.assertFalse(verdict.approved)
        self.assertIn("inspect_artifact_path", verdict.checks_performed[0])


if __name__ == "__main__":
    unittest.main()

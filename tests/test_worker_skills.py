from __future__ import annotations

import tempfile
import unittest
import json
from asyncio import run
from pathlib import Path

from planner_agent.agent_nodes.planner_node import build_full_plan
from planner_agent.agent_nodes.worker_node import (
    _create_worker_started_lineage,
    _create_worker_system_prompt,
    _format_artifact_context,
    _format_available_skill_index,
    _load_task_skills,
    _select_task_tools,
)
from langchain_core.tools import tool
from planner_agent.models import FullPlan, PlannedTask, Task, WorkerPayload
from planner_agent.services.lineage_service import LineageService
from planner_agent.services.skills_service import SkillsService
from planner_agent.tools.skill_tools import build_skill_read_tools
from planner_agent.tools.execute_python_code_tool import build_execute_python_code_tool


class WorkerSkillLoadingTests(unittest.TestCase):
    def test_worker_artifact_context_prompt_is_name_and_schema_only(self) -> None:
        block = _format_artifact_context(
            {
                "artifacts": {
                    "196441adc487458bae281cf686a54bd6": {
                        "artifact_name": "t1_spark_query_table_1",
                        "schema": "event_id:str",
                    },
                },
            }
        )
        self.assertIn("<ARTIFACT t1_spark_query_table_1>", block)
        self.assertIn("schema: event_id:str", block)
        self.assertNotIn("uri:", block)
        self.assertNotIn("metadata:", block)
        self.assertNotIn("[artifact_usage_rules]", block)

    def test_worker_selects_all_non_skill_tools(self) -> None:
        """Проверяет, что worker получает только явно назначенные domain tools."""

        @tool
        def source_tool() -> str:
            """Вернуть тестовый результат source tool."""

            return "source"

        @tool
        def another_source_tool() -> str:
            """Вернуть тестовый результат code tool."""

            return "another"

        selected = _select_task_tools(
            [source_tool, another_source_tool],
            Task(task_id="1", description="Use source", suggested_tools=["source_tool"]),
        )

        self.assertEqual(
            [item.name for item in selected],
            ["source_tool", "another_source_tool"],
        )

        no_suggestions = _select_task_tools(
            [source_tool, another_source_tool],
            Task(task_id="2", description="Analyze from context"),
        )

        self.assertEqual(
            [item.name for item in no_suggestions],
            ["source_tool", "another_source_tool"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            skill_tools = build_skill_read_tools(SkillsService(tmp))
            with_skill_tools = _select_task_tools(
                [source_tool, *skill_tools],
                Task(task_id="3", description="Analyze from context"),
            )
            self.assertEqual(
                [item.name for item in with_skill_tools],
                ["source_tool", "list_skills", "load_skill"],
            )

    def test_worker_includes_execute_python_code_without_suggested_tool_match(self) -> None:
        """Проверяет совместимость старых планов с новым инструментом execute_python_code."""

        class FakeSandbox:
            """Минимальная sandbox для проверки выбора инструмента."""

            globals = {}
            last_dataframe_variable = None

            async def get_all_variable_previews(self) -> dict[str, str]:
                """Возвращает пустые previews переменных."""

                return {}

            async def add_variable(self, name: str, value: object) -> None:
                """Добавляет переменную в sandbox."""

                self.globals[name] = value

            async def get_variable(self, name: str) -> object:
                """Возвращает переменную из sandbox."""

                return self.globals.get(name)

        selected = _select_task_tools(
            [build_execute_python_code_tool(FakeSandbox())],
            Task(
                task_id="1",
                description="Run code",
                suggested_tools=["missing_legacy_tool"],
            ),
        )

        self.assertEqual([item.name for item in selected], ["execute_python_code"])

    def test_planner_preserves_suggested_skills_in_task(self) -> None:
        plan = build_full_plan(
            current_plan={},
            full_plan=FullPlan(
                objective="Find insight",
                tasks=[
                    PlannedTask(
                        task_id="1",
                        description="Analyze repeated behavior",
                        suggested_skills=["insight-design"],
                        suggested_tools=["read_table"],
                        required_artifacts=["worker result"],
                    )
                ],
            ),
            current_run_id="1",
        )

        task = plan["1"]
        self.assertEqual(task.suggested_skills, ["insight-design"])
        self.assertEqual(task.suggested_tools, ["read_table"])
        self.assertEqual(task.required_artifacts, ["worker result"])

    def test_worker_loads_full_skill_content_and_adds_it_to_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "insight-design"
            skill_dir.mkdir(parents=True)
            skill_content = (
                "---\n"
                "name: insight-design\n"
                "description: Build behavioral insights.\n"
                "---\n"
                "# Procedure\n\nSeparate observed facts from interpretation.\n"
            )
            (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")

            task = Task(
                task_id="1",
                description="Find insight",
                suggested_skills=["insight-design", "missing-skill"],
                suggested_tools=["read_table"],
                expected_output="Insight with evidence references.",
                required_artifacts=["dataset export"],
            )
            loaded = _load_task_skills(task, SkillsService(tmp))

            self.assertEqual(list(loaded), ["insight-design"])
            self.assertIn("Separate observed facts", loaded["insight-design"])

            prompt = run(
                _create_worker_system_prompt(
                    WorkerPayload(
                        task=task,
                        context_schemas={"df_current": "shape=(6, 9)"},
                        previous_results="",
                        filesystem_context={
                            "workspace_root": "C:/workspace",
                            "sources_dir": "C:/workspace/data",
                        },
                        skill_previews={
                            "insight-design": "Build behavioral insights.",
                        },
                        artifact_context={
                            "artifacts": {
                                "artifact-1": {
                                    "artifact_name": "df_transactions",
                                    "schema": "event_id:str, amount:float",
                                }
                            },
                        },
                    ),
                    "Task={task_description}\nVars={schema_text}\nConfig={task_config}\nPrev={previous_results}",
                    loaded,
                )
            )

            self.assertIn("<task_contract>", prompt)
            self.assertIn("expected_output: Insight with evidence references.", prompt)
            self.assertIn("required_artifacts:", prompt)
            self.assertIn("- dataset export", prompt)
            self.assertIn("<workspace_context>", prompt)
            self.assertIn("workspace_root: C:/workspace", prompt)
            self.assertIn("<artifact_context>", prompt)
            self.assertIn("<ARTIFACT df_transactions>", prompt)
            self.assertIn("schema: event_id:str, amount:float", prompt)
            self.assertNotIn("C:/workspace/data/transactions.csv", prompt)
            self.assertIn("<available_skill_index>", prompt)
            self.assertIn("load_skill", prompt)
            self.assertIn("<loaded_skills>", prompt)
            self.assertIn("<skill name=\"insight-design\">", prompt)
            self.assertIn("Separate observed facts", prompt)

    def test_worker_does_not_auto_load_skill_without_planner_suggestion(self) -> None:
        """Проверяет fallback-загрузку skill из previews без suggested_skills."""

        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "case-analysis"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                (
                    "---\n"
                    "name: case-analysis\n"
                    "description: Analyze domain cases.\n"
                    "---\n"
                    "# Procedure\n\nUse domain evidence before conclusions.\n"
                ),
                encoding="utf-8",
            )

            loaded = _load_task_skills(
                Task(task_id="1", description="Analyze case"),
                SkillsService(tmp),
            )

            self.assertEqual(loaded, {})
            index = _format_available_skill_index(
                SkillsService(tmp).build_skill_previews()
            )
            self.assertIn("<available_skill_index>", index)
            self.assertIn("case-analysis: Analyze domain cases.", index)
            self.assertIn("load_skill", index)

    def test_load_skill_uses_frontmatter_name_and_runtime_tool_loads_content(self) -> None:
        """Проверяет загрузку skill по имени из frontmatter через сервис и runtime tool."""

        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "folder-name"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                (
                    "---\n"
                    "name: canonical-skill\n"
                    "description: Canonical skill description.\n"
                    "---\n"
                    "# Procedure\n\nFollow canonical instructions.\n"
                ),
                encoding="utf-8",
            )

            service = SkillsService(tmp)
            previews = service.build_skill_previews()
            self.assertEqual(
                previews,
                {"canonical-skill": "Canonical skill description."},
            )
            self.assertTrue(service.skill_view("canonical-skill")["success"])

            tools = {tool.name: tool for tool in build_skill_read_tools(service)}
            result = json.loads(
                tools["load_skill"].invoke({"name": "canonical-skill"})
            )

            self.assertTrue(result["success"])
            self.assertIn("Follow canonical instructions", result["content"])

    def test_worker_started_lineage_snapshot_contains_loaded_skills(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            run_record = lineage.create_run(initial_user_query="Analyze")
            task = Task(
                task_id="1",
                description="Find insight",
                suggested_skills=["insight-design"],
            )
            payload = WorkerPayload(
                task=task,
                context_schemas={},
                previous_results="",
                run_id=run_record.run_id,
            )
            events: list[dict] = []

            node_id = _create_worker_started_lineage(
                payload=payload,
                task=task,
                lineage_service=lineage,
                lineage_events=events,
                loaded_skills={"insight-design": "Full skill body"},
            )

            node = lineage.get_node(run_record.run_id, node_id)
            self.assertEqual(node.metadata["loaded_skill_names"], ["insight-design"])
            snapshot = lineage.load_snapshot(run_record.run_id, node_id)
            self.assertEqual(
                snapshot["loaded_skills"],
                {"insight-design": "Full skill body"},
            )


if __name__ == "__main__":
    unittest.main()

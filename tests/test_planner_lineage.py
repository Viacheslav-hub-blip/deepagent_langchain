from __future__ import annotations

import tempfile
import unittest
from asyncio import run

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import ValidationError
from pathlib import Path

from planner_agent.agent_nodes.planner_node import (
    _apply_plan_patch,
    _build_overwrite_patch,
    _load_planner_skills,
    build_full_plan,
    planner_node,
)
from planner_agent.models import AgentState, FullPlan, PlannedTask, Task, TaskStatus
from planner_agent.services.lineage_service import LineageService
from planner_agent.services.skills_service import SkillsService


class PlannerLineageTests(unittest.TestCase):
    """Проверяет создание плана и фильтрацию задач планировщика."""

    def test_full_plan_rejects_empty_task_list(self) -> None:
        """Проверяет, что структурированная схема планировщика запрещает пустой план."""

        with self.assertRaises(ValidationError):
            FullPlan(objective="Analyze case", tasks=[])
        with self.assertRaises(ValidationError):
            FullPlan(objective="Analyze case")

    def test_planned_task_rejects_text_task_id(self) -> None:
        """Проверяет, что задача в плане принимает только числовой task_id."""

        with self.assertRaises(ValidationError):
            PlannedTask(task_id="load_source", description="Inspect source data")

    def test_planner_creates_plan_created_node(self) -> None:
        """Проверяет, что planner создает plan_created node."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            run_record = lineage.create_run(initial_user_query="Analyze case")
            parent_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="context_snapshot",
                title="Context snapshot",
            )
            state = AgentState(
                run_id=run_record.run_id,
                current_node_id=parent_node.node_id,
                parent_node_ids=[parent_node.node_id],
                messages=[HumanMessage(content="Analyze case")],
                initial_user_query="Analyze case",
            )
            llm = FakeListChatModel(
                responses=[
                    (
                        '{"objective":"Analyze case","tasks":['
                        '{"task_id":"1","description":"Inspect source data","dependencies":[],"config":{}}'
                        ']}'
                    )
                ]
            )

            command = run(
                planner_node(
                    state=state,
                    llm=llm,
                    tools=[],
                    prompt="{plan_str}\n{initial_user_query}\n"
                    "{execution_results}\n{df_info}\n{tools_desc}\n"
                    "{previous_context}\n{schema_str}",
                    lineage_service=lineage,
                )
            )

            update = command.update
            self.assertEqual(command.goto, "scheduler")
            self.assertIn("1", update["plan"])
            self.assertTrue(update["current_node_id"])
            self.assertEqual(len(update["lineage_events"]), 1)

            nodes = lineage.get_nodes(run_record.run_id)
            self.assertEqual(nodes[-1].node_type, "plan_created")
            self.assertEqual(nodes[-1].parent_ids, [parent_node.node_id])

            snapshot = lineage.load_snapshot(run_record.run_id, nodes[-1].node_id)
            self.assertIn("1", snapshot["plan"])

    def test_planner_loads_selected_skill_before_building_plan(self) -> None:
        """Проверяет двухшаговую загрузку skill перед составлением плана."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills_dir = root / "skills"
            skill_dir = skills_dir / "case-routing"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                (
                    "---\n"
                    "name: case-routing\n"
                    "description: Choose sources and dependencies for case analysis.\n"
                    "---\n"
                    "# Procedure\n\nLoad source details before timeline analysis.\n"
                ),
                encoding="utf-8",
            )
            skills_service = SkillsService(skills_dir)
            lineage = LineageService(root / "runs")
            run_record = lineage.create_run(initial_user_query="Analyze case")
            parent_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="context_snapshot",
                title="Context snapshot",
            )
            state = AgentState(
                run_id=run_record.run_id,
                current_node_id=parent_node.node_id,
                parent_node_ids=[parent_node.node_id],
                messages=[HumanMessage(content="Analyze case")],
                initial_user_query="Analyze case",
                skills_index=skills_service.build_skills_index(),
                skill_previews=skills_service.build_skill_previews(),
            )
            llm = FakeListChatModel(
                responses=[
                    '{"skill_names":["case-routing"],"rationale":"Relevant routing skill."}',
                    (
                        '{"objective":"Analyze case","tasks":['
                        '{"task_id":"1","description":"Inspect source data","dependencies":[],"config":{},'
                        '"suggested_skills":["case-routing"]}'
                        ']}'
                    ),
                ]
            )

            command = run(
                planner_node(
                    state=state,
                    llm=llm,
                    tools=[],
                    prompt="{plan_str}\n{initial_user_query}\n"
                    "{execution_results}\n{df_info}\n{tools_desc}\n"
                    "{previous_context}\n{critic_feedback}\n{schema_str}",
                    lineage_service=lineage,
                    skills_service=skills_service,
                )
            )

            update = command.update
            self.assertIn("case-routing", update["loaded_skills"])
            self.assertIn(
                "Load source details",
                update["loaded_skills"]["case-routing"],
            )
            self.assertEqual(
                update["plan"]["1"].suggested_skills,
                ["case-routing"],
            )

    def test_planner_reuses_already_loaded_skill_content(self) -> None:
        """Проверяет, что planner не перечитывает уже загруженный skill."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills_dir = root / "skills"
            skill_dir = skills_dir / "case-routing"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                (
                    "---\n"
                    "name: case-routing\n"
                    "description: Choose sources and dependencies.\n"
                    "---\n"
                    "# Procedure\n\nFresh file content.\n"
                ),
                encoding="utf-8",
            )
            skills_service = SkillsService(skills_dir)
            state = AgentState(
                messages=[HumanMessage(content="Analyze case")],
                initial_user_query="Analyze case",
                skills_index=skills_service.build_skills_index(),
                skill_previews=skills_service.build_skill_previews(),
                loaded_skills={"case-routing": "Cached skill content."},
            )
            llm = FakeListChatModel(
                responses=[
                    '{"skill_names":["case-routing","case-routing"],"rationale":"Still relevant."}',
                ]
            )

            loaded = run(
                _load_planner_skills(
                    llm=llm,
                    state=state,
                    skills_service=skills_service,
                    initial_user_query="Analyze case",
                    plan_str="",
                    execution_results="",
                    tools_desc="",
                    previous_context="",
                    critic_feedback="",
                )
            )

            self.assertEqual(loaded, {"case-routing": "Cached skill content."})

    def test_completed_task_is_preserved_in_effective_plan_patch(self) -> None:
        """Проверяет, что completed-задача не отображается как pending после replanning."""

        current_plan = {
            "1": Task(
                task_id="1",
                description="Получить детали кейса",
                status=TaskStatus.COMPLETED,
                result_preview="case facts",
            ),
            "2": Task(
                task_id="2",
                description="Получить события клиента",
                status=TaskStatus.COMPLETED,
                result_preview="events",
            ),
        }
        next_plan = build_full_plan(
            current_plan=current_plan,
            full_plan=FullPlan(
                objective="Analyze case",
                tasks=[
                    PlannedTask(task_id="1", description="Получить детали кейса"),
                    PlannedTask(
                        task_id="2",
                        description="Получить события клиента с уточненным описанием",
                        dependencies=["1"],
                    ),
                    PlannedTask(
                        task_id="3",
                        description="Сопоставить факты",
                        dependencies=["1", "2"],
                    ),
                ],
            ),
            current_run_id="2",
        )

        plan_patch = _build_overwrite_patch(current_plan, next_plan)
        effective_plan = _apply_plan_patch(current_plan, plan_patch)

        self.assertEqual(effective_plan["2"].status, TaskStatus.COMPLETED)
        self.assertEqual(effective_plan["2"].result_preview, "events")
        self.assertEqual(effective_plan["3"].status, TaskStatus.PENDING)

    def test_build_full_plan_removes_dependencies_on_future_tasks(self) -> None:
        """Проверяет, что задача не может зависеть от задач, объявленных позже."""

        plan = build_full_plan(
            current_plan={},
            full_plan=FullPlan(
                objective="Analyze case",
                tasks=[
                    PlannedTask(
                        task_id="5",
                        description="Сделать вывод",
                        dependencies=["6", "7"],
                    ),
                    PlannedTask(task_id="6", description="Посчитать метрики"),
                    PlannedTask(task_id="7", description="Проверить события"),
                    PlannedTask(
                        task_id="8",
                        description="Финальный synthesis",
                        dependencies=["5", "6", "7"],
                    ),
                ],
            ),
            current_run_id="1",
        )

        self.assertEqual(plan["5"].dependencies, [])
        self.assertEqual(plan["8"].dependencies, ["5", "6", "7"])


if __name__ == "__main__":
    unittest.main()

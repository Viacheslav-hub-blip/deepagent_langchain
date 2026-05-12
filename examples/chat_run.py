"""Minimal no-UI LangChain Runnable run for the research agent.

Run from the repository root:

    python examples\chat_run.py

The example uses FakeListChatModel, so it does not call an external LLM.
It demonstrates the backend flow: invoke request -> graph -> final messages,
with lineage and artifacts written under examples/runs.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pandas as pdfrom langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planner_agent import ResearchAgent  # noqa: E402


class ExampleSandbox:
    """Small in-memory sandbox implementing the methods used by the graph."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.last_dataframe_variable = "df_current"
        self.globals: dict[str, Any] = {"df_current": dataframe}

    async def get_all_variable_previews(self) -> dict[str, str]:
        previews: dict[str, str] = {}
        for name, value in self.globals.items():
            if isinstance(value, pd.DataFrame):
                previews[name] = (
                    f"shape={value.shape}; "
                    f"columns={list(value.columns)}; "
                    f"head={value.head(2).to_dict(orient='records')}"
                )
            else:
                previews[name] = str(value)[:500]
        return previews

    async def add_variable(self, name: str, value: object) -> None:
        self.globals[name] = value
        if isinstance(value, pd.DataFrame):
            self.last_dataframe_variable = name

    async def get_variable(self, name: str) -> object:
        return self.globals.get(name)

    def get_installed_packages(self) -> dict[str, str]:
        """Совместимость с worker_node (sandbox protocol)."""

        return {}


class ToolBindableFakeMessagesListChatModel(FakeMessagesListChatModel):
    """Fake-модель с фиксированными AIMessage и поддержкой bind_tools для ReAct worker/responder."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolBindableFakeMessagesListChatModel":
        """Игнорирует список tools: ответы заданы явно в ``responses``."""

        return self


def _ai(content: str) -> AIMessage:
    return AIMessage(content=content.strip())


def build_fake_model() -> ToolBindableFakeMessagesListChatModel:
    """Собирает fake-модель с ответами для узлов агентного графа.

    Args:
        Отсутствуют.

    Returns:
        Fake-модель, которая последовательно возвращает ответы для планировщика,
        ревьюера плана, worker, валидатора, перепланировщика и финального ответа.
    """

    final_report_md = (
        "The sample run found one anti-fraud hit pattern in the bundled table.\n\n"
        "The bundled hit table exposes event_id, event_time, amount, rule, resolution, "
        "product, and routing fields. Treat this as a smoke-test result only; real "
        "analysis needs approved source exports and stronger evidence checks."
    )

    full_plan_json = """
            {
              "objective": "Find a compact anti-fraud insight in sample hit events.",
              "tasks": [
                {
                  "task_id": "1",
                  "description": "Inspect df_current and summarize anti-fraud hit patterns.",
                  "dependencies": [],
                  "expected_output": "Short insight summary",
                  "suggested_tools": ["show_current_dataframe"],
                  "suggested_skills": ["insight-design"],
                  "required_artifacts": ["worker result"]
                }
              ]
            }
            """

    return ToolBindableFakeMessagesListChatModel(
        responses=[
            _ai('{"skill_names": [], "rationale": ""}'),
            _ai(full_plan_json),
            _ai("""
            {
              "needs_revision": false,
              "summary": "The plan is sufficient for a smoke example.",
              "issues": []
            }
            """),
            _ai(
                "Sample insight: the bundled hit table contains card and non-card "
                "anti-fraud events with amounts, rules, resolutions, and routing "
                "fields. This is a smoke-test conclusion, not a production risk decision."
            ),
            _ai("""
            {
              "approved": true,
              "reasoning": "Smoke example: worker output is sufficient.",
              "issues": [],
              "improvement_instructions": ""
            }
            """),
            _ai("""
            {
              "is_valid": true,
              "confidence": 0.9,
              "reasoning": "The worker result contains anti-fraud hit context and a limitation."
            }
            """),
            _ai('{"skill_names": [], "rationale": ""}'),
            _ai(full_plan_json),
            _ai("""
            {
              "needs_revision": false,
              "summary": "The completed plan is sufficient for the final smoke report.",
              "issues": []
            }
            """),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "submit_final_report",
                        "args": {"report": final_report_md},
                        "id": "submit-final-smoke",
                        "type": "tool_call",
                    }
                ],
            ),
            _ai("Done."),
        ]
    )


async def main() -> None:
    example_root = Path(__file__).resolve().parent
    data_path = example_root / "data" / "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
    runs_dir = example_root / "runs"

    dataframe = pd.read_csv(data_path)
    sandbox = ExampleSandbox(dataframe)
    agent = ResearchAgent(
        model=build_fake_model(),
        sandbox=sandbox,
        tools=[],
        enable_workspace_tools=False,
        workspace_root=str(example_root),
        sources_dir=str(example_root / "data"),
        contexts_dir=str(example_root / "skills"),
        skills_dir=str(example_root / "skills"),
        memory_dir=str(example_root / "memory"),
        runs_dir=str(runs_dir),
    )

    messages = await agent.ainvoke(
        "Find one insight in the bundled anti-fraud hit sample.",
        session_id="example-session",
        user_id="example-user",
    )

    print("\nFinal report\n============")
    print(messages[-1].content if messages else "")
    print("\nRun artifacts and lineage were written to:")
    print(runs_dir)
    print(f"\nRun id: {agent.last_run_id}")


if __name__ == "__main__":
    asyncio.run(main())

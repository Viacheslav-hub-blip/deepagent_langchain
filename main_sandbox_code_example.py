"""Пример запуска агента с execute_python_code и локальной sandbox."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pandas as pd

from model import model as llm
from planner_agent import ResearchAgent
from sandbox import ClientPythonSandbox


PROJECT_ROOT = Path(__file__).resolve().parent


def build_agent() -> tuple[ResearchAgent, ClientPythonSandbox]:
    data_path = (
        PROJECT_ROOT
        / "examples"
        / "data"
        / "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
    )
    df = pd.read_csv(data_path)
    sandbox = ClientPythonSandbox(
        initial_globals={"df_current": df},
        allowed_libraries={"pd": pd},
    )
    sandbox.last_dataframe_variable = "df_current"

    agent = ResearchAgent(
        model=llm,
        sandbox=sandbox,
        tools=[],
        workspace_root=str(PROJECT_ROOT / "examples"),
        runs_dir=str(PROJECT_ROOT / "examples" / "runs"),
        memory_dir=str(PROJECT_ROOT / "examples" / "memory"),
        skills_dir=str(PROJECT_ROOT / "examples" / "skills"),
    )
    return agent, sandbox


async def main() -> None:
    agent, sandbox = build_agent()
    messages = await agent.ainvoke(
        (
            "Посчитай метрики по event_type через execute_python_code. "
            "Сохрани результат в переменную event_type_metrics и объясни вывод."
        ),
    )
    previews = await sandbox.get_all_variable_previews()
    print("\nFinal message:")
    print(messages[-1].content)
    print("\nSandbox variables:")
    for name, preview in previews.items():
        print(f"\n{name}\n{preview}")


if __name__ == "__main__":
    started_at = time.perf_counter()
    asyncio.run(main())
    print(f"\nElapsed seconds: {time.perf_counter() - started_at:.2f}")

from __future__ import annotations

import unittest

from planner_agent.factory import planner_agent


class FakeSandbox:
    last_dataframe_variable = None
    globals = {}

    async def get_all_variable_previews(self) -> dict[str, str]:
        return {}

    async def add_variable(self, name: str, value: object) -> None:
        self.globals[name] = value

    async def get_variable(self, name: str) -> object:
        return self.globals.get(name)


class GraphWiringTests(unittest.TestCase):
    def test_graph_compiles_without_host_sandbox_package_when_no_code_wrapper_needed(self) -> None:
        graph = planner_agent(
            model=object(),
            sandbox=FakeSandbox(),
            tools=[],
            enable_workspace_tools=False,
        )
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()

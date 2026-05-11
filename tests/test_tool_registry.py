from __future__ import annotations

import unittest

from langchain_core.tools import tool

from planner_agent.factory import planner_agent
from planner_agent.tools.registry import ToolRegistry


@tool("alpha_tool")
def alpha_tool(value: str) -> str:
    """Return alpha value."""
    return value


@tool("beta_tool")
def beta_tool(value: str) -> str:
    """Return beta value."""
    return value


class FakeSandbox:
    last_dataframe_variable = None
    globals = {}

    async def get_all_variable_previews(self) -> dict[str, str]:
        return {}

    async def add_variable(self, name: str, value: object) -> None:
        self.globals[name] = value

    async def get_variable(self, name: str) -> object:
        return self.globals.get(name)


class ToolRegistryTests(unittest.TestCase):
    def test_registry_registers_native_langchain_tools(self) -> None:
        registry = ToolRegistry([alpha_tool])
        registry.register(beta_tool, toolset="analysis", tags=["safe"])

        self.assertIs(registry.get("alpha_tool"), alpha_tool)
        self.assertIs(registry.get_tool("beta_tool"), beta_tool)
        self.assertEqual(registry.names(), ["alpha_tool", "beta_tool"])
        self.assertEqual(
            [tool.name for tool in registry.enabled(["beta_tool"])],
            ["beta_tool"],
        )
        self.assertIn("analysis", registry.toolset_names())
        self.assertEqual(
            [tool.name for tool in registry.toolset("analysis")],
            ["beta_tool"],
        )
        self.assertIn("- alpha_tool: Return alpha value.", registry.describe())

    def test_registry_rejects_duplicate_tool_names(self) -> None:
        registry = ToolRegistry([alpha_tool])

        with self.assertRaises(ValueError):
            registry.register(alpha_tool)

    def test_registry_enabled_unknown_tool_can_be_strict_or_lenient(self) -> None:
        registry = ToolRegistry([alpha_tool])

        with self.assertRaises(KeyError):
            registry.enabled(["missing"])

        self.assertEqual(registry.enabled(["missing"], strict=False), [])

    def test_factory_uses_registry_enabled_subset_without_changing_public_api(self) -> None:
        graph = planner_agent(
            model=object(),
            sandbox=FakeSandbox(),
            tools=[alpha_tool, beta_tool],
            enable_workspace_tools=False,
            enabled_tool_names={"alpha_tool"},
        )

        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()

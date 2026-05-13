"""Тесты нативного инструмента выполнения Python-кода.

Содержит:
- FakeSandbox: минимальная песочница для тестов инструмента.
- fake_generate_python_code: legacy-tool, который должен быть заменен.
- SandboxCodeExecutorTests: проверки выполнения, ошибок и factory-интеграции.
"""

from __future__ import annotations

import json
import unittest

import pandas as pd
from langchain_core.tools import tool

from planner_agent.agent_nodes.worker_node import _apply_tool_output
from planner_agent.factory import _prepare_worker_tools
from planner_agent.models import Task
from planner_agent.tools.python_analysis_tool import (
    PYTHON_ANALYSIS_TOOL_NAME,
    PythonAnalysisTool,
    build_python_analysis_tool,
)


class FakeSandbox:
    """Минимальная sandbox-среда с общим словарем переменных.

    Attributes:
        globals: Переменные, доступные Python-коду.
        last_dataframe_variable: Имя последнего созданного DataFrame.
        last_target_variable: Имя последней созданной целевой переменной.
    """

    def __init__(self, initial_globals: dict | None = None) -> None:
        """Создает тестовую песочницу.

        Args:
            initial_globals: Начальные переменные sandbox.

        Returns:
            ``None``.
        """

        self.globals = dict(initial_globals or {})
        self.last_dataframe_variable = None
        self.last_target_variable = None

    async def get_all_variable_previews(self) -> dict[str, str]:
        """Возвращает краткие описания переменных sandbox.

        Returns:
            Словарь ``{имя: описание}``.
        """

        return {name: type(value).__name__ for name, value in self.globals.items()}

    async def add_variable(self, name: str, value: object) -> None:
        """Добавляет переменную в sandbox.

        Args:
            name: Имя переменной.
            value: Значение переменной.

        Returns:
            ``None``.
        """

        self.globals[name] = value

    async def get_variable(self, name: str) -> object:
        """Возвращает переменную из sandbox.

        Args:
            name: Имя переменной.

        Returns:
            Значение переменной или ``None``.
        """

        return self.globals.get(name)

    @staticmethod
    def get_installed_packages() -> dict[str, str]:
        """Возвращает список установленных пакетов для worker prompt.

        Returns:
            Словарь ``{имя_пакета: версия}``.
        """

        return {"pandas": pd.__version__}


@tool("generate_python_code")
def fake_generate_python_code(instruction: str) -> str:
    """Имитирует legacy MCP-tool генерации кода.

    Args:
        instruction: Инструкция для генерации кода.

    Returns:
        Строка Python-кода.
    """

    return "result = 1"


class SandboxCodeExecutorTests(unittest.IsolatedAsyncioTestCase):
    """Проверяет нативный tool выполнения Python-кода в sandbox агента."""

    async def test_python_analysis_executes_code_in_sandbox(self) -> None:
        """Проверяет успешное выполнение кода и создание DataFrame-переменной."""

        df = pd.DataFrame(
            {
                "segment": ["a", "a", "b", None],
                "amount": [10, 20, 30, 40],
            }
        )
        sandbox = FakeSandbox({"df_current": df})
        analysis_tool = build_python_analysis_tool(sandbox)

        raw_result = await analysis_tool.ainvoke(
            {
                "code": (
                    "segment_counts = "
                    "df_current.groupby('segment', dropna=False).size().reset_index(name='count')"
                ),
                "target_variable": "segment_counts",
                "description": "Посчитать количество строк по сегментам.",
            }
        )

        result = json.loads(raw_result)

        self.assertTrue(result["success"])
        self.assertEqual(result["target_variable"], "segment_counts")
        self.assertIn("generated_code", result)
        self.assertIn("DataFrame", result["variable_preview"])
        self.assertIn("segment_counts", sandbox.globals)
        self.assertEqual(sandbox.last_target_variable, "segment_counts")
        self.assertEqual(sandbox.last_dataframe_variable, "segment_counts")

    async def test_python_analysis_returns_compile_error_as_json(self) -> None:
        """Проверяет, что ошибка компиляции возвращается как JSON для retry."""

        sandbox = FakeSandbox()
        analysis_tool = build_python_analysis_tool(sandbox)

        raw_result = await analysis_tool.ainvoke(
            {
                "code": "broken_result =",
                "target_variable": "broken_result",
            }
        )

        result = json.loads(raw_result)

        self.assertFalse(result["success"])
        self.assertIn("SyntaxError", result["error"])
        self.assertIn("traceback", result)
        self.assertEqual(result["target_variable"], "broken_result")
        self.assertIn("possible_causes", result)
        self.assertIn("solution_options", result)
        self.assertIn("retry_guidance", result)

    async def test_python_analysis_normalizes_json_escaped_newlines(self) -> None:
        """Проверяет, что tool исправляет JSON-escaped переносы строк в коде."""

        sandbox = FakeSandbox()
        analysis_tool = build_python_analysis_tool(sandbox)

        raw_result = await analysis_tool.ainvoke(
            {
                "code": "import pandas as pd\\n\\nchannel_value = 'MOBILE'",
                "target_variable": "channel_value",
            }
        )

        result = json.loads(raw_result)

        self.assertTrue(result["success"])
        self.assertEqual(result["target_variable"], "channel_value")
        self.assertEqual(sandbox.globals["channel_value"], "MOBILE")
        self.assertIn("\n\nchannel_value", result["generated_code"])
        self.assertNotIn("\\n\\nchannel_value", result["generated_code"])

    async def test_python_analysis_returns_runtime_error_as_json(self) -> None:
        """Проверяет, что runtime-ошибка видна модели вместе с traceback."""

        sandbox = FakeSandbox()
        analysis_tool = build_python_analysis_tool(sandbox)

        raw_result = await analysis_tool.ainvoke(
            {
                "code": "broken_result = missing_dataframe.copy()",
                "target_variable": "broken_result",
            }
        )

        result = json.loads(raw_result)

        self.assertFalse(result["success"])
        self.assertIn("NameError", result["error"])
        self.assertIn("missing_dataframe", result["generated_code"])
        self.assertIn("missing_dataframe", result["traceback"])
        self.assertTrue(result["possible_causes"])
        self.assertTrue(result["solution_options"])

    async def test_worker_keeps_failed_code_and_error_for_retry(self) -> None:
        """Проверяет сохранение кода и ошибки worker-а после ошибки инструмента."""

        task = Task(
            task_id="t1",
            description="Проверить повтор после ошибки python_analysis.",
        )
        raw_output = json.dumps(
            {
                "success": False,
                "generated_code": "result = missing_dataframe.copy()",
                "target_variable": "result",
                "error": "NameError: name 'missing_dataframe' is not defined",
                "traceback": "Traceback...\nNameError: missing_dataframe",
                "message": "Python code execution failed. Fix generated_code and retry.",
            },
            ensure_ascii=False,
        )

        success = await _apply_tool_output(task, raw_output)

        self.assertFalse(success)
        self.assertEqual(task.generated_code, "result = missing_dataframe.copy()")
        self.assertEqual(task.output_variable_name, "result")
        self.assertIn("missing_dataframe", task.error_log)

    async def test_worker_treats_ok_false_tool_json_as_failure(self) -> None:
        """Проверяет, что tool-ответ ``ok=false`` не считается успешным."""

        task = Task(
            task_id="t2",
            description="Загрузить транзакции из таблицы.",
        )
        raw_output = json.dumps(
            {
                "ok": False,
                "error": {
                    "code": "unknown_columns",
                    "missing_columns": ["account_id"],
                },
                "schema": {"columns": [{"name": "event_id"}]},
            },
            ensure_ascii=False,
        )

        success = await _apply_tool_output(task, raw_output)

        self.assertFalse(success)
        self.assertIn("ok=false", task.error_log)
        self.assertIn("unknown_columns", task.error_log)
        self.assertIn("account_id", task.error_log)

    async def test_factory_replaces_legacy_code_generator_with_python_analysis(self) -> None:
        """Проверяет, что legacy генератор кода скрывается из worker tools."""

        prepared_tools = _prepare_worker_tools(
            tools=[fake_generate_python_code],
            sandbox=FakeSandbox(),
            code_generator_tool_names={"generate_python_code"},
        )

        self.assertEqual(len(prepared_tools), 1)
        self.assertIsInstance(prepared_tools[0], PythonAnalysisTool)
        self.assertEqual(prepared_tools[0].name, PYTHON_ANALYSIS_TOOL_NAME)


if __name__ == "__main__":
    unittest.main()

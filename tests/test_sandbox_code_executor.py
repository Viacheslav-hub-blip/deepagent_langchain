"""Тесты интеграции генератора кода с локальной Python-песочницей.

Содержит:
- FakeCodeGeneratorInput: схема аргументов тестового генератора кода.
- fake_generate_python_code: тестовый LangChain tool, возвращающий Python-код.
- SandboxCodeExecutorTests: проверки wrapper-а и factory-подготовки tools.
"""

from __future__ import annotations

import json
import unittest

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from planner_agent.agent_nodes.worker_node import _apply_tool_output
from planner_agent.factory import _prepare_worker_tools
from planner_agent.models import Task
from sandbox import BaseCodeExecutorTool, ClientPythonSandbox


class FakeCodeGeneratorInput(BaseModel):
    """Аргументы тестового инструмента, который имитирует MCP-генератор кода."""

    instruction: str = Field(description="Текстовая инструкция для генерации кода.")
    target_variable: str = Field(
        default="result",
        description="Имя переменной, которую должен создать сгенерированный код.",
    )


def fake_generate_python_code(
        instruction: str,
        target_variable: str = "result",
) -> str:
    """Возвращает Python-код для проверки выполнения в песочнице.

    Args:
        instruction: Текстовая инструкция. В тесте не используется напрямую.
        target_variable: Имя переменной результата, которую должен создать код.

    Returns:
        Строка Python-кода, создающего агрегированный DataFrame.
    """

    return (
        f"{target_variable} = "
        "df_current.groupby('segment', dropna=False).size().reset_index(name='count')"
    )


def fake_generate_broken_python_code(
        instruction: str,
        target_variable: str = "result",
) -> str:
    """Возвращает Python-код с ошибкой для проверки логирования sandbox.

    Args:
        instruction: Текстовая инструкция. В тесте не используется напрямую.
        target_variable: Имя переменной, которую должен был создать код.

    Returns:
        Строка Python-кода, обращающаяся к отсутствующей переменной.
    """

    return f"{target_variable} = missing_dataframe.copy()"


class SandboxCodeExecutorTests(unittest.IsolatedAsyncioTestCase):
    """Проверяет исполнение кода через BaseCodeExecutorTool и ClientPythonSandbox."""

    async def test_code_generator_tool_executes_code_in_sandbox(self) -> None:
        """Проверяет, что генератор кода создает переменную в песочнице."""

        df = pd.DataFrame(
            {
                "segment": ["a", "a", "b", None],
                "amount": [10, 20, 30, 40],
            }
        )
        sandbox = ClientPythonSandbox(
            initial_globals={"df_current": df},
            allowed_libraries={"pd": pd},
        )
        generator_tool = StructuredTool.from_function(
            func=fake_generate_python_code,
            name="generate_python_code",
            description="Генерирует Python-код для анализа DataFrame.",
            args_schema=FakeCodeGeneratorInput,
        )
        executor_tool = BaseCodeExecutorTool(
            name="generate_python_code",
            description="Генерирует и исполняет Python-код.",
            mcp_tool=generator_tool,
            sandbox=sandbox,
        )

        raw_result = await executor_tool.ainvoke(
            {
                "instruction": "Посчитай количество строк по segment.",
                "target_variable": "segment_counts",
            }
        )

        result = json.loads(raw_result)
        created_value = await sandbox.get_variable("segment_counts")
        previews = await sandbox.get_all_variable_previews()

        self.assertTrue(result["success"])
        self.assertEqual(result["target_variable"], "segment_counts")
        self.assertIsNotNone(created_value)
        self.assertEqual(sandbox.last_target_variable, "segment_counts")
        self.assertEqual(sandbox.last_dataframe_variable, "segment_counts")
        self.assertIn("segment_counts", previews)
        self.assertIn("df_current", previews)
        self.assertIn("pd", sandbox.globals)

    async def test_code_generator_tool_returns_code_and_error_on_failure(self) -> None:
        """Проверяет, что при ошибке sandbox JSON содержит код и traceback."""

        sandbox = ClientPythonSandbox(initial_globals={})
        generator_tool = StructuredTool.from_function(
            func=fake_generate_broken_python_code,
            name="generate_python_code",
            description="Генерирует Python-код с ошибкой.",
            args_schema=FakeCodeGeneratorInput,
        )
        executor_tool = BaseCodeExecutorTool(
            name="generate_python_code",
            description="Генерирует и исполняет Python-код.",
            mcp_tool=generator_tool,
            sandbox=sandbox,
        )

        raw_result = await executor_tool.ainvoke(
            {
                "instruction": "Сгенерируй ошибочный код.",
                "target_variable": "broken_result",
            }
        )

        result = json.loads(raw_result)

        self.assertFalse(result["success"])
        self.assertIn("missing_dataframe", result["generated_code"])
        self.assertEqual(result["target_variable"], "broken_result")
        self.assertIn("NameError", result["error"])
        self.assertIn("execution_output", result)
        self.assertIn("missing_dataframe", executor_tool._previous_code)
        self.assertIn("NameError", executor_tool._error_context)

    async def test_worker_keeps_failed_code_and_error_for_retry(self) -> None:
        """Проверяет сохранение кода и ошибки worker-а после ошибки sandbox."""

        task = Task(
            task_id="t1",
            description="Проверить повтор после ошибки sandbox.",
        )
        raw_output = json.dumps(
            {
                "success": False,
                "generated_code": "result = missing_dataframe.copy()",
                "target_variable": "result",
                "error": "NameError: name 'missing_dataframe' is not defined",
                "message": "Выполнение кода завершилось с ошибкой",
            },
            ensure_ascii=False,
        )

        success = await _apply_tool_output(task, raw_output)

        self.assertFalse(success)
        self.assertEqual(task.generated_code, "result = missing_dataframe.copy()")
        self.assertEqual(task.output_variable_name, "result")
        self.assertIn("missing_dataframe", task.error_log)

    async def test_factory_wraps_named_code_generator_tool(self) -> None:
        """Проверяет, что factory заменяет указанный генератор кода на executor tool."""

        sandbox = ClientPythonSandbox(initial_globals={"df_current": pd.DataFrame()})
        generator_tool = StructuredTool.from_function(
            func=fake_generate_python_code,
            name="generate_python_code",
            description="Генерирует Python-код для анализа DataFrame.",
            args_schema=FakeCodeGeneratorInput,
        )

        prepared_tools = _prepare_worker_tools(
            tools=[generator_tool],
            sandbox=sandbox,
            code_generator_tool_names={"generate_python_code"},
        )

        self.assertEqual(len(prepared_tools), 1)
        self.assertIsInstance(prepared_tools[0], BaseCodeExecutorTool)
        self.assertEqual(prepared_tools[0].name, "generate_python_code")

    async def test_code_generator_task_receives_grounded_data_contract(self) -> None:
        """Проверяет добавление контракта работы с реальными данными."""

        sandbox = ClientPythonSandbox(initial_globals={"df_current": pd.DataFrame()})
        generator_tool = StructuredTool.from_function(
            func=fake_generate_python_code,
            name="generate_python_code",
            description="Генерирует Python-код для анализа DataFrame.",
            args_schema=FakeCodeGeneratorInput,
        )
        executor_tool = BaseCodeExecutorTool(
            name="generate_python_code",
            description="Генерирует и исполняет Python-код.",
            mcp_tool=generator_tool,
            sandbox=sandbox,
        )

        args = executor_tool._prepare_mcp_args(
            instruction="Посчитай количество строк по segment.",
            target_variable="segment_counts",
        )

        self.assertIn("Контракт работы с данными", args["instruction"])
        self.assertIn("Не создавай демонстрационные", args["instruction"])


if __name__ == "__main__":
    unittest.main()

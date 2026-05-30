"""Инструмент генерации и выполнения Python-кода для DeepAgent supervisor."""

from __future__ import annotations

import ast
import json
import traceback
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from deep_agent_test.python_sandbox import DeepAgentPythonSandbox, SANDBOX_HELPER_NAMES
from planner_agent.tools.execute_python_code_tool import (
    EXECUTE_PYTHON_CODE_TOOL_NAME,
    PythonExecutionResult,
    _execute_python_code,
    _normalize_code_text,
    _normalize_target_variable,
    _python_error_possible_causes,
    _python_error_solution_options,
    _python_retry_guidance,
)

MAX_CODE_CHARS = 50_000

ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "collections",
        "datetime",
        "itertools",
        "json",
        "math",
        "matplotlib",
        "numpy",
        "pandas",
        "pathlib",
        "pickle",
        "plotly",
        "re",
        "scipy",
        "seaborn",
        "sklearn",
        "statistics",
    }
)
DENIED_CALL_NAMES: frozenset[str] = frozenset(
    {
        "__import__",
        "compile",
        "eval",
        "exec",
        "input",
    }
)
DENIED_ATTRIBUTE_CALLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("os", "popen"),
        ("os", "remove"),
        ("os", "removedirs"),
        ("os", "rmdir"),
        ("os", "system"),
        ("shutil", "rmtree"),
        ("subprocess", "call"),
        ("subprocess", "check_call"),
        ("subprocess", "check_output"),
        ("subprocess", "run"),
    }
)

EXECUTE_PYTHON_CODE_DESCRIPTION = """
Выполняет Python-код для расчетов, фильтрации, join, агрегаций, нормализации,
разбора pickle/CSV/JSON и подготовки табличных результатов.

Когда использовать:
- нужно прочитать большой `.pkl`, сохраненный middleware после `read_table`;
- нужно обработать `list[dict]`, DataFrame или файл с данными;
- нужно посчитать метрики, отфильтровать строки, отсортировать события;
- нужно подготовить итоговую таблицу или payload для ответа пользователю.

Когда не использовать:
- для чтения таблиц напрямую из источника — используй `task(data-retrieval-agent)`;
- для финального пользовательского ответа вместо supervisor-а.

Доступные helpers в sandbox (уже загружены, импорт не нужен):
- `PROJECT_ROOT` — корень проекта;
- `TOOL_OUTPUTS_DIR` — папка `.pkl` после spill middleware;
- `read_pickle_file(path)` — читает pickle по абсолютному пути из tool output;
- `describe_pickle_file(path)` — тип, rows_count, columns, preview без полной обработки;
- `rows_to_dataframe(rows)` — преобразует list[dict] в pandas DataFrame;
- `pd`, `np` — pandas/numpy, если установлены.

Пример чтения spill-файла:
rows = read_pickle_file(r"C:\\path\\to\\file.pkl")
df = rows_to_dataframe(rows)
print(df.shape)
print(df.head(3).to_string())
result = df

Правила:
- переменные сохраняются между вызовами инструмента в одной сессии;
- для именованного результата укажи `target_variable`, иначе используй `print()` и читай `execution_output`;
- при ошибке tool возвращает JSON с `error`, полным `traceback`, `possible_causes`, `solution_options`, `retry_guidance` — исправь код и повтори вызов;
- не удаляй файлы и директории;
- читай pickle через `read_pickle_file` или `pd.read_pickle` по путям из tool outputs.
""".strip()


class ExecutePythonCodeInput(BaseModel):
    """Аргументы tool ``execute_python_code``: код, имя переменной результата, описание."""

    code: str = Field(
        description=(
            "Python-код для выполнения. Используй helpers `read_pickle_file`, "
            "`rows_to_dataframe`, `pd`, `np` и переменные из предыдущих вызовов."
        ),
    )
    target_variable: str | None = Field(
        default=None,
        description=(
            "Имя переменной, в которую нужно сохранить главный результат. "
            "Если не нужно — опусти и используй print()."
        ),
    )
    description: str = Field(
        default="",
        description="Краткая цель кода на русском языке для трассировки.",
    )


class ExecutePythonCodeTool(BaseTool):
    """LangChain tool выполнения Python-кода в persistent sandbox DeepAgent.

    Перед выполнением код проходит статическую проверку политики (`_validate_code_policy`):
    разрешён только белый список импортов и запрещены опасные вызовы (eval/exec/os.system,
    удаление файлов и т.п.). Само выполнение и формирование информативного результата
    делегируются переиспользуемому ``_execute_python_code`` из ``planner_agent``.

    Любой результат возвращается строкой JSON. При ошибке JSON содержит ``error``,
    полный ``traceback``, ``possible_causes``, ``solution_options``, ``retry_guidance``,
    ``available_variables`` и список ``sandbox_helpers`` — этого достаточно, чтобы модель
    исправила код и повторила вызов.
    """

    name: str = EXECUTE_PYTHON_CODE_TOOL_NAME
    description: str = EXECUTE_PYTHON_CODE_DESCRIPTION
    args_schema: type[BaseModel] = ExecutePythonCodeInput

    _sandbox: DeepAgentPythonSandbox = PrivateAttr()

    def __init__(self, *, sandbox: DeepAgentPythonSandbox) -> None:
        """Создаёт tool поверх готового persistent sandbox.

        Args:
            sandbox: Песочница с общими переменными между вызовами в одной сессии.
        """

        super().__init__()
        self._sandbox = sandbox

    def _run(
        self,
        code: str,
        target_variable: str | None = None,
        description: str = "",
        **_: Any,
    ) -> str:
        """Синхронно проверяет политику, выполняет код и сериализует результат.

        Args:
            code: Python-код для выполнения.
            target_variable: Имя переменной результата или ``None`` для print-вывода.
            description: Краткая цель кода для трассировки.
            **_: Служебные аргументы LangChain, не используются.

        Returns:
            JSON-строка с результатом или с подробным описанием ошибки.
        """

        generated_code = _normalize_code_text(str(code or ""))
        try:
            _validate_code_policy(generated_code)
            _normalize_target_variable(target_variable)
        except Exception as exc:
            return _policy_error_payload(
                generated_code,
                target_variable,
                exc,
                sandbox=self._sandbox,
            )

        result = _execute_python_code(
            sandbox=self._sandbox,
            code=generated_code,
            target_variable=target_variable,
            description=description,
        )
        return _result_to_json(result, sandbox=self._sandbox)

    async def _arun(
        self,
        code: str,
        target_variable: str | None = None,
        description: str = "",
        **_: Any,
    ) -> str:
        """Асинхронная обёртка над :meth:`_run` (выполнение синхронное).

        Args:
            code: Python-код для выполнения.
            target_variable: Имя переменной результата или ``None``.
            description: Краткая цель кода.
            **_: Служебные аргументы LangChain, не используются.

        Returns:
            JSON-строка с результатом или ошибкой.
        """

        return self._run(
            code=code,
            target_variable=target_variable,
            description=description,
        )


def build_execute_python_code_tool(sandbox: DeepAgentPythonSandbox) -> ExecutePythonCodeTool:
    """Фабрика tool ``execute_python_code`` для supervisor.

    Args:
        sandbox: Persistent sandbox с helpers чтения pickle и аналитическими библиотеками.

    Returns:
        Готовый ``ExecutePythonCodeTool`` для регистрации в списке tools.
    """

    return ExecutePythonCodeTool(sandbox=sandbox)


def _validate_code_policy(code: str) -> None:
    """Статически проверяет код перед выполнением.

    Разбирает AST и запрещает: пустой/слишком длинный код, импорт вне
    ``ALLOWED_IMPORT_ROOTS``, вызовы из ``DENIED_CALL_NAMES`` и
    ``DENIED_ATTRIBUTE_CALLS``, а также удаление файлов через ``Path.unlink/rmdir``.

    Args:
        code: Python-код, который нужно проверить.

    Raises:
        ValueError: Код пустой, слишком длинный или содержит запрещённый импорт/вызов.
        SyntaxError: Код не разбирается ``ast.parse``.
    """
    if not str(code or "").strip():
        raise ValueError("code is required")
    if len(code) > MAX_CODE_CHARS:
        raise ValueError(f"code is too long: {len(code)} chars, limit is {MAX_CODE_CHARS}")

    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = node.names if isinstance(node, ast.Import) else []
            module_names = [alias.name for alias in names]
            if isinstance(node, ast.ImportFrom) and node.module:
                module_names.append(node.module)
            for module_name in module_names:
                root = module_name.split(".", maxsplit=1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"Import '{module_name}' is not allowed in execute_python_code")
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in DENIED_CALL_NAMES:
                raise ValueError(f"Call '{call_name}' is not allowed in execute_python_code")
            owner, attr = _attribute_call_name(node.func)
            if owner == "Path" and attr in {"unlink", "rmdir"}:
                raise ValueError(f"Call 'Path.{attr}' is not allowed in execute_python_code")
            if (owner, attr) in DENIED_ATTRIBUTE_CALLS:
                raise ValueError(f"Call '{owner}.{attr}' is not allowed in execute_python_code")


def _call_name(func: ast.AST) -> str:
    """Возвращает имя прямого вызова функции или пустую строку."""

    if isinstance(func, ast.Name):
        return func.id
    return ""


def _attribute_call_name(func: ast.AST) -> tuple[str, str]:
    """Возвращает пару ``(owner, attr)`` для вызова метода атрибута."""

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id, func.attr
    return "", ""


def _policy_error_payload(
    generated_code: str,
    target_variable: str | None,
    exc: Exception,
    *,
    sandbox: DeepAgentPythonSandbox,
) -> str:
    """Формирует информативный JSON-ответ при провале статической проверки кода.

    Args:
        generated_code: Нормализованный код, который не прошёл проверку.
        target_variable: Запрошенное имя переменной результата или ``None``.
        exc: Исключение валидации политики.
        sandbox: Sandbox для перечисления доступных переменных и путей.

    Returns:
        JSON-строка с ``error``, ``traceback``, причинами и вариантами исправления.
    """

    from planner_agent.tools.execute_python_code_tool import _visible_variable_names

    payload = {
        "success": False,
        "message": "Python code did not pass validation. Fix code and retry execute_python_code.",
        "generated_code": generated_code,
        "target_variable": str(target_variable or ""),
        "variable_preview": "",
        "execution_output": "",
        "error": f"{exc.__class__.__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "available_variables": _visible_variable_names(sandbox.globals),
        "possible_causes": _python_error_possible_causes(exc),
        "solution_options": _python_error_solution_options(exc),
        "retry_guidance": _python_retry_guidance(),
        "sandbox_helpers": sorted(SANDBOX_HELPER_NAMES),
        "working_directory": str(sandbox.working_directory),
        "tool_outputs_dir": str(sandbox.tool_outputs_dir),
        "readable_roots": [str(path) for path in sandbox.readable_roots],
    }
    return json.dumps(payload, ensure_ascii=False)


def _result_to_json(result: PythonExecutionResult, *, sandbox: DeepAgentPythonSandbox) -> str:
    """Дополняет результат выполнения контекстом sandbox и сериализует в JSON.

    Args:
        result: Результат ``_execute_python_code`` (успех или ошибка выполнения).
        sandbox: Sandbox для добавления helpers и разрешённых путей в ответ.

    Returns:
        JSON-строка; при неуспехе ``message`` подсказывает прочитать traceback и повторить.
    """

    payload = json.loads(result.to_json())
    payload["sandbox_helpers"] = sorted(SANDBOX_HELPER_NAMES)
    payload["working_directory"] = str(sandbox.working_directory)
    payload["tool_outputs_dir"] = str(sandbox.tool_outputs_dir)
    payload["readable_roots"] = [str(path) for path in sandbox.readable_roots]
    if not payload.get("success"):
        payload["message"] = (
            "Python code execution failed. Read error, traceback, possible_causes "
            "and solution_options, fix the code and retry execute_python_code."
        )
    return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "EXECUTE_PYTHON_CODE_DESCRIPTION",
    "EXECUTE_PYTHON_CODE_TOOL_NAME",
    "ExecutePythonCodeTool",
    "build_execute_python_code_tool",
]

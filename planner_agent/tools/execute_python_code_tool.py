"""Нативный инструмент выполнения Python-кода для аналитического агента.

Содержит:
- ExecutePythonCodeInput: схема аргументов инструмента выполнения кода.
- PythonExecutionResult: внутренний контейнер результата выполнения кода.
- ExecutePythonCodeTool: LangChain-инструмент проверки, компиляции и выполнения кода.
- build_execute_python_code_tool: фабрика инструмента для подключения к worker.
- _validate_target_variable: проверка имени целевой переменной.
- _validate_code_policy: статическая проверка кода перед выполнением.
- _ensure_common_libraries: добавление популярных аналитических библиотек в sandbox.
- _execute_python_code: компиляция и выполнение кода в sandbox.
- _sandbox_working_directory: получение рабочей директории sandbox.
- _temporary_working_directory: временная смена cwd на время выполнения кода.
- _python_error_possible_causes: вероятные причины ошибки execute_python_code.
- _python_error_solution_options: варианты исправления ошибки execute_python_code.
- _python_retry_guidance: краткая инструкция по retry после ошибки.
- _preview_value: компактный предпросмотр значения для ответа модели.
- _json_default: сериализация нестандартных объектов в JSON.
- _limit_text: ограничение длинных текстовых полей.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import keyword
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from ..runtime.sandbox import PythonSandboxProtocol

EXECUTE_PYTHON_CODE_TOOL_NAME = "execute_python_code"
MAX_CODE_CHARS = 50_000
MAX_TEXT_PREVIEW_CHARS = 4_000
MAX_STDIO_CHARS = 8_000
MAX_DATAFRAME_PREVIEW_ROWS = 10
_CWD_EXECUTION_LOCK_ATTR = "_analitic_agent_sandbox_cwd_lock"

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
        "open",
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


class ExecutePythonCodeInput(BaseModel):
    """Аргументы для проверки, компиляции и выполнения Python-кода.

    Attributes:
        code: Python-код, который нужно выполнить в текущем sandbox агента.
        target_variable: Опциональное имя переменной, которую код должен создать.
        description: Краткое описание цели кода для трассировки и самопроверки.
    """

    code: str = Field(
        description=(
            "Python code to compile and execute. Use existing sandbox variables "
            "from the task context. Optionally assign the main result to "
            "target_variable; otherwise use print() and read execution_output."
        ),
    )
    target_variable: str | None = Field(
        default=None,
        description=(
            "Optional name of the variable that the code must create or update. "
            "Omit when the goal is console output or side effects on other variables."
        ),
    )
    description: str = Field(
        default="",
        description="Short natural-language purpose of the code.",
    )


@dataclass
class PythonExecutionResult:
    """Внутреннее представление результата выполнения Python-кода.

    Attributes:
        success: Признак успешной компиляции, выполнения и создания результата.
        message: Краткое человекочитаемое сообщение для модели.
        generated_code: Исходный код, который был проверен и выполнен.
        target_variable: Имя ожидаемой переменной результата.
        variable_preview: Компактное описание результата или пустая строка.
        execution_output: Текст stdout/stderr, полученный при выполнении.
        error: Краткое описание ошибки или пустая строка.
        traceback_text: Полный traceback ошибки или пустая строка.
        available_variables: Имена переменных, доступных в sandbox после вызова.
    """

    success: bool
    message: str
    generated_code: str
    target_variable: str
    variable_preview: str = ""
    execution_output: str = ""
    error: str = ""
    traceback_text: str = ""
    available_variables: list[str] | None = None
    possible_causes: list[str] | None = None
    solution_options: list[str] | None = None
    retry_guidance: str = ""

    def to_json(self) -> str:
        """Сериализует результат выполнения в JSON-строку для модели.

        Returns:
            JSON-строка с полями результата, ошибки и предпросмотра.
        """

        payload = {
            "success": self.success,
            "message": self.message,
            "generated_code": self.generated_code,
            "target_variable": self.target_variable,
            "variable_preview": self.variable_preview,
            "execution_output": self.execution_output,
            "error": self.error,
            "traceback": self.traceback_text,
            "available_variables": self.available_variables or [],
            "possible_causes": self.possible_causes or [],
            "solution_options": self.solution_options or [],
            "retry_guidance": self.retry_guidance,
        }
        return json.dumps(payload, ensure_ascii=False, default=_json_default)


class ExecutePythonCodeTool(BaseTool):
    """LangChain-инструмент для нативного выполнения Python-кода в sandbox агента."""

    name: str = EXECUTE_PYTHON_CODE_TOOL_NAME
    description: str = (
        "Compile and execute Python code in the agent sandbox. Use this tool for "
        "calculations, joins, aggregations, statistics, tabular transformations "
        "and chart/data preparation over variables that already exist in the "
        "task context. target_variable is optional: omit it for print()-based "
        "exploration and read execution_output; set it when you need a named "
        "result variable. Returns structured JSON with generated_code, preview, "
        "stdout/stderr and full traceback when execution fails."
    )
    args_schema: type[BaseModel] = ExecutePythonCodeInput

    _sandbox: PythonSandboxProtocol = PrivateAttr()
    _previous_code: str = PrivateAttr(default="")
    _error_context: str = PrivateAttr(default="")

    def __init__(
        self,
        *,
        sandbox: PythonSandboxProtocol,
        previous_code: str = "",
        error_context: str = "",
    ) -> None:
        """Создает инструмент выполнения Python-кода.

        Args:
            sandbox: Изолированная среда агента с общими переменными.
            previous_code: Код предыдущей неуспешной попытки для retry-контекста.
            error_context: Ошибка предыдущей попытки для retry-контекста.

        Returns:
            ``None``. Экземпляр готов к подключению в список LangChain tools.
        """

        super().__init__()
        self._sandbox = sandbox
        self._previous_code = previous_code or ""
        self._error_context = error_context or ""

    def with_context(
        self,
        *,
        previous_code: str | None = None,
        error_context: str | None = None,
    ) -> "ExecutePythonCodeTool":
        """Возвращает копию инструмента с retry-контекстом worker-задачи.

        Args:
            previous_code: Код предыдущей попытки, если он есть.
            error_context: Ошибка предыдущей попытки, если она есть.

        Returns:
            Новый экземпляр ``ExecutePythonCodeTool`` с тем же sandbox.
        """

        return ExecutePythonCodeTool(
            sandbox=self._sandbox,
            previous_code=previous_code or self._previous_code,
            error_context=error_context or self._error_context,
        )

    def _run(
        self,
        code: str,
        target_variable: str | None = None,
        description: str = "",
        **_: Any,
    ) -> str:
        """Синхронно проверяет, компилирует и выполняет Python-код.

        Args:
            code: Python-код для выполнения.
            target_variable: Имя переменной результата.
            description: Описание цели кода.
            **_: Служебные аргументы LangChain, которые не используются.

        Returns:
            JSON-строка с результатом выполнения или ошибкой.
        """

        result = _execute_python_code(
            sandbox=self._sandbox,
            code=code,
            target_variable=target_variable,
            description=description,
        )
        return result.to_json()

    async def _arun(
        self,
        code: str,
        target_variable: str | None = None,
        description: str = "",
        **_: Any,
    ) -> str:
        """Асинхронно проверяет, компилирует и выполняет Python-код.

        Args:
            code: Python-код для выполнения.
            target_variable: Имя переменной результата.
            description: Описание цели кода.
            **_: Служебные аргументы LangChain, которые не используются.

        Returns:
            JSON-строка с результатом выполнения или ошибкой.
        """

        result = _execute_python_code(
            sandbox=self._sandbox,
            code=code,
            target_variable=target_variable,
            description=description,
        )
        if result.success and result.target_variable:
            value = self._sandbox.globals.get(result.target_variable)
            add_variable = getattr(self._sandbox, "add_variable", None)
            if callable(add_variable):
                await add_variable(result.target_variable, value)
        return result.to_json()


def build_execute_python_code_tool(sandbox: PythonSandboxProtocol) -> ExecutePythonCodeTool:
    """Создает нативный инструмент выполнения Python-кода.

    Args:
        sandbox: Изолированная среда агента с доступными переменными.

    Returns:
        Экземпляр ``ExecutePythonCodeTool`` для регистрации в worker tools.
    """

    return ExecutePythonCodeTool(sandbox=sandbox)


def _normalize_target_variable(target_variable: str | None) -> str | None:
    """Проверяет и нормализует опциональное имя целевой переменной результата.

    Args:
        target_variable: Имя переменной, переданное моделью, или ``None``.

    Returns:
        Очищенное имя переменной или ``None``, если переменная не запрошена.

    Raises:
        ValueError: Если имя не пустое, но не является Python-идентификатором или
            совпадает с ключевым словом Python.
    """

    name = str(target_variable or "").strip()
    if not name:
        return None
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(
            "target_variable must be a valid Python identifier, for example result_df"
        )
    return name


def _normalize_code_text(code: str) -> str:
    """Нормализует текст Python-кода перед валидацией.

    Args:
        code: Исходный текст кода, переданный моделью в ``execute_python_code``.

    Returns:
        Код без изменений, если он уже синтаксически корректен, либо код с
        преобразованными JSON-escaped переносами строк ``\\n``/``\\r\\n``.
    """

    raw_code = str(code or "")
    try:
        ast.parse(raw_code, mode="exec")
        return raw_code
    except SyntaxError as exc:
        message = str(exc)

    if "\\n" not in raw_code:
        return raw_code
    if "unexpected character after line continuation character" not in message:
        return raw_code

    return (
        raw_code
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )


def _validate_code_policy(code: str) -> None:
    """Проверяет код на базовые запрещенные операции перед выполнением.

    Args:
        code: Python-код, который нужно проверить.

    Returns:
        ``None``. При успешной проверке выполнение продолжается.

    Raises:
        ValueError: Если код пустой, слишком длинный или содержит запрещенный
            импорт/вызов.
        SyntaxError: Если код не проходит синтаксический разбор.
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
                    raise ValueError(
                        f"Import '{module_name}' is not allowed in execute_python_code"
                    )
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in DENIED_CALL_NAMES:
                raise ValueError(f"Call '{call_name}' is not allowed in execute_python_code")
            attr_call = _attribute_call_name(node.func)
            if attr_call in DENIED_ATTRIBUTE_CALLS:
                owner, attr = attr_call
                raise ValueError(
                    f"Call '{owner}.{attr}' is not allowed in execute_python_code"
                )


def _ensure_common_libraries(globals_dict: dict[str, Any]) -> None:
    """Добавляет распространенные аналитические библиотеки в globals sandbox.

    Args:
        globals_dict: Словарь глобальных переменных sandbox.

    Returns:
        ``None``. Словарь изменяется на месте, если библиотеки доступны.
    """

    try:
        import pandas as pd  # type: ignore

        globals_dict.setdefault("pd", pd)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore

        globals_dict.setdefault("np", np)
    except Exception:
        pass


def _execute_python_code(
    *,
    sandbox: PythonSandboxProtocol,
    code: str,
    target_variable: str | None = None,
    description: str = "",
) -> PythonExecutionResult:
    """Выполняет полный цикл проверки, компиляции и исполнения кода.

    Args:
        sandbox: Изолированная среда агента с общими переменными.
        code: Python-код для выполнения.
        target_variable: Опциональное имя переменной, которую должен создать код.
        description: Краткое описание цели выполнения.

    Returns:
        ``PythonExecutionResult`` с успехом, предпросмотром или ошибкой.
    """

    generated_code = _normalize_code_text(str(code or ""))
    try:
        target_name = _normalize_target_variable(target_variable)
        _validate_code_policy(generated_code)
        compiled = compile(generated_code, "<execute_python_code>", "exec")
    except Exception as exc:
        return PythonExecutionResult(
            success=False,
            message="Python code did not pass validation or compilation.",
            generated_code=generated_code,
            target_variable=str(target_variable or ""),
            error=f"{exc.__class__.__name__}: {exc}",
            traceback_text=_limit_text(traceback.format_exc(), max_chars=MAX_STDIO_CHARS),
            available_variables=_visible_variable_names(getattr(sandbox, "globals", {})),
            possible_causes=_python_error_possible_causes(exc),
            solution_options=_python_error_solution_options(exc),
            retry_guidance=_python_retry_guidance(),
        )

    globals_dict = getattr(sandbox, "globals", None)
    if not isinstance(globals_dict, dict):
        return PythonExecutionResult(
            success=False,
            message="Sandbox does not expose a globals dictionary.",
            generated_code=generated_code,
            target_variable=target_name or "",
            error="InvalidSandbox: sandbox.globals is not a dictionary",
            available_variables=[],
            possible_causes=["Sandbox не предоставляет словарь globals для выполнения кода."],
            solution_options=["Проверь конфигурацию sandbox в ResearchAgent/factory перед повторным запуском."],
            retry_guidance="Эту ошибку нельзя исправить изменением Python-кода; нужна корректная sandbox-конфигурация.",
        )

    _ensure_common_libraries(globals_dict)
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with (
            _temporary_working_directory(_sandbox_working_directory(sandbox)),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exec(compiled, globals_dict, globals_dict)
    except Exception as exc:
        return PythonExecutionResult(
            success=False,
            message="Python code execution failed. Fix generated_code and retry.",
            generated_code=generated_code,
            target_variable=target_name or "",
            execution_output=_combined_stdio(stdout, stderr),
            error=f"{exc.__class__.__name__}: {exc}",
            traceback_text=_limit_text(traceback.format_exc(), max_chars=MAX_STDIO_CHARS),
            available_variables=_visible_variable_names(globals_dict),
            possible_causes=_python_error_possible_causes(exc),
            solution_options=_python_error_solution_options(exc),
            retry_guidance=_python_retry_guidance(),
        )

    stdio = _combined_stdio(stdout, stderr)
    purpose = f" Purpose: {description.strip()}" if description.strip() else ""

    if target_name is None:
        message = f"Python code executed successfully.{purpose}"
        if stdio.strip():
            message += " Use execution_output as the tool result."
        return PythonExecutionResult(
            success=True,
            message=message,
            generated_code=generated_code,
            target_variable="",
            variable_preview=_preview_stdio_result(stdio),
            execution_output=stdio,
            available_variables=_visible_variable_names(globals_dict),
        )

    if target_name not in globals_dict:
        return PythonExecutionResult(
            success=False,
            message=(
                "Python code executed but did not create target_variable. "
                f"Create variable '{target_name}' and retry."
            ),
            generated_code=generated_code,
            target_variable=target_name,
            execution_output=stdio,
            error=f"MissingTargetVariable: {target_name}",
            available_variables=_visible_variable_names(globals_dict),
            possible_causes=[
                f"Код выполнился, но не создал переменную '{target_name}'.",
            ],
            solution_options=[
                f"Добавь присваивание результата в переменную '{target_name}'.",
                "Проверь, что присваивание выполняется на всех ветках кода.",
                "Если достаточно stdout, повтори вызов без target_variable.",
            ],
            retry_guidance=_python_retry_guidance(),
        )

    value = globals_dict[target_name]
    _update_sandbox_last_variables(sandbox, target_name, value)
    return PythonExecutionResult(
        success=True,
        message=f"Python code executed successfully.{purpose}",
        generated_code=generated_code,
        target_variable=target_name,
        variable_preview=_preview_value(value),
        execution_output=stdio,
        available_variables=_visible_variable_names(globals_dict),
    )


def _sandbox_working_directory(sandbox: PythonSandboxProtocol) -> Path | None:
    """Возвращает рабочую директорию sandbox для относительных файловых путей.

    Args:
        sandbox: Объект песочницы, в котором может быть атрибут
            ``working_directory`` или ``workspace_root``.

    Returns:
        Абсолютный путь рабочей директории или ``None``, если она не задана.
    """

    raw_directory = (
        getattr(sandbox, "working_directory", None)
        or getattr(sandbox, "workspace_root", None)
    )
    if raw_directory is None:
        return None
    directory_text = str(raw_directory).strip()
    if not directory_text:
        return None
    return Path(directory_text).expanduser().resolve()


@contextlib.contextmanager
def _temporary_working_directory(directory: Path | None):
    """Временно переключает текущую директорию процесса для выполнения кода.

    Args:
        directory: Рабочая директория для относительных файловых путей. Если
            ``None``, текущая директория не меняется.

    Yields:
        ``None``. После выхода исходная директория процесса восстанавливается.
    """

    with _get_cwd_execution_lock():
        if directory is None:
            yield
            return
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Sandbox working directory does not exist: {directory}")

        previous_directory = Path.cwd()
        os.chdir(directory)
        try:
            yield
        finally:
            os.chdir(previous_directory)


def _get_cwd_execution_lock() -> threading.RLock:
    """Возвращает общий process-wide lock для временной смены cwd.

    Args:
        Отсутствуют. Lock хранится в ``builtins`` для совместного использования
        разными sandbox-модулями.

    Returns:
        ``threading.RLock`` для защиты ``os.chdir`` и выполнения кода.
    """

    lock = getattr(builtins, _CWD_EXECUTION_LOCK_ATTR, None)
    if lock is None:
        lock = threading.RLock()
        setattr(builtins, _CWD_EXECUTION_LOCK_ATTR, lock)
    return lock


def _call_name(func: ast.AST) -> str:
    """Возвращает имя вызываемой функции для AST-узла.

    Args:
        func: AST-узел функции из ``ast.Call``.

    Returns:
        Имя функции или пустая строка, если его нельзя определить.
    """

    if isinstance(func, ast.Name):
        return func.id
    return ""


def _attribute_call_name(func: ast.AST) -> tuple[str, str]:
    """Возвращает пару ``объект.метод`` для вызова атрибута.

    Args:
        func: AST-узел функции из ``ast.Call``.

    Returns:
        Кортеж ``(owner, attr)`` или пара пустых строк.
    """

    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id, func.attr
    return "", ""


def _update_sandbox_last_variables(
    sandbox: PythonSandboxProtocol,
    target_variable: str,
    value: Any,
) -> None:
    """Обновляет служебные указатели sandbox на последнюю созданную переменную.

    Args:
        sandbox: Изолированная среда агента.
        target_variable: Имя переменной результата.
        value: Значение переменной результата.

    Returns:
        ``None``. Атрибуты sandbox обновляются при наличии.
    """

    try:
        setattr(sandbox, "last_target_variable", target_variable)
    except Exception:
        pass
    if _is_dataframe(value):
        try:
            setattr(sandbox, "last_dataframe_variable", target_variable)
        except Exception:
            pass


def _preview_stdio_result(stdio: str) -> str:
    """Создает компактный preview для успешного выполнения без target_variable.

    Args:
        stdio: Объединенный stdout/stderr после выполнения кода.

    Returns:
        Краткий текст preview или пустая строка.
    """

    text = str(stdio or "").strip()
    if not text:
        return ""
    return _limit_text(f"type: console_output\n{text}", max_chars=MAX_TEXT_PREVIEW_CHARS)


def _preview_value(value: Any) -> str:
    """Создает компактный предпросмотр значения результата.

    Args:
        value: Значение переменной результата после выполнения кода.

    Returns:
        Строка с типом, размерностью и кратким содержимым результата.
    """

    if _is_dataframe(value):
        shape = getattr(value, "shape", None)
        columns = list(getattr(value, "columns", []))
        dtypes = {
            str(column): str(dtype)
            for column, dtype in getattr(value, "dtypes", {}).items()
        }
        head_text = value.head(MAX_DATAFRAME_PREVIEW_ROWS).to_string()
        return _limit_text(
            "\n".join(
                [
                    "type: DataFrame",
                    f"shape: {shape}",
                    f"columns: {columns}",
                    f"dtypes: {dtypes}",
                    "head:",
                    head_text,
                ]
            ),
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    if _is_series(value):
        shape = getattr(value, "shape", None)
        head_text = value.head(MAX_DATAFRAME_PREVIEW_ROWS).to_string()
        return _limit_text(
            f"type: Series\nshape: {shape}\nhead:\n{head_text}",
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)
    except Exception:
        text = repr(value)
    return _limit_text(
        f"type: {type(value).__name__}\nvalue:\n{text}",
        max_chars=MAX_TEXT_PREVIEW_CHARS,
    )


def _combined_stdio(stdout: io.StringIO, stderr: io.StringIO) -> str:
    """Объединяет stdout и stderr в ограниченный текстовый блок.

    Args:
        stdout: Буфер стандартного вывода.
        stderr: Буфер стандартной ошибки.

    Returns:
        Строка с stdout/stderr или пустая строка.
    """

    parts: list[str] = []
    out_text = stdout.getvalue()
    err_text = stderr.getvalue()
    if out_text:
        parts.append(f"stdout:\n{out_text}")
    if err_text:
        parts.append(f"stderr:\n{err_text}")
    return _limit_text("\n".join(parts), max_chars=MAX_STDIO_CHARS)


def _python_error_possible_causes(exc: Exception) -> list[str]:
    """Возвращает вероятные причины ошибки execute_python_code.

    Args:
        exc: Исключение валидации, компиляции или выполнения кода.

    Returns:
        Список причин, которые модель может использовать для исправления.
    """

    if isinstance(exc, SyntaxError):
        return ["Сгенерированный Python-код содержит синтаксическую ошибку."]
    if isinstance(exc, NameError):
        return [
            "Код ссылается на переменную, которой нет в sandbox.",
            "Имя переменной могло быть взято из описания задачи, а не из available_variables.",
        ]
    if isinstance(exc, KeyError):
        return ["В DataFrame/dict отсутствует запрошенная колонка или ключ."]
    if isinstance(exc, ImportError):
        return ["Импортируемая библиотека недоступна в sandbox или запрещена политикой."]
    if isinstance(exc, ValueError):
        return ["Аргумент, имя переменной или операция имеют недопустимое значение."]
    return ["Код столкнулся с ошибкой выполнения; точная причина указана в traceback."]


def _python_error_solution_options(exc: Exception) -> list[str]:
    """Возвращает варианты исправления ошибки execute_python_code.

    Args:
        exc: Исключение валидации, компиляции или выполнения кода.

    Returns:
        Практические варианты следующего вызова инструмента.
    """

    options = [
        "Проверь available_variables и используй только существующие имена переменных.",
        "Исправь generated_code с учетом traceback и повтори execute_python_code.",
        "Если нужна именованная переменная, сохрани результат в target_variable.",
        "Если достаточно print-вывода, повтори вызов без target_variable и читай execution_output.",
    ]
    if isinstance(exc, SyntaxError):
        options.insert(0, "Исправь синтаксис Python-кода перед повторным запуском.")
    if isinstance(exc, NameError):
        options.insert(0, "Замени отсутствующую переменную на существующую из available_variables или создай ее явно.")
    if isinstance(exc, KeyError):
        options.insert(0, "Проверь реальные названия колонок через preview/schema перед обращением к ним.")
    if isinstance(exc, ImportError):
        options.insert(0, "Используй библиотеку из списка available_python_packages или стандартные pandas/numpy.")
    return options


def _python_retry_guidance() -> str:
    """Возвращает краткую инструкцию по retry для execute_python_code.

    Returns:
        Текст, объясняющий как повторять вызов после ошибки.
    """

    return (
        "Не повторяй тот же код без изменений. Используй generated_code, error, "
        "traceback и available_variables из этого ответа, исправь причину и "
        "повтори execute_python_code."
    )


def _visible_variable_names(globals_dict: dict[str, Any]) -> list[str]:
    """Возвращает список пользовательских переменных sandbox.

    Args:
        globals_dict: Словарь глобальных переменных sandbox.

    Returns:
        Отсортированный список имен без служебных ``__dunder__`` переменных.
    """

    return sorted(
        name
        for name in globals_dict
        if not str(name).startswith("__") and name not in vars(builtins)
    )


def _is_dataframe(value: Any) -> bool:
    """Проверяет, является ли значение pandas DataFrame без жесткой зависимости.

    Args:
        value: Проверяемое значение.

    Returns:
        ``True``, если значение похоже на ``pandas.DataFrame``.
    """

    return value.__class__.__name__ == "DataFrame" and hasattr(value, "head")


def _is_series(value: Any) -> bool:
    """Проверяет, является ли значение pandas Series без жесткой зависимости.

    Args:
        value: Проверяемое значение.

    Returns:
        ``True``, если значение похоже на ``pandas.Series``.
    """

    return value.__class__.__name__ == "Series" and hasattr(value, "head")


def _json_default(value: Any) -> Any:
    """Преобразует нестандартные объекты в JSON-совместимое представление.

    Args:
        value: Значение, которое стандартный JSON-сериализатор не обработал.

    Returns:
        JSON-совместимое значение.
    """

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return repr(value)


def _limit_text(value: str | None, *, max_chars: int) -> str:
    """Обрезает текст до заданного лимита с явной пометкой.

    Args:
        value: Исходный текст или ``None``.
        max_chars: Максимальная длина результата.

    Returns:
        Обрезанная строка или пустая строка.
    """

    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


__all__ = [
    "EXECUTE_PYTHON_CODE_TOOL_NAME",
    "ExecutePythonCodeInput",
    "ExecutePythonCodeTool",
    "build_execute_python_code_tool",
]

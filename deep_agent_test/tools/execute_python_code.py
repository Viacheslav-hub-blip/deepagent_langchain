"""Инструмент генерации и выполнения Python-кода для DeepAgent supervisor.

Содержит:
- PythonExecutionResult: контейнер результата выполнения Python-кода.
- ExecutePythonCodeInput: схема аргументов инструмента ``execute_python_code``.
- ExecutePythonCodeTool: LangChain tool выполнения Python-кода в sandbox.
- build_execute_python_code_tool: фабрика инструмента ``execute_python_code``.
- _normalize_code_text: нормализация текста Python-кода.
- _normalize_target_variable: проверка имени целевой переменной результата.
- _execute_python_code: проверка, компиляция и выполнение Python-кода.
- _validate_code_policy: статическая проверка политики безопасности кода.
- _call_name: извлечение имени прямого вызова функции из AST.
- _attribute_call_name: извлечение пары ``(owner, attr)`` из AST-вызова.
- _temporary_working_directory: временная смена рабочей директории процесса.
- _get_cwd_execution_lock: получение общего lock для смены cwd.
- _combined_stdio: объединение stdout и stderr.
- _preview_stdio_result: preview результата без целевой переменной.
- _preview_value: preview значения целевой переменной.
- _python_error_possible_causes: вероятные причины ошибки выполнения.
- _python_error_solution_options: варианты исправления ошибки выполнения.
- _python_retry_guidance: инструкция для повторного запуска после ошибки.
- _visible_variable_names: список пользовательских переменных sandbox.
- _is_dataframe: проверка значения на сходство с pandas DataFrame.
- _is_series: проверка значения на сходство с pandas Series.
- _json_default: JSON-сериализация нестандартных объектов.
- _limit_text: ограничение длинного текста.
- _policy_error_payload: JSON-ответ при ошибке статической политики.
- _result_to_json: дополнение результата контекстом sandbox и сериализация.
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

from deep_agent_test.core.python_sandbox import DeepAgentPythonSandbox, SANDBOX_HELPER_NAMES

EXECUTE_PYTHON_CODE_TOOL_NAME = "execute_python_code"
MAX_CODE_CHARS = 50_000
MAX_TEXT_PREVIEW_CHARS = 4_000
MAX_STDIO_CHARS = 8_000
MAX_DATAFRAME_PREVIEW_ROWS = 10
CWD_EXECUTION_LOCK_ATTR = "_deep_agent_test_sandbox_cwd_lock"

ALLOWED_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "collections",
        "datetime",
        "itertools",
        "json",
        "math",
        "matplotlib",
        "numpy",
        "os",
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
- нужно прочитать большой `.pkl`, сохраненный middleware после `load_data`;
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
rows = read_pickle_file(r"<saved_file из tool output>")
df = rows_to_dataframe(rows)
print(df.shape)
print(df.head(3).to_string())
result = df

Правила:
- переменные сохраняются между вызовами инструмента в одной сессии;
- для именованного результата укажи `target_variable`, иначе используй `print()` и читай `execution_output`;
- при ошибке tool возвращает JSON с `error`, полным `traceback`, `possible_causes`, `solution_options`, `retry_guidance` — исправь код и повтори вызов;
- не удаляй файлы и директории;
- можно импортировать `os` для просмотра директорий, но удаление файлов и shell-вызовы запрещены;
- читай pickle через `read_pickle_file` или `pd.read_pickle` по `saved_file` из tool outputs.
""".strip()


@dataclass
class PythonExecutionResult:
    """Результат выполнения Python-кода в sandbox.

    Attributes:
        success: Признак успешного выполнения кода.
        message: Краткое сообщение о результате выполнения.
        generated_code: Нормализованный Python-код, который был выполнен.
        target_variable: Имя переменной результата или пустая строка.
        variable_preview: Компактное описание значения результата.
        execution_output: Текст stdout/stderr, полученный при выполнении.
        error: Краткое описание ошибки.
        traceback_text: Полный traceback ошибки.
        available_variables: Список доступных переменных sandbox.
        possible_causes: Вероятные причины ошибки.
        solution_options: Практические варианты исправления ошибки.
        retry_guidance: Инструкция для повторного запуска после ошибки.
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
        """Сериализует результат выполнения в JSON-строку.

        Returns:
            JSON-строка с результатом, stdout/stderr, traceback и подсказками.
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
    делегируются локальной переиспользуемой функции ``_execute_python_code``.

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


def _normalize_code_text(code: str) -> str:
    """Нормализует текст Python-кода перед проверкой.

    Args:
        code: Исходный код, переданный в инструмент.

    Returns:
        Исходный код или код с преобразованными JSON-escaped переносами строк.
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
    return raw_code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _normalize_target_variable(target_variable: str | None) -> str | None:
    """Проверяет имя целевой переменной результата.

    Args:
        target_variable: Имя переменной результата или ``None``.

    Returns:
        Нормализованное имя переменной или ``None``.

    Raises:
        ValueError: Имя переменной не является валидным Python-идентификатором.
    """

    name = str(target_variable or "").strip()
    if not name:
        return None
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError("target_variable must be a valid Python identifier, for example result_df")
    return name


def _execute_python_code(
    *,
    sandbox: DeepAgentPythonSandbox,
    code: str,
    target_variable: str | None = None,
    description: str = "",
) -> PythonExecutionResult:
    """Выполняет Python-код в persistent sandbox.

    Args:
        sandbox: Sandbox с общими переменными и рабочими директориями.
        code: Python-код для проверки, компиляции и выполнения.
        target_variable: Опциональное имя переменной результата.
        description: Краткое описание цели кода для сообщения об успехе.

    Returns:
        ``PythonExecutionResult`` с результатом выполнения или подробной ошибкой.
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
            available_variables=_visible_variable_names(sandbox.globals),
            possible_causes=_python_error_possible_causes(exc),
            solution_options=_python_error_solution_options(exc),
            retry_guidance=_python_retry_guidance(),
        )

    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with (
            _temporary_working_directory(sandbox.working_directory),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            exec(compiled, sandbox.globals, sandbox.globals)
    except Exception as exc:
        return PythonExecutionResult(
            success=False,
            message="Python code execution failed. Fix generated_code and retry.",
            generated_code=generated_code,
            target_variable=target_name or "",
            execution_output=_combined_stdio(stdout, stderr),
            error=f"{exc.__class__.__name__}: {exc}",
            traceback_text=_limit_text(traceback.format_exc(), max_chars=MAX_STDIO_CHARS),
            available_variables=_visible_variable_names(sandbox.globals),
            possible_causes=_python_error_possible_causes(exc),
            solution_options=_python_error_solution_options(exc),
            retry_guidance=_python_retry_guidance(),
        )

    execution_output = _combined_stdio(stdout, stderr)
    purpose = f" Purpose: {description.strip()}" if description.strip() else ""
    if target_name is None:
        return PythonExecutionResult(
            success=True,
            message=f"Python code executed successfully.{purpose}",
            generated_code=generated_code,
            target_variable="",
            variable_preview=_preview_stdio_result(execution_output),
            execution_output=execution_output,
            available_variables=_visible_variable_names(sandbox.globals),
        )

    if target_name not in sandbox.globals:
        return PythonExecutionResult(
            success=False,
            message=(
                "Python code executed but did not create target_variable. "
                f"Create variable '{target_name}' and retry."
            ),
            generated_code=generated_code,
            target_variable=target_name,
            execution_output=execution_output,
            error=f"MissingTargetVariable: {target_name}",
            available_variables=_visible_variable_names(sandbox.globals),
            possible_causes=[f"Код выполнился, но не создал переменную '{target_name}'."],
            solution_options=[
                f"Добавь присваивание результата в переменную '{target_name}'.",
                "Проверь, что присваивание выполняется на всех ветках кода.",
                "Если достаточно stdout, повтори вызов без target_variable.",
            ],
            retry_guidance=_python_retry_guidance(),
        )

    return PythonExecutionResult(
        success=True,
        message=f"Python code executed successfully.{purpose}",
        generated_code=generated_code,
        target_variable=target_name,
        variable_preview=_preview_value(sandbox.globals[target_name]),
        execution_output=execution_output,
        available_variables=_visible_variable_names(sandbox.globals),
    )


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


@contextlib.contextmanager
def _temporary_working_directory(directory: Path | None):
    """Временно переключает рабочую директорию процесса.

    Args:
        directory: Рабочая директория sandbox или ``None``.

    Yields:
        ``None``. После выхода исходная директория восстанавливается.
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
    """Возвращает общий lock для временной смены текущей директории.

    Returns:
        ``threading.RLock`` для защиты ``os.chdir`` во время выполнения кода.
    """

    lock = getattr(builtins, CWD_EXECUTION_LOCK_ATTR, None)
    if lock is None:
        lock = threading.RLock()
        setattr(builtins, CWD_EXECUTION_LOCK_ATTR, lock)
    return lock


def _combined_stdio(stdout: io.StringIO, stderr: io.StringIO) -> str:
    """Объединяет stdout и stderr в ограниченную строку.

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


def _preview_stdio_result(stdio: str) -> str:
    """Формирует preview для успешного выполнения без целевой переменной.

    Args:
        stdio: Текст stdout/stderr после выполнения кода.

    Returns:
        Краткий preview консольного вывода или пустая строка.
    """

    text = str(stdio or "").strip()
    if not text:
        return ""
    return _limit_text(f"type: console_output\n{text}", max_chars=MAX_TEXT_PREVIEW_CHARS)


def _preview_value(value: Any) -> str:
    """Формирует компактный preview значения результата.

    Args:
        value: Значение переменной результата.

    Returns:
        Строка с типом и кратким содержимым значения.
    """

    if _is_dataframe(value):
        shape = getattr(value, "shape", None)
        columns = list(getattr(value, "columns", []))
        dtypes = {str(column): str(dtype) for column, dtype in getattr(value, "dtypes", {}).items()}
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
    return _limit_text(f"type: {type(value).__name__}\nvalue:\n{text}", max_chars=MAX_TEXT_PREVIEW_CHARS)


def _python_error_possible_causes(exc: Exception) -> list[str]:
    """Возвращает вероятные причины ошибки выполнения кода.

    Args:
        exc: Исключение валидации, компиляции или выполнения.

    Returns:
        Список причин, которые помогают исправить следующий вызов.
    """

    if isinstance(exc, SyntaxError):
        return ["Сгенерированный Python-код содержит синтаксическую ошибку."]
    if isinstance(exc, NameError):
        return ["Код ссылается на переменную, которой нет в sandbox."]
    if isinstance(exc, KeyError):
        return ["В DataFrame/dict отсутствует запрошенная колонка или ключ."]
    if isinstance(exc, ImportError):
        return ["Импортируемая библиотека недоступна или запрещена политикой инструмента."]
    if isinstance(exc, ValueError):
        return ["Аргумент, имя переменной или операция имеют недопустимое значение."]
    return ["Код столкнулся с ошибкой выполнения; точная причина указана в traceback."]


def _python_error_solution_options(exc: Exception) -> list[str]:
    """Возвращает варианты исправления ошибки выполнения кода.

    Args:
        exc: Исключение валидации, компиляции или выполнения.

    Returns:
        Список практических вариантов для повторного запуска.
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
        options.insert(0, "Используй доступную библиотеку или стандартные pandas/numpy helpers.")
    return options


def _python_retry_guidance() -> str:
    """Возвращает инструкцию по повторному вызову после ошибки.

    Returns:
        Текст с правилом исправления кода перед retry.
    """

    return (
        "Не повторяй тот же код без изменений. Используй generated_code, error, "
        "traceback и available_variables из этого ответа, исправь причину и повтори execute_python_code."
    )


def _visible_variable_names(globals_dict: dict[str, Any]) -> list[str]:
    """Возвращает список пользовательских переменных sandbox.

    Args:
        globals_dict: Словарь глобальных переменных sandbox.

    Returns:
        Отсортированный список имен без служебных и встроенных переменных.
    """

    return sorted(
        name
        for name in globals_dict
        if not str(name).startswith("__") and name not in vars(builtins)
    )


def _is_dataframe(value: Any) -> bool:
    """Проверяет, похоже ли значение на pandas DataFrame.

    Args:
        value: Проверяемое значение.

    Returns:
        ``True``, если значение похоже на DataFrame.
    """

    return value.__class__.__name__ == "DataFrame" and hasattr(value, "head")


def _is_series(value: Any) -> bool:
    """Проверяет, похоже ли значение на pandas Series.

    Args:
        value: Проверяемое значение.

    Returns:
        ``True``, если значение похоже на Series.
    """

    return value.__class__.__name__ == "Series" and hasattr(value, "head")


def _json_default(value: Any) -> Any:
    """Преобразует нестандартное значение к JSON-совместимому виду.

    Args:
        value: Значение, которое не смог сериализовать стандартный JSON.

    Returns:
        JSON-совместимое значение или ``repr(value)``.
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
    """Обрезает текст до заданной длины.

    Args:
        value: Исходный текст или ``None``.
        max_chars: Максимальное количество символов.

    Returns:
        Исходный или обрезанный текст с пометкой ``[truncated]``.
    """

    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


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

"""
Модуль песочницы для безопасного выполнения сгенерированного Python-кода.

Предоставляет:
- Валидацию кода перед выполнением (CodeValidator)
- Изолированную среду выполнения с персистентными глобальными переменными (ClientPythonSandbox)
- Асинхронное выполнение без блокировки event loop
- Предпросмотр созданных переменных
- Информацию об установленных пакетах (get_installed_packages)

"""

import ast
import asyncio
import contextlib
import io
import time
import traceback
from dataclasses import dataclass, field
from importlib.metadata import distributions as _iter_distributions
from typing import Any, Dict, Optional, Set, Tuple

# Максимальная длина текстового превью переменной (символов)
_PREVIEW_MAX_LENGTH: int = 500

# Суффикс при обрезке превью
_PREVIEW_TRUNCATION_SUFFIX: str = "..."

# Количество миллисекунд в одной секунде
_MS_PER_SECOND: int = 1000

# Префикс служебных переменных Python, скрываемых из превью
_DUNDER_PREFIX: str = "__"

# Одиночное подчёркивание — префикс приватных переменных, скрываемых из превью
_PRIVATE_PREFIX: str = "_"

# Имя типа для модулей Python (используется для фильтрации превью)
_MODULE_TYPE_NAME: str = "module"

# Количество первых строк DataFrame, используемых для превью
_DATAFRAME_HEAD_ROWS: int = 1


@dataclass
class ExecutionResult:
    """
    Результат выполнения кода в песочнице.

    Атрибуты:
        success: Флаг успешного выполнения.
        output: Захваченный stdout + stderr.
        error: Сообщение об ошибке (None при успехе).
        new_variable_schemas: Словарь {имя_переменной: превью} для новых переменных.
        execution_time_ms: Время выполнения в миллисекундах.
    """

    success: bool
    output: str = ""
    error: Optional[str] = None
    new_variable_schemas: Dict[str, str] = field(default_factory=dict)
    execution_time_ms: int = 0


class CodeValidator:
    """
    Статический валидатор, блокирующий опасные операции до вызова exec().

    Проверяет AST-дерево кода на наличие запрещённых вызовов функций
    и импортов модулей.
    """

    # Имена функций, вызов которых запрещён в пользовательском коде
    DANGEROUS_CALLS: Set[str] = {"eval", "exec", "compile", "__import__"}

    # Модули, импорт которых запрещён в пользовательском коде
    DANGEROUS_MODULES: Set[str] = {}

    @classmethod
    def validate(cls, code: str) -> Tuple[bool, str]:
        """
        Валидирует строку кода и возвращает результат проверки.

        Аргументы:
            code: Строка с Python-кодом для проверки.

        Возвращает:
            Кортеж (is_valid, error_message):
                - is_valid: True, если код прошёл все проверки.
                - error_message: Описание ошибки (пустая строка при успехе).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            return False, f"Синтаксическая ошибка: {error}"

        for node in ast.walk(tree):
            # Проверка запрещённых вызовов функций
            if isinstance(node, ast.Call):
                if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in cls.DANGEROUS_CALLS
                ):
                    return False, f"Запрещённый вызов функции: {node.func.id}"

            # Проверка прямых импортов запрещённых модулей (import os)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in cls.DANGEROUS_MODULES:
                        return False, f"Запрещённый импорт модуля: {alias.name}"

            # Проверка импортов из запрещённых модулей (from os import path)
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module in cls.DANGEROUS_MODULES:
                    return False, f"Запрещённый импорт модуля: {node.module}"

        return True, ""


class ClientPythonSandbox:
    """
    Песочница выполнения кода в памяти с персистентными глобальными переменными.

    Поддерживает:
    - Асинхронное выполнение кода без блокировки event loop.
    - Захват stdout/stderr.
    - Предпросмотр созданных переменных, включая DataFrame.
    - Блокировку параллельного доступа через asyncio.Lock.
    - Сброс состояния с сохранением базовых переменных.
    """

    def __init__(
            self,
            allowed_libraries: Optional[Set[str] | Dict[str, Any]] = None,
            initial_globals: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Инициализирует песочницу.

        Аргументы:
            allowed_libraries: Множество разрешённых для импорта библиотек.
                               По умолчанию — пустое множество.
            initial_globals:   Начальные глобальные переменные среды выполнения.
                               По умолчанию — пустой словарь.
        """
        if initial_globals is None:
            initial_globals = {}

        if allowed_libraries is None:
            allowed_libraries = set()

        self.globals: Dict[str, Any] = initial_globals.copy()
        if isinstance(allowed_libraries, dict):
            self.allowed_libraries: Set[str] = set(allowed_libraries.keys())
            self.globals.update(allowed_libraries)
        else:
            self.allowed_libraries = set(allowed_libraries)

        # Имена переменных, присутствовавших при инициализации
        self._base_globals_names: Set[str] = set(self.globals.keys())

        # Имена переменных, скрытых из вывода превью
        self._hidden_from_preview_names: Set[str] = set()

        # Лок для защиты от конкурентного изменения globals
        self._lock: asyncio.Lock = asyncio.Lock()

        self.last_target_variable: Optional[str] = None
        self.last_dataframe_variable: Optional[str] = None

    async def execute(
            self,
            code: str,
            target_variable: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Асинхронно выполняет Python-код и возвращает метаданные выполнения.

        Код проходит валидацию перед выполнением. Выполнение происходит
        в отдельном потоке через executor, чтобы не блокировать event loop.

        Аргументы:
            code:            Строка с Python-кодом для выполнения.
            target_variable: Имя ожидаемой переменной-результата.
                             Если указана и не создана — возвращается ошибка.

        Возвращает:
            ExecutionResult с результатами выполнения.
        """
        start_time = time.monotonic()

        # Валидация кода до выполнения
        is_valid, error_message = CodeValidator.validate(code)
        if not is_valid:
            err = f"Валидация кода не пройдена: {error_message}"
            print(f"[sandbox] {err}", flush=True)
            return ExecutionResult(
                success=False,
                error=err,
                execution_time_ms=int(
                    (time.monotonic() - start_time) * _MS_PER_SECOND
                ),
            )

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        async with self._lock:
            keys_before = set(self.globals.keys())

            try:
                # Выполнение в executor чтобы не блокировать event loop
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._exec_code,
                    code,
                    stdout_capture,
                    stderr_capture,
                )
                success = True
                error_message = None
            except Exception:
                success = False
                error_message = traceback.format_exc()
                print(
                    "[sandbox] Исключение при выполнении кода:\n"
                    f"{error_message}",
                    flush=True,
                )

            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            keys_after = set(self.globals.keys())
            new_keys = keys_after - keys_before

            # Формируем превью для всех новых переменных (исключая служебные)
            new_schemas: Dict[str, str] = {
                key: self._get_variable_preview(key)
                for key in new_keys
                if not key.startswith(_DUNDER_PREFIX)
            }

            # Проверяем, что целевая переменная была создана
            if (
                    target_variable
                    and target_variable not in self.globals
                    and not new_keys
            ):
                success = False
                error_message = (
                    f"Целевая переменная '{target_variable}' не найдена "
                    f"после выполнения и новые переменные не были созданы."
                )
                print(f"[sandbox] {error_message}", flush=True)

            return ExecutionResult(
                success=success,
                output=output,
                error=error_message,
                new_variable_schemas=new_schemas,
                execution_time_ms=int(
                    (time.monotonic() - start_time) * _MS_PER_SECOND
                ),
            )

    def _exec_code(
            self,
            code: str,
            stdout_capture: io.StringIO,
            stderr_capture: io.StringIO,
    ) -> None:
        """
        Выполняет код синхронно с перехватом stdout/stderr.

        Метод предназначен для вызова через run_in_executor.
        При возникновении исключения оно пробрасывается наверх.

        Аргументы:
            code:           Строка с Python-кодом.
            stdout_capture: Буфер для перехвата stdout.
            stderr_capture: Буфер для перехвата stderr.
        """
        with (
            contextlib.redirect_stdout(stdout_capture),
            contextlib.redirect_stderr(stderr_capture),
        ):
            exec(code, self.globals, self.globals)  # noqa: S102

    def _get_variable_preview(self, var_name: str) -> str:
        """
        Формирует короткое текстовое превью переменной по её имени.

        Для DataFrame-подобных объектов возвращает структурированное описание
        (форма, колонки, типы, наличие NaN).
        Для прочих объектов возвращает строковое представление,
        обрезанное до _PREVIEW_MAX_LENGTH символов.

        Аргументы:
            var_name: Имя переменной в self.globals.

        Возвращает:
            Строку с превью переменной.
        """
        if var_name not in self.globals:
            return "Переменная не найдена"

        value = self.globals[var_name]
        try:
            # DataFrame-подобный объект (pandas, polars и др.)
            if hasattr(value, "shape") and hasattr(value, "head"):
                columns = list(value.columns.astype(str))
                dtypes = {
                    str(col): str(dtype) for col, dtype in value.dtypes.items()
                }
                nan_flags = {
                    str(col): bool(value[col].isna().any())
                    for col in value.columns
                }
                return (
                    f"DataFrame shape: {value.shape}\n"
                    f"Columns: {columns}\n"
                    f"Dtypes: {dtypes}\n"
                    f"Has NaN: {nan_flags}"
                )

            # Объекты с методом to_string (Series и др.)
            if hasattr(value, "to_string"):
                return value.to_string()

            # Общий случай: строковое представление с обрезкой
            text_value = str(value)
            if len(text_value) > _PREVIEW_MAX_LENGTH:
                return text_value[:_PREVIEW_MAX_LENGTH] + _PREVIEW_TRUNCATION_SUFFIX
            return text_value

        except Exception as error:
            return f"Ошибка получения превью: {error}"

    async def get_all_variable_previews(self) -> Dict[str, str]:
        """
        Возвращает превью всех пользовательских переменных из globals.

        Исключает служебные переменные (с префиксом '_'),
        переменные из _hidden_from_preview_names и модули.

        Возвращает:
            Словарь {имя_переменной: превью}.
        """
        async with self._lock:
            schemas: Dict[str, str] = {}
            for name, value in self.globals.items():
                # Пропускаем приватные и скрытые переменные
                if name.startswith(_PRIVATE_PREFIX):
                    continue
                if name in self._hidden_from_preview_names:
                    continue
                # Пропускаем импортированные модули
                if type(value).__name__ == _MODULE_TYPE_NAME:
                    continue
                schemas[name] = self._get_variable_preview(name)
            return schemas

    async def get_variable(self, var_name: str) -> Any:
        """
        Возвращает значение переменной из globals по имени.

        Аргументы:
            var_name: Имя переменной.

        Возвращает:
            Значение переменной или None, если переменная не найдена.
        """
        async with self._lock:
            return self.globals.get(var_name)

    async def add_variable(
            self,
            name: str,
            value: Any,
            exclude_from_preview: bool = False,
    ) -> None:
        """
        Добавляет переменную в globals песочницы.

        Аргументы:
            name:                Имя переменной.
            value:               Значение переменной.
            exclude_from_preview: Если True — переменная не будет отображаться
                                  в превью (get_all_variable_previews).
        """
        async with self._lock:
            self.globals[name] = value
            if exclude_from_preview:
                self._hidden_from_preview_names.add(name)

    @staticmethod
    def get_installed_packages() -> Dict[str, str]:
        """
        Возвращает словарь установленных пакетов {имя: версия} — аналог pip freeze.

        Использует ``importlib.metadata.distributions``, что работает быстрее
        и безопаснее вызова subprocess. Результат можно передавать в промпт
        worker-а, чтобы он знал, какие библиотеки доступны для импорта
        при генерации кода.

        Returns:
            Словарь вида ``{"pandas": "2.2.0", "numpy": "1.26.0", ...}``.
        """
        return {
            dist.metadata["Name"]: dist.version
            for dist in _iter_distributions()
            if dist.metadata["Name"]
        }

    async def reset(self, keep_base: bool = True) -> None:
        """
        Сбрасывает состояние песочницы.

        Аргументы:
            keep_base: Если True — сохраняет переменные, переданные
                       при инициализации (initial_globals).
                       Если False — полностью очищает globals.
        """
        async with self._lock:
            if keep_base:
                # Оставляем только переменные из начального состояния
                self.globals = {
                    key: val
                    for key, val in self.globals.items()
                    if key in self._base_globals_names
                }
            else:
                self.globals = {}
                self._base_globals_names = set()

            self.last_target_variable = None
            self.last_dataframe_variable = None

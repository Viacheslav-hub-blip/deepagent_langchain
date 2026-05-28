"""Сборка production-ready native DeepAgents агента для аналитики данных.

Содержит:
- clarify_analysis_request: e2e-инструмент уточнения задачи и желаемого результата.
- build_read_table_description: подробное описание read_table tool для LLM.
- build_data_tools: сборка tools чтения данных через фабрику из конфига.
- _load_callable_from_path: загрузка callable по import path из конфига.
- _normalize_data_tools: нормализация результата фабрики tools.
- run_python_analysis: e2e-инструмент выполнения Python-анализа.
- DeleteOperationError: ошибка запрета операций удаления файлов и директорий.
- _validate_python_code_without_delete_operations: проверка кода на операции удаления.
- _call_uses_delete_api: проверка вызова Python API удаления.
- _attribute_root_name: извлечение корневого имени attribute chain.
- _call_uses_delete_shell_command: проверка shell-команды удаления.
- _literal_command_text: извлечение literal shell-команды из AST-вызова.
- _command_text_contains_delete: поиск команды удаления в shell-тексте.
- _temporary_delete_operation_guard: runtime-защита от операций удаления во время exec.
- _install_temporary_patch: временная подмена функции или метода.
- _blocked_delete_callable: создание функции, запрещающей удаление.
- _guarded_shell_callable: создание wrapper-а для shell/subprocess вызова.
- _command_invocation_text: извлечение shell-команды из runtime-аргументов.
- _format_python_analysis_tool_result: форматирование результата Python-анализа.
- _extract_rows_for_artifact: извлечение табличных строк из результата.
- _row_to_mapping: преобразование элемента результата в словарь.
- build_analysis_artifact: e2e-инструмент подготовки текстового artifact payload.
- save_analysis_file: e2e-инструмент сохранения файла в рабочую директорию проекта.
- _is_relative_to: проверка, что путь находится внутри проекта.
- build_skills_backend: создание backend для чтения локальных skills через /skills/.
- build_skills_permissions: запрет записи в локальные skills через filesystem tools.
- build_analytics_deep_agent: сборка DeepAgent supervisor с subagents и HITL.
- get_skills_root: получение папки skills из настроек.
- invoke_agent: отправка пользовательского сообщения агенту.
- resume_with_decision: продолжение выполнения после human-in-the-loop interrupt.
- resume_with_user_answer: продолжение выполнения после уточняющего вопроса агента.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deep_agent_test.agent_logging import DeepAgentEventLogger, build_deep_agent_logger
from deep_agent_test.agent_specs import build_analytics_subagent_specs, build_clarify_interrupt_config
from deep_agent_test.few_shot_examples_index import FewShotExamplesStore, update_few_shot_examples_index
from deep_agent_test.few_shot_examples_middleware import FewShotExamplesMiddleware
from deep_agent_test.plan_approval_middleware import FirstPlanApprovalMiddleware
from deep_agent_test.prompt_debug_middleware import PromptDebugConsoleMiddleware
from deep_agent_test.prompts import READ_TABLE_DESCRIPTION, SYSTEM_PROMPT
from deep_agent_test.settings import PROJECT_ROOT, DeepAgentSettings, load_deep_agent_settings
from deep_agent_test.skills_context_middleware import PreloadedSkillsContextMiddleware
from deep_agent_test.tool_output_file_middleware import ToolOutputFileMiddleware
from deep_agent_test.trace_logging_middleware import ToolTraceLoggingMiddleware


@tool
def clarify_analysis_request(question: str) -> str:
    """Задает пользователю уточняющий вопрос перед продолжением анализа.

    Используй этот инструмент, когда без ответа пользователя невозможно выбрать
    корректный источник данных, период, сущность анализа, формат результата или
    обязательные поля отчета.

    Когда использовать:
    - пользовательский запрос неоднозначен;
    - неизвестен обязательный период, если его нельзя надежно определить из данных;
    - есть несколько допустимых подходов, которые меняют результат;
    - нужна информация, которую нельзя получить из skills или tool outputs.

    Когда не использовать:
    - для простых yes/no подтверждений;
    - если можно действовать по разумному default из domain context;
    - если вопрос можно решить чтением данных или skills;
    - вместо выполнения уже утвержденного плана.

    Args:
        question: Текст уточняющего вопроса с просьбой описать задачу, желаемый
            результат, формат ответа и обязательные элементы результата.

    Returns:
        Текст уточняющего вопроса. В обычном сценарии tool перехватывается
        ``interrupt_on`` и пользователь отвечает через human-in-the-loop ``respond``.
    """

    return question


def build_read_table_description() -> str:
    """Возвращает подробное описание инструмента ``read_table`` для LLM.

    Args:
        Отсутствуют.

    Returns:
        Текстовое описание tool: назначение, ограничения, правила использования
        и формат аргументов.
    """

    return READ_TABLE_DESCRIPTION


def build_data_tools(settings: DeepAgentSettings | None = None) -> list[BaseTool]:
    """Собирает инструменты чтения данных через фабрику из JSON-конфига.

    Args:
        settings: Настройки агента. Если ``None``, загружается JSON-конфиг по умолчанию.

    Returns:
        Список LangChain tools чтения данных.

    Raises:
        ValueError: Если фабрика tools не настроена.
        TypeError: Если фабрика вернула значение не в формате LangChain tools.
    """

    settings = settings or load_deep_agent_settings()
    if not settings.data_tools_factory:
        raise ValueError(
            "Не настроена фабрика tools чтения данных. "
            "Передайте data_tools в build_analytics_deep_agent или укажите "
            "data_tools_factory в deep_agent_test/config/defaults.json или override-конфиге."
        )
    factory = _load_callable_from_path(settings.data_tools_factory)
    return _normalize_data_tools(factory(**settings.data_tools_factory_kwargs))


def _load_callable_from_path(import_path: str) -> Callable[..., Any]:
    """Загружает callable по import path из конфига.

    Args:
        import_path: Строка вида ``package.module:function`` или ``package.module.function``.

    Returns:
        Callable-объект фабрики tools.

    Raises:
        ValueError: Если путь не содержит module и attribute.
        TypeError: Если загруженный объект не callable.
    """

    if ":" in import_path:
        module_name, attribute_name = import_path.split(":", 1)
    else:
        try:
            module_name, attribute_name = import_path.rsplit(".", 1)
        except ValueError:
            raise ValueError(f"Некорректный import path фабрики tools: {import_path}") from None
    if not module_name or not attribute_name:
        raise ValueError(f"Некорректный import path фабрики tools: {import_path}")

    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"Объект {import_path} не является callable.")
    return factory


def _normalize_data_tools(raw_tools: Any) -> list[BaseTool]:
    """Преобразует результат фабрики tools в список LangChain ``BaseTool``.

    Args:
        raw_tools: Один tool или последовательность tools, возвращенная фабрикой.

    Returns:
        Список LangChain tools.

    Raises:
        TypeError: Если результат фабрики не является tool или списком tools.
    """

    if isinstance(raw_tools, BaseTool):
        return [raw_tools]
    if not isinstance(raw_tools, (list, tuple)):
        raise TypeError("Фабрика tools должна вернуть BaseTool или список BaseTool.")

    tools: list[BaseTool] = []
    for item in raw_tools:
        if not isinstance(item, BaseTool):
            raise TypeError(f"Фабрика tools вернула объект не BaseTool: {type(item).__name__}")
        tools.append(item)
    return tools


@tool(response_format="content_and_artifact")
def run_python_analysis(task: str, code: str, input_data: Any | None = None) -> tuple[str, dict[str, Any]]:
    """Выполняет Python-анализ по уже полученным данным.

    Используй этот инструмент для расчетов, объединения, фильтрации, сортировки,
    нормализации, разбора JSON/текста и подготовки табличного результата перед
    сохранением файла.

    Когда использовать:
    - нужно объединить строки из нескольких источников;
    - нужно посчитать метрики, количества, суммы, доли или флаги;
    - нужно отсортировать события по времени;
    - нужно нормализовать названия колонок;
    - нужно подготовить список словарей, CSV-текст или отчетный payload.

    Когда не использовать:
    - для чтения таблиц напрямую;
    - вместо save_analysis_file, если файл уже готов к записи;
    - для финального пользовательского ответа вместо supervisor-а.

    Правила:
    - сначала проверь, какие поля есть в input_data;
    - не используй API-ключи и внешние сервисы;
    - можно использовать любые imports и библиотеки текущего виртуального окружения;
    - можно создавать файлы и директории без ограничений со стороны инструмента;
    - запрещено удалять файлы и директории через Python API или shell-команды;
    - результат должен быть записан в переменную result;
    - если произошла ошибка, верни понятное описание причины.

    Args:
        task: Краткое описание аналитической задачи.
        code: Python-код, который должен записать итог в переменную ``result``.
        input_data: Данные из предыдущих шагов в произвольной структуре.

    Returns:
        Пара ``(content, artifact)``: компактный текст для модели и полный
        структурированный результат выполнения кода.
    """

    normalized_input_data = _normalize_python_analysis_input_data(input_data)
    namespace = {"input_data": normalized_input_data}
    try:
        _validate_python_code_without_delete_operations(code)
        with _temporary_delete_operation_guard():
            exec(code, namespace, namespace)
    except Exception as exc:
        payload = {
            "task": task,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "result": None,
            "artifacts": [],
        }
        return _format_python_analysis_tool_result(payload)
    payload = {
        "task": task,
        "ok": True,
        "result": namespace.get("result"),
        "artifacts": namespace.get("artifacts", []),
    }
    return _format_python_analysis_tool_result(payload)


def _normalize_python_analysis_input_data(input_data: Any) -> Any:
    """Нормализует input_data для python-analysis без потери обратной совместимости.

    Если в инструмент пришла JSON-строка, пытается распарсить ее в dict/list.
    При ошибке парсинга возвращает исходное значение без исключения.
    """

    if not isinstance(input_data, str):
        return input_data
    stripped = input_data.strip()
    if not stripped:
        return input_data
    if stripped[0] not in {"{", "["}:
        return input_data
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return input_data
    return parsed


class DeleteOperationError(ValueError):
    """Ошибка запрета операций удаления файлов и директорий.

    Args:
        message: Описание найденной операции удаления.

    Returns:
        Исключение, которое возвращается агенту как ошибка ``run_python_analysis``.
    """


def _validate_python_code_without_delete_operations(code: str) -> None:
    """Проверяет, что код не содержит операций удаления файлов и директорий.

    Args:
        code: Python-код, который агент передал в ``run_python_analysis``.

    Returns:
        None.

    Raises:
        DeleteOperationError: Если код содержит Python API удаления или shell-команду удаления.
        SyntaxError: Если код не является корректным Python-кодом.
    """

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_uses_delete_api(node):
            raise DeleteOperationError("Код содержит Python API удаления файлов или директорий.")
        if _call_uses_delete_shell_command(node):
            raise DeleteOperationError("Код содержит shell-команду удаления файлов или директорий.")


def _call_uses_delete_api(node: ast.Call) -> bool:
    """Проверяет, вызывает ли AST-узел Python API удаления.

    Args:
        node: AST-узел вызова функции.

    Returns:
        ``True``, если вызов похож на ``os.remove``, ``Path.unlink`` или ``shutil.rmtree``.
    """

    blocked_names = {"remove", "removedirs", "rmdir", "rmtree", "unlink"}
    always_blocked_attrs = {"removedirs", "rmdir", "rmtree", "unlink"}
    if isinstance(node.func, ast.Name):
        return node.func.id in blocked_names
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in always_blocked_attrs:
            return True
        return node.func.attr == "remove" and _attribute_root_name(node.func) == "os"
    return False


def _attribute_root_name(node: ast.Attribute) -> str:
    """Извлекает корневое имя цепочки атрибутов.

    Args:
        node: AST-узел атрибута, например ``os.path.remove``.

    Returns:
        Корневое имя цепочки или пустая строка, если оно не является простым именем.
    """

    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        current = current.value
    if isinstance(current, ast.Name):
        return current.id
    return ""


def _call_uses_delete_shell_command(node: ast.Call) -> bool:
    """Проверяет, запускает ли AST-узел shell-команду удаления.

    Args:
        node: AST-узел вызова функции.

    Returns:
        ``True``, если вызов похож на ``os.system("rm ...")`` или ``subprocess.run(["del", ...])``.
    """

    command_functions = {"call", "check_call", "check_output", "Popen", "popen", "run", "system"}
    function_name = ""
    if isinstance(node.func, ast.Name):
        function_name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        function_name = node.func.attr
    if function_name not in command_functions:
        return False
    command_text = _literal_command_text(node)
    return _command_text_contains_delete(command_text)


def _literal_command_text(node: ast.Call) -> str:
    """Извлекает shell-команду из literal-аргументов AST-вызова.

    Args:
        node: AST-узел вызова ``os.system`` или ``subprocess``.

    Returns:
        Текст команды, если первый аргумент задан строкой или списком строк, иначе пустая строка.
    """

    if not node.args:
        return ""
    first_arg = node.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    if isinstance(first_arg, (ast.List, ast.Tuple)):
        values: list[str] = []
        for item in first_arg.elts:
            if isinstance(item, ast.Constant):
                values.append(str(item.value))
        return " ".join(values)
    return ""


def _command_text_contains_delete(command_text: str) -> bool:
    """Проверяет, содержит ли shell-текст команду удаления.

    Args:
        command_text: Текст shell-команды.

    Returns:
        ``True``, если команда содержит ``rm``, ``del``, ``rmdir``, ``rd`` или ``Remove-Item``.
    """

    normalized = command_text.strip().lower()
    if not normalized:
        return False
    return bool(re.search(r"\b(rm|del|rmdir|rd|remove-item)\b", normalized))


@contextmanager
def _temporary_delete_operation_guard() -> Iterator[None]:
    """Временно запрещает операции удаления во время выполнения Python-кода агента.

    Args:
        Отсутствуют.

    Returns:
        Context manager, который подменяет опасные функции на время ``exec`` и
        восстанавливает их после выполнения.
    """

    import os
    import pathlib
    import shutil
    import subprocess

    restorers: list[Callable[[], None]] = []
    delete_targets: list[tuple[Any, str, str]] = [
        (os, "remove", "os.remove"),
        (os, "unlink", "os.unlink"),
        (os, "rmdir", "os.rmdir"),
        (os, "removedirs", "os.removedirs"),
        (pathlib.Path, "unlink", "Path.unlink"),
        (pathlib.Path, "rmdir", "Path.rmdir"),
        (shutil, "rmtree", "shutil.rmtree"),
    ]
    for target, attribute_name, display_name in delete_targets:
        if hasattr(target, attribute_name):
            restorers.append(
                _install_temporary_patch(
                    target=target,
                    attribute_name=attribute_name,
                    replacement=_blocked_delete_callable(display_name),
                )
            )

    shell_targets: list[tuple[Any, str, str]] = [
        (os, "system", "os.system"),
        (os, "popen", "os.popen"),
        (subprocess, "run", "subprocess.run"),
        (subprocess, "call", "subprocess.call"),
        (subprocess, "check_call", "subprocess.check_call"),
        (subprocess, "check_output", "subprocess.check_output"),
        (subprocess, "Popen", "subprocess.Popen"),
    ]
    for target, attribute_name, display_name in shell_targets:
        if hasattr(target, attribute_name):
            original = getattr(target, attribute_name)
            restorers.append(
                _install_temporary_patch(
                    target=target,
                    attribute_name=attribute_name,
                    replacement=_guarded_shell_callable(display_name, original),
                )
            )

    try:
        yield
    finally:
        for restore in reversed(restorers):
            restore()


def _install_temporary_patch(target: Any, attribute_name: str, replacement: Any) -> Callable[[], None]:
    """Подменяет атрибут объекта и возвращает функцию восстановления.

    Args:
        target: Модуль, класс или объект, у которого нужно временно заменить атрибут.
        attribute_name: Имя заменяемого атрибута.
        replacement: Временное значение атрибута.

    Returns:
        Функция без аргументов, которая восстанавливает исходное значение.
    """

    original = getattr(target, attribute_name)
    setattr(target, attribute_name, replacement)

    def restore() -> None:
        """Восстанавливает исходный атрибут после временной подмены.

        Args:
            Отсутствуют.

        Returns:
            None.
        """

        setattr(target, attribute_name, original)

    return restore


def _blocked_delete_callable(function_name: str) -> Callable[..., Any]:
    """Создает callable, который запрещает конкретную операцию удаления.

    Args:
        function_name: Человекочитаемое имя заблокированной функции.

    Returns:
        Функция, которая всегда выбрасывает ``DeleteOperationError``.
    """

    def blocked(*args: Any, **kwargs: Any) -> Any:
        """Блокирует runtime-вызов удаления внутри ``run_python_analysis``.

        Args:
            args: Позиционные аргументы заблокированной функции.
            kwargs: Именованные аргументы заблокированной функции.

        Returns:
            Не возвращает значение, потому что всегда выбрасывает исключение.

        Raises:
            DeleteOperationError: Всегда, при попытке удалить файл или директорию.
        """

        raise DeleteOperationError(f"Операция удаления запрещена: {function_name}.")

    return blocked


def _guarded_shell_callable(function_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """Создает wrapper для shell/subprocess вызова с проверкой команд удаления.

    Args:
        function_name: Человекочитаемое имя shell/subprocess функции.
        original: Исходная callable-функция, которую нужно вызвать после проверки.

    Returns:
        Wrapper, который блокирует команды удаления и пропускает остальные команды.
    """

    def guarded(*args: Any, **kwargs: Any) -> Any:
        """Проверяет runtime-аргументы shell/subprocess вызова.

        Args:
            args: Позиционные аргументы исходной функции.
            kwargs: Именованные аргументы исходной функции.

        Returns:
            Результат исходной функции, если команда не содержит операций удаления.

        Raises:
            DeleteOperationError: Если shell-команда содержит удаление файлов или директорий.
        """

        command_text = _command_invocation_text(args, kwargs)
        if _command_text_contains_delete(command_text):
            raise DeleteOperationError(f"Shell-команда удаления запрещена через {function_name}.")
        return original(*args, **kwargs)

    return guarded


def _command_invocation_text(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Извлекает текст команды из runtime-аргументов shell/subprocess вызова.

    Args:
        args: Позиционные аргументы shell/subprocess функции.
        kwargs: Именованные аргументы shell/subprocess функции.

    Returns:
        Строковое представление команды для проверки удаления.
    """

    command = args[0] if args else kwargs.get("args", kwargs.get("cmd", ""))
    if isinstance(command, (list, tuple)):
        return " ".join(str(part) for part in command)
    return str(command)


def _format_python_analysis_tool_result(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Формирует текст и artifact для результата ``run_python_analysis``.

    Args:
        payload: Полный результат выполнения Python-кода.

    Returns:
        Кортеж ``(content, artifact)`` для ToolMessage. Если результат содержит
        большую таблицу, middleware сможет сохранить ее в CSV по artifact.
    """

    result = payload.get("result")
    rows = _extract_rows_for_artifact(result)
    artifact = dict(payload)
    if rows:
        artifact["rows"] = rows
        artifact["returned_rows"] = len(rows)

    preview = rows[:5] if rows else result
    content = {
        "task": payload.get("task"),
        "ok": payload.get("ok"),
        "error_type": payload.get("error_type"),
        "error": payload.get("error"),
        "result_type": type(result).__name__,
        "rows": len(rows) if rows else None,
        "preview": preview,
        "artifacts": payload.get("artifacts", []),
    }
    return json.dumps(content, ensure_ascii=False, indent=2, default=str), artifact


def _extract_rows_for_artifact(value: Any) -> list[dict[str, Any]]:
    """Извлекает табличные строки из результата Python-анализа без строгой проверки item-типа.

    Args:
        value: Значение переменной ``result`` из выполненного Python-кода.

    Returns:
        Список словарей. Неструктурированные элементы списка оборачиваются в поле ``value``.
    """

    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
        except TypeError:
            records = None
        if isinstance(records, list):
            return [_row_to_mapping(item) for item in records]
    if isinstance(value, list):
        return [_row_to_mapping(item) for item in value]
    if isinstance(value, dict):
        for key in ("rows", "records", "data", "result"):
            rows = _extract_rows_for_artifact(value.get(key))
            if rows:
                return rows
        return [value]
    return []


def _row_to_mapping(value: Any) -> dict[str, Any]:
    """Преобразует элемент табличного результата в словарь без отбрасывания данных.

    Args:
        value: Строка результата в любом формате.

    Returns:
        Исходный словарь или словарь с единственным полем ``value``.
    """

    if isinstance(value, dict):
        return value
    return {"value": value}


@tool
def build_analysis_artifact(file_name: str, content: str) -> dict[str, str]:
    """Формирует текстовый artifact payload без записи файла на диск.

    Используй этот инструмент только для промежуточного представления результата,
    когда физический файл пользователю не нужен.

    Важно:
    - инструмент не создает файл в рабочей директории;
    - результат нельзя называть сохраненным файлом;
    - если пользователь просит CSV, отчет или файл, используй save_analysis_file.

    Args:
        file_name: Имя будущего файла или artifact в scratch-пространстве агента.
        content: Текстовое содержимое отчета или результата.

    Returns:
        Словарь с именем файла и содержимым. Физический файл на диске не создается.
    """

    return {"file_name": file_name, "content": content}


@tool
def save_analysis_file(file_name: str, content: str, output_dir: str = ".") -> dict[str, Any]:
    """Записывает новый файл в рабочую директорию проекта.

    Используй этот инструмент, когда пользователь просит создать, сохранить или
    выгрузить файл: CSV, отчет, таблицу, историю клиента или текстовый результат.

    Когда использовать:
    - пользователь прямо просит файл;
    - финальный результат должен быть доступен на диске;
    - нужно сохранить CSV после Python-анализа;
    - нужно вернуть пользователю абсолютный путь к созданному файлу.

    Когда не использовать:
    - для промежуточного artifact payload без файла;
    - для чтения или редактирования существующих файлов;
    - если содержимое файла еще не сформировано и не проверено.

    Правила:
    - передавай понятное имя файла с расширением;
    - для CSV передавай уже готовый CSV-текст с заголовком;
    - output_dir должен быть относительной папкой внутри проекта;
    - файл считается созданным только если результат содержит ok=true и
      absolute_path;
    - в финальном ответе указывай absolute_path, size_bytes и encoding.

    Args:
        file_name: Имя файла для сохранения. Допускается относительный путь внутри проекта.
        content: Текстовое содержимое файла.
        output_dir: Относительная папка внутри проекта. По умолчанию используется корень проекта.

    Returns:
        Словарь со статусом сохранения, абсолютным путем, размером файла и кодировкой.

    Raises:
        ValueError: Если путь пытается выйти за пределы рабочей директории проекта.
    """

    root = PROJECT_ROOT.resolve()
    target_dir = (root / output_dir).resolve()
    target_path = (target_dir / file_name).resolve()
    if not _is_relative_to(target_path, root):
        raise ValueError(f"File path escapes project workspace: {file_name}")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = "utf-8-sig" if target_path.suffix.lower() == ".csv" else "utf-8"
    target_path.write_text(content, encoding=encoding, newline="")
    return {
        "ok": True,
        "file_name": target_path.name,
        "absolute_path": str(target_path),
        "size_bytes": target_path.stat().st_size,
        "encoding": encoding,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Проверяет, что путь находится внутри указанной родительской директории.

    Args:
        path: Проверяемый абсолютный путь.
        parent: Абсолютный путь рабочей директории проекта.

    Returns:
        ``True``, если ``path`` находится внутри ``parent``, иначе ``False``.
    """

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def build_analytics_deep_agent(
    model: Any,
    embeddings_model: Any | None = None,
    settings: DeepAgentSettings | None = None,
    event_logger: DeepAgentEventLogger | None = None,
    data_tools: list[BaseTool] | None = None,
) -> Any:
    """Собирает DeepAgent supervisor для аналитики данных.

    Args:
        model: LangChain chat model, например модель из корневого ``model.py``.
        embeddings_model: LangChain embeddings model для few-shot поиска. Если ``None``, импортируется
            ``embeddings`` из корневого ``model.py``.
        settings: Настройки агента. Если ``None``, загружается JSON-конфиг по умолчанию.
        event_logger: Файловый логгер событий. Если ``None``, создается из ``settings``.
        data_tools: Инструменты чтения данных для ``data-retrieval-agent``. Если ``None``,
            используется локальный адаптер ``build_data_tools``.

    Returns:
        Скомпилированный DeepAgent graph с planning, subagents и HITL.
    """

    settings = settings or load_deep_agent_settings()
    event_logger = event_logger or build_deep_agent_logger(settings)
    if embeddings_model is None:
        from model import embeddings as embeddings_model

    if data_tools is None:
        data_tools = build_data_tools(settings)
    analysis_tools = [run_python_analysis, build_analysis_artifact, save_analysis_file]
    tools = [clarify_analysis_request]
    skills_context_middleware = PreloadedSkillsContextMiddleware(
        skills_root=settings.skills_root,
        skills_virtual_dir=settings.skills_virtual_dir,
        max_chars_per_file=settings.max_chars_per_skill,
        model=model,
        event_logger=event_logger,
    )
    update_few_shot_examples_index(
        examples_dir=settings.few_shot_examples_dir,
        index_dir=settings.few_shot_index_dir,
        embeddings=embeddings_model,
    )
    few_shot_examples_middleware = FewShotExamplesMiddleware(
        store=FewShotExamplesStore(index_dir=settings.few_shot_index_dir, embeddings=embeddings_model),
        model=model,
        top_k=settings.few_shot_top_k,
        max_examples=settings.few_shot_max_examples,
        event_logger=event_logger,
    )
    tool_call_trace_middleware = ToolTraceLoggingMiddleware(
        event_logger=event_logger,
        preview_chars=settings.trace_preview_chars,
        log_available_tools=settings.log_available_tools,
        log_model_tool_calls=settings.log_model_tool_calls,
        log_tool_execution=settings.log_tool_execution,
        log_tool_result=settings.log_tool_result,
        print_tool_calls=settings.print_tool_calls,
        print_tool_results=settings.print_tool_results,
    )
    prompt_debug_middleware = PromptDebugConsoleMiddleware(enabled=settings.print_plan_prompts)
    tool_output_file_middleware = ToolOutputFileMiddleware(
        output_dir=settings.tool_outputs_dir,
        min_rows_to_save=settings.tool_output_min_rows_to_save,
        min_content_chars_to_save=settings.tool_output_min_content_chars_to_save,
        preview_rows=settings.tool_output_preview_rows,
        inline_original_content_chars=settings.tool_output_inline_original_chars,
        event_logger=event_logger,
    )
    common_subagent_middleware = [skills_context_middleware, tool_call_trace_middleware, tool_output_file_middleware]
    subagents = build_analytics_subagent_specs(
        settings=settings,
        data_tools=data_tools,
        analysis_tools=analysis_tools,
        common_middleware=common_subagent_middleware,
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        subagents=subagents,
        skills=[settings.skills_virtual_dir],
        backend=build_skills_backend(settings),
        permissions=build_skills_permissions(settings),
        middleware=[
            skills_context_middleware,
            few_shot_examples_middleware,
            prompt_debug_middleware,
            tool_call_trace_middleware,
            tool_output_file_middleware,
            FirstPlanApprovalMiddleware(),
        ],
        interrupt_on=build_clarify_interrupt_config(),
        checkpointer=MemorySaver(),
    )


def get_skills_root(settings: DeepAgentSettings | None = None) -> Path:
    """Возвращает локальную папку проекта со skills.

    Args:
        settings: Настройки агента. Если ``None``, загружается JSON-конфиг по умолчанию.

    Returns:
        Абсолютный путь к папке ``skills`` в корне проекта.
    """

    settings = settings or load_deep_agent_settings()
    return settings.skills_root


def build_skills_backend(settings: DeepAgentSettings | None = None) -> Any:
    """Создает backend для чтения локальной папки ``skills`` через виртуальный путь ``/skills/``.

    Args:
        settings: Настройки агента. Если ``None``, загружается JSON-конфиг по умолчанию.

    Returns:
        ``CompositeBackend`` DeepAgents: scratch-файлы хранятся в state, а ``/skills/``
        маршрутизируется в локальную папку проекта ``skills``.
    """

    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

    settings = settings or load_deep_agent_settings()
    return CompositeBackend(
        default=StateBackend(),
        routes={
            settings.skills_virtual_dir: FilesystemBackend(
                root_dir=settings.skills_root,
                virtual_mode=True,
            )
        },
    )


def build_skills_permissions(settings: DeepAgentSettings | None = None) -> list[Any]:
    """Запрещает агенту изменять локальные skills через встроенные filesystem tools.

    Args:
        settings: Настройки агента. Если ``None``, загружается JSON-конфиг по умолчанию.

    Returns:
        Список permissions DeepAgents, запрещающий операции записи в ``/skills/**``.
    """

    from deepagents import FilesystemPermission

    settings = settings or load_deep_agent_settings()
    return [
        FilesystemPermission(
            operations=["write"],
            paths=[f"{settings.skills_virtual_dir}**"],
            mode="deny",
        )
    ]


def invoke_agent(agent: Any, message: str, thread_id: str) -> Any:
    """Отправляет сообщение пользователю в DeepAgent.

    Args:
        agent: Скомпилированный DeepAgent graph.
        message: Пользовательский вопрос по аналитике данных.
        thread_id: Идентификатор диалога для checkpointer и human-in-the-loop.

    Returns:
        Результат выполнения или interrupt payload, если агент ожидает решение пользователя.
    """

    return agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )


def resume_with_decision(agent: Any, thread_id: str, decision: dict[str, Any]) -> Any:
    """Продолжает выполнение после human-in-the-loop interrupt.

    Args:
        agent: Скомпилированный DeepAgent graph.
        thread_id: Тот же идентификатор диалога, на котором возник interrupt.
        decision: Решение пользователя: ``approve``, ``edit`` или ``reject``.

    Returns:
        Следующий результат выполнения агента.
    """

    return agent.invoke(
        Command(resume={"decisions": [decision]}),
        config={"configurable": {"thread_id": thread_id}},
    )


def resume_with_user_answer(agent: Any, thread_id: str, answer: str) -> Any:
    """Продолжает выполнение после уточняющего вопроса агента.

    Args:
        agent: Скомпилированный DeepAgent graph.
        thread_id: Тот же идентификатор диалога, на котором возник interrupt.
        answer: Ответ пользователя с описанием задачи, формата и ожидаемого результата.

    Returns:
        Следующий результат выполнения агента после передачи ответа пользователя.
    """

    return resume_with_decision(
        agent=agent,
        thread_id=thread_id,
        decision={"type": "respond", "message": answer},
    )


__all__ = [
    "build_skills_backend",
    "build_skills_permissions",
    "build_analytics_deep_agent",
    "build_data_tools",
    "build_read_table_description",
    "clarify_analysis_request",
    "build_analysis_artifact",
    "get_skills_root",
    "invoke_agent",
    "resume_with_decision",
    "resume_with_user_answer",
    "run_python_analysis",
    "save_analysis_file",
]

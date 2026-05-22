"""Классификация ошибок инструментов по типу исключения (без разбора текста сообщения).

Содержит:
- ErrorHintRule: правило подсказок для одного или нескольких типов исключений.
- tool_error_possible_causes: вероятные причины по ``error_type``.
- tool_error_solution_options: варианты исправления по ``error_type`` и имени tool.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tools.execute_python_code_tool import EXECUTE_PYTHON_CODE_TOOL_NAME


@dataclass(frozen=True, slots=True)
class ErrorHintRule:
    """Правило подсказок для группы типов исключений."""

    exception_types: frozenset[str]
    causes: tuple[str, ...]
    solutions: tuple[str, ...]


_DEFAULT_CAUSE = (
    "Инструмент получил некорректные аргументы или столкнулся с внутренней ошибкой выполнения.",
)

_BASE_SOLUTIONS: tuple[str, ...] = (
    "Проверь точные имена доступных переменных, файлов и artifacts перед повтором.",
    "Повтори вызов только после изменения аргументов на основе сообщения об ошибке.",
)

_ERROR_HINT_RULES: tuple[ErrorHintRule, ...] = (
    ErrorHintRule(
        exception_types=frozenset({"FileNotFoundError", "NotFoundError"}),
        causes=("Указанный файл, artifact, таблица или переменная не найдены.",),
        solutions=(
            "Вызови list/get инструмент для доступных файлов, таблиц или artifacts "
            "и выбери существующее имя.",
        ),
    ),
    ErrorHintRule(
        exception_types=frozenset({"PermissionError"}),
        causes=(
            "Запрошенный путь недоступен из текущего workspace или запрещен политикой доступа.",
        ),
        solutions=(
            "Используй путь внутри разрешенного workspace или sources/contexts директории.",
        ),
    ),
    ErrorHintRule(
        exception_types=frozenset({"KeyError"}),
        causes=("В аргументах или данных отсутствует обязательное поле/колонка/ключ.",),
        solutions=(),
    ),
    ErrorHintRule(
        exception_types=frozenset({"ValueError", "TypeError"}),
        causes=("Один из аргументов имеет недопустимое значение или формат.",),
        solutions=("Используй поддерживаемый формат или другой инструмент для этого типа данных.",),
    ),
    ErrorHintRule(
        exception_types=frozenset({"NotImplementedError", "UnsupportedOperation"}),
        causes=("Формат входных данных или файла не поддерживается этим инструментом.",),
        solutions=("Используй поддерживаемый формат или другой инструмент для этого типа данных.",),
    ),
)


def tool_error_possible_causes(error_type: str) -> list[str]:
    """Возвращает вероятные причины ошибки инструмента по типу исключения.

    Args:
        error_type: Имя класса исключения (``exc.__class__.__name__``).

    Returns:
        Список человекочитаемых причин для LLM.
    """

    causes: list[str] = []
    for rule in _ERROR_HINT_RULES:
        if error_type in rule.exception_types:
            causes.extend(rule.causes)
    if not causes:
        causes.append(_DEFAULT_CAUSE)
    return causes


def tool_error_solution_options(tool_name: str, error_type: str) -> list[str]:
    """Возвращает варианты исправления ошибки инструмента.

    Args:
        tool_name: Имя инструмента.
        error_type: Имя класса исключения.

    Returns:
        Список практических действий для следующего шага модели.
    """

    options = list(_BASE_SOLUTIONS)
    for rule in _ERROR_HINT_RULES:
        if error_type in rule.exception_types:
            options.extend(rule.solutions)
    if tool_name == EXECUTE_PYTHON_CODE_TOOL_NAME:
        options.append(
            "Исправь код и повтори execute_python_code. "
            "target_variable указывай только если нужна именованная переменная; "
            "для print-вывода читай execution_output."
        )
    else:
        options.append(
            "Если инструмент не подходит под задачу, выбери другой доступный tool из prompt."
        )
    return options

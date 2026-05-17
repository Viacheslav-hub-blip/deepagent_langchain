"""Текстовые результаты инструментов с внутренними служебными признаками.

Содержит:
- ToolTextResult: строковый результат tool с внутренним признаком ошибки.
- is_tool_error_result: проверка внутреннего признака ошибки без анализа текста.
"""

from __future__ import annotations

from typing import Any


class ToolTextResult(str):
    """Строковый результат инструмента с внутренними метаданными.

    Args:
        text: Текст, который должен увидеть агент или пользователь.
        is_error: Внутренний признак того, что результат является ошибкой tool.

    Returns:
        Строка, совместимая с обычным ``str``, с дополнительным атрибутом
        ``is_tool_error`` для runtime-логики.
    """

    is_tool_error: bool

    def __new__(cls, text: str, *, is_error: bool = False) -> "ToolTextResult":
        """Создает строковый результат с внутренним признаком ошибки.

        Args:
            text: Текст результата инструмента.
            is_error: Нужно ли считать результат ошибкой выполнения tool.

        Returns:
            Экземпляр ``ToolTextResult``.
        """

        value = super().__new__(cls, text)
        value.is_tool_error = is_error
        return value


def is_tool_error_result(value: Any) -> bool:
    """Проверяет, помечен ли результат инструмента как ошибка.

    Args:
        value: Произвольный результат tool или wrapper-а.

    Returns:
        ``True``, если объект несет внутренний признак ошибки tool.
    """

    return bool(getattr(value, "is_tool_error", False))

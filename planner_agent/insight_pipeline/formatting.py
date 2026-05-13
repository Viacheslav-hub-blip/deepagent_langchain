"""Форматирование строк DataFrame в текстовые запросы агенту.

Содержит:
- parse_complex_string: разбирает JSON-like строки.
- format_complex_value: форматирует сложные значения в читаемый текст.
- format_case_row_to_text: превращает одну строку DataFrame в текстовый блок.
- build_case_prompt: формирует полный prompt для single-case агента.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd


def parse_complex_string(value: Any) -> Any:
    """Пробует разобрать строку, если она похожа на JSON, список или словарь.

    Args:
        value: Произвольное значение из строки DataFrame.

    Returns:
        Исходное значение или разобранный Python-объект.
    """

    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except Exception:
            try:
                return ast.literal_eval(stripped)
            except Exception:
                return value
    return value


def format_complex_value(
    value: Any,
    *,
    level: int = 0,
    max_string_length: int | None = None,
) -> str:
    """Форматирует сложное значение в читаемый многострочный текст.

    Args:
        value: Значение ячейки DataFrame.
        level: Уровень вложенности для отступов.
        max_string_length: Максимальная длина строкового значения. `None` отключает обрезку.

    Returns:
        Строковое представление значения для prompt-а агента.
    """

    indent = "  " * level
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return f"{indent}Значение отсутствует"
    if isinstance(value, (datetime, pd.Timestamp)):
        return f"{indent}{value.strftime('%Y-%m-%d %H:%M:%S')}"
    if isinstance(value, dict):
        if not value:
            return f"{indent}{{}}"
        lines = [f"{indent}{{"]
        for key, nested_value in value.items():
            nested_text = format_complex_value(
                nested_value,
                level=level + 1,
                max_string_length=max_string_length,
            )
            lines.append(f"{indent}  {key}: {nested_text.lstrip()}")
        lines.append(f"{indent}}}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return f"{indent}[]"
        lines = [f"{indent}["]
        for item in value:
            item_text = format_complex_value(
                item,
                level=level + 1,
                max_string_length=max_string_length,
            )
            lines.append(f"{indent}  {item_text.lstrip()}")
        lines.append(f"{indent}]")
        return "\n".join(lines)
    if isinstance(value, str):
        parsed = parse_complex_string(value)
        if parsed != value:
            return format_complex_value(
                parsed,
                level=level,
                max_string_length=max_string_length,
            )
        text = (
            value
            if max_string_length is None or len(value) <= max_string_length
            else f"{value[:max_string_length]}..."
        )
        return f'{indent}"{text}"'
    return f"{indent}{str(value)}"


def format_case_row_to_text(
    row: Mapping[str, Any],
    *,
    row_index: int | None = None,
    max_string_length: int | None = None,
) -> str:
    """Преобразует одну строку с контекстом кейса в читаемый текстовый блок.

    Args:
        row: Словарь или pandas Series с полями кейса.
        row_index: Индекс строки, если его нужно показать в тексте.
        max_string_length: Максимальная длина строковых значений.

    Returns:
        Текстовое представление одного кейса.
    """

    title = f"Строка №{row_index}" if row_index is not None else "Строка кейса"
    row_lines = ["=" * 100, title, "=" * 100]
    for column, value in row.items():
        row_lines.append(f"\n{column}:")
        formatted = format_complex_value(value, max_string_length=max_string_length)
        row_lines.extend(f"  {line}" for line in formatted.split("\n"))
    row_lines.append("=" * 100)
    return "\n".join(row_lines)


def build_case_prompt(
    row: Mapping[str, Any],
    *,
    group_name: str,
    row_index: int,
    max_string_length: int | None = None,
) -> str:
    """Формирует полный текстовый запрос агенту по одной записи.

    Args:
        row: Данные кейса в формате словаря или pandas Series.
        group_name: Название группы после кластеризации.
        row_index: Индекс строки в DataFrame.
        max_string_length: Максимальная длина строковых значений.

    Returns:
        Готовый prompt, который можно передать в `ResearchAgent.invoke`.
    """

    case_text = format_case_row_to_text(
        row,
        row_index=row_index,
        max_string_length=max_string_length,
    )
    return f"""
Разбери антифрод-сработку по базовому контексту ниже.

Предварительная группа проблемы: {group_name}

Задача:
1. Определи, достаточно ли текущего набора данных для объяснения кейса.
2. Если данных достаточно, опиши, что произошло, какие факты это подтверждают и какие есть ограничения.
3. Если данных не хватает, перечисли, какие данные нужно дополнительно загрузить, по каким ключам, за какой период и зачем.
4. Не выдумывай факты. Разделяй подтвержденные факты, гипотезы и ограничения.
5. В конце верни структурированный JSON-блок в тегах <structured_result>...</structured_result>.

Ожидаемый JSON внутри structured_result:
{{
  "case_id": "строка",
  "event_id": "строка или null",
  "group_name": "строка",
  "facts": ["подтвержденные факты"],
  "hypotheses": ["гипотезы"],
  "missing_data_requests": [
    {{
      "source_name": "источник",
      "reason": "зачем нужны данные",
      "lookup_keys": {{"event_id": "...", "epk_id": "..."}},
      "period": "период или null",
      "priority": "low|medium|high"
    }}
  ],
  "limitations": ["ограничения"],
  "final_summary": "краткий итог"
}}

Базовый контекст:
{case_text}
""".strip()

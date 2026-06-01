"""Обёртка инструментов чтения данных: прозрачность того, что именно выполнено.

Зачем нужна обёртка:
- агент должен понимать, ЧТО произошло при чтении: какой запрос фактически выполнен и
  сколько строк под него подошло (а не только видеть усечённый repr таблицы);
- агент часто путает «выборку строк» с «уникальными значениями» и принимает сэмпл за
  полный справочник.

Что делает обёртка для каждого data-tool (например ``read_table``):
1. строит человекочитаемый SQL-подобный код фактического запроса из аргументов вызова;
2. вызывает исходный инструмент;
3. при табличном результате возвращает агенту вместе с данными:
   - сгенерированный код запроса (строкой);
   - счётчики строк: всего в таблице / подошло под фильтры / возвращено;
   - данные в виде JSON-записей (для офлоада большие результаты заменяются ссылкой на pkl).

Обёртка использует ``response_format="content_and_artifact"``: ``content`` — текст для
модели (код запроса + счётчики + данные), ``artifact`` — структура с ``rows`` и метаданными,
которую читает ``ToolOutputFileMiddleware`` для офлоада больших таблиц.

Сам базовый инструмент не меняется: вся прозрачность добавляется здесь, в слое
``deep_agent_test``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

try:  # pandas нужен только для распознавания DataFrame-результата.
    import pandas as pd
except Exception:  # pragma: no cover - pandas всегда есть в проекте, но обёртка не падает без него.
    pd = None  # type: ignore[assignment]


def wrap_data_tools_with_query_code(tools: list[BaseTool]) -> list[BaseTool]:
    """Оборачивает инструменты чтения данных в слой прозрачности запроса.

    Args:
        tools: Базовые data-tools (например результат ``build_spark_data_tools``).

    Returns:
        Список инструментов с тем же ``name``/``description``/``args_schema``, но с
        ``response_format="content_and_artifact"`` и добавленным кодом запроса и счётчиками.
    """

    return [_wrap_single_tool(tool) for tool in tools]


def _wrap_single_tool(tool: BaseTool) -> BaseTool:
    """Создаёт обёртку над одним инструментом чтения данных."""

    description = f"{tool.description}\n\n{_TRANSPARENCY_NOTE}"

    async def _arun(**kwargs: Any) -> tuple[str, Any]:
        """Асинхронно вызывает базовый tool и оборачивает результат в (content, artifact)."""

        raw = await tool.ainvoke(kwargs)
        return _format_result(kwargs, raw)

    def _run(**kwargs: Any) -> tuple[str, Any]:
        """Синхронно вызывает базовый tool и оборачивает результат в (content, artifact)."""

        raw = tool.invoke(kwargs)
        return _format_result(kwargs, raw)

    factory_kwargs: dict[str, Any] = {
        "name": tool.name,
        "description": description,
        "args_schema": tool.args_schema,
        "response_format": "content_and_artifact",
    }
    if getattr(tool, "func", None) is not None:
        factory_kwargs["func"] = _run
    if getattr(tool, "coroutine", None) is not None:
        factory_kwargs["coroutine"] = _arun
    if "func" not in factory_kwargs and "coroutine" not in factory_kwargs:
        # Базовый инструмент без явного func/coroutine — поддержим хотя бы async-путь.
        factory_kwargs["coroutine"] = _arun
    return StructuredTool.from_function(**factory_kwargs)


_TRANSPARENCY_NOTE = (
    "Прозрачность результата: вместе с данными инструмент возвращает сгенерированный "
    "SQL-подобный код фактического запроса и число строк результата. Инструмент возвращает "
    "ВСЕ строки, попавшие под запрос: полный набор всегда сохраняется в pickle для "
    "переиспользования без повторного read_table; если результат помещается в контекст, "
    "он также приходит inline, если большой — в контекст только preview, а полный набор в "
    "файле. "
    "Обычный select возвращает строки как есть, БЕЗ устранения дублей: "
    "уникальные значения выводи сам по полному набору."
)


def _format_result(kwargs: dict[str, Any], raw: Any) -> tuple[str, Any]:
    """Готовит пару (content, artifact) для агента из результата базового инструмента."""

    query_code = _build_query_code(kwargs)

    if pd is not None and isinstance(raw, pd.DataFrame):
        rows = _dataframe_to_rows(raw)
        returned_rows = len(raw)
        columns = [str(column) for column in raw.columns]
        is_aggregation = bool(_split_instruction_text(_get(kwargs, "aggregations")))
        content = _build_success_content(
            query_code=query_code,
            returned_rows=returned_rows,
            columns=columns,
            rows=rows,
            is_aggregation=is_aggregation,
        )
        artifact = {
            "rows": rows,
            "query_code": query_code,
            "returned_rows": returned_rows,
            "columns": columns,
            "is_aggregation": is_aggregation,
        }
        return content, artifact

    # Ошибка или текстовый результат базового инструмента: показываем, что пытались выполнить.
    content = f"Сгенерированный SQL-подобный запрос:\n{query_code}\n\nРезультат инструмента:\n{raw}"
    return content, None


def _build_success_content(
    *,
    query_code: str,
    returned_rows: int,
    columns: list[str],
    rows: list[dict[str, Any]],
    is_aggregation: bool = False,
) -> str:
    """Строит текст ответа для модели при успешной выборке.

    Намеренно НЕ сообщаем «всего в таблице» и «подошло под фильтры» — эти числа из
    исходной таблицы только путают модель. Этот текст — инлайн-результат, целиком
    переданный в контекст; если результат большой, его заменит summary offload-middleware
    с пометкой, что в контексте лишь preview, а полный набор лежит в файле.
    """

    if is_aggregation:
        status_line = (
            f"Это ПОЛНЫЙ результат запроса в контексте: {returned_rows} групп "
            "(уникальных значений) — все строки результата здесь, это не сэмпл."
        )
    else:
        status_line = (
            f"Это ПОЛНЫЙ результат запроса в контексте: {returned_rows} строк — "
            "все строки результата переданы здесь целиком."
        )
    rows_json = json.dumps(rows, ensure_ascii=False, default=str)
    return (
        "Сгенерированный SQL-подобный запрос:\n"
        f"{query_code}\n\n"
        f"{status_line}\n"
        f"Колонки результата ({len(columns)}): {', '.join(columns)}.\n"
        "Данные (JSON records):\n"
        f"{rows_json}"
    )


def _dataframe_to_rows(frame: Any) -> list[dict[str, Any]]:
    """Конвертирует DataFrame в список JSON-совместимых записей."""

    records = frame.to_dict(orient="records")
    cleaned: list[dict[str, Any]] = []
    for record in records:
        cleaned.append({str(key): _clean_scalar(value) for key, value in record.items()})
    return cleaned


def _clean_scalar(value: Any) -> Any:
    """Приводит pandas/numpy скаляр к JSON-совместимому типу."""

    if pd is not None:
        try:
            if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _build_query_code(kwargs: dict[str, Any]) -> str:
    """Строит человекочитаемый SQL-подобный код фактического запроса из аргументов вызова."""

    table_name = _get(kwargs, "table_name") or "<table>"
    select_columns = _parse_columns(_get(kwargs, "select_columns"))
    group_by = _parse_columns(_get(kwargs, "group_by"))
    aggregations = _split_instruction_text(_get(kwargs, "aggregations"))
    filters = _get(kwargs, "filters")
    derived_columns = _split_instruction_text(_get(kwargs, "derived_columns"))
    order_by = _split_instruction_text(_get(kwargs, "order_by"))
    max_rows = _get(kwargs, "max_rows")

    if aggregations:
        select_parts = [*group_by]
        select_parts.extend(aggregations)
        select_clause = ", ".join(select_parts) or "*"
    else:
        select_clause = ", ".join(select_columns) or "*"

    lines = [f"SELECT {select_clause}", f"FROM {table_name}"]

    for derived in derived_columns:
        lines.append(f"-- derived: {derived}")

    where_clause = _build_where_clause(filters)
    if where_clause:
        lines.append(f"WHERE {where_clause}")
    if group_by:
        lines.append(f"GROUP BY {', '.join(group_by)}")
    if order_by:
        lines.append(f"ORDER BY {', '.join(order_by)}")
    if isinstance(max_rows, int):
        lines.append(f"LIMIT {max_rows}")
    return "\n".join(lines)


def _build_where_clause(filters: Any) -> str:
    """Преобразует строковые фильтры в SQL-подобное условие WHERE."""

    predicates = [_build_predicate(filter_item) for filter_item in _split_instruction_text(filters)]
    return " AND ".join(predicate for predicate in predicates if predicate)


def _build_predicate(filter_item: Any) -> str:
    """Строит один SQL-подобный предикат из фильтра."""

    if isinstance(filter_item, str):
        return filter_item

    column = _get(filter_item, "column")
    operator = _get(filter_item, "operator") or "eq"
    value = _get(filter_item, "value")
    values = _as_list(_get(filter_item, "values"))

    if operator == "is_null":
        return f"{column} IS NULL"
    if operator == "not_null":
        return f"{column} IS NOT NULL"
    if operator == "in":
        return f"{column} IN ({', '.join(_literal(item) for item in values)})"
    if operator == "between" and len(values) == 2:
        return f"{column} BETWEEN {_literal(values[0])} AND {_literal(values[1])}"
    if operator == "contains":
        return f"{column} LIKE {_literal(f'%{value}%')}"
    sql_operator = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}.get(operator, operator)
    return f"{column} {sql_operator} {_literal(value)}"


def _literal(value: Any) -> str:
    """Возвращает SQL-литерал для значения фильтра."""

    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _parse_columns(value: Any) -> list[str]:
    """Разбирает строку 'col1, col2' в список имён колонок."""

    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _split_instruction_text(value: Any) -> list[str]:
    """Разбирает строку инструкций через ``;`` или перенос строки."""

    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    normalized = str(value).replace("\n", ";")
    return [part.strip() for part in normalized.split(";") if part.strip()]


def _as_list(value: Any) -> list[Any]:
    """Возвращает значение как список (пустой, если None)."""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get(source: Any, key: str) -> Any:
    """Достаёт поле из dict или pydantic-объекта по имени."""

    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


__all__ = ["wrap_data_tools_with_query_code"]

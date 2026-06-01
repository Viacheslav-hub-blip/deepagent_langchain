"""Production-инструмент ``load_data`` поверх общей Spark session.

Содержит:
- ReadTableInput: строковая схема аргументов инструмента ``load_data``.
- build_spark_data_tools: сборка LangChain tool поверх готовой Spark session.
- _read_table: выполнение выборки через Spark DataFrame API.
- _apply_derived_columns: добавление вычисляемых колонок.
- _build_derived_column: построение одной вычисляемой колонки.
- _apply_filters: применение строковых фильтров.
- _build_filter_expression: построение одного Spark-предиката.
- _apply_aggregations: применение агрегатов.
- _build_aggregation_expression: построение одного Spark-агрегата.
- _apply_order_by: сортировка результата.
- _parse_columns: разбор строки колонок.
- _split_items: разбор строки со списком инструкций.
- _parse_filter_item: разбор одного фильтра.
- _parse_derived_item: разбор одной вычисляемой колонки.
- _parse_aggregation_item: разбор одного агрегата.
- _parse_order_item: разбор одной сортировки.
- _parse_scalar: приведение строкового значения к простому типу.
- _validate_columns: проверка наличия колонок в DataFrame.
- _format_missing_columns: человекочитаемая ошибка по отсутствующим колонкам.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

READ_TABLE_DESCRIPTION = (
    "load_data\n"
    "---\n"
    "Описание: универсальная выборка из Spark-таблиц. "
    "Инструмент принимает имя таблицы, строку со списком полей, строковые фильтры, "
    "вычисляемые колонки, группировки, агрегации, сортировку и лимит строк. "
    "При успешной выборке возвращает pandas DataFrame.\n"
    "Выгрузка всех столбцов запрещена: агент должен явно указать минимально "
    "достаточный набор колонок.\n\n"
    "Параметры:\n"
    "  table_name (str, обяз.) - имя Spark-таблицы или view.\n"
    "  select_columns (str, обяз.) - поля результата: 'col1, col2, col3'. "
    "Пустая строка, '*' и 'all' запрещены, если нет aggregations.\n"
    "  filters (str, опц.) - фильтры через ';' или перенос строки. "
    "Формат одного фильтра: 'column operator value'. "
    "Пример: 'epk_id eq 123; event_dt in 20260123,20260124'. "
    "Операторы: eq, ne, gt, gte, lt, lte, contains, in, between, is_null, not_null.\n"
    "  derived_columns (str, опц.) - вычисляемые поля через ';'. "
    "Формат: 'new_col = operation(source_col)'. Операции: year, month, "
    "year_month, date, lower, upper, length, abs.\n"
    "  group_by (str, опц.) - поля группировки: 'col1, col2'.\n"
    "  aggregations (str, опц.) - агрегаты через ';'. "
    "Формат: 'function(column) as alias'. Функции: count, count_distinct, "
    "min, max, sum, mean.\n"
    "  order_by (str, опц.) - сортировка через ';'. Формат: 'column asc'.\n"
    "  max_rows (int, опц.) - максимум строк в ответе; если не передан, лимит не применяется.\n"
    "  include_schema (bool, опц.) - вернуть схему результата в metadata."
)

_FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "between", "is_null", "not_null"}
_DERIVED_OPERATIONS = {"year", "month", "year_month", "date", "lower", "upper", "length", "abs"}
_AGGREGATION_FUNCTIONS = {"count", "count_distinct", "min", "max", "sum", "mean"}


class ReadTableInput(BaseModel):
    """Строковые аргументы инструмента чтения таблиц из Spark.

    Args:
        table_name: Имя таблицы Spark или view, доступной через ``spark.table``.
        select_columns: Поля результата в формате ``col1, col2``.
        filters: Фильтры одной строкой через ``;`` или перенос строки.
        derived_columns: Вычисляемые колонки одной строкой через ``;``.
        group_by: Поля группировки в формате ``col1, col2``.
        aggregations: Агрегаты одной строкой через ``;``.
        order_by: Сортировка одной строкой через ``;``.
        max_rows: Максимальное число строк, которое нужно вернуть.
        include_schema: Нужно ли приложить схему результата в metadata DataFrame.

    Returns:
        Валидированные строковые параметры для ``load_data``.
    """

    table_name: str = Field(description="Имя Spark-таблицы или view, например csp_afpc_sss_inc.uko_event.")
    select_columns: str = Field(
        default="",
        description="Поля результата через запятую: 'event_id, event_dt, epk_id'. Не используй '*' и 'all'.",
    )
    filters: str = Field(
        default="",
        description="Фильтры через ';' или перенос строки. Пример: 'epk_id eq 123; event_dt in 20260123,20260124'.",
    )
    derived_columns: str = Field(
        default="",
        description="Вычисляемые поля через ';'. Пример: 'event_month = year_month(event_dt)'.",
    )
    group_by: str = Field(default="", description="Поля группировки через запятую.")
    aggregations: str = Field(
        default="",
        description="Агрегаты через ';'. Пример: 'count(event_id) as events_count; sum(amount) as total_amount'.",
    )
    order_by: str = Field(default="", description="Сортировка через ';'. Пример: 'event_dt asc; event_time desc'.")
    max_rows: int | None = Field(default=None, ge=0, description="Максимальное число строк результата.")
    include_schema: bool = Field(default=False, description="Если True, добавить схему результата в metadata.")


def build_spark_data_tools(spark: Any) -> list[BaseTool]:
    """Создает инструмент ``load_data`` поверх готовой Spark session.

    Args:
        spark: Активная ``pyspark.sql.SparkSession``, созданная один раз при старте приложения.

    Returns:
        Список с одним LangChain tool ``load_data``.
    """

    def read_table(
        table_name: str,
        select_columns: str = "",
        filters: str = "",
        derived_columns: str = "",
        group_by: str = "",
        aggregations: str = "",
        order_by: str = "",
        max_rows: int | None = None,
        include_schema: bool = False,
    ) -> Any:
        """Выполняет выборку из Spark-таблицы через переданную Spark session.

        Args:
            table_name: Имя Spark-таблицы или view.
            select_columns: Поля результата в формате ``col1, col2``.
            filters: Фильтры одной строкой через ``;`` или перенос строки.
            derived_columns: Вычисляемые колонки одной строкой через ``;``.
            group_by: Поля группировки в формате ``col1, col2``.
            aggregations: Агрегаты одной строкой через ``;``.
            order_by: Сортировка одной строкой через ``;``.
            max_rows: Максимальное число строк результата.
            include_schema: Нужно ли приложить схему результата в metadata DataFrame.

        Returns:
            pandas DataFrame с результатом или текст ошибки, который агент может исправить.
        """

        return _read_table(
            spark=spark,
            table_name=table_name,
            select_columns=select_columns,
            filters=filters,
            derived_columns=derived_columns,
            group_by=group_by,
            aggregations=aggregations,
            order_by=order_by,
            max_rows=max_rows,
            include_schema=include_schema,
        )

    return [
        StructuredTool.from_function(
            func=read_table,
            name="load_data",
            description=READ_TABLE_DESCRIPTION,
            args_schema=ReadTableInput,
        )
    ]


def _read_table(
    *,
    spark: Any,
    table_name: str,
    select_columns: str,
    filters: str,
    derived_columns: str,
    group_by: str,
    aggregations: str,
    order_by: str,
    max_rows: int | None,
    include_schema: bool,
) -> Any:
    """Выполняет Spark-запрос и возвращает pandas DataFrame.

    Args:
        spark: Активная Spark session.
        table_name: Имя таблицы Spark или view.
        select_columns: Поля результата строкой.
        filters: Фильтры строкой.
        derived_columns: Вычисляемые колонки строкой.
        group_by: Поля группировки строкой.
        aggregations: Агрегаты строкой.
        order_by: Сортировка строкой.
        max_rows: Максимальное число строк результата.
        include_schema: Нужно ли приложить схему результата.

    Returns:
        pandas DataFrame с metadata в ``attrs`` или текст ошибки.
    """

    try:
        table = spark.table(table_name.strip())
        total_rows = table.count()
        table = _apply_derived_columns(table=table, derived_columns=derived_columns)
        table = _apply_filters(table=table, filters=filters)
        matched_rows = table.count()

        group_columns = _parse_columns(group_by)
        aggregation_items = _split_items(aggregations)
        if aggregation_items:
            result = _apply_aggregations(table=table, group_columns=group_columns, aggregations=aggregation_items)
        else:
            columns = _parse_columns(select_columns)
            select_error = _validate_columns(columns=columns, available_columns=table.columns, allow_empty=False)
            if select_error:
                return select_error
            result = table.select(*columns)

        order_items = _split_items(order_by)
        if order_items:
            order_error = _validate_columns(
                columns=[_parse_order_item(item)[0] for item in order_items],
                available_columns=result.columns,
                allow_empty=True,
            )
            if order_error:
                return order_error
            result = _apply_order_by(table=result, order_by=order_items)

        if max_rows is not None:
            result = result.limit(max(0, int(max_rows)))

        frame = result.toPandas()
        frame.attrs["spark_table_name"] = table_name.strip()
        frame.attrs["spark_source_file"] = table_name.strip()
        frame.attrs["spark_total_rows"] = int(total_rows)
        frame.attrs["spark_matched_rows"] = int(matched_rows)
        if include_schema:
            frame.attrs["spark_schema"] = {
                "table_name": table_name.strip(),
                "columns_count": len(result.columns),
                "columns": [{"name": name, "type": str(dtype)} for name, dtype in result.dtypes],
            }
        return frame
    except ValueError as exc:
        return f"Ошибка load_data: {exc}"


def _apply_derived_columns(*, table: Any, derived_columns: str) -> Any:
    """Добавляет вычисляемые колонки к Spark DataFrame.

    Args:
        table: Исходный Spark DataFrame.
        derived_columns: Описания вычисляемых колонок строкой.

    Returns:
        Spark DataFrame с добавленными колонками.
    """

    result = table
    for item in _split_items(derived_columns):
        name, source_column, operation = _parse_derived_item(item)
        missing = _validate_columns(columns=[source_column], available_columns=result.columns, allow_empty=False)
        if missing:
            raise ValueError(missing)
        result = result.withColumn(name, _build_derived_column(source_column=source_column, operation=operation))
    return result


def _build_derived_column(*, source_column: str, operation: str) -> Any:
    """Строит выражение Spark Column для вычисляемой колонки.

    Args:
        source_column: Исходная колонка.
        operation: Имя операции.

    Returns:
        Spark Column с вычисленным значением.
    """

    from pyspark.sql import functions as functions

    source = functions.col(source_column)
    if operation == "lower":
        return functions.lower(source.cast("string"))
    if operation == "upper":
        return functions.upper(source.cast("string"))
    if operation == "length":
        return functions.length(source.cast("string"))
    if operation == "abs":
        return functions.abs(source.cast("double"))

    digits = functions.regexp_replace(source.cast("string"), r"\D", "")
    if operation == "year":
        return digits.substr(1, 4)
    if operation == "month":
        return digits.substr(5, 2)
    if operation == "year_month":
        return digits.substr(1, 6)
    if operation == "date":
        return digits.substr(1, 8)
    raise ValueError(f"Неподдерживаемая операция вычисляемой колонки: {operation}")


def _apply_filters(*, table: Any, filters: str) -> Any:
    """Применяет строковые фильтры к Spark DataFrame.

    Args:
        table: Исходный Spark DataFrame.
        filters: Фильтры одной строкой.

    Returns:
        Отфильтрованный Spark DataFrame.
    """

    result = table
    for item in _split_items(filters):
        column, _, _ = _parse_filter_item(item)
        missing = _validate_columns(columns=[column], available_columns=result.columns, allow_empty=False)
        if missing:
            raise ValueError(missing)
        result = result.filter(_build_filter_expression(item))
    return result


def _build_filter_expression(item: str) -> Any:
    """Строит Spark Column-предикат из одного строкового фильтра.

    Args:
        item: Один фильтр в формате ``column operator value``.

    Returns:
        Spark Column с булевым условием.
    """

    from pyspark.sql import functions as functions

    column, operator, raw_value = _parse_filter_item(item)
    spark_column = functions.col(column)
    if operator == "is_null":
        return spark_column.isNull()
    if operator == "not_null":
        return spark_column.isNotNull()
    if operator == "contains":
        return spark_column.cast("string").contains(raw_value)
    if operator == "in":
        return spark_column.isin([_parse_scalar(value) for value in raw_value.split(",") if value.strip()])
    if operator == "between":
        values = [_parse_scalar(value) for value in raw_value.split(",") if value.strip()]
        if len(values) != 2:
            raise ValueError("Для оператора between нужны два значения через запятую.")
        return spark_column.between(values[0], values[1])

    value = _parse_scalar(raw_value)
    if operator == "eq":
        return spark_column == value
    if operator == "ne":
        return spark_column != value
    if operator == "gt":
        return spark_column > value
    if operator == "gte":
        return spark_column >= value
    if operator == "lt":
        return spark_column < value
    if operator == "lte":
        return spark_column <= value
    raise ValueError(f"Неподдерживаемый оператор фильтра: {operator}")


def _apply_aggregations(*, table: Any, group_columns: list[str], aggregations: list[str]) -> Any:
    """Применяет агрегаты к Spark DataFrame.

    Args:
        table: Отфильтрованный Spark DataFrame.
        group_columns: Поля группировки.
        aggregations: Строковые описания агрегатов.

    Returns:
        Spark DataFrame с результатом агрегаций.
    """

    missing = _validate_columns(
        columns=[*group_columns, *[_parse_aggregation_item(item)[1] for item in aggregations]],
        available_columns=table.columns,
        allow_empty=True,
    )
    if missing:
        raise ValueError(missing)

    expressions = [_build_aggregation_expression(item) for item in aggregations]
    if group_columns:
        return table.groupBy(*group_columns).agg(*expressions)
    return table.agg(*expressions)


def _build_aggregation_expression(item: str) -> Any:
    """Строит Spark Column для одного агрегата.

    Args:
        item: Агрегат в формате ``function(column) as alias``.

    Returns:
        Spark Column с alias.
    """

    from pyspark.sql import functions as functions

    function, column, alias = _parse_aggregation_item(item)
    if function == "count":
        expression = functions.count(functions.col(column))
    elif function == "count_distinct":
        expression = functions.countDistinct(functions.col(column))
    elif function == "min":
        expression = functions.min(functions.col(column))
    elif function == "max":
        expression = functions.max(functions.col(column))
    elif function == "sum":
        expression = functions.sum(functions.col(column))
    elif function == "mean":
        expression = functions.avg(functions.col(column))
    else:
        raise ValueError(f"Неподдерживаемая агрегатная функция: {function}")
    return expression.alias(alias or f"{function}_{column}")


def _apply_order_by(*, table: Any, order_by: list[str]) -> Any:
    """Сортирует Spark DataFrame.

    Args:
        table: Spark DataFrame результата.
        order_by: Строковые правила сортировки.

    Returns:
        Отсортированный Spark DataFrame.
    """

    from pyspark.sql import functions as functions

    expressions = []
    for item in order_by:
        column, direction = _parse_order_item(item)
        expression = functions.col(column).asc() if direction == "asc" else functions.col(column).desc()
        expressions.append(expression)
    return table.orderBy(*expressions)


def _parse_columns(value: str | None) -> list[str]:
    """Разбирает строку колонок через запятую.

    Args:
        value: Строка вида ``col1, col2``.

    Returns:
        Список колонок без пустых значений.
    """

    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _split_items(value: str | None) -> list[str]:
    """Разбирает строку инструкций через ``;`` или перенос строки.

    Args:
        value: Строка с несколькими инструкциями.

    Returns:
        Список непустых инструкций.
    """

    if not value:
        return []
    normalized = str(value).replace("\n", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def _parse_filter_item(item: str) -> tuple[str, str, str]:
    """Разбирает один строковый фильтр.

    Args:
        item: Фильтр в формате ``column operator value`` или ``column=value``.

    Returns:
        Кортеж ``(column, operator, value)``.
    """

    if "=" in item and not re.search(r"\s(eq|ne|gt|gte|lt|lte|contains|in|between)\s", item, flags=re.I):
        column, value = item.split("=", 1)
        return column.strip(), "eq", value.strip()

    parts = item.split(None, 2)
    if len(parts) < 2:
        raise ValueError(f"Некорректный фильтр: {item}")
    column = parts[0].strip()
    operator = parts[1].strip().lower()
    if operator not in _FILTER_OPERATORS:
        raise ValueError(f"Неподдерживаемый оператор фильтра: {operator}")
    value = parts[2].strip() if len(parts) > 2 else ""
    if operator not in {"is_null", "not_null"} and not value:
        raise ValueError(f"Для фильтра {item!r} нужно передать значение.")
    return column, operator, value


def _parse_derived_item(item: str) -> tuple[str, str, str]:
    """Разбирает описание вычисляемой колонки.

    Args:
        item: Строка вида ``new_col = operation(source_col)``.

    Returns:
        Кортеж ``(name, source_column, operation)``.
    """

    match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*=\s*([A-Za-z_][\w]*)\(([^)]+)\)\s*", item)
    if match is None:
        raise ValueError(f"Некорректное описание derived_columns: {item}")
    name, operation, source_column = match.groups()
    operation = operation.lower()
    if operation not in _DERIVED_OPERATIONS:
        raise ValueError(f"Неподдерживаемая операция derived_columns: {operation}")
    return name.strip(), source_column.strip(), operation


def _parse_aggregation_item(item: str) -> tuple[str, str, str]:
    """Разбирает описание агрегата.

    Args:
        item: Строка вида ``function(column) as alias``.

    Returns:
        Кортеж ``(function, column, alias)``.
    """

    match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\(([^)]+)\)(?:\s+as\s+([A-Za-z_][\w]*))?\s*", item, flags=re.I)
    if match is None:
        raise ValueError(f"Некорректное описание aggregations: {item}")
    function, column, alias = match.groups()
    function = function.lower()
    if function not in _AGGREGATION_FUNCTIONS:
        raise ValueError(f"Неподдерживаемая агрегатная функция: {function}")
    return function, column.strip(), (alias or "").strip()


def _parse_order_item(item: str) -> tuple[str, str]:
    """Разбирает одно правило сортировки.

    Args:
        item: Строка вида ``column asc`` или ``column desc``.

    Returns:
        Кортеж ``(column, direction)``.
    """

    parts = item.replace(":", " ").split()
    if not parts:
        raise ValueError("Пустое правило сортировки.")
    column = parts[0].strip()
    direction = parts[1].strip().lower() if len(parts) > 1 else "asc"
    if direction not in {"asc", "desc"}:
        raise ValueError(f"Направление сортировки должно быть asc или desc: {item}")
    return column, direction


def _parse_scalar(value: str) -> str | int | float | bool:
    """Приводит строковое значение фильтра к простому типу.

    Args:
        value: Строковое значение из фильтра.

    Returns:
        Строка, число или bool.
    """

    text = value.strip().strip("'\"")
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"-?\d+", text) and len(text.lstrip("-")) not in {8} and len(text.lstrip("-")) <= 15:
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _validate_columns(*, columns: list[str], available_columns: list[str], allow_empty: bool) -> str:
    """Проверяет наличие колонок в Spark DataFrame.

    Args:
        columns: Колонки, которые нужны запросу.
        available_columns: Колонки текущего Spark DataFrame.
        allow_empty: Можно ли передать пустой список.

    Returns:
        Пустая строка, если ошибок нет, иначе текст ошибки для агента.
    """

    normalized = [column for column in columns if column]
    if not normalized and not allow_empty:
        return "Ошибка load_data: нужно явно указать select_columns или aggregations. '*' и 'all' запрещены."
    forbidden = {column.lower() for column in normalized} & {"*", "all"}
    if forbidden:
        return "Ошибка load_data: нельзя запрашивать все поля. Укажи минимально нужные колонки."
    missing = sorted({column for column in normalized if column not in set(available_columns)})
    return _format_missing_columns(missing=missing, available_columns=available_columns) if missing else ""


def _format_missing_columns(*, missing: list[str], available_columns: list[str]) -> str:
    """Формирует текст ошибки по отсутствующим колонкам.

    Args:
        missing: Колонки, которых нет в DataFrame.
        available_columns: Доступные колонки DataFrame.

    Returns:
        Текст ошибки для повторного вызова инструмента.
    """

    return (
        "Ошибка load_data: в таблице нет колонок из запроса.\n"
        f"Отсутствующие поля: {', '.join(missing)}.\n"
        f"Доступные поля ({len(available_columns)}): {', '.join(available_columns)}."
    )


__all__ = [
    "READ_TABLE_DESCRIPTION",
    "ReadTableInput",
    "build_spark_data_tools",
]

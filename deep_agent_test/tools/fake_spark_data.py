"""Временный fake-инструмент ``read_table`` для тестов без Spark.

Содержит:
- FakeReadTableInput: строковая схема аргументов fake-инструмента ``read_table``.
- build_fake_spark_data_tools: сборка LangChain tool поверх CSV-файлов из ``data``.
- _fake_read_table: выполнение выборки через pandas DataFrame API.
- _load_table_frame: чтение CSV-файла по жестко заданной карте таблиц.
- _apply_derived_columns: добавление вычисляемых колонок.
- _build_derived_series: построение одной вычисляемой pandas Series.
- _apply_filters: применение строковых фильтров.
- _build_filter_mask: построение одного pandas-предиката.
- _apply_aggregations: применение агрегатов.
- _apply_order_by: сортировка результата.
- _parse_columns: разбор строки колонок.
- _split_items: разбор строки со списком инструкций.
- _parse_filter_item: разбор одного фильтра.
- _parse_filter_values: разбор списка значений для операторов ``in`` и ``between``.
- _parse_derived_item: разбор одной вычисляемой колонки.
- _parse_aggregation_item: разбор одного агрегата.
- _parse_order_item: разбор одной сортировки.
- _parse_scalar: приведение строкового значения к простому типу.
- _coerce_filter_value: приведение значения фильтра к типу pandas Series.
- _validate_columns: проверка наличия колонок в DataFrame.
- _format_missing_columns: человекочитаемая ошибка по отсутствующим колонкам.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

FAKE_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
FAKE_TABLE_FILES: dict[str, str] = {
    "csp_afpc_sss_inc.cards_event": "csp_afpc_sss_inc.cards_event.csv",
    "csp_afpc_sss_inc.uko_event": "csp_afpc_sss_inc.uko_event.csv",
    "csp_repo_features.history_automarking_big_148078_155487": (
        "csp_repo_features.history_automarking_big_148078_155487.csv"
    ),
    "cspfs_repo_features3.hits_extra_info_129372427_view": (
        "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
    ),
    "demo_client_timeline": "demo_client_timeline.csv",
}

FAKE_READ_TABLE_DESCRIPTION = (
    "read_table\n"
    "---\n"
    "Описание: временный fake-инструмент для тестирования агента без Spark. "
    "Читает CSV-файлы из локальной папки data и поддерживает тот же строковый "
    "интерфейс, что production read_table: select_columns, filters, derived_columns, "
    "group_by, aggregations, order_by, max_rows и include_schema. "
    "Инструмент временный, использует хардкод таблиц и не должен встраиваться "
    "в production-логику проекта."
)

_FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "between", "is_null", "not_null"}
_DERIVED_OPERATIONS = {"year", "month", "year_month", "date", "lower", "upper", "length", "abs"}
_AGGREGATION_FUNCTIONS = {"count", "count_distinct", "min", "max", "sum", "mean"}


class FakeReadTableInput(BaseModel):
    """Строковые аргументы fake-инструмента чтения таблиц из CSV.

    Args:
        table_name: Имя таблицы из жестко заданной карты ``FAKE_TABLE_FILES``.
        select_columns: Поля результата в формате ``col1, col2``.
        filters: Фильтры одной строкой через ``;`` или перенос строки.
        derived_columns: Вычисляемые колонки одной строкой через ``;``.
        group_by: Поля группировки в формате ``col1, col2``.
        aggregations: Агрегаты одной строкой через ``;``.
        order_by: Сортировка одной строкой через ``;``.
        max_rows: Максимальное число строк, которое нужно вернуть.
        include_schema: Нужно ли приложить схему результата в metadata DataFrame.

    Returns:
        Валидированные строковые параметры для fake ``read_table``.
    """

    table_name: str = Field(description="Имя fake-таблицы, например csp_afpc_sss_inc.uko_event.")
    select_columns: str = Field(default="", description="Поля результата через запятую.")
    filters: str = Field(default="", description="Фильтры через ';' или перенос строки.")
    derived_columns: str = Field(default="", description="Вычисляемые поля через ';'.")
    group_by: str = Field(default="", description="Поля группировки через запятую.")
    aggregations: str = Field(default="", description="Агрегаты через ';'.")
    order_by: str = Field(default="", description="Сортировка через ';'.")
    max_rows: int | None = Field(default=None, ge=0, description="Максимальное число строк результата.")
    include_schema: bool = Field(default=False, description="Если True, добавить схему результата в metadata.")


def build_fake_spark_data_tools() -> list[BaseTool]:
    """Создает временный fake ``read_table`` поверх CSV-файлов из папки ``data``.

    Args:
        Отсутствуют. Папка ``data`` и имена таблиц заданы хардкодом в этом модуле.

    Returns:
        Список с одним LangChain tool ``read_table`` для тестов агента без Spark.
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
        """Выполняет выборку из локального CSV-файла через pandas.

        Args:
            table_name: Имя fake-таблицы.
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

        return _fake_read_table(
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
            name="read_table",
            description=FAKE_READ_TABLE_DESCRIPTION,
            args_schema=FakeReadTableInput,
        )
    ]


def _fake_read_table(
    *,
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
    """Выполняет fake-запрос к CSV-таблице и возвращает pandas DataFrame.

    Args:
        table_name: Имя таблицы из ``FAKE_TABLE_FILES``.
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
        table_name = table_name.strip()
        table = _load_table_frame(table_name)
        total_rows = len(table)
        table = _apply_derived_columns(table=table, derived_columns=derived_columns)
        table = _apply_filters(table=table, filters=filters)
        matched_rows = len(table)

        group_columns = _parse_columns(group_by)
        aggregation_items = _split_items(aggregations)
        if aggregation_items:
            result = _apply_aggregations(table=table, group_columns=group_columns, aggregations=aggregation_items)
        else:
            columns = _parse_columns(select_columns)
            select_error = _validate_columns(columns=columns, available_columns=list(table.columns), allow_empty=False)
            if select_error:
                return select_error
            result = table.loc[:, columns].copy()

        order_items = _split_items(order_by)
        if order_items:
            order_error = _validate_columns(
                columns=[_parse_order_item(item)[0] for item in order_items],
                available_columns=list(result.columns),
                allow_empty=True,
            )
            if order_error:
                return order_error
            result = _apply_order_by(table=result, order_by=order_items)

        if max_rows is not None:
            result = result.head(max(0, int(max_rows))).copy()

        result.attrs["spark_table_name"] = table_name
        result.attrs["spark_source_file"] = str((FAKE_DATA_ROOT / FAKE_TABLE_FILES[table_name]).resolve())
        result.attrs["spark_total_rows"] = int(total_rows)
        result.attrs["spark_matched_rows"] = int(matched_rows)
        if include_schema:
            result.attrs["spark_schema"] = {
                "table_name": table_name,
                "columns_count": len(result.columns),
                "columns": [{"name": str(name), "type": str(dtype)} for name, dtype in result.dtypes.items()],
            }
        return result.reset_index(drop=True)
    except ValueError as exc:
        return f"Ошибка read_table: {exc}"


def _load_table_frame(table_name: str) -> pd.DataFrame:
    """Читает CSV-файл для fake-таблицы.

    Args:
        table_name: Имя таблицы из жестко заданной карты.

    Returns:
        pandas DataFrame с содержимым CSV.

    Raises:
        ValueError: Таблица неизвестна или CSV-файл отсутствует.
    """

    if table_name not in FAKE_TABLE_FILES:
        available = ", ".join(sorted(FAKE_TABLE_FILES))
        raise ValueError(f"неизвестная fake-таблица {table_name!r}. Доступные таблицы: {available}.")
    path = FAKE_DATA_ROOT / FAKE_TABLE_FILES[table_name]
    if not path.exists():
        raise ValueError(f"CSV-файл fake-таблицы не найден: {path}")
    return pd.read_csv(path)


def _apply_derived_columns(*, table: pd.DataFrame, derived_columns: str) -> pd.DataFrame:
    """Добавляет вычисляемые колонки к pandas DataFrame.

    Args:
        table: Исходный pandas DataFrame.
        derived_columns: Описания вычисляемых колонок строкой.

    Returns:
        pandas DataFrame с добавленными колонками.
    """

    result = table.copy()
    for item in _split_items(derived_columns):
        name, source_column, operation = _parse_derived_item(item)
        missing = _validate_columns(columns=[source_column], available_columns=list(result.columns), allow_empty=False)
        if missing:
            raise ValueError(missing)
        result[name] = _build_derived_series(source=result[source_column], operation=operation)
    return result


def _build_derived_series(*, source: pd.Series, operation: str) -> pd.Series:
    """Строит вычисляемую pandas Series.

    Args:
        source: Исходная pandas Series.
        operation: Имя операции.

    Returns:
        pandas Series с вычисленным значением.
    """

    text = source.astype("string")
    if operation == "lower":
        return text.str.lower()
    if operation == "upper":
        return text.str.upper()
    if operation == "length":
        return text.str.len()
    if operation == "abs":
        return pd.to_numeric(source, errors="coerce").abs()

    digits = text.str.replace(r"\D", "", regex=True)
    if operation == "year":
        return digits.str.slice(0, 4)
    if operation == "month":
        return digits.str.slice(4, 6)
    if operation == "year_month":
        return digits.str.slice(0, 6)
    if operation == "date":
        return digits.str.slice(0, 8)
    raise ValueError(f"Неподдерживаемая операция вычисляемой колонки: {operation}")


def _apply_filters(*, table: pd.DataFrame, filters: str) -> pd.DataFrame:
    """Применяет строковые фильтры к pandas DataFrame.

    Args:
        table: Исходный pandas DataFrame.
        filters: Фильтры одной строкой.

    Returns:
        Отфильтрованный pandas DataFrame.
    """

    result = table
    for item in _split_items(filters):
        column, _, _ = _parse_filter_item(item)
        missing = _validate_columns(columns=[column], available_columns=list(result.columns), allow_empty=False)
        if missing:
            raise ValueError(missing)
        result = result.loc[_build_filter_mask(table=result, item=item)].copy()
    return result


def _build_filter_mask(*, table: pd.DataFrame, item: str) -> pd.Series:
    """Строит pandas mask из одного строкового фильтра.

    Args:
        table: pandas DataFrame, к которому применяется фильтр.
        item: Один фильтр в формате ``column operator value``.

    Returns:
        pandas Series с булевым условием.
    """

    column, operator, raw_value = _parse_filter_item(item)
    series = table[column]
    if operator == "is_null":
        return series.isna()
    if operator == "not_null":
        return series.notna()
    if operator == "contains":
        return series.astype("string").str.contains(str(raw_value), case=False, na=False, regex=False)
    if operator == "in":
        values = [_coerce_filter_value(series, _parse_scalar(value)) for value in _parse_filter_values(raw_value)]
        return series.isin(values)
    if operator == "between":
        values = [_coerce_filter_value(series, _parse_scalar(value)) for value in _parse_filter_values(raw_value)]
        if len(values) != 2:
            raise ValueError("Для оператора between нужны два значения через запятую или and.")
        return series.between(values[0], values[1])

    value = _coerce_filter_value(series, _parse_scalar(raw_value))
    if operator == "eq":
        return series == value
    if operator == "ne":
        return series != value
    if operator == "gt":
        return series > value
    if operator == "gte":
        return series >= value
    if operator == "lt":
        return series < value
    if operator == "lte":
        return series <= value
    raise ValueError(f"Неподдерживаемый оператор фильтра: {operator}")


def _apply_aggregations(*, table: pd.DataFrame, group_columns: list[str], aggregations: list[str]) -> pd.DataFrame:
    """Применяет агрегаты к pandas DataFrame.

    Args:
        table: Отфильтрованный pandas DataFrame.
        group_columns: Поля группировки.
        aggregations: Строковые описания агрегатов.

    Returns:
        pandas DataFrame с результатом агрегаций.
    """

    source_columns = [_parse_aggregation_item(item)[1] for item in aggregations]
    missing = _validate_columns(columns=[*group_columns, *source_columns], available_columns=list(table.columns), allow_empty=True)
    if missing:
        raise ValueError(missing)

    if group_columns:
        grouped = table.groupby(group_columns, dropna=False)
        result = grouped.size().reset_index().iloc[:, : len(group_columns)]
        for item in aggregations:
            function, column, alias = _parse_aggregation_item(item)
            result[alias or f"{function}_{column}"] = grouped[column].agg(_aggregation_name(function)).to_numpy()
        return result

    payload: dict[str, list[Any]] = {}
    for item in aggregations:
        function, column, alias = _parse_aggregation_item(item)
        payload[alias or f"{function}_{column}"] = [getattr(table[column], _aggregation_name(function))()]
    return pd.DataFrame(payload)


def _aggregation_name(function: str) -> str:
    """Возвращает имя pandas-агрегата для функции DSL.

    Args:
        function: Имя агрегатной функции DSL.

    Returns:
        Имя метода pandas для агрегации.
    """

    if function == "count":
        return "count"
    if function == "count_distinct":
        return "nunique"
    if function == "mean":
        return "mean"
    if function in {"min", "max", "sum"}:
        return function
    raise ValueError(f"Неподдерживаемая агрегатная функция: {function}")


def _apply_order_by(*, table: pd.DataFrame, order_by: list[str]) -> pd.DataFrame:
    """Сортирует pandas DataFrame.

    Args:
        table: pandas DataFrame результата.
        order_by: Строковые правила сортировки.

    Returns:
        Отсортированный pandas DataFrame.
    """

    columns: list[str] = []
    ascending: list[bool] = []
    for item in order_by:
        column, direction = _parse_order_item(item)
        columns.append(column)
        ascending.append(direction == "asc")
    return table.sort_values(by=columns, ascending=ascending, kind="mergesort")


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


def _parse_filter_values(raw_value: str) -> list[str]:
    """Разбирает строку значений фильтра ``in`` или ``between``.

    Args:
        raw_value: Значения фильтра в формате ``a,b``, ``(a,b)``, ``[a,b]`` или ``a and b`` для ``between``.

    Returns:
        Список очищенных строковых значений без внешних скобок и кавычек.
    """

    text = raw_value.strip()
    if (text.startswith("(") and text.endswith(")")) or (text.startswith("[") and text.endswith("]")):
        text = text[1:-1].strip()
    if "," in text:
        parts = text.split(",")
    else:
        parts = re.split(r"\s+and\s+", text, maxsplit=1, flags=re.I)
    return [part.strip() for part in parts if part.strip()]


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


def _coerce_filter_value(series: pd.Series, value: str | int | float | bool) -> Any:
    """Приводит значение фильтра к типу колонки pandas.

    Args:
        series: Колонка DataFrame, с которой сравнивается значение.
        value: Значение фильтра после базового парсинга.

    Returns:
        Значение, приведенное к числу или bool при необходимости.
    """

    if pd.api.types.is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes", "да"}
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return value if pd.isna(numeric) else numeric
    return str(value)


def _validate_columns(*, columns: list[str], available_columns: list[str], allow_empty: bool) -> str:
    """Проверяет наличие колонок в pandas DataFrame.

    Args:
        columns: Колонки, которые нужны запросу.
        available_columns: Колонки текущего pandas DataFrame.
        allow_empty: Можно ли передать пустой список.

    Returns:
        Пустая строка, если ошибок нет, иначе текст ошибки для агента.
    """

    normalized = [column for column in columns if column]
    if not normalized and not allow_empty:
        return "Ошибка read_table: нужно явно указать select_columns или aggregations. '*' и 'all' запрещены."
    forbidden = {column.lower() for column in normalized} & {"*", "all"}
    if forbidden:
        return "Ошибка read_table: нельзя запрашивать все поля. Укажи минимально нужные колонки."
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
        "Ошибка read_table: в таблице нет колонок из запроса.\n"
        f"Отсутствующие поля: {', '.join(missing)}.\n"
        f"Доступные поля ({len(available_columns)}): {', '.join(available_columns)}."
    )


__all__ = [
    "FAKE_DATA_ROOT",
    "FAKE_READ_TABLE_DESCRIPTION",
    "FAKE_TABLE_FILES",
    "FakeReadTableInput",
    "build_fake_spark_data_tools",
]

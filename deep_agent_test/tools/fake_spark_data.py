"""Временный fake-инструмент ``load_data`` для тестов без Spark.

Содержит:
- ReadTableInput: структурированная схема аргументов fake-инструмента ``load_data``.
- build_fake_spark_data_tools: сборка LangChain tool поверх CSV-файлов из ``data``.
- _extract_query_args_with_llm: LLM-разбор SQL-подобного запроса в аргументы выборки.
- _fake_read_table: выполнение выборки через pandas DataFrame API.
- _load_table_frame: чтение CSV-файла по жестко заданной карте таблиц.
- _resolve_table_name: преобразование короткого alias таблицы в ключ CSV-файла.
- _available_table_aliases_text: форматирование списка доступных alias таблиц.
- _apply_derived_columns: добавление вычисляемых колонок.
- _build_derived_series: построение одной вычисляемой pandas Series.
- _apply_filters: применение структурированных фильтров.
- _build_filter_mask: построение одного pandas-предиката.
- _apply_aggregations: применение агрегатов.
- _apply_order_by: сортировка результата.
- _parse_columns: разбор списка колонок.
- _split_items: разбор списка инструкций.
- _parse_filter_item: разбор одного фильтра.
- _parse_filter_values: разбор списка значений для операторов ``in`` и ``between``.
- _parse_derived_item: разбор одной вычисляемой колонки.
- _parse_aggregation_item: разбор одного агрегата.
- _parse_order_item: разбор одной сортировки.
- _parse_scalar: приведение строкового значения к простому типу.
- _coerce_filter_value: приведение значения фильтра к типу pandas Series.
- _validate_columns: проверка наличия колонок в DataFrame.
- _format_empty_select_error: человекочитаемая ошибка для пустой обычной выборки.
- _format_missing_columns: человекочитаемая ошибка по отсутствующим колонкам.
- _get_field: чтение поля из dict или pydantic-модели.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool

from deep_agent_test.tools.data_query_schema import ReadTableInput, normalize_filter_operator
from deep_agent_test.tools.spark_data import READ_TABLE_DESCRIPTION, _extract_query_args_with_llm

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
FAKE_TABLE_ALIASES: dict[str, str] = {
    "cards": "csp_afpc_sss_inc.cards_event",
    "uko": "csp_afpc_sss_inc.uko_event",
    "history_automarking": "csp_repo_features.history_automarking_big_148078_155487",
    "hits": "cspfs_repo_features3.hits_extra_info_129372427_view",
    "demo_client_timeline": "demo_client_timeline",
}

FAKE_READ_TABLE_DESCRIPTION = READ_TABLE_DESCRIPTION

_FILTER_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "contains_any",
    "in",
    "between",
    "is_null",
    "not_null",
}
_DERIVED_OPERATIONS = {"year", "month", "year_month", "date", "lower", "upper", "length", "abs"}
_AGGREGATION_FUNCTIONS = {"count", "count_distinct", "min", "max", "sum", "mean"}


def build_fake_spark_data_tools(query_parser_model: Any | None = None) -> list[BaseTool]:
    """Создает временный fake ``load_data`` поверх CSV-файлов из папки ``data``.

    Args:
        query_parser_model: Chat-модель LangChain для внутреннего разбора SQL-подобного ``query``.

    Returns:
        Список с одним LangChain tool ``load_data`` для тестов агента без Spark.
    """

    def read_table(query: str) -> Any:
        """Выполняет SQL-подобный запрос к локальному CSV-файлу через pandas.

        Args:
            query: SQL-подобный запрос с alias таблицы, явным периодом и колонками результата.

        Returns:
            pandas DataFrame с результатом или текст ошибки, который агент может исправить.
        """

        try:
            parsed = _extract_query_args_with_llm(query=query, query_parser_model=query_parser_model)
        except ValueError as exc:
            return f"Ошибка load_data: {exc}"

        result = _fake_read_table(**parsed)
        if hasattr(result, "attrs"):
            result.attrs["spark_query_code"] = query.strip()
            result.attrs["spark_is_aggregation"] = bool(parsed["aggregations"])
        return result

    return [
        StructuredTool.from_function(
            func=read_table,
            name="load_data",
            description=FAKE_READ_TABLE_DESCRIPTION,
            args_schema=ReadTableInput,
        )
    ]


def _fake_read_table(
    *,
    table_name: str,
    select_columns: Any,
    filters: Any,
    derived_columns: Any,
    group_by: Any,
    aggregations: Any,
    order_by: Any,
    max_rows: int | None,
) -> Any:
    """Выполняет fake-запрос к CSV-таблице и возвращает pandas DataFrame.

    Args:
        table_name: Имя таблицы из ``FAKE_TABLE_FILES``.
        select_columns: Поля результата списком.
        filters: Фильтры списком объектов.
        derived_columns: Вычисляемые колонки списком объектов.
        group_by: Поля группировки списком.
        aggregations: Агрегаты списком объектов.
        order_by: Сортировка списком объектов.
        max_rows: Максимальное число строк результата.

    Returns:
        pandas DataFrame с metadata в ``attrs`` или текст ошибки.
    """

    try:
        table_alias = table_name.strip()
        resolved_table_name = _resolve_table_name(table_alias)
        table = _load_table_frame(resolved_table_name)
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

        result.attrs["spark_table_name"] = table_alias
        result.attrs["spark_resolved_table_name"] = resolved_table_name
        result.attrs["spark_source_file"] = table_alias
        result.attrs["spark_total_rows"] = int(total_rows)
        result.attrs["spark_matched_rows"] = int(matched_rows)
        return result.reset_index(drop=True)
    except ValueError as exc:
        return f"Ошибка load_data: {exc}"


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


def _resolve_table_name(table_name: str) -> str:
    """Преобразует короткое имя таблицы в ключ CSV-файла.

    Args:
        table_name: Короткий alias таблицы, который передала модель.

    Returns:
        Полное внутреннее имя таблицы для поиска CSV-файла.

    Raises:
        ValueError: Передано неизвестное или похожее на файл значение ``table_name``.
    """

    normalized = table_name.strip()
    if not normalized:
        raise ValueError(f"нужно указать alias таблицы. Доступные таблицы: {_available_table_aliases_text()}.")
    suspicious_fragments = (".", "saved_file", "virtual_file", "select_columns=", "/", "\\", "=")
    if any(fragment in normalized for fragment in suspicious_fragments) or len(normalized) > 80:
        raise ValueError(
            "table_name должен быть коротким alias таблицы, а не путём к файлу, именем артефакта "
            f"или сгенерированным view. Доступные таблицы: {_available_table_aliases_text()}."
        )
    if normalized not in FAKE_TABLE_ALIASES:
        raise ValueError(f"неизвестная таблица {normalized!r}. Доступные таблицы: {_available_table_aliases_text()}.")
    return FAKE_TABLE_ALIASES[normalized]


def _available_table_aliases_text() -> str:
    """Возвращает человекочитаемый список alias таблиц для сообщений инструмента.

    Args:
        Отсутствуют.

    Returns:
        Строка с короткими именами таблиц через запятую.
    """

    return ", ".join(sorted(FAKE_TABLE_ALIASES))


def _apply_derived_columns(*, table: pd.DataFrame, derived_columns: Any) -> pd.DataFrame:
    """Добавляет вычисляемые колонки к pandas DataFrame.

    Args:
        table: Исходный pandas DataFrame.
        derived_columns: Описания вычисляемых колонок списком объектов или строкой.

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


def _apply_filters(*, table: pd.DataFrame, filters: Any) -> pd.DataFrame:
    """Применяет строковые фильтры к pandas DataFrame.

    Args:
        table: Исходный pandas DataFrame.
        filters: Фильтры списком объектов или одной строкой.

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


def _build_filter_mask(*, table: pd.DataFrame, item: Any) -> pd.Series:
    """Строит pandas mask из одного строкового фильтра.

    Args:
        table: pandas DataFrame, к которому применяется фильтр.
        item: Один фильтр в структурированном или строковом формате.

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
    if operator == "contains_any":
        mask = pd.Series(False, index=series.index)
        for value in _parse_filter_values(raw_value):
            mask = mask | series.astype("string").str.contains(str(value), case=False, na=False, regex=False)
        return mask
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


def _apply_aggregations(*, table: pd.DataFrame, group_columns: list[str], aggregations: list[Any]) -> pd.DataFrame:
    """Применяет агрегаты к pandas DataFrame.

    Args:
        table: Отфильтрованный pandas DataFrame.
        group_columns: Поля группировки.
        aggregations: Описания агрегатов списком объектов или строк.

    Returns:
        pandas DataFrame с результатом агрегаций.
    """

    source_columns = [
        column
        for item in aggregations
        for function, column, _alias in [_parse_aggregation_item(item)]
        if not (function == "count" and column == "*")
    ]
    missing = _validate_columns(columns=[*group_columns, *source_columns], available_columns=list(table.columns), allow_empty=True)
    if missing:
        raise ValueError(missing)

    if group_columns:
        grouped = table.groupby(group_columns, dropna=False)
        result = grouped.size().reset_index().iloc[:, : len(group_columns)]
        for item in aggregations:
            function, column, alias = _parse_aggregation_item(item)
            result[alias or f"{function}_{column}"] = (
                grouped.size().to_numpy() if function == "count" and column == "*" else grouped[column].agg(_aggregation_name(function)).to_numpy()
            )
        return result

    payload: dict[str, list[Any]] = {}
    for item in aggregations:
        function, column, alias = _parse_aggregation_item(item)
        payload[alias or f"{function}_{column}"] = [
            len(table) if function == "count" and column == "*" else getattr(table[column], _aggregation_name(function))()
        ]
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


def _apply_order_by(*, table: pd.DataFrame, order_by: list[Any]) -> pd.DataFrame:
    """Сортирует pandas DataFrame.

    Args:
        table: pandas DataFrame результата.
        order_by: Правила сортировки списком объектов или строк.

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


def _parse_columns(value: Any) -> list[str]:
    """Разбирает строку колонок через запятую.

    Args:
        value: Список колонок или строка вида ``col1, col2``.

    Returns:
        Список колонок без пустых значений.
    """

    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _split_items(value: Any) -> list[Any]:
    """Разбирает строку инструкций через ``;`` или перенос строки.

    Args:
        value: Список инструкций или строка с несколькими инструкциями.

    Returns:
        Список непустых инструкций.
    """

    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item]
    normalized = str(value).replace("\n", ";")
    return [item.strip() for item in normalized.split(";") if item.strip()]


def _parse_filter_item(item: Any) -> tuple[str, str, str]:
    """Разбирает один фильтр.

    Args:
        item: Фильтр в структурированном формате или строка ``column operator value``.

    Returns:
        Кортеж ``(column, operator, value)``.
    """

    if not isinstance(item, str):
        column = str(_get_field(item, "column") or "").strip()
        operator = normalize_filter_operator(_get_field(item, "operator"))
        values = _get_field(item, "values") or []
        value = _get_field(item, "value")
        if operator not in _FILTER_OPERATORS:
            raise ValueError(f"Неподдерживаемый оператор фильтра: {operator}")
        if operator in {"in", "contains_any"}:
            raw_values = values if values else ([] if value is None else [value])
            raw_value = ",".join(str(part) for part in raw_values)
        elif operator == "between":
            raw_value = ",".join(str(part) for part in values)
        else:
            if value is not None:
                raw_value = str(value)
            elif values:
                raw_value = str(values[0])
            else:
                raw_value = ""
        if not column:
            raise ValueError(f"В фильтре не указана колонка: {item}")
        if operator not in {"is_null", "not_null"} and not raw_value:
            raise ValueError(f"Для фильтра {item!r} нужно передать value или values.")
        return column, operator, raw_value

    symbolic_match = re.fullmatch(r"\s*([A-Za-z_][\w.]*)\s*(==|=|!=|<>|>=|<=|>|<)\s*(.+)\s*", item)
    if symbolic_match is not None:
        column, operator, value = symbolic_match.groups()
        return column.strip(), normalize_filter_operator(operator), value.strip()

    parts = item.split(None, 2)
    if len(parts) < 2:
        raise ValueError(f"Некорректный фильтр: {item}")
    column = parts[0].strip()
    operator = normalize_filter_operator(parts[1])
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


def _parse_derived_item(item: Any) -> tuple[str, str, str]:
    """Разбирает описание вычисляемой колонки.

    Args:
        item: Структурированное описание или строка вида ``new_col = operation(source_col)``.

    Returns:
        Кортеж ``(name, source_column, operation)``.
    """

    if not isinstance(item, str):
        name = str(_get_field(item, "name") or "").strip()
        source_column = str(_get_field(item, "source_column") or "").strip()
        operation = str(_get_field(item, "operation") or "").strip().lower()
        if not name or not source_column or operation not in _DERIVED_OPERATIONS:
            raise ValueError(f"Некорректное описание derived_columns: {item}")
        return name, source_column, operation

    match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*=\s*([A-Za-z_][\w]*)\(([^)]+)\)\s*", item)
    if match is None:
        raise ValueError(f"Некорректное описание derived_columns: {item}")
    name, operation, source_column = match.groups()
    operation = operation.lower()
    if operation not in _DERIVED_OPERATIONS:
        raise ValueError(f"Неподдерживаемая операция derived_columns: {operation}")
    return name.strip(), source_column.strip(), operation


def _parse_aggregation_item(item: Any) -> tuple[str, str, str]:
    """Разбирает описание агрегата.

    Args:
        item: Структурированное описание или строка вида ``function(column) as alias``.

    Returns:
        Кортеж ``(function, column, alias)``.
    """

    if not isinstance(item, str):
        function = str(_get_field(item, "function") or "").strip().lower()
        column = str(_get_field(item, "column") or "").strip()
        alias = str(_get_field(item, "alias") or "").strip()
        if function not in _AGGREGATION_FUNCTIONS or not column:
            raise ValueError(f"Некорректное описание aggregations: {item}")
        return function, column, alias

    match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\(([^)]+)\)(?:\s+as\s+([A-Za-z_][\w]*))?\s*", item, flags=re.I)
    if match is None:
        raise ValueError(f"Некорректное описание aggregations: {item}")
    function, column, alias = match.groups()
    function = function.lower()
    if function not in _AGGREGATION_FUNCTIONS:
        raise ValueError(f"Неподдерживаемая агрегатная функция: {function}")
    return function, column.strip(), (alias or "").strip()


def _parse_order_item(item: Any) -> tuple[str, str]:
    """Разбирает одно правило сортировки.

    Args:
        item: Структурированное правило или строка вида ``column asc``.

    Returns:
        Кортеж ``(column, direction)``.
    """

    if not isinstance(item, str):
        column = str(_get_field(item, "column") or "").strip()
        direction = str(_get_field(item, "direction") or "asc").strip().lower()
        if not column:
            raise ValueError(f"В сортировке не указана колонка: {item}")
        if direction not in {"asc", "desc"}:
            raise ValueError(f"Направление сортировки должно быть asc или desc: {item}")
        return column, direction

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
        return _format_empty_select_error()
    forbidden = {column.lower() for column in normalized} & {"*", "all"}
    if forbidden:
        return (
            "Ошибка load_data: нельзя запрашивать все поля через '*' или 'all'.\n"
            "Исправление: в query укажи минимальный список колонок в SELECT.\n"
            "Пример: LOAD hits\\nPERIOD event_dt FROM '20260101' TO '20260131'\\n"
            "SELECT event_id, event_dt, event_time\\nWHERE event_id = '<event_id>'\\nLIMIT 1."
        )
    missing = sorted({column for column in normalized if column not in set(available_columns)})
    return _format_missing_columns(missing=missing, available_columns=available_columns) if missing else ""


def _format_empty_select_error() -> str:
    """Формирует точечную ошибку для вызова ``load_data`` без колонок результата.

    Args:
        Отсутствуют.

    Returns:
        Текст ошибки с шаблонами исправленного вызова.
    """

    return (
        "Ошибка load_data: обычная выборка без явного SELECT запрещена. "
        "Инструмент не выполняет SELECT *.\n"
        "Исправление для чтения строк: добавь в query SELECT с минимально нужными полями "
        "и, если есть ключ из задачи, добавь WHERE.\n"
        "Пример точечного поиска по event_id: LOAD hits\\n"
        "PERIOD event_dt FROM '20260101' TO '20260131'\\n"
        "SELECT event_id, event_dt, event_time\\nWHERE event_id = '<event_id>'\\nLIMIT 1.\n"
        "Исправление для расчёта: укажи агрегат прямо в SELECT, например "
        "SELECT event_description, count(event_id) AS events_count."
    )


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
        f"Доступные поля ({len(available_columns)}): {', '.join(available_columns)}.\n"
        "Исправление: перепиши query с существующими полями из списка выше или проверь нужный alias "
        "по skills; не повторяй тот же набор отсутствующих колонок."
    )


def _get_field(source: Any, key: str) -> Any:
    """Достаёт поле из dict или pydantic-модели.

    Args:
        source: Объект с данными фильтра, агрегации, вычисляемой колонки или сортировки.
        key: Имя поля.

    Returns:
        Значение поля или ``None``, если поле отсутствует.
    """

    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


__all__ = [
    "FAKE_DATA_ROOT",
    "FAKE_READ_TABLE_DESCRIPTION",
    "FAKE_TABLE_ALIASES",
    "FAKE_TABLE_FILES",
    "build_fake_spark_data_tools",
]

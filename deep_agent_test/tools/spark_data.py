"""Production-инструмент ``load_data`` поверх общей Spark session.

Содержит:
- ReadTableInput: структурированная схема аргументов инструмента ``load_data``.
- build_spark_data_tools: сборка LangChain tool поверх готовой Spark session.
- _parse_sql_like_query: разбор SQL-подобного запроса в аргументы выборки.
- _read_table: выполнение выборки через Spark DataFrame API.
- _resolve_table_name: преобразование короткого alias таблицы в полное Spark-имя.
- _available_table_aliases_text: форматирование списка доступных alias таблиц.
- _apply_derived_columns: добавление вычисляемых колонок.
- _build_derived_column: построение одной вычисляемой колонки.
- _apply_filters: применение структурированных фильтров.
- _build_filter_expression: построение одного Spark-предиката.
- _apply_aggregations: применение агрегатов.
- _build_aggregation_expression: построение одного Spark-агрегата.
- _apply_order_by: сортировка результата.
- _parse_columns: разбор списка колонок.
- _split_items: разбор списка инструкций.
- _parse_filter_item: разбор одного фильтра.
- _parse_derived_item: разбор одной вычисляемой колонки.
- _parse_aggregation_item: разбор одного агрегата.
- _parse_order_item: разбор одной сортировки.
- _parse_scalar: приведение строкового значения к простому типу.
- _validate_columns: проверка наличия колонок в DataFrame.
- _format_empty_select_error: человекочитаемая ошибка для пустой обычной выборки.
- _format_missing_columns: человекочитаемая ошибка по отсутствующим колонкам.
- _get_field: чтение поля из dict или pydantic-модели.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from deep_agent_test.tools.data_query_schema import ReadTableInput

READ_TABLE_DESCRIPTION = (
    "load_data\n"
    "---\n"
    "Описание: универсальная выборка из Spark-таблиц. "
    "Инструмент принимает один параметр query: SQL-подобный текст запроса. "
    "Агент сам пишет запрос по известным ему skills, а инструмент разбирает query, "
    "проверяет обязательный период и выполняет выборку. При успешной выборке возвращает pandas DataFrame.\n"
    "Выгрузка всех столбцов запрещена: SELECT * и SELECT all не выполняются. "
    "Период выборки обязателен для каждого запроса.\n\n"
    "Параметры:\n"
    "  query (str, обяз.) - SQL-подобный запрос в формате ниже.\n\n"
    "Формат:\n"
    "  LOAD <table_alias>\n"
    "  PERIOD <date_column> FROM '<YYYYMMDD>' TO '<YYYYMMDD>'\n"
    "  SELECT <column_1>, <column_2> [, count(<column>) AS <alias>]\n"
    "  WHERE <column> = '<value>' AND <column> CONTAINS '<value>'\n"
    "  GROUP BY <column>\n"
    "  ORDER BY <column> ASC|DESC\n"
    "  LIMIT <int>\n\n"
    "Допустимые alias: hits, cards, uko, history_automarking, demo_client_timeline. "
    "Вместо LOAD можно использовать обычный FROM, но alias должен быть коротким. "
    "Период задавай только через PERIOD или через WHERE <date_column> BETWEEN '<from>' AND '<to>'. "
    "SELECT * и SELECT all запрещены. Фильтры WHERE поддерживают =, !=, >, >=, <, <=, CONTAINS, IN, BETWEEN."
)

_FILTER_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "between", "is_null", "not_null"}
_DERIVED_OPERATIONS = {"year", "month", "year_month", "date", "lower", "upper", "length", "abs"}
_AGGREGATION_FUNCTIONS = {"count", "count_distinct", "min", "max", "sum", "mean"}
TABLE_ALIASES: dict[str, str] = {
    "cards": "csp_afpc_sss_inc.cards_event",
    "uko": "csp_afpc_sss_inc.uko_event",
    "history_automarking": "csp_repo_features.history_automarking_big_148078_155487",
    "hits": "cspfs_repo_features3.hits_extra_info_129372427_view",
    "demo_client_timeline": "demo_client_timeline",
}


def build_spark_data_tools(spark: Any) -> list[BaseTool]:
    """Создает инструмент ``load_data`` поверх готовой Spark session.

    Args:
        spark: Активная ``pyspark.sql.SparkSession``, созданная один раз при старте приложения.

    Returns:
        Список с одним LangChain tool ``load_data``.
    """

    def read_table(query: str) -> Any:
        """Выполняет SQL-подобный запрос к Spark-таблице через переданную Spark session.

        Args:
            query: SQL-подобный запрос с alias таблицы, явным периодом и колонками результата.

        Returns:
            pandas DataFrame с результатом или текст ошибки, который агент может исправить.
        """

        try:
            parsed = _parse_sql_like_query(query)
        except ValueError as exc:
            return f"Ошибка load_data: {exc}"

        result = _read_table(
            spark=spark,
            **parsed,
        )
        if hasattr(result, "attrs"):
            result.attrs["spark_query_code"] = query.strip()
            result.attrs["spark_is_aggregation"] = bool(parsed["aggregations"])
        return result

    return [
        StructuredTool.from_function(
            func=read_table,
            name="load_data",
            description=READ_TABLE_DESCRIPTION,
            args_schema=ReadTableInput,
        )
    ]


def _parse_sql_like_query(query: str) -> dict[str, Any]:
    """Разбирает SQL-подобный запрос ``load_data`` в старые структурированные аргументы.

    Args:
        query: SQL-подобный текст с ``LOAD`` или ``FROM``, ``PERIOD`` и ``SELECT``.

    Returns:
        Словарь аргументов, совместимый с внутренней функцией ``_read_table``.

    Raises:
        ValueError: Запрос не содержит обязательных частей или использует неподдерживаемый синтаксис.
    """

    text = query.strip().rstrip(";")
    if not text:
        raise ValueError("query пустой. Передай SQL-подобный запрос с LOAD, PERIOD и SELECT.")

    table_name = _extract_table_alias(text)
    select_text = _extract_clause(text, "SELECT", ["FROM", "WHERE", "PERIOD", "GROUP BY", "ORDER BY", "LIMIT"])
    if not select_text:
        raise ValueError("в query нет SELECT. Укажи явные колонки результата или агрегаты.")

    group_by = _parse_columns(_extract_clause(text, "GROUP BY", ["ORDER BY", "LIMIT"]))
    order_by = _parse_order_by_clause(_extract_clause(text, "ORDER BY", ["LIMIT"]))
    max_rows = _parse_limit(_extract_clause(text, "LIMIT", []))
    derived_columns = _parse_derive_clause(_extract_clause(text, "DERIVE", ["SELECT", "WHERE", "GROUP BY", "ORDER BY", "LIMIT"]))
    select_columns, aggregations = _parse_select_clause(select_text=select_text, group_by=group_by)

    filters = _parse_where_clause(_extract_clause(text, "WHERE", ["GROUP BY", "ORDER BY", "LIMIT"]))
    period_filter = _parse_period_clause(text)
    if period_filter is None:
        period_filter = _find_between_filter(filters)
    if period_filter is None:
        raise ValueError(
            "в query нет обязательного периода. Добавь строку вида "
            "PERIOD event_dt FROM '20260101' TO '20260131'."
        )
    if period_filter not in filters:
        filters.insert(0, period_filter)

    return {
        "table_name": table_name,
        "select_columns": select_columns,
        "filters": filters,
        "derived_columns": derived_columns,
        "group_by": group_by,
        "aggregations": aggregations,
        "order_by": order_by,
        "max_rows": max_rows,
        "include_schema": False,
    }


def _extract_table_alias(text: str) -> str:
    """Извлекает короткий alias таблицы из ``LOAD`` или ``FROM``.

    Args:
        text: Полный SQL-подобный запрос.

    Returns:
        Короткий alias таблицы.

    Raises:
        ValueError: Alias таблицы не найден.
    """

    match = re.search(r"(?im)^\s*LOAD\s+([A-Za-z_][\w]*)\b", text)
    if match is None:
        match = re.search(r"(?is)\bFROM\s+([A-Za-z_][\w]*)\b", text)
    if match is None:
        raise ValueError("в query нет LOAD <table_alias> или FROM <table_alias>.")
    return match.group(1).strip()


def _extract_clause(text: str, clause: str, stop_clauses: list[str]) -> str:
    """Извлекает тело одной SQL-подобной секции.

    Args:
        text: Полный SQL-подобный запрос.
        clause: Имя секции, которую нужно найти.
        stop_clauses: Секции, на которых нужно остановить чтение.

    Returns:
        Текст секции без имени или пустая строка, если секция отсутствует.
    """

    stop_pattern = "|".join(re.escape(stop) for stop in stop_clauses)
    pattern = rf"(?is)\b{re.escape(clause)}\b\s+(.+?)(?=\b(?:{stop_pattern})\b|\Z)" if stop_pattern else rf"(?is)\b{re.escape(clause)}\b\s+(.+?)\Z"
    match = re.search(pattern, text)
    return match.group(1).strip().rstrip(";") if match else ""


def _parse_period_clause(text: str) -> dict[str, Any] | None:
    """Разбирает обязательную секцию периода.

    Args:
        text: Полный SQL-подобный запрос.

    Returns:
        Фильтр ``between`` по дате или ``None``, если секция отсутствует.
    """

    match = re.search(
        r"(?is)\bPERIOD\s+([A-Za-z_][\w]*)\s+FROM\s+(['\"]?)([^'\"\s]+)\2\s+TO\s+(['\"]?)([^'\"\s]+)\4",
        text,
    )
    if match is None:
        return None
    return {"column": match.group(1), "operator": "between", "values": [match.group(3), match.group(5)]}


def _parse_select_clause(*, select_text: str, group_by: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    """Разбирает ``SELECT`` на обычные колонки и агрегаты.

    Args:
        select_text: Текст после ``SELECT``.
        group_by: Колонки группировки, которые не нужно дублировать в ``select_columns`` при агрегации.

    Returns:
        Кортеж ``(select_columns, aggregations)``.

    Raises:
        ValueError: ``SELECT`` пустой или запрашивает все поля.
    """

    items = _split_csv(select_text)
    if not items:
        raise ValueError("SELECT пустой. Укажи конкретные поля результата.")
    forbidden = {item.strip().lower() for item in items} & {"*", "all"}
    if forbidden:
        raise ValueError("SELECT * и SELECT all запрещены. Укажи минимальный список конкретных колонок.")

    select_columns: list[str] = []
    aggregations: list[dict[str, str]] = []
    for item in items:
        aggregation = _parse_aggregation_expression(item)
        if aggregation:
            aggregations.append(aggregation)
        elif item not in group_by:
            select_columns.append(item)
    return ([] if aggregations else select_columns), aggregations


def _parse_aggregation_expression(item: str) -> dict[str, str] | None:
    """Разбирает один агрегат из ``SELECT``.

    Args:
        item: Элемент списка ``SELECT``.

    Returns:
        Описание агрегата или ``None``, если элемент является обычной колонкой.
    """

    match = re.fullmatch(
        r"\s*(count|count_distinct|min|max|sum|mean)\s*\(\s*([A-Za-z_][\w]*)\s*\)(?:\s+AS\s+([A-Za-z_][\w]*))?\s*",
        item,
        flags=re.I,
    )
    if match is None:
        return None
    function, column, alias = match.groups()
    return {"function": function.lower(), "column": column, "alias": alias or ""}


def _parse_where_clause(where_text: str) -> list[dict[str, Any]]:
    """Разбирает ``WHERE`` в список структурированных фильтров.

    Args:
        where_text: Текст после ``WHERE``.

    Returns:
        Список фильтров для внутренней выборки.
    """

    return [_parse_where_condition(condition) for condition in _split_where_conditions(where_text)]


def _split_where_conditions(where_text: str) -> list[str]:
    """Разделяет условия ``WHERE`` по ``AND`` без разрыва оператора ``BETWEEN``.

    Args:
        where_text: Текст после ``WHERE``.

    Returns:
        Список отдельных условий.
    """

    if not where_text:
        return []

    def protect_between(match: re.Match[str]) -> str:
        """Заменяет внутренний ``AND`` в ``BETWEEN`` на технический маркер."""

        return f"BETWEEN {match.group(1)} __BETWEEN_AND__ {match.group(2)}"

    protected = re.sub(
        r"(?is)BETWEEN\s+('[^']*'|\"[^\"]*\"|\S+)\s+AND\s+('[^']*'|\"[^\"]*\"|\S+)",
        protect_between,
        where_text,
    )
    return [part.replace("__BETWEEN_AND__", "AND").strip() for part in re.split(r"(?i)\s+AND\s+", protected) if part.strip()]


def _parse_where_condition(condition: str) -> dict[str, Any]:
    """Разбирает одно условие ``WHERE``.

    Args:
        condition: SQL-подобное условие.

    Returns:
        Структурированный фильтр для внутренней выборки.

    Raises:
        ValueError: Условие использует неподдерживаемый синтаксис.
    """

    between_match = re.fullmatch(
        r"\s*([A-Za-z_][\w]*)\s+BETWEEN\s+(.+?)\s+AND\s+(.+?)\s*",
        condition,
        flags=re.I,
    )
    if between_match:
        return {
            "column": between_match.group(1),
            "operator": "between",
            "values": [_strip_quotes(between_match.group(2)), _strip_quotes(between_match.group(3))],
        }

    in_match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s+IN\s*\((.+)\)\s*", condition, flags=re.I)
    if in_match:
        return {"column": in_match.group(1), "operator": "in", "values": [_strip_quotes(item) for item in _split_csv(in_match.group(2))]}

    contains_match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s+CONTAINS\s+(.+?)\s*", condition, flags=re.I)
    if contains_match:
        return {"column": contains_match.group(1), "operator": "contains", "value": _strip_quotes(contains_match.group(2))}

    compare_match = re.fullmatch(r"\s*([A-Za-z_][\w]*)\s*(=|!=|<>|>=|<=|>|<)\s*(.+?)\s*", condition)
    if compare_match:
        operator = {"=": "eq", "!=": "ne", "<>": "ne", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte"}[
            compare_match.group(2)
        ]
        return {"column": compare_match.group(1), "operator": operator, "value": _strip_quotes(compare_match.group(3))}

    raise ValueError(f"неподдерживаемое условие WHERE: {condition!r}.")


def _find_between_filter(filters: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Ищет фильтр периода среди уже разобранных условий ``WHERE``.

    Args:
        filters: Список фильтров из ``WHERE``.

    Returns:
        Первый фильтр ``between`` или ``None``.
    """

    for item in filters:
        if item.get("operator") == "between" and len(item.get("values", [])) == 2:
            return item
    return None


def _parse_order_by_clause(order_text: str) -> list[dict[str, str]]:
    """Разбирает ``ORDER BY`` в список правил сортировки.

    Args:
        order_text: Текст после ``ORDER BY``.

    Returns:
        Список правил сортировки.
    """

    result: list[dict[str, str]] = []
    for item in _split_csv(order_text):
        parts = item.split()
        if parts:
            result.append({"column": parts[0], "direction": parts[1].lower() if len(parts) > 1 else "asc"})
    return result


def _parse_derive_clause(derive_text: str) -> list[dict[str, str]]:
    """Разбирает ``DERIVE`` в список вычисляемых колонок.

    Args:
        derive_text: Текст после ``DERIVE``.

    Returns:
        Список вычисляемых колонок.
    """

    result: list[dict[str, str]] = []
    for item in _split_csv(derive_text):
        parsed = _parse_derived_item(item)
        result.append({"name": parsed[0], "source_column": parsed[1], "operation": parsed[2]})
    return result


def _parse_limit(limit_text: str) -> int | None:
    """Разбирает ``LIMIT``.

    Args:
        limit_text: Текст после ``LIMIT``.

    Returns:
        Целочисленный лимит или ``None``.
    """

    if not limit_text:
        return None
    match = re.match(r"\s*(\d+)\b", limit_text)
    return int(match.group(1)) if match else None


def _split_csv(text: str) -> list[str]:
    """Разделяет строку по запятым вне кавычек и скобок.

    Args:
        text: Строка со списком значений.

    Returns:
        Список очищенных элементов.
    """

    result: list[str] = []
    current: list[str] = []
    quote = ""
    depth = 0
    for char in text:
        if char in {"'", '"'} and not quote:
            quote = char
        elif char == quote:
            quote = ""
        elif not quote and char == "(":
            depth += 1
        elif not quote and char == ")" and depth:
            depth -= 1
        if char == "," and not quote and depth == 0:
            item = "".join(current).strip()
            if item:
                result.append(item)
            current = []
        else:
            current.append(char)
    item = "".join(current).strip()
    if item:
        result.append(item)
    return result


def _strip_quotes(value: str) -> str:
    """Удаляет внешние кавычки у значения фильтра.

    Args:
        value: Значение из SQL-подобного запроса.

    Returns:
        Очищенное строковое значение.
    """

    text = value.strip()
    if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
        return text[1:-1]
    return text


def _read_table(
    *,
    spark: Any,
    table_name: str,
    select_columns: Any,
    filters: Any,
    derived_columns: Any,
    group_by: Any,
    aggregations: Any,
    order_by: Any,
    max_rows: int | None,
    include_schema: bool,
) -> Any:
    """Выполняет Spark-запрос и возвращает pandas DataFrame.

    Args:
        spark: Активная Spark session.
        table_name: Имя таблицы Spark или view.
        select_columns: Поля результата списком.
        filters: Фильтры списком объектов.
        derived_columns: Вычисляемые колонки списком объектов.
        group_by: Поля группировки списком.
        aggregations: Агрегаты списком объектов.
        order_by: Сортировка списком объектов.
        max_rows: Максимальное число строк результата.
        include_schema: Нужно ли приложить схему результата.

    Returns:
        pandas DataFrame с metadata в ``attrs`` или текст ошибки.
    """

    try:
        table_alias = table_name.strip()
        resolved_table_name = _resolve_table_name(table_alias)
        table = spark.table(resolved_table_name)
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
        frame.attrs["spark_table_name"] = table_alias
        frame.attrs["spark_resolved_table_name"] = resolved_table_name
        frame.attrs["spark_source_file"] = table_alias
        frame.attrs["spark_total_rows"] = int(total_rows)
        frame.attrs["spark_matched_rows"] = int(matched_rows)
        if include_schema:
            frame.attrs["spark_schema"] = {
                "table_name": table_alias,
                "resolved_table_name": resolved_table_name,
                "columns_count": len(result.columns),
                "columns": [{"name": name, "type": str(dtype)} for name, dtype in result.dtypes],
            }
        return frame
    except ValueError as exc:
        return f"Ошибка load_data: {exc}"


def _resolve_table_name(table_name: str) -> str:
    """Преобразует короткое имя таблицы в полное Spark-имя.

    Args:
        table_name: Короткий alias таблицы, который передала модель.

    Returns:
        Полное имя Spark-таблицы для ``spark.table``.

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
    if normalized not in TABLE_ALIASES:
        raise ValueError(f"неизвестная таблица {normalized!r}. Доступные таблицы: {_available_table_aliases_text()}.")
    return TABLE_ALIASES[normalized]


def _available_table_aliases_text() -> str:
    """Возвращает человекочитаемый список alias таблиц для сообщений инструмента.

    Args:
        Отсутствуют.

    Returns:
        Строка с короткими именами таблиц через запятую.
    """

    return ", ".join(sorted(TABLE_ALIASES))


def _apply_derived_columns(*, table: Any, derived_columns: Any) -> Any:
    """Добавляет вычисляемые колонки к Spark DataFrame.

    Args:
        table: Исходный Spark DataFrame.
        derived_columns: Описания вычисляемых колонок списком объектов или строкой.

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


def _apply_filters(*, table: Any, filters: Any) -> Any:
    """Применяет строковые фильтры к Spark DataFrame.

    Args:
        table: Исходный Spark DataFrame.
        filters: Фильтры списком объектов или одной строкой.

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


def _build_filter_expression(item: Any) -> Any:
    """Строит Spark Column-предикат из одного строкового фильтра.

    Args:
        item: Один фильтр в структурированном или строковом формате.

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
        return spark_column.isin([_parse_scalar(value) for value in _parse_filter_values(raw_value)])
    if operator == "between":
        values = [_parse_scalar(value) for value in _parse_filter_values(raw_value)]
        if len(values) != 2:
            raise ValueError("Для оператора between нужны два значения.")
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


def _apply_aggregations(*, table: Any, group_columns: list[str], aggregations: list[Any]) -> Any:
    """Применяет агрегаты к Spark DataFrame.

    Args:
        table: Отфильтрованный Spark DataFrame.
        group_columns: Поля группировки.
        aggregations: Описания агрегатов списком объектов или строк.

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


def _build_aggregation_expression(item: Any) -> Any:
    """Строит Spark Column для одного агрегата.

    Args:
        item: Агрегат в структурированном или строковом формате.

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


def _apply_order_by(*, table: Any, order_by: list[Any]) -> Any:
    """Сортирует Spark DataFrame.

    Args:
        table: Spark DataFrame результата.
        order_by: Правила сортировки списком объектов или строк.

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
        operator = str(_get_field(item, "operator") or "eq").strip().lower()
        values = _get_field(item, "values") or []
        value = _get_field(item, "value")
        second_value = _get_field(item, "second_value")
        if operator not in _FILTER_OPERATORS:
            raise ValueError(f"Неподдерживаемый оператор фильтра: {operator}")
        if operator == "in":
            raw_values = values if values else ([] if value is None else [value])
            raw_value = ",".join(str(part) for part in raw_values)
        elif operator == "between":
            raw_values = values if values else [part for part in (value, second_value) if part is not None]
            raw_value = ",".join(str(part) for part in raw_values)
        else:
            raw_value = "" if value is None else str(value)
        if not column:
            raise ValueError(f"В фильтре не указана колонка: {item}")
        if operator not in {"is_null", "not_null"} and not raw_value:
            raise ValueError(f"Для фильтра {item!r} нужно передать value или values.")
        return column, operator, raw_value

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
        raw_value: Значения фильтра в формате ``a,b`` или ``a and b``.

    Returns:
        Список очищенных строковых значений.
    """

    text = str(raw_value).strip()
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
    "READ_TABLE_DESCRIPTION",
    "ReadTableInput",
    "TABLE_ALIASES",
    "build_spark_data_tools",
    "_parse_sql_like_query",
]

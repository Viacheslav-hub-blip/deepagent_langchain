"""Production-инструмент ``load_data`` поверх общей Spark session.

Содержит:
- ReadTableInput: структурированная схема аргументов инструмента ``load_data``.
- build_spark_data_tools: сборка LangChain tool поверх готовой Spark session.
- _extract_query_args_with_llm: LLM-разбор SQL-подобного запроса в аргументы выборки.
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

import json
import re
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from deep_agent_test.tools.data_query_schema import ParsedDataQuery, ReadTableInput, normalize_filter_operator

READ_TABLE_DESCRIPTION = (
    "load_data\n"
    "---\n"
    "Описание: универсальная безопасная выборка из доступных Spark-таблиц по короткому alias. "
    "Инструмент принимает один параметр query: SQL-подобный текст запроса. Агент пишет query по skills, "
    "а внутренний нормализатор преобразует его в структурированные аргументы и выполняет выборку. "
    "При успешной выборке возвращается pandas DataFrame с полным результатом запроса.\n\n"
    "Когда использовать:\n"
    "- нужно прочитать строки, события или агрегаты из таблиц hits, cards, uko, history_automarking "
    "или demo_client_timeline;\n"
    "- известны таблица, период, нужные колонки и фильтры по ключам/значениям;\n"
    "- нужно проверить наличие записей, получить фактические поля события или посчитать агрегат "
    "по данным источника.\n\n"
    "Когда не использовать:\n"
    "- нет периода, даты начала или даты конца: сначала запроси недостающие данные;\n"
    "- нужно обработать уже выгруженный pickle/offload-файл: используй код поверх сохраненного результата, "
    "а не повторный load_data;\n"
    "- нужна произвольная Spark SQL-команда, join нескольких источников, запись данных, удаление данных "
    "или изменение таблиц;\n"
    "- требуется SELECT * / SELECT all: перечисли только нужные колонки.\n\n"
    "Параметры:\n"
    "- query (str, обяз.): SQL-подобный запрос. В query обязательно укажи LOAD/FROM с коротким alias, "
    "PERIOD или date BETWEEN, SELECT с явными колонками или агрегатами, при необходимости WHERE/GROUP BY/"
    "ORDER BY/LIMIT.\n\n"
    "Формат query:\n"
    "  LOAD <table_alias>\n"
    "  PERIOD <date_column> FROM '<YYYYMMDD>' TO '<YYYYMMDD>'\n"
    "  SELECT <column_1>, <column_2> [, COUNT(*) AS <alias>] [, count(<column>) AS <alias>]\n"
    "  WHERE <column> = '<value>' AND (<column> LIKE '%value%' OR <column> CONTAINS '<value>')\n"
    "  GROUP BY <column>\n"
    "  ORDER BY <column> ASC|DESC\n"
    "  LIMIT <int>\n\n"
    "Допустимые таблицы: hits, cards, uko, history_automarking, demo_client_timeline. "
    "Вместо LOAD можно использовать FROM, но имя источника должно быть коротким alias, а не Spark-путем, "
    "именем файла, saved_file, virtual_file или pkl.\n\n"
    "Операторы WHERE:\n"
    "- равенство: =, ==, eq, equals -> внутренне нормализуется в eq;\n"
    "- не равно: !=, <>, ne, not_equals -> ne;\n"
    "- сравнения: >, >=, <, <=, gt, gte, lt, lte;\n"
    "- текстовый поиск: LIKE '%value%' или CONTAINS 'value' -> contains;\n"
    "- списки и интервалы: IN (...), BETWEEN <from> AND <to>;\n"
    "- несколько условий можно соединять через AND и OR.\n\n"
    "Ограничения:\n"
    "- период обязателен для каждого запроса и задается через PERIOD или WHERE <date_column> BETWEEN '<from>' AND '<to>';\n"
    "- SELECT * и SELECT all запрещены для обычной выборки, но COUNT(*) разрешен в агрегатах;\n"
    "- длинные идентификаторы передавай строками в кавычках, чтобы не потерять точность."
)

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
TABLE_ALIASES: dict[str, str] = {
    "cards": "csp_afpc_sss_inc.cards_event",
    "uko": "csp_afpc_sss_inc.uko_event",
    "history_automarking": "csp_repo_features.history_automarking_big_148078_155487",
    "hits": "cspfs_repo_features3.hits_extra_info_129372427_view",
    "demo_client_timeline": "demo_client_timeline",
}


def build_spark_data_tools(spark: Any, query_parser_model: Any | None = None) -> list[BaseTool]:
    """Создает инструмент ``load_data`` поверх готовой Spark session.

    Args:
        spark: Активная ``pyspark.sql.SparkSession``, созданная один раз при старте приложения.
        query_parser_model: Chat-модель LangChain для внутреннего разбора SQL-подобного ``query``.

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
            parsed = _extract_query_args_with_llm(query=query, query_parser_model=query_parser_model)
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


def _extract_query_args_with_llm(*, query: str, query_parser_model: Any | None) -> dict[str, Any]:
    """Извлекает аргументы выборки из SQL-подобного запроса с помощью LLM.

    Args:
        query: SQL-подобный запрос, который написал data-retrieval-agent.
        query_parser_model: Chat-модель LangChain для JSON-разбора ``query``.

    Returns:
        Словарь аргументов, совместимый с внутренней функцией ``_read_table``.

    Raises:
        ValueError: Модель разбора не передана или LLM вернул неполную структуру запроса.
    """

    if query_parser_model is None:
        raise ValueError(
            "для load_data не передана query_parser_model. "
            "Собери data tool с query_parser_model=model."
        )

    messages = [
        ("system", _QUERY_PARSER_SYSTEM_PROMPT),
        ("human", _normalize_query_text_for_parser(query)),
    ]
    parsed = _invoke_query_parser_json_fallback(query_parser_model=query_parser_model, messages=messages)
    try:
        return _parsed_query_to_read_args(parsed)
    except ValueError as first_error:
        repaired = _repair_parsed_query_with_llm(
            query_parser_model=query_parser_model,
            query=query,
            parsed=parsed,
            validation_error=str(first_error),
        )
        if repaired is not None:
            try:
                return _parsed_query_to_read_args(repaired)
            except ValueError as repaired_error:
                raise ValueError(
                    f"{repaired_error}\nLLM parser output: {_format_parsed_query_debug(repaired)}"
                ) from repaired_error
        raise ValueError(f"{first_error}\nLLM parser output: {_format_parsed_query_debug(parsed)}") from first_error


_QUERY_PARSER_SYSTEM_PROMPT = """
Ты внутренний нормализатор запроса для инструмента load_data.
На входе только SQL-подобный query. Не выполняй анализ данных и не отвечай текстом.
Верни ParsedDataQuery.

Правила извлечения:
- status="ready" только если есть короткое имя таблицы, явные колонки/агрегации и временной интервал.
- table_name — только короткое имя источника: hits, cards, uko, history_automarking, demo_client_timeline.
- Если первая строка содержит один из известных alias рядом со служебным словом или опечаткой,
  извлекай известный alias и игнорируй лишний токен.
- Не требуй SQL-alias. `LOAD hits`, `LOAD hits AS h`, `FROM hits h` и `FROM <hits> AS t`
  означают table_name="hits".
- Игнорируй SQL-alias и служебные префиксы: `h.event_dt`, `t.event_dt`, `<hits>.event_dt`
  должны стать `event_dt`.
- Если нет периода начала/конца, верни status="needs_more_input" и missing_inputs.
- Если указана неизвестная таблица, неизвестный синтаксис или неподдерживаемая агрегация,
  верни status="schema_error" и problem.
- SELECT * как выборка колонок запрещён. Для COUNT(*) используй aggregation:
  {"function": "count", "column": "*", "alias": "..."}.
- Если в query есть SELECT col1, col2, эти имена обязательно должны попасть в select_columns.
- Не возвращай needs_more_input из-за отсутствия колонок, если после SELECT указаны колонки или агрегаты.
- Равенство через `=`, `==`, `eq`, `equals`, `equal` преобразуй в operator="eq".
- Неравенство через `!=`, `<>`, `ne`, `not_equals`, `not_equal` преобразуй в operator="ne".
- Сравнения `>`, `>=`, `<`, `<=` преобразуй в operator="gt", "gte", "lt", "lte".
- LIKE '%x%' преобразуй в operator="contains", value="x".
- CONTAINS 'x' преобразуй в operator="contains", value="x".
- Цепочку OR по одной колонке вида col LIKE '%a%' OR col LIKE '%b%' преобразуй в один
  фильтр operator="contains_any", values=["a", "b"].
- IN (...) преобразуй в operator="in".
- BETWEEN преобразуй в operator="between", values=[start, end].
- Сохраняй строковые идентификаторы строками.
- Не выдумывай поля, которых нет в query. Если данных недостаточно, верни needs_more_input.

Примеры:

query:
LOAD hits
PERIOD event_dt FROM '20260101' TO '20260131'
SELECT event_id, event_dt, event_description

JSON:
{
  "status": "ready",
  "table_name": "hits",
  "select_columns": ["event_id", "event_dt", "event_description"],
  "filters": [
    {"column": "event_dt", "operator": "between", "values": ["20260101", "20260131"]}
  ],
  "derived_columns": [],
  "group_by": [],
  "aggregations": [],
  "order_by": [],
  "max_rows": null,
  "problem": "",
  "missing_inputs": []
}

query:
LOAD hits
PERIOD event_dt FROM '20260101' TO '20260131'
SELECT event_description, COUNT(*) AS events_count
WHERE event_description LIKE '%обучение%' OR event_description LIKE '%курсы%'
GROUP BY event_description
ORDER BY events_count DESC

JSON:
{
  "status": "ready",
  "table_name": "hits",
  "select_columns": ["event_description"],
  "filters": [
    {"column": "event_dt", "operator": "between", "values": ["20260101", "20260131"]},
    {"column": "event_description", "operator": "contains_any", "values": ["обучение", "курсы"]}
  ],
  "derived_columns": [],
  "group_by": ["event_description"],
  "aggregations": [
    {"function": "count", "column": "*", "alias": "events_count"}
  ],
  "order_by": [
    {"column": "events_count", "direction": "desc"}
  ],
  "max_rows": null,
  "problem": "",
  "missing_inputs": []
}

query:
LOAD cards AS c
PERIOD c.event_dt FROM '20260101' TO '20260107'
SELECT c.event_id, c.event_dt, c.atm_mcc
WHERE c.atm_mcc IN ('8299', '8244')
LIMIT 50

JSON:
{
  "status": "ready",
  "table_name": "cards",
  "select_columns": ["event_id", "event_dt", "atm_mcc"],
  "filters": [
    {"column": "event_dt", "operator": "between", "values": ["20260101", "20260107"]},
    {"column": "atm_mcc", "operator": "in", "values": ["8299", "8244"]}
  ],
  "derived_columns": [],
  "group_by": [],
  "aggregations": [],
  "order_by": [],
  "max_rows": 50,
  "problem": "",
  "missing_inputs": []
}
""".strip()


def _invoke_query_parser_json_fallback(*, query_parser_model: Any, messages: list[tuple[str, str]]) -> ParsedDataQuery:
    """Вызывает модель без structured-output и разбирает JSON из текстового ответа.

    Args:
        query_parser_model: Chat-модель LangChain.
        messages: Сообщения system/human для внутреннего нормализатора.

    Returns:
        Pydantic-модель ``ParsedDataQuery``.

    Raises:
        ValueError: Модель не вернула JSON, совместимый с ``ParsedDataQuery``.
    """

    fallback_messages = [
        messages[0],
        (
            "human",
            f"{messages[1][1]}\n\nВерни только JSON-объект без Markdown и пояснений.",
        ),
    ]
    raw = query_parser_model.invoke(fallback_messages)
    try:
        return ParsedDataQuery.model_validate(_extract_json_object(_message_text(raw)))
    except Exception as exc:
        raise ValueError(f"LLM parser не вернул валидный ParsedDataQuery JSON: {exc}") from exc


def _repair_parsed_query_with_llm(
    *,
    query_parser_model: Any,
    query: str,
    parsed: ParsedDataQuery,
    validation_error: str,
) -> ParsedDataQuery | None:
    """Повторно просит LLM исправить результат разбора, если первый JSON не прошёл валидацию.

    Args:
        query_parser_model: Chat-модель LangChain для внутреннего разбора запроса.
        query: Исходный SQL-подобный запрос, который передал агент.
        parsed: Первый структурированный результат разбора.
        validation_error: Ошибка обязательной валидации первого результата.

    Returns:
        Исправленная модель ``ParsedDataQuery`` или ``None``, если repair-вызов не дал валидный JSON.
    """

    parsed_json = json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False)
    repair_prompt = (
        "Исправь только структурированный JSON-разбор query для load_data.\n"
        "Не анализируй данные и не добавляй поля, которых нет в query.\n"
        "Если в query есть SELECT col1, col2, скопируй эти имена в select_columns.\n"
        "Если в query есть COUNT(*) или count(col), перенеси это в aggregations.\n"
        "Если в query есть PERIOD date_col FROM 'start' TO 'end', добавь фильтр between по date_col.\n"
        "Если в query есть равенство через =, ==, eq или equals, используй operator=\"eq\".\n"
        "Если в query есть неравенство через !=, <> или not_equals, используй operator=\"ne\".\n"
        "Если первая строка содержит известный alias таблицы рядом с лишним словом или опечаткой, используй alias.\n"
        "Игнорируй посторонние SQL-символы и alias: <table>, AS t, t.column должны стать table и column.\n\n"
        f"Validation error:\n{validation_error}\n\n"
        f"Original query:\n{_normalize_query_text_for_parser(query)}\n\n"
        f"Current ParsedDataQuery JSON:\n{parsed_json}"
    )
    try:
        return _invoke_query_parser_json_fallback(
            query_parser_model=query_parser_model,
            messages=[("system", _QUERY_PARSER_SYSTEM_PROMPT), ("human", repair_prompt)],
        )
    except ValueError:
        return None


def _format_parsed_query_debug(parsed: ParsedDataQuery) -> str:
    """Формирует короткий JSON-дамп результата LLM-разбора для диагностики ошибок инструмента.

    Args:
        parsed: Структурированный результат LLM-разбора.

    Returns:
        JSON-строка с ключевыми полями ``ParsedDataQuery``.
    """

    return json.dumps(parsed.model_dump(mode="json"), ensure_ascii=False)


def _message_text(message: Any) -> str:
    """Извлекает текст из ответа chat-модели.

    Args:
        message: Ответ LangChain chat model или строка.

    Returns:
        Текстовое содержимое ответа.
    """

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Извлекает первый JSON-объект из текста модели.

    Args:
        text: Текстовый ответ LLM.

    Returns:
        Распарсенный JSON-объект.

    Raises:
        ValueError: В тексте нет JSON-объекта.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("в ответе модели нет JSON-объекта.")
    return json.loads(cleaned[start : end + 1])


def _normalize_query_text_for_parser(query: str) -> str:
    """Убирает из SQL-подобного запроса шум, который не должен влиять на LLM-разбор.

    Args:
        query: Исходный SQL-подобный запрос.

    Returns:
        Запрос без угловых скобок вокруг таблиц и без SQL-alias после ``LOAD``/``FROM``.
    """

    text = query.strip()
    text = re.sub(r"<([A-Za-z_][\w]*)>", r"\1", text)
    text = re.sub(r"(?im)^(\s*LOAD\s+)([A-Za-z_][\w]*)(?:\s+AS)?\s+[A-Za-z_][\w]*(\s*)$", r"\1\2\3", text)
    text = re.sub(
        r"(?is)\bFROM\s+([A-Za-z_][\w]*)(?:\s+AS)?\s+[A-Za-z_][\w]*\b",
        r"FROM \1",
        text,
    )
    return text


def _parsed_query_to_read_args(parsed: ParsedDataQuery) -> dict[str, Any]:
    """Преобразует результат LLM-разбора в аргументы внутренней выборки.

    Args:
        parsed: Структурированный результат LLM-разбора запроса.

    Returns:
        Словарь аргументов для ``_read_table``.

    Raises:
        ValueError: LLM сообщил проблему или вернул неполный запрос.
    """

    if parsed.status != "ready":
        details = parsed.problem or ", ".join(parsed.missing_inputs) or "query нельзя выполнить."
        raise ValueError(f"{parsed.status}: {details}")
    _validate_parsed_query(parsed)
    table_name = _normalize_table_alias(parsed.table_name)
    filters = [_normalize_filter_item(_dump_model(item)) for item in parsed.filters]
    derived_columns = [_normalize_derived_item(_dump_model(item)) for item in parsed.derived_columns]
    aggregations = [_normalize_aggregation_item(_dump_model(item)) for item in parsed.aggregations]
    order_by = [_normalize_order_item(_dump_model(item)) for item in parsed.order_by]
    return {
        "table_name": table_name,
        "select_columns": [_normalize_column_name(column) for column in parsed.select_columns if str(column).strip()],
        "filters": filters,
        "derived_columns": derived_columns,
        "group_by": [_normalize_column_name(column) for column in parsed.group_by if str(column).strip()],
        "aggregations": aggregations,
        "order_by": order_by,
        "max_rows": parsed.max_rows,
    }


def _validate_parsed_query(parsed: ParsedDataQuery) -> None:
    """Проверяет обязательные части запроса после LLM-разбора.

    Args:
        parsed: Структурированный результат LLM-разбора запроса.

    Returns:
        ``None``, если запрос можно выполнять.

    Raises:
        ValueError: В запросе нет обязательных колонок, периода или таблицы.
    """

    table_name = _normalize_table_alias(parsed.table_name)
    if not table_name:
        raise ValueError("needs_more_input: в query не указан alias таблицы.")
    if table_name not in TABLE_ALIASES:
        raise ValueError(
            f"schema_error: неизвестная таблица {parsed.table_name!r}. Доступные таблицы: {_available_table_aliases_text()}."
        )
    select_columns = [str(column).strip() for column in parsed.select_columns if str(column).strip()]
    if {column.lower() for column in select_columns} & {"*", "all"}:
        raise ValueError("schema_error: SELECT * и SELECT all запрещены для обычной выборки.")
    if not select_columns and not parsed.aggregations:
        raise ValueError("needs_more_input: в query нет явных колонок результата или агрегаций.")
    if not _has_required_period(parsed.filters):
        raise ValueError("needs_more_input: в query нет обязательного временного интервала с двумя границами.")


def _normalize_table_alias(value: Any) -> str:
    """Очищает имя таблицы от SQL-alias и служебных символов.

    Args:
        value: Значение ``table_name``, которое вернул LLM.

    Returns:
        Короткий alias таблицы или пустую строку.
    """

    text = str(value or "").strip().strip("`\"'").strip()
    text = re.sub(r"[<>]", "", text)
    text = re.sub(r"(?i)^\s*(LOAD|FROM)\s+", "", text).strip()
    text = re.split(r"(?i)\s+AS\s+|\s+", text, maxsplit=1)[0]
    if "." in text:
        text = text.split(".")[-1]
    return text.strip()


def _normalize_column_name(value: Any) -> str:
    """Очищает имя колонки от SQL-alias, кавычек и угловых скобок.

    Args:
        value: Имя колонки, которое вернул LLM.

    Returns:
        Имя колонки без префикса таблицы или SQL-alias.
    """

    text = str(value or "").strip().strip("`\"'").strip()
    text = re.sub(r"[<>]", "", text)
    if "." in text:
        text = text.split(".")[-1]
    return text.strip()


def _normalize_filter_item(item: dict[str, Any]) -> dict[str, Any]:
    """Очищает колонку фильтра от SQL-префиксов.

    Args:
        item: Фильтр, который вернул LLM.

    Returns:
        Фильтр с нормализованным именем колонки.
    """

    result = dict(item)
    result["column"] = _normalize_column_name(result.get("column"))
    result["operator"] = normalize_filter_operator(result.get("operator"))
    return result


def _normalize_derived_item(item: dict[str, Any]) -> dict[str, Any]:
    """Очищает вычисляемую колонку от SQL-префиксов.

    Args:
        item: Описание вычисляемой колонки.

    Returns:
        Описание с нормализованными именами колонок.
    """

    result = dict(item)
    result["name"] = _normalize_column_name(result.get("name"))
    result["source_column"] = _normalize_column_name(result.get("source_column"))
    return result


def _normalize_aggregation_item(item: dict[str, Any]) -> dict[str, Any]:
    """Очищает агрегат от SQL-префиксов.

    Args:
        item: Описание агрегата.

    Returns:
        Описание агрегата с нормализованной колонкой.
    """

    result = dict(item)
    column = str(result.get("column") or "").strip()
    result["column"] = "*" if column == "*" else _normalize_column_name(column)
    return result


def _normalize_order_item(item: dict[str, Any]) -> dict[str, Any]:
    """Очищает сортировку от SQL-префиксов.

    Args:
        item: Описание сортировки.

    Returns:
        Описание сортировки с нормализованной колонкой.
    """

    result = dict(item)
    result["column"] = _normalize_column_name(result.get("column"))
    return result


def _has_required_period(filters: list[Any]) -> bool:
    """Проверяет наличие фильтра временного интервала.

    Args:
        filters: Фильтры, которые вернул LLM-разбор.

    Returns:
        ``True``, если найден хотя бы один ``between`` с двумя границами.
    """

    for item in filters:
        operator = normalize_filter_operator(_get_field(item, "operator"))
        values = _get_field(item, "values") or []
        value = _get_field(item, "value")
        if operator == "between" and len(values) == 2:
            return True
    return False


def _dump_model(value: Any) -> dict[str, Any]:
    """Преобразует pydantic-модель или dict в обычный словарь.

    Args:
        value: Pydantic-модель или словарь.

    Returns:
        Словарь с полями модели.
    """

    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return dict(value)


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
    if operator == "contains_any":
        expression = None
        for value in _parse_filter_values(raw_value):
            item_expression = spark_column.cast("string").contains(value)
            expression = item_expression if expression is None else expression | item_expression
        if expression is None:
            raise ValueError("Для оператора contains_any нужно хотя бы одно значение.")
        return expression
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

    aggregation_columns = [
        column
        for item in aggregations
        for function, column, _alias in [_parse_aggregation_item(item)]
        if not (function == "count" and column == "*")
    ]
    missing = _validate_columns(
        columns=[*group_columns, *aggregation_columns],
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
        expression = functions.count("*") if column == "*" else functions.count(functions.col(column))
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
    "_extract_query_args_with_llm",
    "_parsed_query_to_read_args",
]

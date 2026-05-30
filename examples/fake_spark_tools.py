"""Spark-like инструмент для чтения локальных CSV-таблиц из examples/data.

Содержит:
- SparkTableFilter: схема одного фильтра для read_table.
- SparkTableAggregation: схема одной агрегации для read_table.
- SparkTableOrderBy: схема сортировки результата read_table.
- SparkColumnOperation: схема вычисляемой колонки read_table.
- SparkTableQueryInput: схема входа для read_table.
- build_fake_spark_tools: фабрика одного LangChain tool для запросов к Spark-like таблицам.
- _read_table_query: выполнение выборки по таблице, полям, фильтрам и лимиту.
- _apply_derived_columns: добавление вычисляемых колонок к DataFrame.
- _build_derived_column: расчет одной вычисляемой колонки.
- _apply_aggregations: выполнение агрегаций по всей таблице или группам.
- _aggregate_series: расчет одной агрегатной функции по Series.
- _aggregation_alias: получение имени итоговой колонки агрегации.
- _apply_order_by: сортировка результата.
- _load_spark_table: загрузка таблицы по логическому имени.
- _get_table_registry: создание реестра доступных Spark-like таблиц.
- _get_table_schema: получение схемы таблицы.
- _format_unknown_table_error: текст ошибки для неизвестной таблицы.
- _format_select_columns_error: текст ошибки для пустого или запрещенного select_columns.
- _format_unknown_columns_error: текст ошибки для отсутствующих колонок.
- _format_invalid_event_dt_filter_error: текст ошибки для некорректного формата фильтра event_dt.
- _format_schema_columns_text: компактное описание доступных колонок.
- _format_close_columns_text: подсказки по похожим именам колонок.
- _parse_select_columns: разбор строки колонок select_columns в список имен полей.
- _validate_select_columns_present: проверка, что агент явно указал нужные поля.
- _validate_query_columns: проверка наличия полей в таблице.
- _validate_aggregation_columns: проверка наличия полей агрегаций в таблице.
- _validate_order_columns: проверка наличия полей сортировки в таблице.
- _validate_derived_columns: проверка наличия исходных полей вычисляемых колонок.
- _validate_event_dt_filters: проверка формата значений фильтра event_dt.
- _is_yyyymmdd_value: проверка значения на формат YYYYMMDD.
- _apply_filters: применение списка фильтров к DataFrame.
- _apply_filter: применение одного фильтра к DataFrame.
- _should_compare_as_text: проверка необходимости строкового сравнения фильтра.
- _get_comparable_series: подготовка колонки к сравнению со значением фильтра.
- _coerce_filter_value: приведение значения фильтра к типу колонки.
- _clean_value: преобразование pandas/numpy значения к JSON-совместимому типу.
- _fake_sleep: имитация задержки Spark-запроса.
"""

from __future__ import annotations

import asyncio
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

from planner_agent.runtime.tool_text import ToolTextResult

DATA_DIR = Path(__file__).resolve().parent / "data"
HITS_FILE = "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
UKO_FILE = "csp_afpc_sss_inc.uko_event.csv"
CARDS_FILE = "csp_afpc_sss_inc.cards_event.csv"
HISTORY_AUTOMARKING_FILE = "csp_repo_features.history_automarking_big_148078_155487.csv"
DEMO_TIMELINE_FILE = "demo_client_timeline.csv"


FilterOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "in",
    "between",
    "is_null",
    "not_null",
]

# Только JSON-примитивы: GigaChat и др. провайдеры отклоняют схему с Any
# (в tool JSON Schema получается anyOf: [{}, {"type": "null"}] без type у первой ветки).
FilterScalar = str | int | float | bool
AggregationFunction = Literal["count", "count_distinct", "min", "max", "sum", "mean"]
OrderDirection = Literal["asc", "desc"]
ColumnOperation = Literal["year", "month", "year_month", "date", "lower", "upper", "length", "abs"]


class SparkTableFilter(BaseModel):
    """Описывает одно ограничение для выборки из Spark-like таблицы.

    Args:
        column: Имя поля, по которому нужно применить фильтр.
        operator: Оператор фильтрации: eq, ne, gt, gte, lt, lte, contains, in,
            between, is_null или not_null.
        value: Значение для операторов eq, ne, gt, gte, lt, lte и contains.
        values: Список значений для операторов in и between.

    Returns:
        Валидированное описание одного условия фильтрации.
    """

    column: str = Field(description="Имя поля таблицы для фильтрации.")
    operator: FilterOperator = Field(
        default="eq",
        description="Оператор фильтра: eq, ne, gt, gte, lt, lte, contains, in, between, is_null, not_null.",
    )
    value: FilterScalar | None = Field(
        default=None,
        description="Значение фильтра для операторов eq/ne/gt/gte/lt/lte/contains.",
    )
    values: list[FilterScalar] | None = Field(
        default=None,
        description="Список значений для операторов in и between.",
    )

    @model_validator(mode="after")
    def validate_filter_values(self) -> "SparkTableFilter":
        """Проверяет, что для выбранного оператора переданы нужные значения.

        Args:
            Отсутствуют.

        Returns:
            Текущий объект фильтра, если параметры заданы корректно.
        """

        if self.operator in {"is_null", "not_null"}:
            return self
        if self.operator == "between":
            if self.values is None or len(self.values) != 2:
                raise ValueError("Для оператора between нужно передать ровно два значения в values.")
            return self
        if self.operator == "in":
            if not self.values:
                raise ValueError("Для оператора in нужно передать непустой список values.")
            return self
        if self.value is None:
            raise ValueError(f"Для оператора {self.operator} нужно передать value.")
        return self


class SparkTableAggregation(BaseModel):
    """Описывает одну агрегатную операцию для выборки из Spark-like таблицы.

    Args:
        function: Агрегатная функция: count, count_distinct, min, max, sum или mean.
        column: Имя поля, по которому нужно выполнить агрегацию.
        alias: Необязательное имя итоговой колонки.

    Returns:
        Валидированное описание агрегатной операции.
    """

    function: AggregationFunction = Field(description="Агрегатная функция: count, count_distinct, min, max, sum, mean.")
    column: str = Field(description="Имя поля для агрегации.")
    alias: str | None = Field(default=None, description="Имя итоговой колонки. Если не задано, будет создано автоматически.")


class SparkTableOrderBy(BaseModel):
    """Описывает сортировку результата выборки.

    Args:
        column: Имя поля результата, по которому нужно сортировать.
        direction: Направление сортировки: asc или desc.

    Returns:
        Валидированное описание сортировки.
    """

    column: str = Field(description="Имя поля результата для сортировки.")
    direction: OrderDirection = Field(default="asc", description="Направление сортировки: asc или desc.")


class SparkColumnOperation(BaseModel):
    """Описывает вычисляемую колонку на основе одного исходного поля.

    Args:
        name: Имя новой колонки.
        source_column: Исходная колонка таблицы.
        operation: Операция над исходной колонкой: year, month, year_month, date, lower, upper, length или abs.

    Returns:
        Валидированное описание вычисляемой колонки.
    """

    name: str = Field(description="Имя новой вычисляемой колонки.")
    source_column: str = Field(description="Исходная колонка таблицы.")
    operation: ColumnOperation = Field(
        description="Операция над колонкой: year, month, year_month, date, lower, upper, length или abs.",
    )


class SparkTableQueryInput(BaseModel):
    """Параметры универсальной выборки из Spark-like таблицы.

    Args:
        table_name: Логическое имя таблицы из списка доступных таблиц.
        select_columns: Минимально достаточный непустой список полей в формате строки "col1, col2, col3".
        filters: Ограничения для отбора строк.
        derived_columns: Вычисляемые колонки, которые нужно добавить перед фильтрацией, агрегацией или сортировкой.
        group_by: Поля группировки для агрегаций в формате строки "col1, col2".
        aggregations: Агрегатные операции, которые нужно выполнить после фильтрации.
        order_by: Правила сортировки результата.
        max_rows: Максимальное число строк в ответе.
        include_schema: Если True, добавить схему таблицы даже при успешной выборке.

    Returns:
        Валидированные параметры запроса к Spark-like таблице.
    """

    table_name: str = Field(description="Имя Spark-like таблицы, например hits_extra_info, uko_event или cards_event.")
    select_columns: str = Field(
        default="",
        description=(
            "Минимально достаточный список полей для выборки в формате строки 'col1, col2, col3'. "
            "Выгрузка всех полей запрещена."
        ),
    )
    filters: list[SparkTableFilter] = Field(
        default_factory=list,
        description="Список фильтров, которые нужно применить к строкам таблицы.",
    )
    derived_columns: list[SparkColumnOperation] = Field(
        default_factory=list,
        description="Список вычисляемых колонок, которые добавляются перед фильтрацией, агрегацией и сортировкой.",
    )
    group_by: str | None = Field(
        default=None,
        description="Поля группировки для aggregations в формате 'col1, col2'. Без aggregations не применяется.",
    )
    aggregations: list[SparkTableAggregation] = Field(
        default_factory=list,
        description="Список агрегатных функций после фильтрации: count, count_distinct, min, max, sum, mean.",
    )
    order_by: list[SparkTableOrderBy] = Field(
        default_factory=list,
        description="Список правил сортировки итогового результата.",
    )
    max_rows: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Максимальное число строк в ответе. Если значение не передано, "
            "возвращаются все строки, подходящие под фильтры. Значение 0 вернет пустую выборку."
        ),
    )
    include_schema: bool = Field(
        default=False,
        description="Если True, вернуть схему таблицы вместе с результатом выборки.",
    )


def build_fake_spark_tools(
        *,
        delay_seconds: float = 1.5,
        data_dir: str | Path | None = None,
) -> list[BaseTool]:
    """Создает один универсальный Spark-like tool для запросов к локальным CSV-таблицам.

    Args:
        delay_seconds: Искусственная задержка каждого tool-вызова в секундах.
        data_dir: Директория с CSV-файлами. По умолчанию используется examples/data.

    Returns:
        Список с одним LangChain tool: read_table.
    """

    resolved_data_dir = Path(data_dir).resolve() if data_dir else DATA_DIR

    async def read_table(
            table_name: str,
            select_columns: str | None = None,
            filters: list[SparkTableFilter] | None = None,
            derived_columns: list[SparkColumnOperation] | None = None,
            group_by: str | None = None,
            aggregations: list[SparkTableAggregation] | None = None,
            order_by: list[SparkTableOrderBy] | None = None,
            max_rows: int | None = None,
            include_schema: bool = False,
    ) -> pd.DataFrame | str:
        """Выполняет универсальную выборку из Spark-like таблицы.

        Args:
            table_name: Имя таблицы.
            select_columns: Минимально достаточный список полей результата в формате строки "col1, col2, col3".
            filters: Ограничения выборки.
            derived_columns: Вычисляемые колонки, которые добавляются перед фильтрацией, агрегацией и сортировкой.
            group_by: Поля группировки для агрегаций в формате строки "col1, col2".
            aggregations: Агрегатные операции после фильтрации.
            order_by: Правила сортировки итогового результата.
            max_rows: Максимальное число строк. Если не передано, ограничение не применяется.
            include_schema: Признак возврата схемы таблицы.

        Returns:
            DataFrame с результатом выборки или текстовое описание ошибки и способа исправления.
        """

        return await _read_table_query(
            table_name=table_name,
            select_columns=select_columns or "",
            filters=filters or [],
            derived_columns=derived_columns or [],
            group_by=group_by,
            aggregations=aggregations or [],
            order_by=order_by or [],
            max_rows=max_rows,
            include_schema=include_schema,
            data_dir=resolved_data_dir,
            delay_seconds=delay_seconds,
        )

    return [
        StructuredTool.from_function(
            coroutine=read_table,
            name="read_table",
            description=(
                "read_table\n"
                "---\n"
                "Описание: универсальная выборка из Spark-like таблиц. "
                "Инструмент принимает имя таблицы, строку со списком полей, фильтры, "
                "вычисляемые колонки, группировки, агрегации, сортировку и лимит строк, "
                "а при успешной выборке возвращает pandas DataFrame.\n"
                "Если в select_columns или filters указанного поля нет в таблице, инструмент "
                "вернет текстовую ошибку с кодом, причиной, доступными полями и подсказкой для повтора. "
                "Выгрузка всех столбцов запрещена: агент должен явно указать "
                "минимально достаточный набор колонок.\n\n"
                "Семантика выборки (важно):\n"
                "  - select_columns возвращает СТРОКИ как есть, без устранения дублей (это не DISTINCT). "
                "Чтобы увидеть ВСЕ подошедшие строки, не задавай ограничивающий max_rows: инструмент "
                "вернёт все строки под фильтр, а большие результаты уходят в offload-файл.\n"
                "  - Уникальные значения поля (например перечень event_description) НЕ обязательно "
                "получать через group_by: можно прочитать все строки этого поля и вывести уникальные "
                "значения самостоятельно (drop_duplicates/value_counts) по полному набору. group_by + "
                "aggregations — лишь удобный способ сразу получить группы со счётчиками.\n"
                "  - Инструмент возвращает ВСЕ строки результата. Если они помещаются, они целиком "
                "приходят в контекст; если результат большой, он сохраняется в offload-файл, а в "
                "контекст приходит preview с числом строк в файле и инструкцией прочитать его "
                "целиком через execute_python_code.\n\n"
                "Параметры:\n"
                "  table_name (str, обяз.) — имя таблицы или алиас.\n"
                "  select_columns (str, обяз.) — минимально достаточные поля результата "
                "в формате 'col1, col2, col3'. Пустая строка, '*' и 'all' запрещены.\n"
                "  filters (list[dict], опц.) — фильтры вида "
                "{column, operator, value/values}. Операторы: eq, ne, gt, gte, lt, lte, "
                "contains, in, between, is_null, not_null. Для event_dt значения должны быть "
                "датами YYYYMMDD; месяц YYYYMM нужно задавать через between по границам месяца "
                "или через derived_columns year_month.\n"
                "  derived_columns (list[dict], опц.) — вычисляемые поля вида "
                "{name, source_column, operation}. Операции: year, month, year_month, date, lower, upper, length, abs.\n"
                "  group_by (str, опц.) — поля группировки для aggregations в формате 'col1, col2'.\n"
                "  aggregations (list[dict], опц.) — агрегаты вида {function, column, alias}. "
                "Функции: count, count_distinct, min, max, sum, mean.\n"
                "  order_by (list[dict], опц.) — сортировка вида {column, direction}, direction: asc или desc.\n"
                "  max_rows (int, опц.) — максимум строк в ответе; если не передан, лимит не применяется.\n"
                "  include_schema (bool, опц., False) — вернуть схему при успешной выборке."
            ),
            args_schema=SparkTableQueryInput,
        ),
    ]


async def _read_table_query(
        *,
        table_name: str,
        select_columns: str,
        filters: list[SparkTableFilter],
        derived_columns: list[SparkColumnOperation],
        group_by: str | None,
        aggregations: list[SparkTableAggregation],
        order_by: list[SparkTableOrderBy],
        max_rows: int | None,
        include_schema: bool,
        data_dir: Path,
        delay_seconds: float,
) -> pd.DataFrame | str:
    """Выполняет выборку из Spark-like таблицы с проверкой полей.

    Args:
        table_name:  имя таблицы.
        select_columns: Минимально достаточные поля результата в формате строки "col1, col2, col3".
        filters: Список фильтров.
        derived_columns: Вычисляемые колонки, которые добавляются перед фильтрацией, агрегацией и сортировкой.
        group_by: Поля группировки для агрегаций в формате строки "col1, col2".
        aggregations: Список агрегатных операций после фильтрации.
        order_by: Правила сортировки итогового результата.
        max_rows: Максимальное число строк. Если не передано, ограничение не применяется.
        include_schema: Признак возврата схемы при успешном ответе.
        data_dir: Директория с CSV-файлами.
        delay_seconds: Искусственная задержка запроса.

    Returns:
        DataFrame с результатом выборки или текстовое описание ошибки и способа исправления.
    """

    await _fake_sleep(delay_seconds)
    registry = _get_table_registry()
    normalized_table_name = table_name.strip()
    table_meta = registry.get(normalized_table_name)
    if table_meta is None:
        return _format_unknown_table_error(table_name=table_name, registry=registry)

    table = _load_spark_table(data_dir=data_dir, table_name=normalized_table_name)
    source_schema = _get_table_schema(table_name=normalized_table_name, source_file=table_meta["file"], table=table)
    missing_derived_columns = _validate_derived_columns(table=table, derived_columns=derived_columns)
    if missing_derived_columns:
        return _format_unknown_columns_error(
            table_name=normalized_table_name,
            source_file=table_meta["file"],
            missing_columns=sorted(set(missing_derived_columns)),
            schema=source_schema,
        )
    table = _apply_derived_columns(table=table, derived_columns=derived_columns)
    schema = _get_table_schema(table_name=normalized_table_name, source_file=table_meta["file"], table=table)
    parsed_select_columns = _parse_select_columns(select_columns)
    group_by_columns = _parse_select_columns(group_by or "")
    select_error = _validate_select_columns_present(parsed_select_columns)
    if select_error is not None:
        return _format_select_columns_error(
            table_name=normalized_table_name,
            source_file=table_meta["file"],
            select_error=select_error,
            schema=schema,
        )
    aggregate_output_columns = [*group_by_columns, *[_aggregation_alias(item) for item in aggregations]]
    known_output_columns = aggregate_output_columns if aggregations else []
    missing_columns = _validate_query_columns(
        table=table,
        select_columns=parsed_select_columns,
        filters=filters,
        known_output_columns=known_output_columns,
    )
    missing_columns.extend(_validate_aggregation_columns(table=table, aggregations=aggregations, group_by=group_by_columns))
    if missing_columns:
        return _format_unknown_columns_error(
            table_name=normalized_table_name,
            source_file=table_meta["file"],
            missing_columns=sorted(set(missing_columns)),
            schema=schema,
        )
    invalid_event_dt_filters = _validate_event_dt_filters(filters=filters)
    if invalid_event_dt_filters:
        return _format_invalid_event_dt_filter_error(
            table_name=normalized_table_name,
            source_file=table_meta["file"],
            invalid_filters=invalid_event_dt_filters,
        )

    filtered = _apply_filters(table=table, filters=filters)
    if aggregations:
        result = _apply_aggregations(table=filtered, group_by=group_by_columns, aggregations=aggregations)
    else:
        result_columns = parsed_select_columns
        result = filtered.loc[:, result_columns].copy()

    missing_order_columns = _validate_order_columns(table=result, order_by=order_by)
    if missing_order_columns:
        result_schema = _get_table_schema(table_name=normalized_table_name, source_file=table_meta["file"], table=result)
        return _format_unknown_columns_error(
            table_name=normalized_table_name,
            source_file=table_meta["file"],
            missing_columns=missing_order_columns,
            schema=result_schema,
        )
    result = _apply_order_by(table=result, order_by=order_by)
    if max_rows is not None:
        result = result.head(max(0, int(max_rows))).copy()
    if include_schema:
        result.attrs["spark_schema"] = schema
    result.attrs["spark_table_name"] = normalized_table_name
    result.attrs["spark_source_file"] = table_meta["file"]
    result.attrs["spark_total_rows"] = int(len(table))
    result.attrs["spark_matched_rows"] = int(len(filtered))
    return result


def _load_spark_table(*, data_dir: Path, table_name: str) -> pd.DataFrame:
    """Загружает Spark-like таблицу по логическому имени.

    Args:
        data_dir: Директория с CSV-файлами.
        table_name: Имя таблицы или алиас из реестра.

    Returns:
        DataFrame с содержимым CSV-файла.
    """

    table_meta = _get_table_registry()[table_name]
    path = data_dir / table_meta["file"]
    return _load_csv_table(str(path.resolve())).copy()


def _apply_derived_columns(*, table: pd.DataFrame, derived_columns: list[SparkColumnOperation]) -> pd.DataFrame:
    """Добавляет вычисляемые колонки к таблице.

    Args:
        table: Исходная таблица.
        derived_columns: Описания вычисляемых колонок.

    Returns:
        Копия таблицы с добавленными вычисляемыми колонками.
    """

    result = table.copy()
    for operation in derived_columns:
        result[operation.name] = _build_derived_column(table=result, operation=operation)
    return result


def _build_derived_column(*, table: pd.DataFrame, operation: SparkColumnOperation) -> pd.Series:
    """Рассчитывает одну вычисляемую колонку.

    Args:
        table: Таблица с исходной колонкой.
        operation: Описание операции над колонкой.

    Returns:
        Series с рассчитанными значениями новой колонки.
    """

    source = table[operation.source_column]
    if operation.operation == "lower":
        return source.astype("string").str.lower()
    if operation.operation == "upper":
        return source.astype("string").str.upper()
    if operation.operation == "length":
        return source.astype("string").str.len()
    if operation.operation == "abs":
        return pd.to_numeric(source, errors="coerce").abs()

    source_text = source.astype("string").str.replace(r"\D", "", regex=True)
    if operation.operation == "year":
        return source_text.str.slice(0, 4)
    if operation.operation == "month":
        return source_text.str.slice(4, 6)
    if operation.operation == "year_month":
        return source_text.str.slice(0, 6)
    if operation.operation == "date":
        return source_text.str.slice(0, 8)
    raise ValueError(f"Неподдерживаемая операция вычисляемой колонки: {operation.operation}")


def _apply_aggregations(
        *,
        table: pd.DataFrame,
        group_by: list[str],
        aggregations: list[SparkTableAggregation],
) -> pd.DataFrame:
    """Выполняет агрегатные операции по всей таблице или по группам.

    Args:
        table: Отфильтрованная таблица.
        group_by: Поля группировки.
        aggregations: Список агрегатных операций.

    Returns:
        DataFrame с результатом агрегаций.
    """

    if not group_by:
        row = {
            _aggregation_alias(aggregation): _aggregate_series(
                series=table[aggregation.column],
                function=aggregation.function,
            )
            for aggregation in aggregations
        }
        return pd.DataFrame([row])

    rows: list[dict[str, Any]] = []
    grouped = table.groupby(group_by, dropna=False, sort=False)
    for group_key, group in grouped:
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(group_by, key_values, strict=False))
        for aggregation in aggregations:
            row[_aggregation_alias(aggregation)] = _aggregate_series(
                series=group[aggregation.column],
                function=aggregation.function,
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=[*group_by, *[_aggregation_alias(item) for item in aggregations]])


def _aggregate_series(*, series: pd.Series, function: AggregationFunction) -> Any:
    """Рассчитывает одну агрегатную функцию по колонке.

    Args:
        series: Колонка для агрегации.
        function: Имя агрегатной функции.

    Returns:
        Скалярный результат агрегации.
    """

    if function == "count":
        return int(series.count())
    if function == "count_distinct":
        return int(series.nunique(dropna=True))
    if function in {"sum", "mean"}:
        numeric = pd.to_numeric(series, errors="coerce")
        value = numeric.sum() if function == "sum" else numeric.mean()
        return _clean_value(value)
    if function == "min":
        return _clean_value(series.min())
    if function == "max":
        return _clean_value(series.max())
    raise ValueError(f"Неподдерживаемая агрегатная функция: {function}")


def _aggregation_alias(aggregation: SparkTableAggregation) -> str:
    """Возвращает имя итоговой колонки для агрегатной операции.

    Args:
        aggregation: Описание агрегатной операции.

    Returns:
        Явный alias или автоматически построенное имя колонки.
    """

    return aggregation.alias or f"{aggregation.function}_{aggregation.column}"


def _apply_order_by(*, table: pd.DataFrame, order_by: list[SparkTableOrderBy]) -> pd.DataFrame:
    """Сортирует таблицу по одному или нескольким полям.

    Args:
        table: Таблица результата.
        order_by: Правила сортировки.

    Returns:
        Отсортированная копия таблицы.
    """

    if not order_by:
        return table
    columns = [item.column for item in order_by]
    ascending = [item.direction == "asc" for item in order_by]
    return table.sort_values(by=columns, ascending=ascending, na_position="last").copy()


@lru_cache(maxsize=16)
def _load_csv_table(path_text: str) -> pd.DataFrame:
    """Загружает CSV-файл с кешированием по абсолютному пути.

    Args:
        path_text: Абсолютный путь к CSV-файлу.

    Returns:
        DataFrame с содержимым CSV-файла.
    """

    return pd.read_csv(path_text, low_memory=False)


def _get_table_registry() -> dict[str, dict[str, str]]:
    """Возвращает реестр доступных Spark-like таблиц и алиасов.

    Args:
        Отсутствуют.

    Returns:
        Словарь, где ключ — имя таблицы, а значение содержит имя CSV-файла.
    """

    return {
        "hits": {"file": HITS_FILE},
        "hits_extra_info": {"file": HITS_FILE},
        "cspfs_repo_features3.hits_extra_info_129372427_view": {"file": HITS_FILE},
        "hits_extra_info_129372427_view": {"file": HITS_FILE},
        "uko_event": {"file": UKO_FILE},
        "csp_afpc_sss_inc.uko_event": {"file": UKO_FILE},
        "cspfs_repo_features3.uko_event": {"file": UKO_FILE},
        "cards_event": {"file": CARDS_FILE},
        "csp_afpc_sss_inc.cards_event": {"file": CARDS_FILE},
        "cspfs_repo_features3.cards_event": {"file": CARDS_FILE},
        "history_automarking": {"file": HISTORY_AUTOMARKING_FILE},
        "demo_client_timeline": {"file": DEMO_TIMELINE_FILE},

    }


def _get_table_schema(*, table_name: str, source_file: str, table: pd.DataFrame) -> dict[str, Any]:
    """Формирует схему Spark-like таблицы по DataFrame.

    Args:
        table_name: Логическое имя таблицы.
        source_file: Имя CSV-файла источника.
        table: DataFrame, для которого нужна схема.

    Returns:
        Словарь со списком колонок, типами и признаком nullable.
    """

    return {
        "table_name": table_name,
        "source_file": source_file,
        "columns_count": int(len(table.columns)),
        "columns": [
            {
                "name": column,
                "type": str(table[column].dtype),
                "nullable": bool(table[column].isna().any()),
            }
            for column in table.columns
        ],
    }


def _format_unknown_table_error(*, table_name: str, registry: dict[str, dict[str, str]]) -> str:
    """Формирует текстовую ошибку для неизвестного имени таблицы.

    Args:
        table_name: Имя таблицы, переданное агентом.
        registry: Реестр доступных таблиц и алиасов.

    Returns:
        Человекочитаемый текст ошибки с доступными вариантами table_name.
    """

    available_tables = ", ".join(sorted(registry))
    return ToolTextResult(
        "Ошибка инструмента read_table: таблица не найдена.\n"
        "Код ошибки: unknown_table.\n"
        f"Запрошенная таблица: {table_name!r}.\n"
        f"Доступные таблицы и алиасы: {available_tables}.\n"
        "Как исправить: выбери одно из доступных имен table_name и повтори запрос "
        "с явным минимальным списком select_columns.",
        is_error=True,
    )


def _format_select_columns_error(
        *,
        table_name: str,
        source_file: str,
        select_error: dict[str, Any],
        schema: dict[str, Any],
) -> str:
    """Формирует текстовую ошибку для пустого или запрещенного select_columns.

    Args:
        table_name: Имя таблицы или алиаса.
        source_file: CSV-файл источника.
        select_error: Описание ошибки валидации select_columns.
        schema: Схема таблицы с доступными колонками.

    Returns:
        Человекочитаемый текст ошибки с подсказкой, как выбрать поля.
    """

    forbidden_columns = select_error.get("forbidden_columns")
    forbidden_text = ""
    if forbidden_columns:
        forbidden_text = f"\nЗапрещенные маркеры в select_columns: {', '.join(map(str, forbidden_columns))}."
    return ToolTextResult(
        "Ошибка инструмента read_table: некорректный список select_columns.\n"
        f"Код ошибки: {select_error['code']}.\n"
        f"Таблица: {table_name}; источник: {source_file}.\n"
        f"Причина: {select_error['message']}{forbidden_text}\n"
        f"{_format_schema_columns_text(schema=schema)}\n"
        "Как исправить: повтори запрос с минимальным списком реально нужных колонок, "
        "например 'event_id, event_dt, epk_id'. Не используй пустую строку, '*' или 'all'.",
        is_error=True,
    )


def _format_unknown_columns_error(
        *,
        table_name: str,
        source_file: str,
        missing_columns: list[str],
        schema: dict[str, Any],
) -> str:
    """Формирует текстовую ошибку для колонок, которых нет в таблице.

    Args:
        table_name: Имя таблицы или алиаса.
        source_file: CSV-файл источника.
        missing_columns: Отсутствующие колонки из select_columns или filters.
        schema: Схема таблицы с доступными колонками.

    Returns:
        Человекочитаемый текст ошибки с похожими колонками и вариантом повтора.
    """

    missing_text = ", ".join(missing_columns)
    close_columns = _format_close_columns_text(missing_columns=missing_columns, schema=schema)
    close_block = f"\nПохожие доступные поля: {close_columns}." if close_columns else ""
    return ToolTextResult(
        "Ошибка инструмента read_table: в таблице нет одного или нескольких полей из запроса.\n"
        "Код ошибки: unknown_columns.\n"
        f"Таблица: {table_name}; источник: {source_file}.\n"
        f"Отсутствующие поля: {missing_text}.{close_block}\n"
        f"{_format_schema_columns_text(schema=schema)}\n"
        "Как исправить: замени отсутствующие поля на доступные из схемы либо явно сообщи, "
        "что skill/план ожидает поля, которых нет в текущей таблице.",
        is_error=True,
    )


def _format_invalid_event_dt_filter_error(
        *,
        table_name: str,
        source_file: str,
        invalid_filters: list[SparkTableFilter],
) -> str:
    """Формирует текст ошибки для некорректного формата фильтра по event_dt.

    Args:
        table_name: Имя таблицы или алиаса.
        source_file: CSV-файл источника.
        invalid_filters: Фильтры по event_dt с некорректным форматом значений.

    Returns:
        Человекочитаемый текст ошибки с правилом корректного фильтра по event_dt.
    """

    filters_text = "; ".join(
        f"operator={item.operator}, value={item.value}, values={item.values}"
        for item in invalid_filters
    )
    return ToolTextResult(
        "Ошибка инструмента read_table: некорректный формат значения для поля event_dt.\n"
        "Код ошибки: invalid_event_dt_filter.\n"
        f"Таблица: {table_name}; источник: {source_file}.\n"
        f"Некорректные фильтры: {filters_text}.\n"
        "Поле event_dt хранит дневную дату в формате YYYYMMDD. "
        "Для фильтра за месяц используй operator=between и values вида "
        "[\"YYYYMM01\", \"YYYYMMDD\"] с последним днем месяца либо создай "
        "derived_columns с operation=year_month и фильтруй вычисляемую колонку по YYYYMM. "
        "Не используй event_dt eq YYYYMM.",
        is_error=True,
    )


def _format_schema_columns_text(*, schema: dict[str, Any]) -> str:
    """Формирует компактный текст со списком доступных колонок таблицы.

    Args:
        schema: Схема таблицы, сформированная ``_get_table_schema``.

    Returns:
        Текст со всеми именами колонок в порядке схемы.
    """

    column_names = [str(column["name"]) for column in schema.get("columns", [])]
    return f"Доступные поля ({len(column_names)}): {', '.join(column_names)}."


def _format_close_columns_text(*, missing_columns: list[str], schema: dict[str, Any]) -> str:
    """Подбирает похожие имена колонок для отсутствующих полей.

    Args:
        missing_columns: Имена колонок, которых нет в таблице.
        schema: Схема таблицы с доступными колонками.

    Returns:
        Строка с подсказками вида ``missing -> candidate1, candidate2`` или пустая строка.
    """

    available_columns = [str(column["name"]) for column in schema.get("columns", [])]
    hints: list[str] = []
    for missing_column in missing_columns:
        matches = get_close_matches(missing_column, available_columns, n=3, cutoff=0.45)
        if matches:
            hints.append(f"{missing_column} -> {', '.join(matches)}")
    return "; ".join(hints)


def _parse_select_columns(select_columns: str) -> list[str]:
    """Разбирает строку колонок select_columns в список имен полей.

    Args:
        select_columns: Строка с колонками в формате "col1, col2, col3".

    Returns:
        Список имен колонок без пробелов по краям и пустых элементов.
    """

    return [column.strip() for column in select_columns.split(",") if column.strip()]


def _validate_select_columns_present(select_columns: list[str]) -> dict[str, Any] | None:
    """Проверяет, что запрошен явный минимальный набор колонок.

    Args:
        select_columns: Поля, которые агент хочет выгрузить.

    Returns:
        None для корректного списка или словарь ошибки для ответа tool.
    """

    normalized = {str(column).strip().lower() for column in select_columns}
    if not normalized or "" in normalized:
        return {
            "code": "select_columns_required",
            "message": (
                "Нужно явно указать минимально достаточный список полей в select_columns. "
                "Выгрузка всех столбцов запрещена."
            ),
        }
    if normalized & {"*", "all"}:
        return {
            "code": "select_all_forbidden",
            "message": (
                "Запрещено запрашивать все столбцы таблицы. Укажите только поля, "
                "которые действительно нужны для текущей задачи."
            ),
            "forbidden_columns": sorted(normalized & {"*", "all"}),
        }
    return None


def _validate_query_columns(
        *,
        table: pd.DataFrame,
        select_columns: list[str],
        filters: list[SparkTableFilter],
        known_output_columns: list[str] | None = None,
) -> list[str]:
    """Проверяет, что все запрошенные поля есть в таблице.

    Args:
        table: Таблица для проверки.
        select_columns: Поля результата.
        filters: Фильтры, поля которых нужно проверить.
        known_output_columns: Дополнительные поля результата, которые появятся после агрегаций.

    Returns:
        Отсортированный список отсутствующих полей.
    """

    available_columns = set(table.columns)
    available_columns.update(known_output_columns or [])
    requested_columns = set(select_columns)
    requested_columns.update(filter_item.column for filter_item in filters)
    return sorted(column for column in requested_columns if column not in available_columns)


def _validate_aggregation_columns(
        *,
        table: pd.DataFrame,
        aggregations: list[SparkTableAggregation],
        group_by: list[str],
) -> list[str]:
    """Проверяет наличие колонок, используемых в агрегациях и группировках.

    Args:
        table: Таблица после добавления вычисляемых колонок.
        aggregations: Агрегатные операции.
        group_by: Поля группировки.

    Returns:
        Отсортированный список отсутствующих колонок.
    """

    requested_columns = set(group_by)
    requested_columns.update(aggregation.column for aggregation in aggregations)
    return sorted(column for column in requested_columns if column not in table.columns)


def _validate_order_columns(*, table: pd.DataFrame, order_by: list[SparkTableOrderBy]) -> list[str]:
    """Проверяет наличие колонок сортировки в итоговом результате.

    Args:
        table: Итоговая таблица до сортировки.
        order_by: Правила сортировки.

    Returns:
        Отсортированный список отсутствующих колонок сортировки.
    """

    requested_columns = {item.column for item in order_by}
    return sorted(column for column in requested_columns if column not in table.columns)


def _validate_derived_columns(*, table: pd.DataFrame, derived_columns: list[SparkColumnOperation]) -> list[str]:
    """Проверяет наличие исходных колонок для вычисляемых полей.

    Args:
        table: Исходная таблица.
        derived_columns: Описания вычисляемых колонок.

    Returns:
        Отсортированный список отсутствующих исходных колонок.
    """

    requested_columns = {item.source_column for item in derived_columns}
    return sorted(column for column in requested_columns if column not in table.columns)


def _validate_event_dt_filters(*, filters: list[SparkTableFilter]) -> list[SparkTableFilter]:
    """Проверяет, что фильтры по event_dt используют дневной формат YYYYMMDD.

    Args:
        filters: Список фильтров read_table.

    Returns:
        Список фильтров по event_dt, где значения переданы не в формате YYYYMMDD.
    """

    invalid_filters: list[SparkTableFilter] = []
    for filter_item in filters:
        if filter_item.column != "event_dt" or filter_item.operator in {"is_null", "not_null"}:
            continue
        if filter_item.operator == "between":
            values = filter_item.values or []
            if any(not _is_yyyymmdd_value(value) for value in values):
                invalid_filters.append(filter_item)
            continue
        if filter_item.operator == "in":
            values = filter_item.values or []
            if any(not _is_yyyymmdd_value(value) for value in values):
                invalid_filters.append(filter_item)
            continue
        if filter_item.operator == "contains" or not _is_yyyymmdd_value(filter_item.value):
            invalid_filters.append(filter_item)
    return invalid_filters


def _is_yyyymmdd_value(value: FilterScalar | None) -> bool:
    """Проверяет, что значение выглядит как дата YYYYMMDD.

    Args:
        value: Значение фильтра, переданное агентом.

    Returns:
        True, если значение является строкой из восьми цифр или целым числом из восьми цифр.
    """

    if value is None or isinstance(value, bool):
        return False
    text_value = str(value).strip()
    return len(text_value) == 8 and text_value.isdigit()


def _apply_filters(*, table: pd.DataFrame, filters: list[SparkTableFilter]) -> pd.DataFrame:
    """Последовательно применяет фильтры к таблице.

    Args:
        table: Исходная таблица.
        filters: Список фильтров.

    Returns:
        Отфильтрованный DataFrame.
    """

    result = table.copy()
    for filter_item in filters:
        result = _apply_filter(table=result, filter_item=filter_item)
    return result


def _apply_filter(*, table: pd.DataFrame, filter_item: SparkTableFilter) -> pd.DataFrame:
    """Применяет один фильтр к таблице.

    Args:
        table: Таблица для фильтрации.
        filter_item: Описание фильтра.

    Returns:
        DataFrame со строками, которые соответствуют фильтру.
    """

    series = table[filter_item.column]
    operator = filter_item.operator
    if operator == "is_null":
        return table[series.isna()].copy()
    if operator == "not_null":
        return table[series.notna()].copy()
    if operator == "contains":
        value = "" if filter_item.value is None else str(filter_item.value)
        return table[series.astype(str).str.contains(value, case=False, na=False, regex=False)].copy()
    if operator == "in":
        if any(_should_compare_as_text(series=series, value=value) for value in filter_item.values or []):
            values = [str(value) for value in filter_item.values or []]
            return table[series.astype(str).isin(values)].copy()
        values = [_coerce_filter_value(series=series, value=value) for value in filter_item.values or []]
        comparable = _get_comparable_series(series=series, value=values[0] if values else None)
        return table[comparable.isin(values)].copy()
    if operator == "between":
        values = filter_item.values or []
        left = _coerce_filter_value(series=series, value=values[0])
        right = _coerce_filter_value(series=series, value=values[1])
        comparable = _get_comparable_series(series=series, value=left)
        return table[(comparable >= left) & (comparable <= right)].copy()

    if operator in {"eq", "ne"} and _should_compare_as_text(series=series, value=filter_item.value):
        value = str(filter_item.value)
        comparable = series.astype(str)
    else:
        value = _coerce_filter_value(series=series, value=filter_item.value)
        comparable = _get_comparable_series(series=series, value=value)
    if operator == "eq":
        mask = comparable == value
    elif operator == "ne":
        mask = comparable != value
    elif operator == "gt":
        mask = comparable > value
    elif operator == "gte":
        mask = comparable >= value
    elif operator == "lt":
        mask = comparable < value
    elif operator == "lte":
        mask = comparable <= value
    else:
        raise ValueError(f"Неподдерживаемый оператор фильтра: {operator}")
    return table[mask].copy()


def _should_compare_as_text(*, series: pd.Series, value: Any) -> bool:
    """Проверяет, нужно ли сравнивать значение фильтра как текст.

    Args:
        series: Колонка, к которой применяется фильтр.
        value: Значение фильтра из запроса агента.

    Returns:
        ``True``, если строковое сравнение безопаснее числового приведения.
    """

    if value is None or pd.api.types.is_numeric_dtype(series):
        return False
    if isinstance(value, str):
        return True
    return isinstance(value, int) and abs(value) > 2**53 - 1


def _get_comparable_series(*, series: pd.Series, value: Any) -> pd.Series:
    """Готовит колонку к сравнению со значением фильтра.

    Args:
        series: Исходная колонка таблицы.
        value: Значение фильтра после базового приведения.

    Returns:
        Колонка, приведенная к числовому типу или строке, если это нужно для сравнения.
    """

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    if value is None:
        return series
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric_value) and numeric_series.notna().any():
        return numeric_series
    return series.astype(str)


def _coerce_filter_value(*, series: pd.Series, value: Any) -> Any:
    """Приводит значение фильтра к типу колонки, если это возможно.

    Args:
        series: Колонка, с которой сравнивается значение.
        value: Исходное значение фильтра.

    Returns:
        Значение, приведенное к числу для числовых колонок, иначе исходное значение.
    """

    if value is None:
        return None
    numeric_series = pd.to_numeric(series, errors="coerce")
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.api.types.is_numeric_dtype(series) or (not pd.isna(numeric_value) and numeric_series.notna().any()):
        return _clean_value(numeric_value)
    return str(value) if pd.api.types.is_string_dtype(series) else value


def _clean_value(value: Any) -> Any:
    """Преобразует значение pandas/numpy к JSON-совместимому типу.

    Args:
        value: Значение из DataFrame.

    Returns:
        JSON-совместимое значение.
    """

    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


async def _fake_sleep(delay_seconds: float) -> None:
    """Выполняет безопасную асинхронную задержку.

    Args:
        delay_seconds: Количество секунд задержки.

    Returns:
        None.
    """

    await asyncio.sleep(max(0.0, float(delay_seconds)))


__all__ = [
    "SparkTableFilter",
    "SparkTableQueryInput",
    "build_fake_spark_tools",
]

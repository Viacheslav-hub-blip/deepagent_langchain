"""Spark-like инструмент для чтения локальных CSV-таблиц из examples/data.

Содержит:
- SparkTableFilter: схема одного фильтра для spark_query_table.
- SparkTableQueryInput: схема входа для spark_query_table.
- build_fake_spark_tools: фабрика одного LangChain tool для запросов к Spark-like таблицам.
- _spark_query_table: выполнение выборки по таблице, полям, фильтрам и лимиту.
- _load_spark_table: загрузка таблицы по логическому имени.
- _get_table_registry: создание реестра доступных Spark-like таблиц.
- _get_table_schema: получение схемы таблицы.
- _validate_select_columns_present: проверка, что агент явно указал нужные поля.
- _validate_query_columns: проверка наличия полей в таблице.
- _apply_filters: применение списка фильтров к DataFrame.
- _apply_filter: применение одного фильтра к DataFrame.
- _get_comparable_series: подготовка колонки к сравнению со значением фильтра.
- _coerce_filter_value: приведение значения фильтра к типу колонки.
- _clean_value: преобразование pandas/numpy значения к JSON-совместимому типу.
- _fake_sleep: имитация задержки Spark-запроса.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, model_validator

DATA_DIR = Path(__file__).resolve().parent / "data"
HITS_FILE = "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
UKO_FILE = "csp_afpc_sss_inc.uko_event.csv"
CARDS_FILE = "csp_afpc_sss_inc.cards_event.csv"
HISTORY_AUTOMARKING_FILE = "csp_repo_features.history_automarking_big_148078_155487.csv"
DEMO_TIMELINE_FILE = "demo_client_timeline.csv"
SOURCE_1_FILE = "source_1.csv"
SOURCE_2_FILE = "source_2.csv"
SOURCE_3_FILE = "source_3.csv"

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


class SparkTableQueryInput(BaseModel):
    """Параметры универсальной выборки из Spark-like таблицы.

    Args:
        table_name: Логическое имя таблицы из списка доступных таблиц.
        select_columns: Минимально достаточный непустой список полей для выборки.
        filters: Ограничения для отбора строк.
        max_rows: Максимальное число строк в ответе.
        include_schema: Если True, добавить схему таблицы даже при успешной выборке.

    Returns:
        Валидированные параметры запроса к Spark-like таблице.
    """

    table_name: str = Field(description="Имя Spark-like таблицы, например hits_extra_info, uko_event или cards_event.")
    select_columns: list[str] = Field(
        default_factory=list,
        description="Минимально достаточный список полей для выборки. Выгрузка всех полей запрещена.",
    )
    filters: list[SparkTableFilter] = Field(
        default_factory=list,
        description="Список фильтров, которые нужно применить к строкам таблицы.",
    )
    max_rows: int = Field(
        default=50,
        ge=0,
        le=1000,
        description="Максимальное число строк в ответе. Значение 0 вернет только метаданные.",
    )
    include_schema: bool = Field(
        default=False,
        description="Если True, вернуть схему таблицы вместе с результатом выборки.",
    )


def build_fake_spark_tools(
        *,
        delay_seconds: float = 1.5,
        data_dir: str | Path | None = None,
        transaction_count: int | None = None,
        day_event_count: int | None = None,
) -> list[BaseTool]:
    """Создает один универсальный Spark-like tool для запросов к локальным CSV-таблицам.

    Args:
        delay_seconds: Искусственная задержка каждого tool-вызова в секундах.
        data_dir: Директория с CSV-файлами. По умолчанию используется examples/data.
        transaction_count: Устаревший параметр совместимости, не влияет на результат.
        day_event_count: Устаревший параметр совместимости, не влияет на результат.

    Returns:
        Список с одним LangChain tool: spark_query_table.
    """

    del transaction_count, day_event_count
    resolved_data_dir = Path(data_dir).resolve() if data_dir else DATA_DIR

    async def spark_query_table(
            table_name: str,
            select_columns: list[str] | None = None,
            filters: list[SparkTableFilter] | None = None,
            max_rows: int = 50,
            include_schema: bool = False,
    ) -> pd.DataFrame | dict[str, Any]:
        """Выполняет универсальную выборку из Spark-like таблицы.

        Args:
            table_name: Имя таблицы.
            select_columns: Минимально достаточный список полей результата.
            filters: Ограничения выборки.
            max_rows: Максимальное число строк.
            include_schema: Признак возврата схемы таблицы.

        Returns:
            DataFrame с результатом выборки или словарь с ошибкой и схемой таблицы.
        """

        return await _spark_query_table(
            table_name=table_name,
            select_columns=select_columns or [],
            filters=filters or [],
            max_rows=max_rows,
            include_schema=include_schema,
            data_dir=resolved_data_dir,
            delay_seconds=delay_seconds,
        )

    return [
        StructuredTool.from_function(
            coroutine=spark_query_table,
            name="spark_query_table",
            description=(
                "spark_query_table\n"
                "---\n"
                "Описание: универсальная выборка из Spark-like таблиц. "
                "Инструмент принимает имя таблицы, список полей, фильтры и лимит строк, "
                "а при успешной выборке возвращает pandas DataFrame.\n"
                "Если в select_columns или filters указанного поля нет в таблице, инструмент "
                "вернет ok=False, описание ошибки и актуальную схему таблицы. "
                "Выгрузка всех столбцов запрещена: агент должен явно указать "
                "минимально достаточный набор колонок.\n\n"
                "Параметры:\n"
                "  table_name (str, обяз.) — имя таблицы или алиас.\n"
                "  select_columns (list[str], обяз.) — минимально достаточные поля результата. "
                "Пустой список, '*' и 'all' запрещены.\n"
                "  filters (list[dict], опц.) — фильтры вида "
                "{column, operator, value/values}. Операторы: eq, ne, gt, gte, lt, lte, "
                "contains, in, between, is_null, not_null.\n"
                "  max_rows (int, опц., 50) — максимум строк в ответе, от 0 до 1000.\n"
                "  include_schema (bool, опц., False) — вернуть схему при успешной выборке."
            ),
            args_schema=SparkTableQueryInput,
        ),
    ]


async def _spark_query_table(
        *,
        table_name: str,
        select_columns: list[str],
        filters: list[SparkTableFilter],
        max_rows: int,
        include_schema: bool,
        data_dir: Path,
        delay_seconds: float,
) -> pd.DataFrame | dict[str, Any]:
    """Выполняет выборку из Spark-like таблицы с проверкой полей.

    Args:
        table_name:  имя таблицы.
        select_columns: Минимально достаточные поля результата.
        filters: Список фильтров.
        max_rows: Максимальное число строк.
        include_schema: Признак возврата схемы при успешном ответе.
        data_dir: Директория с CSV-файлами.
        delay_seconds: Искусственная задержка запроса.

    Returns:
        DataFrame с результатом выборки или словарь с ошибкой и схемой таблицы.
    """

    await _fake_sleep(delay_seconds)
    registry = _get_table_registry()
    normalized_table_name = table_name.strip()
    table_meta = registry.get(normalized_table_name)
    if table_meta is None:
        return {
            "ok": False,
            "error": {
                "code": "unknown_table",
                "message": f"Таблица '{table_name}' не найдена.",
                "available_tables": sorted(registry),
            },
        }

    table = _load_spark_table(data_dir=data_dir, table_name=normalized_table_name)
    schema = _get_table_schema(table_name=normalized_table_name, source_file=table_meta["file"], table=table)
    select_error = _validate_select_columns_present(select_columns)
    if select_error is not None:
        return {
            "ok": False,
            "table_name": normalized_table_name,
            "source_file": table_meta["file"],
            "error": select_error,
            "schema": schema,
        }
    missing_columns = _validate_query_columns(table=table, select_columns=select_columns, filters=filters)
    if missing_columns:
        return {
            "ok": False,
            "table_name": normalized_table_name,
            "source_file": table_meta["file"],
            "error": {
                "code": "unknown_columns",
                "message": "В таблице нет одного или нескольких полей из запроса.",
                "missing_columns": missing_columns,
            },
            "schema": schema,
        }

    filtered = _apply_filters(table=table, filters=filters)
    result_columns = select_columns
    result = filtered.loc[:, result_columns].head(max(0, int(max_rows))).copy()
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
        "uko_event": {"file": UKO_FILE},
        "cards_event": {"file": CARDS_FILE},
        "history_automarking": {"file": HISTORY_AUTOMARKING_FILE},
        "demo_client_timeline": {"file": DEMO_TIMELINE_FILE},
        "source_1": {"file": SOURCE_1_FILE},
        "source_2": {"file": SOURCE_2_FILE},
        "source_3": {"file": SOURCE_3_FILE},
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
) -> list[str]:
    """Проверяет, что все запрошенные поля есть в таблице.

    Args:
        table: Таблица для проверки.
        select_columns: Поля результата.
        filters: Фильтры, поля которых нужно проверить.

    Returns:
        Отсортированный список отсутствующих полей.
    """

    requested_columns = set(select_columns)
    requested_columns.update(filter_item.column for filter_item in filters)
    return sorted(column for column in requested_columns if column not in table.columns)


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
        values = [_coerce_filter_value(series=series, value=value) for value in filter_item.values or []]
        comparable = _get_comparable_series(series=series, value=values[0] if values else None)
        return table[comparable.isin(values)].copy()
    if operator == "between":
        values = filter_item.values or []
        left = _coerce_filter_value(series=series, value=values[0])
        right = _coerce_filter_value(series=series, value=values[1])
        comparable = _get_comparable_series(series=series, value=left)
        return table[(comparable >= left) & (comparable <= right)].copy()

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

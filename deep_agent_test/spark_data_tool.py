"""Заглушка production-инструмента ``read_table`` поверх общей Spark session.

Это точка замены тестового ``examples.fake_spark_tools.build_fake_spark_tools`` на
реальный источник данных. Предполагается, что ``SparkSession`` создаётся один раз при
старте приложения (агента) и передаётся в фабрику.

Имя (`read_table`), описание и схема аргументов (`SparkTableQueryInput`) полностью
совпадают с текущим инструментом, поэтому при переключении источника данных prompts,
skills и поведение агента менять не нужно.

Как подключить (собери tools кодом, чтобы передать живую session):

    from pyspark.sql import SparkSession
    from deep_agent_test.spark_data_tool import build_spark_data_tools
    from deep_agent_test import build_analytics_deep_agent, load_deep_agent_settings
    from model import model

    spark = SparkSession.builder.getOrCreate()  # один раз при старте приложения
    settings = load_deep_agent_settings()
    data_tools = build_spark_data_tools(spark)
    agent = build_analytics_deep_agent(model, settings=settings, data_tools=data_tools)

Схема аргументов переиспользуется из ``examples.fake_spark_tools``. При вынесении в
production перенеси ``SparkTableQueryInput`` и связанные модели в собственный модуль.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from examples.fake_spark_tools import (
    SparkColumnOperation,
    SparkTableAggregation,
    SparkTableFilter,
    SparkTableOrderBy,
    SparkTableQueryInput,
)

# Описание полностью совпадает с текущим read_table (build_fake_spark_tools), чтобы
# контракт инструмента для модели не менялся при переключении на Spark.
READ_TABLE_DESCRIPTION = (
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
)


def build_spark_data_tools(spark: Any) -> list[BaseTool]:
    """Создаёт инструмент ``read_table`` поверх готовой Spark session.

    Сигнатура повторяет ``build_fake_spark_tools``, но вместо чтения локальных CSV
    использует переданный объект Spark session.

    Args:
        spark: Активная ``pyspark.sql.SparkSession``, созданная один раз при старте
            приложения и переданная во все вызовы инструмента.

    Returns:
        Список с одним LangChain tool ``read_table`` (как и у тестовой фабрики).
    """

    def read_table(
        table_name: str,
        select_columns: str | None = None,
        filters: list[SparkTableFilter] | None = None,
        derived_columns: list[SparkColumnOperation] | None = None,
        group_by: str | None = None,
        aggregations: list[SparkTableAggregation] | None = None,
        order_by: list[SparkTableOrderBy] | None = None,
        max_rows: int | None = None,
        include_schema: bool = False,
    ) -> Any:
        """ЗАГЛУШКА: реальная выгрузка через Spark session ещё не реализована.

        Имя, описание и сигнатура совпадают с текущим ``read_table``. Реальную
        реализацию см. в закомментированном фрагменте ниже — он повторяет логику
        выборки, но вместо чтения CSV использует переданный объект ``spark``.

        Args:
            table_name: Имя таблицы или алиас.
            select_columns: Минимально достаточные поля в формате "col1, col2, col3".
            filters: Список фильтров отбора строк.
            derived_columns: Вычисляемые колонки до фильтрации/агрегации.
            group_by: Поля группировки для агрегаций в формате "col1, col2".
            aggregations: Агрегатные операции после фильтрации.
            order_by: Правила сортировки результата.
            max_rows: Максимум строк в ответе; если ``None`` — лимит не применяется.
            include_schema: Признак возврата схемы вместе с результатом.

        Returns:
            pandas DataFrame с результатом выборки (после реализации).
        """

        raise NotImplementedError(
            "build_spark_data_tools.read_table — заглушка. Раскомментируй и адаптируй "
            "Spark-реализацию ниже под свои таблицы, схему и формат полей."
        )

        # ----------------------------------------------------------------------
        # РЕАЛЬНАЯ РЕАЛИЗАЦИЯ НА SPARK SESSION (раскомментировать и адаптировать):
        #
        # from pyspark.sql import functions as F
        #
        # # 1. Источник: логическое имя таблицы -> Spark DataFrame.
        # #    spark создан один раз при старте приложения и передан в фабрику.
        # sdf = spark.table(table_name)            # или spark.sql(f"SELECT * FROM {table_name}")
        # total_rows = sdf.count()
        #
        # # 2. derived_columns: вычисляемые поля до фильтрации/агрегации/сортировки.
        # for op in derived_columns or []:
        #     src = F.col(op.source_column)
        #     if op.operation == "lower":
        #         sdf = sdf.withColumn(op.name, F.lower(src.cast("string")))
        #     elif op.operation == "upper":
        #         sdf = sdf.withColumn(op.name, F.upper(src.cast("string")))
        #     elif op.operation == "length":
        #         sdf = sdf.withColumn(op.name, F.length(src.cast("string")))
        #     elif op.operation == "abs":
        #         sdf = sdf.withColumn(op.name, F.abs(src.cast("double")))
        #     else:
        #         digits = F.regexp_replace(src.cast("string"), r"\D", "")
        #         spans = {"year": (1, 4), "month": (5, 2), "year_month": (1, 6), "date": (1, 8)}
        #         start, length = spans[op.operation]
        #         sdf = sdf.withColumn(op.name, digits.substr(start, length))
        #
        # # 3. filters: список условий -> предикаты Spark Column.
        # def _predicate(f: SparkTableFilter):
        #     col = F.col(f.column)
        #     if f.operator == "eq":
        #         return col == f.value
        #     if f.operator == "ne":
        #         return col != f.value
        #     if f.operator == "gt":
        #         return col > f.value
        #     if f.operator == "gte":
        #         return col >= f.value
        #     if f.operator == "lt":
        #         return col < f.value
        #     if f.operator == "lte":
        #         return col <= f.value
        #     if f.operator == "contains":
        #         return col.cast("string").contains(str(f.value))
        #     if f.operator == "in":
        #         return col.isin(list(f.values or []))
        #     if f.operator == "between":
        #         lo, hi = (f.values or [None, None])[0], (f.values or [None, None])[1]
        #         return col.between(lo, hi)
        #     if f.operator == "is_null":
        #         return col.isNull()
        #     if f.operator == "not_null":
        #         return col.isNotNull()
        #     raise ValueError(f"Неподдерживаемый оператор фильтра: {f.operator}")
        #
        # for f in filters or []:
        #     sdf = sdf.filter(_predicate(f))
        # matched_rows = sdf.count()
        #
        # # 4. select ИЛИ group_by + aggregations.
        # if aggregations:
        #     agg_fns = {
        #         "count": F.count, "count_distinct": F.countDistinct,
        #         "min": F.min, "max": F.max, "sum": F.sum, "mean": F.avg,
        #     }
        #     agg_exprs = [
        #         agg_fns[a.function](F.col(a.column)).alias(a.alias or f"{a.function}_{a.column}")
        #         for a in aggregations
        #     ]
        #     group_cols = [c.strip() for c in (group_by or "").split(",") if c.strip()]
        #     sdf = sdf.groupBy(*group_cols).agg(*agg_exprs) if group_cols else sdf.agg(*agg_exprs)
        # else:
        #     cols = [c.strip() for c in (select_columns or "").split(",") if c.strip()]
        #     # Выгрузка всех полей запрещена: cols должен быть непустым.
        #     sdf = sdf.select(*cols)
        #
        # # 5. order_by.
        # for ob in order_by or []:
        #     ordering = F.col(ob.column).asc() if ob.direction == "asc" else F.col(ob.column).desc()
        #     sdf = sdf.orderBy(ordering)
        #
        # # 6. limit + сбор в pandas (тот же тип результата, что у текущего инструмента).
        # if max_rows is not None:
        #     sdf = sdf.limit(max(0, int(max_rows)))
        # result = sdf.toPandas()
        #
        # # 7. метаданные в attrs — контракт совпадает с build_fake_spark_tools, поэтому
        # #    offload middleware и форматирование ответа работают без изменений.
        # result.attrs["spark_table_name"] = table_name
        # result.attrs["spark_source_file"] = table_name
        # result.attrs["spark_total_rows"] = int(total_rows)
        # result.attrs["spark_matched_rows"] = int(matched_rows)
        # if include_schema:
        #     result.attrs["spark_schema"] = {
        #         "table_name": table_name,
        #         "columns_count": len(result.columns),
        #         "columns": [{"name": c, "type": str(result[c].dtype)} for c in result.columns],
        #     }
        # return result
        # ----------------------------------------------------------------------

    return [
        StructuredTool.from_function(
            func=read_table,
            name="read_table",
            description=READ_TABLE_DESCRIPTION,
            args_schema=SparkTableQueryInput,
        )
    ]


__all__ = [
    "READ_TABLE_DESCRIPTION",
    "build_spark_data_tools",
]

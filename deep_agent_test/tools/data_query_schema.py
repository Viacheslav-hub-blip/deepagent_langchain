"""Схемы структурированных аргументов для инструмента ``load_data``.

Содержит:
- FilterCondition: схема одного фильтра строк таблицы.
- DerivedColumnSpec: схема одной вычисляемой колонки.
- AggregationSpec: схема одного агрегата.
- OrderBySpec: схема одного правила сортировки.
- ReadTableInput: полная схема аргументов инструмента ``load_data``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ScalarValue = str | int | float | bool
FilterOperator = Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "in", "between", "is_null", "not_null"]
DerivedOperation = Literal["year", "month", "year_month", "date", "lower", "upper", "length", "abs"]
AggregationFunction = Literal["count", "count_distinct", "min", "max", "sum", "mean"]
SortDirection = Literal["asc", "desc"]


class FilterCondition(BaseModel):
    """Один фильтр для отбора строк таблицы.

    Args:
        column: Имя колонки, по которой применяется фильтр.
        operator: Оператор сравнения или проверки значения.
        value: Одно значение для операторов ``eq``, ``ne``, ``gt``, ``gte``, ``lt``,
            ``lte`` и ``contains``.
        values: Несколько значений для операторов ``in`` и ``between``.
        second_value: Вторая граница для оператора ``between``, если первая передана через ``value``.

    Returns:
        Валидированное описание одного фильтра.
    """

    column: str = Field(description="Имя колонки для фильтра.")
    operator: FilterOperator = Field(description="Оператор фильтра: eq, ne, gt, gte, lt, lte, contains, in, between, is_null или not_null.")
    value: ScalarValue | None = Field(default=None, description="Одно значение фильтра. Для in используй values; для between можно указать первую границу.")
    values: list[ScalarValue] = Field(default_factory=list, description="Список значений для операторов in и between.")
    second_value: ScalarValue | None = Field(default=None, description="Вторая граница для оператора between, если первая граница передана в value.")


class DerivedColumnSpec(BaseModel):
    """Одна вычисляемая колонка.

    Args:
        name: Имя новой колонки.
        source_column: Исходная колонка.
        operation: Операция преобразования исходной колонки.

    Returns:
        Валидированное описание вычисляемой колонки.
    """

    name: str = Field(description="Имя новой вычисляемой колонки.")
    source_column: str = Field(description="Исходная колонка для преобразования.")
    operation: DerivedOperation = Field(description="Операция: year, month, year_month, date, lower, upper, length или abs.")


class AggregationSpec(BaseModel):
    """Один агрегат для группировки или общей сводки.

    Args:
        function: Агрегатная функция.
        column: Колонка, к которой применяется агрегат.
        alias: Имя результирующей колонки. Если пусто, инструмент создаст имя сам.

    Returns:
        Валидированное описание агрегата.
    """

    function: AggregationFunction = Field(description="Агрегатная функция: count, count_distinct, min, max, sum или mean.")
    column: str = Field(description="Колонка для агрегации.")
    alias: str = Field(default="", description="Имя результирующей колонки.")


class OrderBySpec(BaseModel):
    """Одно правило сортировки результата.

    Args:
        column: Колонка для сортировки.
        direction: Направление сортировки.

    Returns:
        Валидированное правило сортировки.
    """

    column: str = Field(description="Колонка для сортировки.")
    direction: SortDirection = Field(default="asc", description="Направление сортировки: asc или desc.")


class ReadTableInput(BaseModel):
    """Структурированные аргументы инструмента чтения данных ``load_data``.

    Args:
        table_name: Короткое имя таблицы.
        select_columns: Колонки результата для обычной выборки.
        filters: Список фильтров.
        derived_columns: Список вычисляемых колонок.
        group_by: Колонки группировки.
        aggregations: Список агрегатов.
        order_by: Правила сортировки.
        max_rows: Максимальное число строк результата.
        include_schema: Нужно ли приложить схему результата.

    Returns:
        Валидированные аргументы для ``load_data``.
    """

    table_name: str = Field(description="Короткое имя таблицы: hits, cards, uko, history_automarking или demo_client_timeline.")
    select_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Обязательные колонки результата для обычной выборки. "
            "Не оставляй пустым без aggregations; не используй '*' и 'all'."
        ),
    )
    filters: list[FilterCondition] = Field(default_factory=list, description="Фильтры строк таблицы.")
    derived_columns: list[DerivedColumnSpec] = Field(default_factory=list, description="Вычисляемые колонки.")
    group_by: list[str] = Field(default_factory=list, description="Колонки группировки.")
    aggregations: list[AggregationSpec] = Field(default_factory=list, description="Агрегаты для расчёта.")
    order_by: list[OrderBySpec] = Field(default_factory=list, description="Правила сортировки результата.")
    max_rows: int | None = Field(default=None, ge=0, description="Максимальное число строк результата.")
    include_schema: bool = Field(default=False, description="Если True, добавить схему результата в metadata.")

    @model_validator(mode="after")
    def validate_select_or_aggregations(self) -> "ReadTableInput":
        """Проверяет, что обычная выборка не превращается в неявный ``SELECT *``.

        Args:
            Отсутствуют. Метод использует поля текущей модели.

        Returns:
            Текущую модель, если указан явный список колонок или агрегаты.

        Raises:
            ValueError: Вызов ``load_data`` не содержит ни ``select_columns``, ни ``aggregations``.
        """

        normalized_select_columns = [str(column).strip() for column in self.select_columns if str(column).strip()]
        forbidden_columns = {column.lower() for column in normalized_select_columns} & {"*", "all"}
        if forbidden_columns:
            raise ValueError(
                "select_columns не может содержать '*' или 'all'. "
                "Укажи минимальный список конкретных колонок результата."
            )
        has_select_columns = bool(normalized_select_columns)
        has_aggregations = bool(self.aggregations)
        if not has_select_columns and not has_aggregations:
            raise ValueError(
                "Для load_data обязательно укажи select_columns или aggregations. "
                "Вызов только с table_name запрещён, потому что SELECT * не поддерживается."
            )
        return self


__all__ = [
    "AggregationFunction",
    "AggregationSpec",
    "DerivedColumnSpec",
    "DerivedOperation",
    "FilterCondition",
    "FilterOperator",
    "OrderBySpec",
    "ReadTableInput",
    "ScalarValue",
    "SortDirection",
]

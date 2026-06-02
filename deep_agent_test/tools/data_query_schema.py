"""Схемы аргументов для инструмента ``load_data``.

Содержит:
- FilterCondition: схема одного фильтра строк таблицы.
- DerivedColumnSpec: схема одной вычисляемой колонки.
- AggregationSpec: схема одного агрегата.
- OrderBySpec: схема одного правила сортировки.
- ReadTableInput: схема одного SQL-подобного запроса инструмента ``load_data``.
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

    column: str = Field(description="Имя колонки для фильтра. Пример: event_id.", examples=["event_id"])
    operator: FilterOperator = Field(description="Оператор фильтра. Пример: eq.", examples=["eq"])
    value: ScalarValue | None = Field(
        default=None,
        description="Одно значение фильтра для eq, ne, gt, gte, lt, lte или contains. Пример: 3486d84b-4eba-4ba4-b044-94764fc9e7a4.",
        examples=["3486d84b-4eba-4ba4-b044-94764fc9e7a4"],
    )
    values: list[ScalarValue] = Field(
        default_factory=list,
        description="Список значений для операторов in и between. Пример: ['20260101', '20260131'].",
        examples=[["20260101", "20260131"]],
    )
    second_value: ScalarValue | None = Field(
        default=None,
        description="Вторая граница для between, если первая граница передана в value. Пример: 20260131.",
        examples=["20260131"],
    )


class DerivedColumnSpec(BaseModel):
    """Одна вычисляемая колонка.

    Args:
        name: Имя новой колонки.
        source_column: Исходная колонка.
        operation: Операция преобразования исходной колонки.

    Returns:
        Валидированное описание вычисляемой колонки.
    """

    name: str = Field(description="Имя новой вычисляемой колонки. Пример: event_month.", examples=["event_month"])
    source_column: str = Field(description="Исходная колонка для преобразования. Пример: event_dt.", examples=["event_dt"])
    operation: DerivedOperation = Field(description="Операция преобразования. Пример: year_month.", examples=["year_month"])


class AggregationSpec(BaseModel):
    """Один агрегат для группировки или общей сводки.

    Args:
        function: Агрегатная функция.
        column: Колонка, к которой применяется агрегат.
        alias: Имя результирующей колонки. Если пусто, инструмент создаст имя сам.

    Returns:
        Валидированное описание агрегата.
    """

    function: AggregationFunction = Field(description="Агрегатная функция. Пример: count.", examples=["count"])
    column: str = Field(description="Колонка для агрегации. Пример: event_id.", examples=["event_id"])
    alias: str = Field(default="", description="Имя результирующей колонки. Пример: events_count.", examples=["events_count"])


class OrderBySpec(BaseModel):
    """Одно правило сортировки результата.

    Args:
        column: Колонка для сортировки.
        direction: Направление сортировки.

    Returns:
        Валидированное правило сортировки.
    """

    column: str = Field(description="Колонка для сортировки. Пример: event_dt.", examples=["event_dt"])
    direction: SortDirection = Field(default="asc", description="Направление сортировки. Пример: asc.", examples=["asc"])


class ReadTableInput(BaseModel):
    """Аргументы инструмента чтения данных ``load_data``.

    Args:
        query: SQL-подобный текст запроса с alias таблицы, явным периодом и колонками результата.

    Returns:
        Валидированный SQL-подобный запрос для ``load_data``.
    """

    query: str = Field(
        description=(
            "SQL-подобный запрос. Обязателен alias таблицы, явный SELECT без '*', "
            "и обязательный период через PERIOD <date_column> FROM 'YYYYMMDD' TO 'YYYYMMDD'."
        ),
        examples=[
            (
                "LOAD uko\n"
                "PERIOD event_dt FROM '20260101' TO '20260131'\n"
                "SELECT event_id, event_dt, event_description\n"
                "WHERE event_description CONTAINS 'фон'"
            )
        ],
    )

    @model_validator(mode="after")
    def validate_query_text(self) -> "ReadTableInput":
        """Проверяет, что запрос не пустой.

        Args:
            Отсутствуют. Метод использует поля текущей модели.

        Returns:
            Текущую модель, если передан непустой текст запроса.

        Raises:
            ValueError: Вызов ``load_data`` содержит пустой запрос.
        """

        if not self.query.strip():
            raise ValueError("Для load_data обязательно передай непустой SQL-подобный query.")
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

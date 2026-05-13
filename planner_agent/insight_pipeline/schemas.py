"""Схемы данных для pipeline поиска инсайтов.

Содержит:
- MissingDataRequest: запрос на дозагрузку данных, который может вернуть агент.
- CaseAnalysisRecord: результат обработки одной записи агентом.
- GroupSelectionDecision: бинарное решение о полезности группы для дальнейшего анализа.
- GroupProblemSummary: итоговая сводка по проблемной группе.
- InsightPipelineConfig: конфигурация полного pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MissingDataRequest(BaseModel):
    """Запрос на дозагрузку данных для одного кейса.

    Args:
        source_name: Название таблицы, tool или внешнего источника.
        reason: Зачем агенту нужны эти данные.
        lookup_keys: Ключи, по которым нужно загрузить данные.
        period: Период, который нужно использовать при дозагрузке.
        priority: Приоритет запроса: `low`, `medium` или `high`.

    Returns:
        Валидированный объект запроса на дополнительные данные.
    """

    source_name: str = Field(default="", description="Название источника данных.")
    reason: str = Field(default="", description="Причина дозагрузки данных.")
    lookup_keys: dict[str, Any] = Field(
        default_factory=dict,
        description="Ключи поиска: event_id, epk_id, event_dt и другие поля.",
    )
    period: str | None = Field(
        default=None,
        description="Период дозагрузки в свободном, но читаемом формате.",
    )
    priority: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Приоритет запроса на дозагрузку.",
    )


class CaseAnalysisRecord(BaseModel):
    """Результат обработки одной записи агентом.

    Args:
        row_index: Индекс строки в исходном DataFrame.
        case_id: Идентификатор кейса, переданный агенту.
        group_name: Название группы после кластеризации.
        status: Статус обработки записи.
        agent_prompt: Текстовый запрос, который был передан агенту.
        report_markdown: Markdown или обычный текст отчета агента.
        structured_result: Машиночитаемый результат, если агент вернул JSON.
        missing_data_requests: Список запросов на дозагрузку данных.
        error: Текст ошибки, если обработка завершилась неуспешно.

    Returns:
        Валидированный результат обработки одной записи.
    """

    model_config = ConfigDict(extra="ignore")

    row_index: int = Field(description="Индекс строки в DataFrame.")
    case_id: str = Field(default="", description="Идентификатор кейса.")
    group_name: str = Field(default="", description="Название группы.")
    status: Literal["processed", "skipped", "failed"] = Field(
        default="processed",
        description="Статус обработки записи.",
    )
    agent_prompt: str = Field(default="", description="Prompt, переданный агенту.")
    report_markdown: str = Field(default="", description="Текстовый отчет агента.")
    structured_result: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-результат агента, если он был найден и распарсен.",
    )
    missing_data_requests: list[MissingDataRequest] = Field(
        default_factory=list,
        description="Запросы агента на дозагрузку дополнительных данных.",
    )
    error: str = Field(default="", description="Ошибка обработки записи.")


class GroupSelectionDecision(BaseModel):
    """Бинарное решение о полезности группы для дальнейшего анализа.

    Args:
        group_name: Название группы после кластеризации.
        is_meaningful: Признак, что группа имеет аналитический смысл и должна попасть в filtered_df.
        reason: Краткое объяснение решения.
        confidence: Уверенность решения от 0 до 1.
        total_records: Количество записей в группе.

    Returns:
        Валидированное решение selector-а по одной группе.
    """

    group_name: str = Field(description="Название группы.")
    is_meaningful: bool = Field(description="Нужно ли включить группу в дальнейший анализ.")
    reason: str = Field(default="", description="Краткое объяснение решения.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Уверенность от 0 до 1.")
    total_records: int = Field(default=0, ge=0, description="Количество записей в группе.")


class GroupProblemSummary(BaseModel):
    """Итоговая сводка по одной группе проблем.

    Args:
        group_name: Название группы.
        total_records: Количество строк в группе после кластеризации.
        processed_records: Количество записей, обработанных агентом.
        failed_records: Количество записей с ошибкой обработки.
        missing_data_requests_count: Количество запросов на дозагрузку данных.
        problem_summary: Краткое описание проблемы в группе.
        evidence_case_ids: Примеры case_id, на которых основана сводка.
        limitations: Ограничения и пробелы данных.

    Returns:
        Валидированная сводка по группе.
    """

    group_name: str = Field(description="Название группы.")
    total_records: int = Field(default=0, description="Количество строк в группе.")
    processed_records: int = Field(default=0, description="Количество обработанных строк.")
    failed_records: int = Field(default=0, description="Количество ошибок обработки.")
    missing_data_requests_count: int = Field(
        default=0,
        description="Количество запросов на дозагрузку данных.",
    )
    problem_summary: str = Field(default="", description="Краткое описание проблемы.")
    evidence_case_ids: list[str] = Field(
        default_factory=list,
        description="Примеры кейсов, подтверждающих сводку.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Ограничения и пробелы данных.",
    )


class InsightPipelineConfig(BaseModel):
    """Конфигурация pipeline поиска инсайтов.

    Args:
        text_column: Колонка с комментарием или текстом проблемы.
        group_column: Колонка, куда кластеризатор запишет название группы.
        case_id_column: Колонка с бизнес-идентификатором кейса.
        event_id_column: Колонка с event_id для prompt-а агента.
        selected_group_column: Колонка-флаг значимой группы.
        group_selection_reason_column: Колонка с причиной выбора или отклонения группы.
        agent_report_column: Колонка с текстовым описанием записи агентом.
        agent_status_column: Колонка со статусом обработки записи агентом.
        agent_error_column: Колонка с ошибкой обработки записи агентом.
        agent_structured_result_column: Колонка со структурированным результатом агента.
        agent_missing_data_requests_column: Колонка с запросами агента на дозагрузку данных.
        max_cases_per_group: Максимальное число записей, которое агент обрабатывает в каждой группе.
            Если `None`, обрабатываются все строки отфильтрованного DataFrame.
        min_group_size: Минимальный размер группы для заглушки выбора значимых групп.
        max_groups: Максимальное число групп для обработки.
        agent_recursion_limit: Лимит рекурсии LangGraph/агента при вызове.
        include_full_row_prompt: Передавать ли агенту всю строку как обычный текстовый запрос.

    Returns:
        Валидированная конфигурация pipeline.
    """

    text_column: str = Field(default="comment_text", description="Колонка с текстом комментария.")
    group_column: str = Field(default="problem_group", description="Колонка с группой проблемы.")
    case_id_column: str = Field(default="case_id", description="Колонка с id кейса.")
    event_id_column: str = Field(default="event_id", description="Колонка с event_id.")
    selected_group_column: str = Field(
        default="is_significant_group",
        description="Колонка-флаг выбранной значимой группы.",
    )
    group_selection_reason_column: str = Field(
        default="group_selection_reason",
        description="Колонка с причиной выбора или отклонения группы.",
    )
    agent_report_column: str = Field(
        default="agent_record_description",
        description="Колонка с описанием записи агентом.",
    )
    agent_status_column: str = Field(
        default="agent_processing_status",
        description="Колонка со статусом агентной обработки.",
    )
    agent_error_column: str = Field(
        default="agent_processing_error",
        description="Колонка с ошибкой агентной обработки.",
    )
    agent_structured_result_column: str = Field(
        default="agent_structured_result",
        description="Колонка со структурированным JSON-результатом агента.",
    )
    agent_missing_data_requests_column: str = Field(
        default="agent_missing_data_requests",
        description="Колонка с запросами агента на дозагрузку данных.",
    )
    max_cases_per_group: int | None = Field(
        default=None,
        description="Максимальное число кейсов на группу для агентного разбора.",
    )
    min_group_size: int = Field(
        default=3,
        ge=1,
        description="Минимальный размер группы для детерминированной заглушки selector-а.",
    )
    max_groups: int | None = Field(
        default=10,
        description="Максимальное число групп для обработки. `None` означает без лимита.",
    )
    agent_recursion_limit: int = Field(
        default=60,
        ge=1,
        description="Лимит рекурсии при запуске агента.",
    )
    include_full_row_prompt: bool = Field(
        default=True,
        description="Передавать агенту полное строковое представление строки.",
    )

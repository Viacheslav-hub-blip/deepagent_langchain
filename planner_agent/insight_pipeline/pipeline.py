"""Orchestrator полного pipeline поиска инсайтов.

Содержит:
- InsightPipelineRun: контейнер результата выполнения pipeline.
- InsightPipeline: загрузка файла, обогащение, кластеризация, бинарный выбор групп, агентная обработка и суммаризация.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from planner_agent.insight_pipeline.components import (
    BaseDataEnricher,
    CaseAgentProcessor,
    GroupProblemSummarizer,
    NoopDataEnricher,
    StubCommentClusterer,
    StubSignificantGroupSelector,
)
from planner_agent.insight_pipeline.io import load_dataframe
from planner_agent.insight_pipeline.schemas import (
    CaseAnalysisRecord,
    GroupProblemSummary,
    GroupSelectionDecision,
    InsightPipelineConfig,
)


@dataclass
class InsightPipelineRun:
    """Результат выполнения pipeline поиска инсайтов.

    Args:
        loaded_df: DataFrame после загрузки файла.
        enriched_df: DataFrame после обогащения.
        clustered_df: DataFrame после кластеризации с колонкой группы.
        filtered_df: DataFrame только с группами, выбранными selector-ом.
        analyzed_df: DataFrame после агентной обработки с добавленными колонками отчета.
        group_selection_decisions: Бинарные решения selector-а по всем группам.
        selected_groups: Список групп, выбранных для агентной обработки.
        case_results: Результаты обработки записей агентом.
        group_summaries: Сводки проблем по выбранным группам.
        metadata: Техническая информация о запуске.

    Returns:
        Контейнер, который можно использовать в notebook, API или batch job.
    """

    loaded_df: pd.DataFrame
    enriched_df: pd.DataFrame
    clustered_df: pd.DataFrame
    filtered_df: pd.DataFrame
    analyzed_df: pd.DataFrame
    group_selection_decisions: list[GroupSelectionDecision] = field(default_factory=list)
    selected_groups: list[str] = field(default_factory=list)
    case_results: list[CaseAnalysisRecord] = field(default_factory=list)
    group_summaries: list[GroupProblemSummary] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def case_results_df(self) -> pd.DataFrame:
        """Возвращает результаты агентной обработки как pandas DataFrame.

        Args:
            Нет входных аргументов.

        Returns:
            DataFrame с результатами обработки кейсов.
        """

        return pd.DataFrame([record.model_dump(mode="json") for record in self.case_results])

    def group_summaries_df(self) -> pd.DataFrame:
        """Возвращает групповые сводки как pandas DataFrame.

        Args:
            Нет входных аргументов.

        Returns:
            DataFrame со сводками по группам.
        """

        return pd.DataFrame([summary.model_dump(mode="json") for summary in self.group_summaries])

    def group_selection_decisions_df(self) -> pd.DataFrame:
        """Возвращает решения selector-а как pandas DataFrame.

        Args:
            Нет входных аргументов.

        Returns:
            DataFrame с бинарными решениями по группам.
        """

        return pd.DataFrame(
            [decision.model_dump(mode="json") for decision in self.group_selection_decisions]
        )


class InsightPipeline:
    """Полный pipeline поиска инсайтов поверх single-case агента.

    Args:
        config: Конфигурация колонок и лимитов pipeline.
        enricher: Компонент обогащения данных после загрузки файла.
        clusterer: Компонент кластеризации комментариев.
        group_selector: Компонент бинарного выбора значимых групп.
        case_processor: Компонент агентной обработки записей.
        group_summarizer: Компонент суммаризации проблем внутри групп.

    Returns:
        Объект, который запускает pipeline методом `run`.
    """

    def __init__(
        self,
        *,
        config: InsightPipelineConfig | None = None,
        enricher: BaseDataEnricher | None = None,
        clusterer: Any | None = None,
        group_selector: Any | None = None,
        case_processor: CaseAgentProcessor | None = None,
        group_summarizer: GroupProblemSummarizer | None = None,
    ) -> None:
        self.config = config or InsightPipelineConfig()
        self.enricher = enricher or NoopDataEnricher()
        self.clusterer = clusterer or StubCommentClusterer()
        self.group_selector = group_selector or StubSignificantGroupSelector()
        self.case_processor = case_processor or CaseAgentProcessor(agent=None)
        self.group_summarizer = group_summarizer or GroupProblemSummarizer()

    def run(
        self,
        source: str | Path | pd.DataFrame,
        *,
        read_kwargs: dict[str, Any] | None = None,
    ) -> InsightPipelineRun:
        """Запускает полный pipeline по файлу или готовому DataFrame.

        Args:
            source: Путь к входному файлу или готовый pandas DataFrame.
            read_kwargs: Дополнительные параметры чтения файла pandas.

        Returns:
            `InsightPipelineRun` с промежуточными DataFrame и итоговыми результатами.
        """

        loaded_df = self._load_source(source, read_kwargs=read_kwargs or {})
        enriched_df = self.enricher.enrich(loaded_df)
        clustered_df = self._cluster(enriched_df)
        group_selection_decisions = self._classify_groups(clustered_df)
        filtered_df = self._filter_selected_groups(clustered_df, group_selection_decisions)
        analyzed_df, case_results = self._process_filtered_cases(filtered_df)
        selected_groups = [
            decision.group_name
            for decision in group_selection_decisions
            if decision.is_meaningful
        ]
        group_summaries = self.group_summarizer.summarize(
            analyzed_df,
            case_results,
            selected_groups=selected_groups,
            config=self.config,
        )

        return InsightPipelineRun(
            loaded_df=loaded_df,
            enriched_df=enriched_df,
            clustered_df=clustered_df,
            filtered_df=filtered_df,
            analyzed_df=analyzed_df,
            group_selection_decisions=group_selection_decisions,
            selected_groups=selected_groups,
            case_results=case_results,
            group_summaries=group_summaries,
            metadata={
                "loaded_rows": len(loaded_df),
                "enriched_rows": len(enriched_df),
                "clustered_rows": len(clustered_df),
                "filtered_rows": len(filtered_df),
                "analyzed_rows": len(analyzed_df),
                "selected_groups_count": len(selected_groups),
                "case_results_count": len(case_results),
            },
        )

    def _load_source(
        self,
        source: str | Path | pd.DataFrame,
        *,
        read_kwargs: dict[str, Any],
    ) -> pd.DataFrame:
        """Загружает источник данных в DataFrame.

        Args:
            source: Путь к файлу или готовый DataFrame.
            read_kwargs: Параметры чтения файла.

        Returns:
            pandas DataFrame после загрузки.
        """

        if isinstance(source, pd.DataFrame):
            return source.copy()
        return load_dataframe(source, **read_kwargs)

    def _cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """Запускает кластеризацию комментариев.

        Args:
            df: DataFrame после обогащения.

        Returns:
            DataFrame с колонкой группы.
        """

        if callable(getattr(self.clusterer, "cluster", None)):
            return self.clusterer.cluster(
                df,
                text_column=self.config.text_column,
                group_column=self.config.group_column,
            )
        if callable(self.clusterer):
            return self.clusterer(
                df,
                text_column=self.config.text_column,
                group_column=self.config.group_column,
            )
        raise TypeError("clusterer должен иметь метод cluster(...) или быть callable.")

    def _classify_groups(self, df: pd.DataFrame) -> list[GroupSelectionDecision]:
        """Классифицирует каждую группу как значимую или незначимую.

        Args:
            df: DataFrame после кластеризации.

        Returns:
            Список бинарных решений по всем группам.
        """

        normalized_df = df.copy()
        normalized_df[self.config.group_column] = (
            normalized_df[self.config.group_column].fillna("Без группы").astype(str)
        )

        if callable(getattr(self.group_selector, "classify_group", None)):
            decisions = [
                GroupSelectionDecision.model_validate(
                    self.group_selector.classify_group(str(group_name), group_df, self.config)
                )
                for group_name, group_df in normalized_df.groupby(self.config.group_column, dropna=False)
            ]
            return self._apply_group_limit(decisions)

        if callable(getattr(self.group_selector, "select_groups", None)):
            selected_groups = set(self.group_selector.select_groups(normalized_df, self.config))
            return [
                GroupSelectionDecision(
                    group_name=str(group_name),
                    is_meaningful=str(group_name) in selected_groups,
                    reason=(
                        "Группа выбрана legacy select_groups."
                        if str(group_name) in selected_groups
                        else "Группа не выбрана legacy select_groups."
                    ),
                    confidence=0.0,
                    total_records=len(group_df),
                )
                for group_name, group_df in normalized_df.groupby(self.config.group_column, dropna=False)
            ]

        if callable(self.group_selector):
            raw_decisions = self.group_selector(normalized_df, self.config)
            return [
                GroupSelectionDecision.model_validate(decision)
                for decision in raw_decisions
            ]

        raise TypeError(
            "group_selector должен иметь classify_group(...), select_groups(...) или быть callable."
        )

    def _apply_group_limit(
        self,
        decisions: list[GroupSelectionDecision],
    ) -> list[GroupSelectionDecision]:
        """Применяет лимит `max_groups` к положительным решениям selector-а.

        Args:
            decisions: Решения selector-а по всем группам.

        Returns:
            Решения после ограничения числа выбранных групп.
        """

        if self.config.max_groups is None:
            return decisions

        kept_count = 0
        limited: list[GroupSelectionDecision] = []
        for decision in decisions:
            if not decision.is_meaningful:
                limited.append(decision)
                continue

            kept_count += 1
            if kept_count <= self.config.max_groups:
                limited.append(decision)
                continue

            limited.append(
                decision.model_copy(
                    update={
                        "is_meaningful": False,
                        "reason": (
                            f"{decision.reason} Группа отклонена из-за max_groups="
                            f"{self.config.max_groups}."
                        ),
                    }
                )
            )
        return limited

    def _filter_selected_groups(
        self,
        df: pd.DataFrame,
        decisions: list[GroupSelectionDecision],
    ) -> pd.DataFrame:
        """Фильтрует DataFrame, оставляя только значимые группы.

        Args:
            df: DataFrame после кластеризации.
            decisions: Бинарные решения selector-а по всем группам.

        Returns:
            DataFrame только с выбранными группами и колонками решения.
        """

        selected_groups = {
            decision.group_name
            for decision in decisions
            if decision.is_meaningful
        }
        reason_by_group = {decision.group_name: decision.reason for decision in decisions}

        result = df.copy()
        result[self.config.group_column] = result[self.config.group_column].fillna("Без группы").astype(str)
        result[self.config.selected_group_column] = result[self.config.group_column].isin(selected_groups)
        result[self.config.group_selection_reason_column] = (
            result[self.config.group_column].map(reason_by_group).fillna("")
        )
        return result[result[self.config.selected_group_column]].copy()

    def _process_filtered_cases(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[CaseAnalysisRecord]]:
        """Обрабатывает отфильтрованные строки через single-case агента.

        Args:
            df: DataFrame только с выбранными группами.

        Returns:
            Кортеж `(analyzed_df, records)`, где `analyzed_df` содержит исходные
            колонки плюс колонки результата агента.
        """

        analyzed_df = self._prepare_agent_columns(df.copy())
        records: list[CaseAnalysisRecord] = []

        for group_name, group_df in analyzed_df.groupby(self.config.group_column, dropna=False):
            group_rows = (
                group_df
                if self.config.max_cases_per_group is None
                else group_df.head(self.config.max_cases_per_group)
            )

            for row_index, row in group_rows.iterrows():
                record = self.case_processor.process_row(
                    row,
                    row_index=int(row_index),
                    group_name=str(group_name),
                    config=self.config,
                )
                records.append(record)
                self._write_record_to_analyzed_df(analyzed_df, row_index, record)

        return analyzed_df, records

    def _prepare_agent_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет пустые колонки для результата агентной обработки.

        Args:
            df: Отфильтрованный DataFrame.

        Returns:
            DataFrame с колонками отчета, статуса, ошибки и structured result.
        """

        df[self.config.agent_report_column] = ""
        df[self.config.agent_status_column] = ""
        df[self.config.agent_error_column] = ""
        df[self.config.agent_structured_result_column] = None
        df[self.config.agent_missing_data_requests_column] = None
        return df

    def _write_record_to_analyzed_df(
        self,
        df: pd.DataFrame,
        row_index: int,
        record: CaseAnalysisRecord,
    ) -> None:
        """Записывает результат агентной обработки в строку `analyzed_df`.

        Args:
            df: DataFrame с колонками агентного результата.
            row_index: Индекс строки, которую обработал агент.
            record: Результат агентной обработки.

        Returns:
            None. Функция изменяет `df` на месте.
        """

        df.at[row_index, self.config.agent_report_column] = record.report_markdown
        df.at[row_index, self.config.agent_status_column] = record.status
        df.at[row_index, self.config.agent_error_column] = record.error
        df.at[row_index, self.config.agent_structured_result_column] = record.structured_result
        df.at[row_index, self.config.agent_missing_data_requests_column] = [
            request.model_dump(mode="json")
            for request in record.missing_data_requests
        ]

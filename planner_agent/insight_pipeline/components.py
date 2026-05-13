"""Компоненты и заглушки для pipeline поиска инсайтов.

Содержит:
- BaseDataEnricher: интерфейс обогащения входного DataFrame.
- NoopDataEnricher: обогащение без изменений.
- StubCommentClusterer: заглушка кластеризации комментариев.
- StubSignificantGroupSelector: заглушка выбора значимых групп.
- CaseAgentProcessor: адаптер запуска single-case агента по строкам DataFrame.
- GroupProblemSummarizer: суммаризатор проблем внутри групп.
- _content_to_text: извлекает текст из ответа агента.
- _extract_structured_result: извлекает JSON из ответа агента.
- _build_case_id: определяет case_id для строки.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Protocol

import pandas as pd

from planner_agent.insight_pipeline.formatting import build_case_prompt
from planner_agent.insight_pipeline.schemas import (
    CaseAnalysisRecord,
    GroupSelectionDecision,
    GroupProblemSummary,
    InsightPipelineConfig,
    MissingDataRequest,
)


class BaseDataEnricher(Protocol):
    """Интерфейс компонента обогащения данных.

    Args:
        df: pandas DataFrame после загрузки файла.

    Returns:
        pandas DataFrame с добавленными колонками и нормализованными значениями.
    """

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обогащает входной DataFrame.

        Args:
            df: Входной DataFrame с базовыми строками.

        Returns:
            DataFrame после обогащения.
        """


class NoopDataEnricher:
    """Заглушка обогащения, которая возвращает входной DataFrame без изменений."""

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает копию входного DataFrame без дополнительных колонок.

        Args:
            df: Входной DataFrame.

        Returns:
            Копия входного DataFrame.
        """

        return df.copy()


class StubCommentClusterer:
    """Заглушка кластеризатора комментариев.

    Если во входном DataFrame уже есть колонка группы, она сохраняется. Если колонки нет,
    создается одна группа `Без группы`. В реальном проекте этот компонент нужно заменить
    адаптером к готовой библиотеке кластеризации.
    """

    def cluster(self, df: pd.DataFrame, *, text_column: str, group_column: str) -> pd.DataFrame:
        """Добавляет колонку группы проблемы.

        Args:
            df: DataFrame после обогащения.
            text_column: Колонка с текстом комментария.
            group_column: Колонка, куда нужно записать название группы.

        Returns:
            DataFrame с колонкой `group_column`.
        """

        _ = text_column
        result = df.copy()
        if group_column not in result.columns:
            result[group_column] = "Без группы"
        result[group_column] = result[group_column].fillna("Без группы").astype(str)
        return result


class StubSignificantGroupSelector:
    """Детерминированная заглушка бинарной классификации групп вместо LLM.

    Заглушка считает группу значимой, если ее размер не меньше `min_group_size`.
    В реальном проекте этот компонент можно заменить LLM-классификатором, который
    получает одну группу, примеры комментариев и возвращает бинарное решение.
    """

    def classify_group(
        self,
        group_name: str,
        group_df: pd.DataFrame,
        config: InsightPipelineConfig,
    ) -> GroupSelectionDecision:
        """Классифицирует одну группу как значимую или незначимую.

        Args:
            group_name: Название группы после кластеризации.
            group_df: DataFrame только с записями этой группы.
            config: Конфигурация pipeline.

        Returns:
            Решение о включении группы в filtered_df.
        """

        total_records = len(group_df)
        is_meaningful = total_records >= config.min_group_size
        reason = (
            f"Размер группы {total_records} >= min_group_size={config.min_group_size}."
            if is_meaningful
            else f"Размер группы {total_records} < min_group_size={config.min_group_size}."
        )
        return GroupSelectionDecision(
            group_name=group_name,
            is_meaningful=is_meaningful,
            reason=reason,
            confidence=1.0,
            total_records=total_records,
        )

    def select_groups(self, df: pd.DataFrame, config: InsightPipelineConfig) -> list[str]:
        """Возвращает список значимых групп через бинарную классификацию каждой группы.

        Args:
            df: DataFrame после кластеризации.
            config: Конфигурация pipeline.

        Returns:
            Список названий групп, которые нужно обработать агентом.
        """

        decisions: list[GroupSelectionDecision] = []
        for group_name, group_df in df.groupby(config.group_column, dropna=False):
            decisions.append(self.classify_group(str(group_name), group_df, config))

        selected = [decision.group_name for decision in decisions if decision.is_meaningful]
        if config.max_groups is not None:
            selected = selected[: config.max_groups]
        return selected


def _content_to_text(response: Any) -> str:
    """Извлекает текст из ответа агента или LangChain messages.

    Args:
        response: Результат `invoke`, строка, список сообщений или произвольный объект.

    Returns:
        Текстовое содержимое ответа агента.
    """

    if isinstance(response, str):
        return response
    if isinstance(response, list) and response:
        last_item = response[-1]
        content = getattr(last_item, "content", None)
        if content is not None:
            return str(content)
        return str(last_item)
    content = getattr(response, "content", None)
    if content is not None:
        return str(content)
    return str(response)


def _extract_structured_result(text: str) -> dict[str, Any]:
    """Извлекает JSON-результат из тегов `<structured_result>`.

    Args:
        text: Текстовый ответ агента.

    Returns:
        Словарь с распарсенным JSON или пустой словарь, если JSON не найден.
    """

    tag_match = re.search(
        r"<structured_result>\s*(?P<payload>.*?)\s*</structured_result>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if tag_match:
        payload = tag_match.group("payload").strip()
        try:
            return json.loads(payload)
        except Exception:
            return {}

    fenced_match = re.search(
        r"```json\s*(?P<payload>\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced_match:
        try:
            return json.loads(fenced_match.group("payload"))
        except Exception:
            return {}

    return {}


def _build_case_id(row: pd.Series, config: InsightPipelineConfig, row_index: int) -> str:
    """Определяет идентификатор кейса для строки.

    Args:
        row: Строка DataFrame.
        config: Конфигурация pipeline.
        row_index: Индекс строки в DataFrame.

    Returns:
        Значение `case_id` или технический id на основе индекса строки.
    """

    if config.case_id_column in row and pd.notna(row[config.case_id_column]):
        return str(row[config.case_id_column])
    if config.event_id_column in row and pd.notna(row[config.event_id_column]):
        return str(row[config.event_id_column])
    return f"row_{row_index}"


class CaseAgentProcessor:
    """Адаптер обработки строк текущим single-case агентом.

    Args:
        agent: Runnable-совместимый агент с методом `invoke`. Если агент не передан,
            обработка возвращает `skipped`, но prompt будет сформирован.
        prompt_builder: Опциональная функция сборки prompt-а по строке.
        max_string_length: Максимальная длина строковых значений в prompt-е.

    Returns:
        Объект, который умеет обрабатывать одну строку или группу строк.
    """

    def __init__(
        self,
        agent: Any | None = None,
        *,
        prompt_builder: Callable[..., str] = build_case_prompt,
        max_string_length: int | None = None,
    ) -> None:
        self.agent = agent
        self.prompt_builder = prompt_builder
        self.max_string_length = max_string_length

    def process_row(
        self,
        row: pd.Series,
        *,
        row_index: int,
        group_name: str,
        config: InsightPipelineConfig,
    ) -> CaseAnalysisRecord:
        """Обрабатывает одну строку через агента.

        Args:
            row: Строка DataFrame с базовым контекстом кейса.
            row_index: Индекс строки в DataFrame.
            group_name: Название группы проблемы.
            config: Конфигурация pipeline.

        Returns:
            Результат обработки одной записи.
        """

        case_id = _build_case_id(row, config, row_index)
        prompt = self.prompt_builder(
            row.to_dict(),
            group_name=group_name,
            row_index=row_index,
            max_string_length=self.max_string_length,
        )

        if self.agent is None:
            return CaseAnalysisRecord(
                row_index=row_index,
                case_id=case_id,
                group_name=group_name,
                status="skipped",
                agent_prompt=prompt,
                report_markdown="Агент не передан. Сформирован только prompt для будущей обработки.",
            )

        try:
            response = self.agent.invoke(
                prompt,
                config={"recursion_limit": config.agent_recursion_limit},
            )
            report_text = _content_to_text(response)
            structured_result = _extract_structured_result(report_text)
            missing_data_requests = [
                MissingDataRequest.model_validate(item)
                for item in structured_result.get("missing_data_requests", [])
                if isinstance(item, dict)
            ]
            return CaseAnalysisRecord(
                row_index=row_index,
                case_id=case_id,
                group_name=group_name,
                status="processed",
                agent_prompt=prompt,
                report_markdown=report_text,
                structured_result=structured_result,
                missing_data_requests=missing_data_requests,
            )
        except Exception as exc:
            return CaseAnalysisRecord(
                row_index=row_index,
                case_id=case_id,
                group_name=group_name,
                status="failed",
                agent_prompt=prompt,
                error=str(exc),
            )


class GroupProblemSummarizer:
    """Суммаризатор проблем внутри выбранных групп.

    Args:
        summarizer: Опциональная функция или Runnable для LLM-суммаризации группы.
            Если не передана, используется детерминированная сводка по отчетам и counts.

    Returns:
        Объект, который строит список `GroupProblemSummary`.
    """

    def __init__(self, summarizer: Any | None = None) -> None:
        self.summarizer = summarizer

    def summarize(
        self,
        df: pd.DataFrame,
        records: list[CaseAnalysisRecord],
        *,
        selected_groups: list[str],
        config: InsightPipelineConfig,
    ) -> list[GroupProblemSummary]:
        """Строит итоговые сводки по выбранным группам.

        Args:
            df: DataFrame после кластеризации.
            records: Результаты агентной обработки записей.
            selected_groups: Список выбранных групп.
            config: Конфигурация pipeline.

        Returns:
            Список сводок по группам.
        """

        records_by_group: dict[str, list[CaseAnalysisRecord]] = {}
        for record in records:
            records_by_group.setdefault(record.group_name, []).append(record)

        summaries: list[GroupProblemSummary] = []
        for group_name in selected_groups:
            group_df = df[df[config.group_column].astype(str) == group_name]
            group_records = records_by_group.get(group_name, [])
            if self.summarizer is not None:
                summaries.append(
                    self._summarize_with_external(
                        group_name,
                        group_df,
                        group_records,
                    )
                )
                continue
            summaries.append(self._summarize_deterministic(group_name, group_df, group_records))
        return summaries

    def _summarize_deterministic(
        self,
        group_name: str,
        group_df: pd.DataFrame,
        records: list[CaseAnalysisRecord],
    ) -> GroupProblemSummary:
        """Строит детерминированную сводку группы без LLM.

        Args:
            group_name: Название группы.
            group_df: Строки DataFrame в этой группе.
            records: Результаты агентной обработки этой группы.

        Returns:
            Сводка по группе.
        """

        processed = [record for record in records if record.status == "processed"]
        failed = [record for record in records if record.status == "failed"]
        missing_count = sum(len(record.missing_data_requests) for record in records)
        evidence_case_ids = [record.case_id for record in processed[:5]]

        summaries = [
            str(record.structured_result.get("final_summary", "")).strip()
            for record in processed
            if record.structured_result.get("final_summary")
        ]
        if summaries:
            most_common_summary = Counter(summaries).most_common(1)[0][0]
        elif processed:
            most_common_summary = processed[0].report_markdown[:700]
        else:
            most_common_summary = (
                "Группа выбрана для анализа, но агентные отчеты пока отсутствуют "
                "или не были успешно обработаны."
            )

        limitations = []
        if missing_count:
            limitations.append(f"Агент запросил дозагрузку данных: {missing_count} запросов.")
        if failed:
            limitations.append(f"Ошибки обработки кейсов: {len(failed)}.")
        if len(processed) < len(group_df):
            limitations.append(
                f"Разобрано {len(processed)} из {len(group_df)} строк группы; вывод ограничен покрытием."
            )

        return GroupProblemSummary(
            group_name=group_name,
            total_records=int(len(group_df)),
            processed_records=len(processed),
            failed_records=len(failed),
            missing_data_requests_count=missing_count,
            problem_summary=most_common_summary,
            evidence_case_ids=evidence_case_ids,
            limitations=limitations,
        )

    def _summarize_with_external(
        self,
        group_name: str,
        group_df: pd.DataFrame,
        records: list[CaseAnalysisRecord],
    ) -> GroupProblemSummary:
        """Строит сводку группы через внешний summarizer.

        Args:
            group_name: Название группы.
            group_df: Строки DataFrame в этой группе.
            records: Результаты агентной обработки этой группы.

        Returns:
            Сводка по группе. Если внешний summarizer вернул текст, он попадет в `problem_summary`.
        """

        payload = {
            "group_name": group_name,
            "total_records": len(group_df),
            "case_reports": [record.model_dump(mode="json") for record in records],
        }
        if callable(self.summarizer):
            raw_summary = self.summarizer(payload)
        elif callable(getattr(self.summarizer, "invoke", None)):
            raw_summary = self.summarizer.invoke(payload)
        else:
            raw_summary = str(payload)

        return GroupProblemSummary(
            group_name=group_name,
            total_records=int(len(group_df)),
            processed_records=len([record for record in records if record.status == "processed"]),
            failed_records=len([record for record in records if record.status == "failed"]),
            missing_data_requests_count=sum(len(record.missing_data_requests) for record in records),
            problem_summary=_content_to_text(raw_summary),
            evidence_case_ids=[record.case_id for record in records[:5]],
            limitations=[],
        )

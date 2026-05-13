"""Пример запуска pipeline поиска инсайтов без API-ключей.

Содержит:
- DemoClusterer: пример адаптера к внешней библиотеке кластеризации.
- DemoAgent: локальная заглушка single-case агента.
- build_demo_dataframe: создает тестовый DataFrame.
- main: запускает pipeline и печатает результаты.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from planner_agent.insight_pipeline import (
    CaseAgentProcessor,
    InsightPipeline,
    InsightPipelineConfig,
)


class DemoClusterer:
    """Демонстрационный адаптер кластеризации комментариев.

    В реальном проекте вместо этой заглушки нужно вызвать готовую библиотеку,
    которая принимает DataFrame и возвращает тот же DataFrame с колонкой группы.
    """

    def cluster(self, df: pd.DataFrame, *, text_column: str, group_column: str) -> pd.DataFrame:
        """Добавляет группу проблемы по простым ключевым словам.

        Args:
            df: Входной DataFrame.
            text_column: Колонка с комментарием клиента.
            group_column: Колонка, куда нужно записать группу.

        Returns:
            DataFrame с добавленной колонкой группы.
        """

        result = df.copy()
        text = result[text_column].fillna("").astype(str).str.lower()
        result[group_column] = "Другое"
        result.loc[text.str.contains("себе|свой счет|свой счёт"), group_column] = "Перевод себе"
        result.loc[text.str.contains("оплата|платеж|платёж"), group_column] = "Регулярная оплата"
        return result


class DemoAgent:
    """Локальная заглушка агента с методом `invoke`.

    Заглушка нужна только для проверки pipeline без внешних LLM и API-ключей.
    """

    def invoke(self, prompt: str, config: dict | None = None) -> str:
        """Возвращает тестовый отчет со structured_result.

        Args:
            prompt: Текстовый запрос, сформированный pipeline.
            config: Техническая конфигурация запуска агента.

        Returns:
            Строка, похожая на ответ single-case агента.
        """

        _ = config
        needs_more_data = "events_day" not in prompt
        missing = (
            """
    {
      "source_name": "client_transaction_history",
      "reason": "Нужно проверить историю похожих операций клиента.",
      "lookup_keys": {"epk_id": "из prompt"},
      "period": "180 дней до даты сработки",
      "priority": "medium"
    }
"""
            if needs_more_data
            else ""
        )
        missing_block = f"[{missing}]" if needs_more_data else "[]"
        return f"""
# Отчет по кейсу

Кейс разобран в демонстрационном режиме. Реальный агент должен подтянуть факты через tools.

<structured_result>
{{
  "case_id": "demo_case",
  "event_id": null,
  "group_name": "demo_group",
  "facts": ["Получен базовый контекст строки."],
  "hypotheses": ["Требуется проверка полной истории клиента."],
  "missing_data_requests": {missing_block},
  "limitations": ["Это демонстрационный агент без доступа к источникам."],
  "final_summary": "Группа содержит повторяющиеся жалобы, требующие проверки фактического контекста операций."
}}
</structured_result>
""".strip()


def build_demo_dataframe() -> pd.DataFrame:
    """Создает тестовый DataFrame для локальной проверки pipeline.

    Args:
        Нет входных аргументов.

    Returns:
        DataFrame с минимальным набором колонок для pipeline.
    """

    return pd.DataFrame(
        [
            {
                "case_id": "case_001",
                "event_id": "event_001",
                "epk_id": "1001",
                "event_dt": "20260401",
                "comment_text": "Заблокировали перевод самому себе",
                "amount": 15000,
            },
            {
                "case_id": "case_002",
                "event_id": "event_002",
                "epk_id": "1002",
                "event_dt": "20260401",
                "comment_text": "Почему не проходит перевод на свой счет",
                "amount": 22000,
            },
            {
                "case_id": "case_003",
                "event_id": "event_003",
                "epk_id": "1003",
                "event_dt": "20260402",
                "comment_text": "Не прошла регулярная оплата",
                "amount": 3500,
            },
        ]
    )


def main() -> None:
    """Запускает демонстрационный pipeline.

    Args:
        Нет входных аргументов.

    Returns:
        None. Функция печатает выбранные группы и сводки.
    """

    config = InsightPipelineConfig(
        text_column="comment_text",
        group_column="problem_group",
        case_id_column="case_id",
        min_group_size=1,
        max_cases_per_group=2,
    )
    pipeline = InsightPipeline(
        config=config,
        clusterer=DemoClusterer(),
        case_processor=CaseAgentProcessor(agent=DemoAgent()),
    )

    run = pipeline.run(build_demo_dataframe())
    print("Selected groups:", run.selected_groups)
    print(run.case_results_df()[["case_id", "group_name", "status"]])
    print(run.group_summaries_df()[["group_name", "total_records", "problem_summary"]])


if __name__ == "__main__":
    main()

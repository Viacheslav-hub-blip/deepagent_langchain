"""Публичные entrypoints pipeline поиска инсайтов.

Содержит:
- InsightPipeline: основной orchestrator загрузки, обогащения, кластеризации, case-разбора и суммаризации.
- InsightPipelineConfig: конфигурация колонок, лимитов и форматов входных данных.
- InsightPipelineRun: результат выполнения pipeline с DataFrame и структурированными итогами.
- GroupSelectionDecision: решение LLM/classifier о том, включать ли группу в дальнейший разбор.
- BaseDataEnricher: базовый интерфейс обогащения данных.
- NoopDataEnricher: заглушка обогащения без изменения данных.
- StubCommentClusterer: заглушка кластеризации комментариев.
- StubSignificantGroupSelector: заглушка выбора значимых групп.
- CaseAgentProcessor: адаптер запуска single-case агента по строкам DataFrame.
- GroupProblemSummarizer: суммаризатор проблем внутри групп.
"""

from planner_agent.insight_pipeline.components import (
    BaseDataEnricher,
    CaseAgentProcessor,
    GroupProblemSummarizer,
    NoopDataEnricher,
    StubCommentClusterer,
    StubSignificantGroupSelector,
)
from planner_agent.insight_pipeline.pipeline import InsightPipeline, InsightPipelineRun
from planner_agent.insight_pipeline.schemas import (
    CaseAnalysisRecord,
    GroupProblemSummary,
    GroupSelectionDecision,
    InsightPipelineConfig,
    MissingDataRequest,
)

__all__ = [
    "BaseDataEnricher",
    "CaseAgentProcessor",
    "CaseAnalysisRecord",
    "GroupProblemSummary",
    "GroupProblemSummarizer",
    "GroupSelectionDecision",
    "InsightPipeline",
    "InsightPipelineConfig",
    "InsightPipelineRun",
    "MissingDataRequest",
    "NoopDataEnricher",
    "StubCommentClusterer",
    "StubSignificantGroupSelector",
]

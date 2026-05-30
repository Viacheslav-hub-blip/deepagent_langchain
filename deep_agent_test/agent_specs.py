"""Контракт supervisor-а и subagent data-retrieval аналитического DeepAgent."""

from __future__ import annotations

from pydantic import BaseModel, Field

DATA_RETRIEVAL_AGENT_NAME = "data-retrieval-agent"
DATA_RETRIEVAL_CRITIC_AGENT_NAME = "data-retrieval-critic"


class DataRetrievalCriticVerdict(BaseModel):
    """Structured output внутреннего critic-а для data-retrieval-agent."""

    approved: bool = Field(
        description="Можно ли считать шаг чтения данных завершённым и отдать результат supervisor-у.",
    )
    reasoning: str = Field(
        description="Краткое обоснование с опорой на проверенные факты и tool output.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Список выявленных проблем без обязательного фиксированного формата.",
    )
    revision_instructions: str = Field(
        default="",
        description="Что data-retrieval-agent должен сделать следующим, если approved=false.",
    )
    checks_performed: list[str] = Field(
        default_factory=list,
        description="Какие проверки или tools critic выполнил (для аудита).",
    )


__all__ = [
    "DATA_RETRIEVAL_AGENT_NAME",
    "DATA_RETRIEVAL_CRITIC_AGENT_NAME",
    "DataRetrievalCriticVerdict",
]

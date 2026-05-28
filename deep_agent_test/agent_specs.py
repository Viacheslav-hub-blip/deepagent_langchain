"""Контракт supervisor-а, subagents и HITL-действий аналитического DeepAgent.

Содержит:
- CLARIFY_ANALYSIS_REQUEST_TOOL: имя tool уточнения пользовательского запроса.
- DATA_RETRIEVAL_AGENT_NAME: имя subagent-а чтения данных.
- PYTHON_ANALYSIS_AGENT_NAME: имя subagent-а Python-анализа.
- build_clarify_interrupt_config: сборка Human-in-the-loop правила для уточняющего вопроса.
- build_analytics_subagent_specs: сборка спецификаций subagents для ``create_deep_agent``.
"""

from __future__ import annotations

from typing import Any

from deep_agent_test.prompts import DATA_RETRIEVAL_PROMPT, PYTHON_ANALYSIS_PROMPT
from deep_agent_test.settings import DeepAgentSettings

CLARIFY_ANALYSIS_REQUEST_TOOL = "clarify_analysis_request"
DATA_RETRIEVAL_AGENT_NAME = "data-retrieval-agent"
PYTHON_ANALYSIS_AGENT_NAME = "python-analysis-agent"

DATA_RETRIEVAL_AGENT_DESCRIPTION = (
    "Читает фактические строки данных через read_table по таблицам, которые описаны в skills. "
    "Используй этого subagent-а для выборок по конкретным полям, фильтрам, ключам и периоду. "
    "Не используй его для расчетов, join, отчетов или создания файлов."
)
PYTHON_ANALYSIS_AGENT_DESCRIPTION = (
    "Выполняет Python-анализ по уже полученным данным: объединение источников, сортировку, "
    "нормализацию, расчеты, подготовку CSV/отчетов и сохранение готового файла."
)


def build_clarify_interrupt_config() -> dict[str, dict[str, Any]]:
    """Собирает конфигурацию interrupt для уточняющего вопроса supervisor-а.

    Args:
        Отсутствуют.

    Returns:
        Словарь ``interrupt_on`` для ``create_deep_agent``. Пользователь отвечает
        текстом через решение ``respond``.
    """

    return {
        CLARIFY_ANALYSIS_REQUEST_TOOL: {
            "allowed_decisions": ["respond"],
            "description": "Опишите задачу анализа, ожидаемый результат, формат ответа и обязательные элементы.",
        },
    }


def build_analytics_subagent_specs(
    *,
    settings: DeepAgentSettings,
    data_tools: list[Any],
    analysis_tools: list[Any],
    common_middleware: list[Any],
) -> list[dict[str, Any]]:
    """Собирает спецификации subagents аналитического DeepAgent.

    Args:
        settings: Настройки агента с виртуальной папкой skills.
        data_tools: Tools чтения данных для ``data-retrieval-agent``.
        analysis_tools: Tools расчетов и сохранения файлов для ``python-analysis-agent``.
        common_middleware: Middleware, которые должны работать внутри обоих subagents.

    Returns:
        Список спецификаций subagents в формате ``create_deep_agent``.
    """

    return [
        {
            "name": DATA_RETRIEVAL_AGENT_NAME,
            "description": DATA_RETRIEVAL_AGENT_DESCRIPTION,
            "system_prompt": DATA_RETRIEVAL_PROMPT,
            "tools": data_tools,
            "skills": [settings.skills_virtual_dir],
            "middleware": common_middleware,
        },
        {
            "name": PYTHON_ANALYSIS_AGENT_NAME,
            "description": PYTHON_ANALYSIS_AGENT_DESCRIPTION,
            "system_prompt": PYTHON_ANALYSIS_PROMPT,
            "tools": analysis_tools,
            "skills": [settings.skills_virtual_dir],
            "middleware": common_middleware,
        },
    ]


__all__ = [
    "CLARIFY_ANALYSIS_REQUEST_TOOL",
    "DATA_RETRIEVAL_AGENT_NAME",
    "PYTHON_ANALYSIS_AGENT_NAME",
    "build_analytics_subagent_specs",
    "build_clarify_interrupt_config",
]

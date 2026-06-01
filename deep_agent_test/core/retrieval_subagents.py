"""Сборка data-retrieval subagent с вложенным critic через нативный DeepAgents ``task``.

Содержит:
- build_data_retrieval_critic_tools: инструменты critic-а.
- build_critic_filesystem_permissions: права critic-а на чтение артефактов.
- build_data_retrieval_subagent_spec: спецификация data-retrieval-agent.
- build_analytics_subagent_specs: список subagents для supervisor-а.
"""

from __future__ import annotations

from typing import Any

from deepagents import FilesystemPermission
from deepagents.middleware.subagents import SubAgentMiddleware

from deep_agent_test.core.agent_specs import (
    DATA_RETRIEVAL_AGENT_NAME,
    DATA_RETRIEVAL_CRITIC_AGENT_NAME,
    DataRetrievalCriticVerdict,
)
from deep_agent_test.tools.inspect_artifact import build_inspect_artifact_tool
from deep_agent_test.core.prompts import (
    DATA_RETRIEVAL_CRITIC_PROMPT,
    DATA_RETRIEVAL_INNER_TASK_PROMPT,
    DATA_RETRIEVAL_PROMPT,
    DATA_RETRIEVAL_PROMPT_WITHOUT_CRITIC,
)
from deep_agent_test.core.settings import DeepAgentSettings


def build_data_retrieval_critic_tools(settings: DeepAgentSettings) -> list[Any]:
    """Возвращает tools критика: только ``inspect_artifact_path`` для проверки файлов."""

    return [build_inspect_artifact_tool(settings)]


def build_critic_filesystem_permissions(settings: DeepAgentSettings) -> list[FilesystemPermission]:
    """Возвращает permissions критика для чтения файлов с результатами инструментов."""

    return [
        FilesystemPermission(operations=["read", "list"], paths=["/tool_outputs/**"], mode="allow"),
    ]


def build_data_retrieval_subagent_spec(
    *,
    settings: DeepAgentSettings,
    model: Any,
    backend: Any,
    data_tools: list[Any],
    common_middleware: list[Any],
    critic_middleware: list[Any] | None = None,
    enable_critic: bool = True,
) -> dict[str, Any]:
    """data-retrieval-agent: load_data + (опционально) внутренний task(critic) в одном invoke.

    ``critic_middleware`` намеренно не содержит skills-middleware: critic не должен
    загружать domain skills, он только верифицирует ответ по реальным tool results.
    Если не передан, используется ``common_middleware``.

    ``enable_critic`` управляет вложенным critic. При ``False`` critic полностью
    отключается: tool ``task(data-retrieval-critic)`` не подключается, а data-retrieval-agent
    получает prompt без инструкций critic-цикла и отдаёт отчёт supervisor-у напрямую.
    """

    if not enable_critic:
        return {
            "name": DATA_RETRIEVAL_AGENT_NAME,
            "description": (
                "Читает табличные данные через load_data и возвращает структурированный "
                "отчёт supervisor-у. Используй для выборок по полям, фильтрам, ключам и периоду."
            ),
            "system_prompt": DATA_RETRIEVAL_PROMPT_WITHOUT_CRITIC,
            "model": model,
            "tools": data_tools,
            "skills": [settings.skills_virtual_dir],
            "middleware": list(common_middleware),
        }

    critic_spec: dict[str, Any] = {
        "name": DATA_RETRIEVAL_CRITIC_AGENT_NAME,
        "description": (
            "Проверяет, что ответ data-retrieval-agent основан на реальных tool results "
            "и что заявленные файлы/артефакты существуют. Вызывается только изнутри "
            "data-retrieval-agent через task."
        ),
        "system_prompt": DATA_RETRIEVAL_CRITIC_PROMPT,
        "model": model,
        "tools": build_data_retrieval_critic_tools(settings),
        "response_format": DataRetrievalCriticVerdict,
        "permissions": build_critic_filesystem_permissions(settings),
        "middleware": list(common_middleware if critic_middleware is None else critic_middleware),
    }

    inner_task_middleware = SubAgentMiddleware(
        backend=backend,
        subagents=[critic_spec],
        system_prompt=DATA_RETRIEVAL_INNER_TASK_PROMPT,
    )

    return {
        "name": DATA_RETRIEVAL_AGENT_NAME,
        "description": (
            "Читает табличные данные через load_data. Внутри выполняет проверку critic-ом "
            "перед ответом supervisor-у. Используй для выборок по полям, фильтрам, ключам и периоду."
        ),
        "system_prompt": DATA_RETRIEVAL_PROMPT,
        "model": model,
        "tools": data_tools,
        "skills": [settings.skills_virtual_dir],
        "middleware": [*common_middleware, inner_task_middleware],
    }


def build_analytics_subagent_specs(
    *,
    settings: DeepAgentSettings,
    data_tools: list[Any],
    common_middleware: list[Any],
    model: Any,
    backend: Any,
    critic_middleware: list[Any] | None = None,
    enable_critic: bool = True,
) -> list[dict[str, Any]]:
    """Собирает список спеков субагентов supervisor-а (сейчас один data-retrieval-agent).

    Args:
        settings: Настройки агента.
        data_tools: Инструменты чтения данных для data-retrieval-agent.
        common_middleware: Middleware data-retrieval-agent.
        model: Chat model для субагента и критика.
        backend: Backend skills/spill-файлов.
        critic_middleware: Middleware критика (без skills); ``None`` — как у субагента.
        enable_critic: Подключать ли внутренний critic. ``False`` — без критика.

    Returns:
        Список спеков субагентов для ``create_deep_agent(subagents=...)``.
    """

    return [
        build_data_retrieval_subagent_spec(
            settings=settings,
            model=model,
            backend=backend,
            data_tools=data_tools,
            common_middleware=common_middleware,
            critic_middleware=critic_middleware,
            enable_critic=enable_critic,
        ),
    ]


__all__ = [
    "build_analytics_subagent_specs",
    "build_critic_filesystem_permissions",
    "build_data_retrieval_critic_tools",
    "build_data_retrieval_subagent_spec",
]

"""Запуск UI API вместе с ResearchAgent и нативным инструментом выполнения кода.

Содержит:
- _build_sandbox: создание пустой Python-песочницы с разрешенными библиотеками.
- _build_agent: сборка ResearchAgent с sandbox, fake Spark tools и python_analysis.
- _run_async_before_server_start: синхронный запуск async-кода до старта uvicorn.
- create_app_with_agent: factory FastAPI приложения для `uvicorn --factory`.

Файл нужен для локального запуска analyst UI так, чтобы кнопка запуска анализа
работала через реальные endpoints `/api/v1/runs/invoke` и `/api/v1/branches/invoke`,
а worker мог выполнять Python-код через нативный инструмент `python_analysis`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from examples.fake_spark_tools import build_fake_spark_tools
from model import model as deepseek_model
from model import gigachat 
from planner_agent import ResearchAgent
from sandbox import ClientPythonSandbox
from planner_agent.http_api import ApiSettings, create_app
from planner_agent.http_api.config import ApiServices



PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_ROOT = PROJECT_ROOT / "examples"


def _suppress_uvicorn_access_log() -> None:
    """Отключает построчный access-log uvicorn (GET /api/... 200 OK)."""

    logging.getLogger("uvicorn.access").disabled = True


def _build_sandbox() -> ClientPythonSandbox:
    """Создает Python-песочницу без предзагруженных пользовательских таблиц.

    Args:
        Отсутствуют.

    Returns:
        ClientPythonSandbox с доступными библиотеками `pd`, `px`.
    """

    return ClientPythonSandbox(allowed_libraries={"pd": pd, "px": px})


async def _build_agent() -> ResearchAgent:
    """Собирает ResearchAgent для UI с fake Spark tools и python_analysis.

    Args:
        Отсутствуют.

    Returns:
        ResearchAgent, совместимый с LangChain Runnable API и подключенный к
        локальной Python-песочнице. Нативный инструмент ``python_analysis``
        добавляется внутри фабрики агента.
    """

    sandbox = _build_sandbox()
    spark_tools = build_fake_spark_tools(
        delay_seconds=0.5,
        transaction_count=120,
        day_event_count=40,
    )

    return ResearchAgent(
        model=gigachat,
        sandbox=sandbox,
        tools=spark_tools,
        code_generator_tool_names=set(),
        enable_workspace_tools=True,
        workspace_root=str(PROJECT_ROOT),
        sources_dir=str(EXAMPLE_ROOT / "data"),
        contexts_dir=str(PROJECT_ROOT / "skills"),
        skills_dir=str(PROJECT_ROOT / "skills"),
        memory_dir=str(EXAMPLE_ROOT / "memory"),
        runs_dir=str(EXAMPLE_ROOT / "runs"),
        stream_console=True,
    )


def _run_async_before_server_start(coro: Any) -> Any:
    """Выполняет coroutine до запуска event loop FastAPI/uvicorn.

    Args:
        coro: Coroutine, которую нужно выполнить синхронно.

    Returns:
        Результат выполнения coroutine.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}

    def runner() -> None:
        """Запускает coroutine в отдельном потоке с отдельным event loop.

        Args:
            Отсутствуют. Использует coroutine из замыкания.

        Returns:
            None. Результат или ошибка сохраняются в словарь `result`.
        """

        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, name="research-agent-factory-loader")
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def create_app_with_agent():
    """Создает FastAPI приложение с подключенным ResearchAgent.

    Args:
        Отсутствуют. Агент собирается из текущего проекта, `model.py`,
        fake Spark tools и нативного инструмента `python_analysis`.

    Returns:
        FastAPI приложение, которое раздает статический UI по `/app/` и умеет
        запускать агента через API endpoints.
    """

    _suppress_uvicorn_access_log()
    agent = _run_async_before_server_start(_build_agent())
    services = ApiServices(
        lineage_service=agent.lineage_service,
        artifact_service=agent.artifact_service,
        inspection_service=agent.inspection_service,
        dialog_context_service=agent.dialog_context_service,
        skills_service=agent.skills_service,
        agent=agent,
    )
    return create_app(
        settings=ApiSettings(
            workspace_root=str(PROJECT_ROOT),
            runs_dir=str(EXAMPLE_ROOT / "runs"),
            api_prefix="/api/v1",
        ),
        services=services,
    )


if __name__ == "__main__":
    import uvicorn

    _suppress_uvicorn_access_log()
    uvicorn.run(
        "main_ui_agent_server:create_app_with_agent",
        factory=True,
        host="127.0.0.1",
        port=8000,
        access_log=False,
    )

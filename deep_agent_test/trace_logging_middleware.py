"""Middleware файловой трассировки tool calls аналитического DeepAgent.

Содержит:
- ToolTraceLoggingMiddleware: middleware записи видимых tools и tool calls в JSONL.
- ToolTraceLoggingMiddleware.wrap_model_call: логирование tools перед вызовом модели.
- ToolTraceLoggingMiddleware.after_model: логирование tool calls после ответа модели.
- ToolTraceLoggingMiddleware.wrap_tool_call: логирование sync tool call.
- ToolTraceLoggingMiddleware.awrap_tool_call: логирование async tool call.
- _agent_name: извлечение имени агента из runtime metadata.
- _strip_service_state_from_task_command: очистка служебных state-полей из результата task.
- _serialize_tool_result: структурированное представление результата tool call.
- _format_json: компактное JSON-представление значения.
- _format_tool_result: компактное описание результата tool call.
- _print_tool_call: вывод фактического вызова tool в консоль.
- _print_tool_result: вывод ответа tool в консоль.
- _preview_text: ограничение длины текста для логов.
- _tool_names: извлечение имен tools из ModelRequest.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from deep_agent_test.agent_logging import DeepAgentEventLogger

TASK_TOOL_NAME = "task"
RESPONSE_FORMAT_TOOL_NAMES = {"DataRetrievalResponse", "PythonAnalysisResponse"}
SERVICE_STATE_KEYS_FROM_TASK = {
    "approved_plan_user_key",
    "few_shot_example_names",
    "few_shot_examples",
    "few_shot_examples_user_key",
    "memory_contents",
    "preloaded_skill_paths",
    "preloaded_skills_context",
    "preloaded_skills_selection_user_key",
    "skills_context_loaded",
    "skills_load_errors",
    "skills_metadata",
}


@dataclass(frozen=True)
class ToolTraceLoggingMiddleware(AgentMiddleware):
    """Пишет трассировку tools и очищает служебный state из результата ``task``.

    Args:
        event_logger: Файловый логгер DeepAgent.
        preview_chars: Максимальная длина preview для аргументов и результатов.
        log_available_tools: Нужно ли логировать tools, доступные модели.
        log_model_tool_calls: Нужно ли логировать tool calls из ответа модели.
        log_tool_execution: Нужно ли логировать старт выполнения tool.
        log_tool_result: Нужно ли логировать результат выполнения tool.
        print_tool_calls: Нужно ли печатать фактические вызовы tools в консоль.
        print_tool_results: Нужно ли печатать ответы tools в консоль.

    Returns:
        Middleware LangChain, которое пишет JSONL-логи и не передает supervisor-у
        большие служебные state-поля subagent-а из результата ``task``.
    """

    event_logger: DeepAgentEventLogger
    preview_chars: int = 1200
    log_available_tools: bool = False
    log_model_tool_calls: bool = True
    log_tool_execution: bool = True
    log_tool_result: bool = True
    print_tool_calls: bool = False
    print_tool_results: bool = False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Логирует список tools перед вызовом модели, если это включено настройками.

        Args:
            request: Запрос модели с сообщениями, tools, state и runtime.
            handler: Функция реального вызова модели.

        Returns:
            Ответ модели без изменений.
        """

        if self.log_available_tools:
            self.event_logger.log_tool_event(
                "available_tools",
                {
                    "agent": _agent_name(request.runtime),
                    "tools": _tool_names(request.tools),
                },
            )
        return handler(request)

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Логирует tool calls, которые модель запросила после ответа.

        Args:
            state: Текущий state агента с историей сообщений.
            runtime: Runtime LangGraph текущего запуска.

        Returns:
            None, потому что middleware не меняет state.
        """

        if not self.log_model_tool_calls:
            return None

        messages = state.get("messages") or []
        last_ai_message = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
        if last_ai_message is None or not last_ai_message.tool_calls:
            return None

        calls = [
            {
                "name": tool_call.get("name"),
                "args": tool_call.get("args", {}),
                "id": tool_call.get("id"),
            }
            for tool_call in last_ai_message.tool_calls
        ]
        self.event_logger.log_tool_event(
            "model_tool_calls",
            {
                "agent": _agent_name(runtime),
                "calls": calls,
            },
        )
        if self.print_tool_calls:
            for call in calls:
                if call.get("name") in RESPONSE_FORMAT_TOOL_NAMES:
                    _print_model_tool_call(
                        agent=_agent_name(runtime),
                        tool_name=str(call.get("name")),
                        args_preview=_format_json(call.get("args", {}), self.preview_chars),
                    )
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Логирует синхронное выполнение tool call.

        Args:
            request: Запрос на выполнение tool.
            handler: Функция реального выполнения tool.

        Returns:
            Результат tool без изменений.
        """

        tool_name = str(request.tool_call.get("name") or "tool")
        agent_name = _agent_name(request.runtime)
        args_preview = _format_json(request.tool_call.get("args", {}), self.preview_chars)
        if self.log_tool_execution:
            self.event_logger.log_tool_event(
                "tool_start",
                {
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": request.tool_call.get("id"),
                    "args_preview": args_preview,
                },
            )
        if self.print_tool_calls:
            _print_tool_call(agent=agent_name, tool_name=tool_name, args_preview=args_preview)
        result = handler(request)
        result = _strip_service_state_from_task_command(result, tool_name=tool_name)
        result_payload = _serialize_tool_result(result)
        result_preview = _format_tool_result(result, self.preview_chars)
        if self.log_tool_result:
            self.event_logger.log_tool_event(
                "tool_end",
                {
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": request.tool_call.get("id"),
                    "result_preview": result_preview,
                },
            )
            self.event_logger.log_tool_event(
                "tool_io",
                {
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": request.tool_call.get("id"),
                    "args": request.tool_call.get("args", {}),
                    "args_preview": args_preview,
                    "result": result_payload,
                    "result_preview": result_preview,
                },
            )
        if self.print_tool_results:
            _print_tool_result(agent=agent_name, tool_name=tool_name, result_preview=result_preview)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Логирует асинхронное выполнение tool call.

        Args:
            request: Запрос на асинхронное выполнение tool.
            handler: Асинхронная функция реального выполнения tool.

        Returns:
            Результат tool без изменений.
        """

        tool_name = str(request.tool_call.get("name") or "tool")
        agent_name = _agent_name(request.runtime)
        args_preview = _format_json(request.tool_call.get("args", {}), self.preview_chars)
        if self.log_tool_execution:
            self.event_logger.log_tool_event(
                "tool_start",
                {
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": request.tool_call.get("id"),
                    "args_preview": args_preview,
                },
            )
        if self.print_tool_calls:
            _print_tool_call(agent=agent_name, tool_name=tool_name, args_preview=args_preview)
        result = await handler(request)
        result = _strip_service_state_from_task_command(result, tool_name=tool_name)
        result_payload = _serialize_tool_result(result)
        result_preview = _format_tool_result(result, self.preview_chars)
        if self.log_tool_result:
            self.event_logger.log_tool_event(
                "tool_end",
                {
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": request.tool_call.get("id"),
                    "result_preview": result_preview,
                },
            )
            self.event_logger.log_tool_event(
                "tool_io",
                {
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "tool_call_id": request.tool_call.get("id"),
                    "args": request.tool_call.get("args", {}),
                    "args_preview": args_preview,
                    "result": result_payload,
                    "result_preview": result_preview,
                },
            )
        if self.print_tool_results:
            _print_tool_result(agent=agent_name, tool_name=tool_name, result_preview=result_preview)
        return result


def _agent_name(runtime: Any) -> str:
    """Извлекает имя агента из runtime metadata.

    Args:
        runtime: Runtime LangGraph или ToolRuntime.

    Returns:
        Имя агента из metadata или ``main``.
    """

    config = getattr(runtime, "config", {}) or {}
    metadata = config.get("metadata", {})
    return str(metadata.get("lc_agent_name") or "main")


def _strip_service_state_from_task_command(value: ToolMessage | Command[Any], *, tool_name: str) -> ToolMessage | Command[Any]:
    """Удаляет служебные state-поля subagent-а из результата ``task``.

    Args:
        value: Результат tool call, который может быть ``ToolMessage`` или ``Command``.
        tool_name: Имя выполненного инструмента.

    Returns:
        Исходный результат или новый ``Command`` с очищенным ``update``.
    """

    if tool_name != TASK_TOOL_NAME or not isinstance(value, Command):
        return value
    update = value.update
    if not isinstance(update, dict):
        return value

    cleaned_update = {key: item for key, item in update.items() if key not in SERVICE_STATE_KEYS_FROM_TASK}
    if len(cleaned_update) == len(update):
        return value
    return Command(
        graph=value.graph,
        update=cleaned_update,
        resume=value.resume,
        goto=value.goto,
    )


def _serialize_tool_result(value: Any) -> dict[str, Any]:
    """Преобразует результат tool call в структуру для JSONL-трассировки.

    Args:
        value: Результат tool call, например ``ToolMessage`` или ``Command``.

    Returns:
        Словарь с типом результата и основными данными, пригодными для JSONL.
    """

    payload: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, ToolMessage):
        payload.update(
            {
                "content": value.content,
                "name": value.name,
                "tool_call_id": value.tool_call_id,
                "status": getattr(value, "status", None),
            }
        )
        artifact = getattr(value, "artifact", None)
        if artifact is not None:
            payload["artifact"] = artifact
        return payload

    if isinstance(value, Command):
        payload.update(
            {
                "graph": value.graph,
                "goto": value.goto,
                "resume": value.resume,
                "update": _serialize_command_update(value.update),
            }
        )
        return payload

    payload["value"] = value
    return payload


def _serialize_command_update(update: Any) -> Any:
    """Сериализует update из ``Command`` без тяжелых служебных полей.

    Args:
        update: Значение ``Command.update``.

    Returns:
        JSON-совместимое представление update.
    """

    if not isinstance(update, dict):
        return update

    serialized: dict[str, Any] = {}
    for key, item in update.items():
        if key in SERVICE_STATE_KEYS_FROM_TASK:
            continue
        if key == "messages" and isinstance(item, list):
            serialized[key] = [_serialize_message(message) for message in item]
        else:
            serialized[key] = item
    return serialized


def _serialize_message(message: Any) -> dict[str, Any]:
    """Преобразует сообщение LangChain в компактный словарь для лога.

    Args:
        message: Сообщение из state update.

    Returns:
        Словарь с типом, содержимым и служебными идентификаторами сообщения.
    """

    payload = {
        "type": type(message).__name__,
        "content": getattr(message, "content", None),
        "name": getattr(message, "name", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


def _format_json(value: Any, max_chars: int) -> str:
    """Форматирует значение как короткий JSON.

    Args:
        value: Объект для логирования.
        max_chars: Максимальная длина результата.

    Returns:
        JSON-строка или строковое представление объекта.
    """

    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    return _preview_text(text, max_chars)


def _format_tool_result(value: Any, max_chars: int) -> str:
    """Форматирует результат tool call для файлового лога.

    Args:
        value: Результат tool call.
        max_chars: Максимальная длина preview.

    Returns:
        Короткое описание результата tool call.
    """

    content = getattr(value, "content", value)
    status = getattr(value, "status", None)
    prefix = f"status={status} " if status else ""
    return prefix + _preview_text(str(content), max_chars)


def _print_tool_call(*, agent: str, tool_name: str, args_preview: str) -> None:
    """Печатает фактический вызов tool в консоль.

    Args:
        agent: Имя агента, который выполняет tool.
        tool_name: Имя вызываемого инструмента.
        args_preview: Краткое JSON-представление аргументов tool.

    Returns:
        None.
    """

    print()
    print("Агент:")
    print(f"Вызов инструмента `{tool_name}` ({agent}):")
    print(args_preview)
    print()


def _print_tool_result(*, agent: str, tool_name: str, result_preview: str) -> None:
    """Печатает ответ tool в консоль.

    Args:
        agent: Имя агента, который выполнял tool.
        tool_name: Имя инструмента.
        result_preview: Краткое текстовое представление ответа tool.

    Returns:
        None.
    """

    print()
    print("Агент:")
    print(f"Ответ инструмента `{tool_name}` ({agent}):")
    print(result_preview)
    print()


def _print_model_tool_call(*, agent: str, tool_name: str, args_preview: str) -> None:
    """Печатает tool call из ответа модели, который не исполняется ToolNode.

    Args:
        agent: Имя агента, который запросил structured output.
        tool_name: Имя response-format tool.
        args_preview: Краткое JSON-представление аргументов.

    Returns:
        None.
    """

    print()
    print("Агент:")
    print(f"Вызов structured output `{tool_name}` ({agent}):")
    print(args_preview)
    print()


def _preview_text(text: str, max_chars: int) -> str:
    """Обрезает текст до заданной длины.

    Args:
        text: Исходный текст.
        max_chars: Максимальная длина текста.

    Returns:
        Исходный или обрезанный текст.
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... <truncated {len(text) - max_chars} chars>"


def _tool_names(tools: list[BaseTool | dict[str, Any]]) -> list[str]:
    """Извлекает имена tools из списка LangChain tools.

    Args:
        tools: Tools из ``ModelRequest.tools``.

    Returns:
        Список имен tools, которые видит модель.
    """

    names: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            names.append(str(tool.get("name") or tool.get("function", {}).get("name") or "<dict_tool>"))
        else:
            names.append(str(getattr(tool, "name", "<tool>")))
    return names


__all__ = ["ToolTraceLoggingMiddleware"]

"""Минимальный запуск аналитического DeepAgent на fake-данных.

Содержит:
- main: инициализация fake-инструмента чтения данных, trace-логгера, агента и один invoke.
- main_stream: запуск агента со стримингом человекочитаемых промежуточных шагов.
- _print_v3_progress: вывод typed-событий ``stream_events(version="v3")``.
- _print_tool_call_progress: вывод статуса одного tool call.
- _subagent_status: форматирование статуса subagent-а.
- _tool_call_status: форматирование статуса вызова инструмента.
- _compact_args_preview: компактное представление аргументов инструмента.
- _last_message_text: извлечение текста последнего ответа агента.
"""

from __future__ import annotations

import json
from typing import Any

from deep_agent_test import build_analytics_deep_agent, load_deep_agent_settings
from deep_agent_test.core.trace_logging import FileTraceCallbackHandler, build_trace_file_path
from deep_agent_test.tools.fake_spark_data import build_fake_spark_data_tools
from model import model

#USER_MESSAGE = "что делал клиент в день сработки и за день до сработки? id сработки 3486d84b-4eba-4ba4-b044-94764fc9e7a4"
USER_MESSAGE = "найди все сработки связанные с образованием за январь 2026"
#USER_MESSAGE = "построй график распределения количества сработок по age category за январь 2026"

TOOL_STATUS_LABELS = {
    "write_todos": "Составляю план",
    "load_skills": "Читаю skills",
    "task": "Запускаю subagent",
    "load_data": "Читаю данные",
    "execute_python_code": "Анализирую данные",
    "write_file": "Сохраняю файл",
    "edit_file": "Обновляю файл",
}


def main() -> int:
    """Запускает один запрос к агенту.

    Args:
        Отсутствуют. Скрипт не принимает параметры командной строки.

    Returns:
        Код завершения процесса: ``0`` при успешном invoke.
    """

    settings = load_deep_agent_settings()
    data_tools = build_fake_spark_data_tools(query_parser_model=model)
    agent = build_analytics_deep_agent(model=model, settings=settings, data_tools=data_tools)
    trace_file_path = build_trace_file_path(settings.trace_log_dir)
    trace_handler = FileTraceCallbackHandler(trace_file_path)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": USER_MESSAGE}]},
        config={
            "callbacks": [trace_handler],
            "configurable": {"thread_id": settings.thread_id},
            "recursion_limit": settings.graph_recursion_limit,
        },
    )
    print(_last_message_text(result))
    print(f"Trace log: {trace_file_path}")
    return 0


def main_stream() -> int:
    """Запускает один запрос к агенту и печатает промежуточные шаги.

    Args:
        Отсутствуют. Скрипт не принимает параметры командной строки.

    Returns:
        Код завершения процесса: ``0`` при успешном stream-запуске.
    """

    settings = load_deep_agent_settings()
    data_tools = build_fake_spark_data_tools(query_parser_model=model)
    agent = build_analytics_deep_agent(model=model, settings=settings, data_tools=data_tools)
    trace_file_path = build_trace_file_path(settings.trace_log_dir)
    trace_handler = FileTraceCallbackHandler(trace_file_path)
    config = {
        "callbacks": [trace_handler],
        "configurable": {"thread_id": settings.thread_id},
        "recursion_limit": settings.graph_recursion_limit,
    }

    print("Запускаю агента...")
    final_result = None
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": USER_MESSAGE}]},
        config=config,
        version="v3",
    )
    _print_v3_progress(stream)
    final_result = stream.output

    if final_result is not None:
        print("\nИтоговый ответ:")
        print(_last_message_text(final_result))
    print(f"Trace log: {trace_file_path}")
    return 0


def _print_v3_progress(stream: Any, *, prefix: str = "") -> None:
    """Печатает промежуточные шаги из typed stream-проекций v3.

    Args:
        stream: ``GraphRunStream`` или ``SubagentRunStream`` из ``stream_events(version="v3")``.
        prefix: Префикс для вложенных subagent-сообщений.

    Returns:
        ``None``. Функция печатает статусы по мере прихода событий.
    """

    for channel_name, item in stream.interleave("tool_calls", "subagents", "lifecycle"):
        if channel_name == "tool_calls":
            _print_tool_call_progress(item, prefix=prefix)
        elif channel_name == "subagents":
            status = _subagent_status(item)
            print(f"{prefix}{status}")
            _print_v3_progress(item, prefix=f"{prefix}[{item.name or 'subagent'}] ")
        elif channel_name == "lifecycle":
            if item.get("event") == "started" and item.get("graph_name"):
                print(f"{prefix}Запускаю graph: {item['graph_name']}")


def _print_tool_call_progress(tool_call: Any, *, prefix: str = "") -> None:
    """Печатает старт и завершение одного вызова инструмента.

    Args:
        tool_call: ``ToolCallStream`` из v3-проекции ``tool_calls``.
        prefix: Префикс для вложенных subagent-сообщений.

    Returns:
        ``None``. Функция печатает статусы и дожидается завершения tool call.
    """

    tool_name = str(getattr(tool_call, "tool_name", "") or "")
    if tool_name == "task":
        return
    print(f"{prefix}{_tool_call_status(tool_name, getattr(tool_call, 'input', None))}")
    for _ in tool_call.output_deltas:
        pass
    label = TOOL_STATUS_LABELS.get(tool_name, f"Инструмент {tool_name}")
    if getattr(tool_call, "error", None):
        print(f"{prefix}{label}: ошибка - {tool_call.error}")
    else:
        print(f"{prefix}{label}: завершено")


def _subagent_status(subagent: Any) -> str:
    """Формирует статус запуска subagent-а.

    Args:
        subagent: ``SubagentRunStream`` из v3-проекции ``subagents``.

    Returns:
        Текст статуса с именем subagent-а и кратким описанием задачи.
    """

    name = getattr(subagent, "name", None) or "subagent"
    task_input = _compact_args_preview(getattr(subagent, "task_input", None))
    return f"Запускаю subagent {name}: {task_input}" if task_input else f"Запускаю subagent {name}"


def _tool_call_status(tool_name: str, args: Any) -> str:
    """Формирует статус старта инструмента.

    Args:
        tool_name: Имя инструмента из события LangChain.
        args: Аргументы вызова инструмента.

    Returns:
        Строка статуса с компактным preview аргументов.
    """

    label = TOOL_STATUS_LABELS.get(tool_name, f"Вызываю инструмент {tool_name}")
    preview = _compact_args_preview(args)
    return f"{label}: {preview}" if preview else label


def _compact_args_preview(args: Any, *, max_chars: int = 300) -> str:
    """Возвращает короткое JSON-представление аргументов инструмента.

    Args:
        args: Любые аргументы tool call.
        max_chars: Максимальная длина строки preview.

    Returns:
        Компактная строка без переносов или пустая строка, если аргументов нет.
    """

    if args in (None, "", {}, []):
        return ""
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except TypeError:
        text = str(args)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def _last_message_text(result: Any) -> str:
    """Достает текст последнего сообщения агента из результата invoke.

    Args:
        result: Словарь состояния, который вернул ``agent.invoke``.

    Returns:
        Текст последнего сообщения или строковое представление результата.
    """

    if not isinstance(result, dict):
        return str(result)
    messages = result.get("messages") or []
    if not messages:
        return str(result)
    last_message = messages[-1]
    text = getattr(last_message, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(last_message, "content", None)
    if isinstance(content, str):
        return content
    return str(last_message)


if __name__ == "__main__":
    raise SystemExit(main_stream())

"""Терминальный чат для проверки native DeepAgents аналитического агента."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool
from langgraph.errors import GraphRecursionError

from deep_agent_test.analytics_deep_agent import build_analytics_deep_agent
from deep_agent_test.settings import DeepAgentSettings, load_deep_agent_settings
from deep_agent_test.trace_logging import build_file_trace_handler

EXIT_COMMANDS = {"exit", "quit", "q", "выход", "стоп"}
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
TOOL_ARGS_PREVIEW_CHARS = 2500
TOOL_RESULT_PREVIEW_CHARS = 3500
#DEFAULT_DEMO_QUERY = "Какой город по IP у сработки 3486d84b-4eba-4ba4-b044-94764fc9e7a4?"
#DEFAULT_DEMO_QUERY = "найди все сработки связанные с образованием за январь 2026 года"
#DEFAULT_DEMO_QUERY = "создай новый файл для записи skill, назови папку test-skill"
#DEFAULT_DEMO_QUERY = "средняя сумма транзакции у сработок с правилом 'DENY оплата обучения после смены устройства'. Если тебе удастся получить ответ, то сохрани правильную последовательность действий в виде skill"
#DEFAULT_DEMO_QUERY = "построй распределение сработок по age category в виде графика за январь 2026, сохрани в файл"
DEFAULT_DEMO_QUERY = "что делал клиент в день сработки и за день до сработки? id сработки 3486d84b-4eba-4ba4-b044-94764fc9e7a4"

def build_chat_agent(settings: DeepAgentSettings | None = None, data_tools: list[BaseTool] | None = None) -> Any:
    """Собирает аналитический DeepAgent на модели из корневого ``model.py``."""

    from model import model as openrouter_model

    return build_analytics_deep_agent(
        openrouter_model,
        settings=settings,
        data_tools=data_tools,
    )


def build_test_data_tools(data_dir: Path = TEST_DATA_DIR) -> list[BaseTool]:
    """Собирает demo-инструмент ``read_table`` поверх тестовых CSV из ``data_dir``."""

    from examples.fake_spark_tools import build_fake_spark_tools

    raw_tool = build_fake_spark_tools(delay_seconds=0.0, data_dir=data_dir)[0]

    def read_table_sync(**kwargs: Any) -> Any:
        """Синхронно вызывает базовый tool и нормализует результат."""

        return _normalize_read_table_output(asyncio.run(raw_tool.ainvoke(kwargs)))

    async def read_table_async(**kwargs: Any) -> Any:
        """Асинхронно вызывает базовый tool и нормализует результат."""

        return _normalize_read_table_output(await raw_tool.ainvoke(kwargs))

    return [
        StructuredTool.from_function(
            func=read_table_sync,
            coroutine=read_table_async,
            name=raw_tool.name,
            description=raw_tool.description,
            args_schema=raw_tool.args_schema,
        )
    ]


def _normalize_read_table_output(value: Any) -> Any:
    """Возвращает сырой результат read_table как есть (DataFrame со всеми строками).

    Инструмент НЕ должен сам форматировать вывод или считать строки: за код запроса,
    счётчики (всего в таблице / подошло под фильтры / возвращено) и offload больших
    результатов отвечает обёртка ``wrap_data_tools_with_query_code``. Поэтому здесь мы
    отдаём DataFrame целиком, лишь нормализуя идентификаторы к строкам (чтобы не терять
    точность длинных ключей) и сохраняя ``attrs`` со счётчиками для обёртки.
    """

    if not hasattr(value, "to_dict") or not hasattr(value, "columns"):
        return value

    attrs = dict(getattr(value, "attrs", {}) or {})
    for column in value.columns:
        value[column] = [_format_read_table_cell(column, item) for item in value[column].tolist()]
    value.attrs = attrs
    return value


def _format_read_table_cell(column: str, value: Any) -> Any:
    """Нормализует ячейку: идентификаторы — в строку, скаляры numpy — в Python-типы."""

    if value is None or value != value:
        return None
    normalized_column = column.lower()
    if (
        normalized_column.endswith("_id")
        or normalized_column in {"event_id", "epk_id", "user_id", "event_dt", "event_time", "operation_id"}
        or "transaction_id" in normalized_column
    ):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def make_config(
    thread_id: str,
    callbacks: list[Any] | None = None,
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """Собирает config для ``invoke``: thread_id, callbacks и recursion_limit."""

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if callbacks:
        config["callbacks"] = callbacks
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    return config


def run_chat(settings: DeepAgentSettings | None = None, data_tools: list[BaseTool] | None = None) -> None:
    """Запускает интерактивный терминальный чат с агентом до команды выхода."""

    settings = settings or load_deep_agent_settings()
    agent = build_chat_agent(settings=settings, data_tools=data_tools)
    trace_handler = build_file_trace_handler(settings.trace_log_dir, label="chat")
    config = make_config(
        settings.thread_id,
        callbacks=[trace_handler],
        recursion_limit=settings.graph_recursion_limit,
    )
    print(f"Лог хода агента пишется в: {trace_handler.file_path}")
    loaded_skills_printed = False
    message_cursor = _initial_message_cursor(agent, config)

    print("Native Analytics DeepAgent. Команды выхода: exit, quit, q.")
    while True:
        user_message = input("Вы: ").strip()
        if user_message.lower() in EXIT_COMMANDS:
            return
        if not user_message:
            continue

        print("Агент: обрабатываю запрос...", flush=True)
        try:
            result = invoke_user_message(agent, config, user_message)
        except Exception as error:
            print()
            print(f"Ошибка выполнения агента: {error}")
            print()
            continue

        state = resolve_agent_state(agent, config, result)
        loaded_skills_printed = print_loaded_skills_once(state, already_printed=loaded_skills_printed)
        message_cursor = print_messages(state, start_index=message_cursor)


def invoke_user_message(agent: Any, config: dict[str, Any], message: str) -> Any:
    """Вызывает агента на сообщении; при исчерпании бюджета графа отдаёт частичный прогресс."""

    try:
        return agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    except GraphRecursionError:
        # Бюджет графа исчерпан: не роняем прогон, а отдаём то, что уже накоплено в state.
        # Частичный прогресс печатается вызывающим кодом через resolve_agent_state.
        print()
        print(
            "Достигнут лимит шагов графа (recursion_limit): агент не успел сформировать "
            "финальный ответ. Ниже — частичный прогресс. Уточните запрос или увеличьте "
            "graph_recursion_limit в конфиге."
        )
        print()
        return None


def print_loaded_skills_once(result: Any, *, already_printed: bool) -> bool:
    """Печатает список предзагруженных skills один раз за сессию."""

    if already_printed or not isinstance(result, dict):
        return already_printed

    skill_paths = result.get("preloaded_skill_paths") or []
    if not skill_paths:
        return already_printed

    print()
    print("Агент:")
    print("Загруженные skills:")
    for index, skill_path in enumerate(skill_paths, start=1):
        print(f"{index}. {skill_path}")
    print()
    return True


def print_messages(state: Any, *, start_index: int = 0) -> int:
    """Печатает новые сообщения из state начиная с ``start_index``."""

    if not isinstance(state, dict):
        return start_index

    messages = state.get("messages") or []
    if not isinstance(messages, list) or start_index >= len(messages):
        return len(messages)

    printed = False
    for message in messages[start_index:]:
        if isinstance(message, HumanMessage):
            continue

        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            for tool_call in tool_calls:
                if not printed:
                    print()
                printed = True
                print(format_tool_call(str(tool_call.get("name") or "tool"), tool_call.get("args")))

            text = message_to_text(message).strip()
            if text:
                if not printed:
                    print()
                printed = True
                print("Агент:")
                print(text)
            continue

        if type(message).__name__ == "ToolMessage":
            if not printed:
                print()
            printed = True
            status = str(getattr(message, "status", "") or "success")
            print(
                format_tool_result(
                    str(getattr(message, "name", "") or "tool"),
                    message_to_text(message),
                    status=status,
                )
            )

    if printed:
        print()
    return len(messages)


def resolve_agent_state(agent: Any, config: dict[str, Any], result: Any) -> Any:
    """Возвращает актуальный state агента из чекпойнтера, иначе — исходный результат."""

    get_state = getattr(agent, "get_state", None)
    if get_state is None:
        return result

    try:
        snapshot = get_state(config)
    except Exception:
        return result

    values = getattr(snapshot, "values", None)
    if isinstance(values, dict):
        return values
    if isinstance(snapshot, dict):
        snapshot_values = snapshot.get("values")
        if isinstance(snapshot_values, dict):
            return snapshot_values
    return result


def last_agent_response_text(result: Any) -> str:
    """Возвращает текст последнего финального ответа агента (AIMessage без tool_calls)."""

    if not isinstance(result, dict):
        return ""

    messages = result.get("messages") or []
    if not messages:
        return ""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = message_to_text(message).strip()
        if text:
            return text
    return ""


def format_todos_for_user(todos: list[dict[str, Any]]) -> str:
    """Форматирует список todo в нумерованный план с человекочитаемыми статусами."""

    if not todos:
        return "План не указан."

    status_labels = {
        "pending": "ожидает",
        "in_progress": "в работе",
        "completed": "готово",
    }
    lines = []
    for index, todo in enumerate(todos, start=1):
        content = str(todo.get("content") or "").strip() or "Без описания"
        status = status_labels.get(str(todo.get("status") or ""), str(todo.get("status") or ""))
        suffix = f" [{status}]" if status else ""
        lines.append(f"{index}. {content}{suffix}")
    return "\n".join(lines)


def message_to_text(message: Any) -> str:
    """Приводит content сообщения (строку, список блоков или объект) к тексту."""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(block) for block in content)
    if content is not None:
        return str(content)
    return str(message)


def _message_count(state: Any) -> int:
    """Возвращает число сообщений в state (0, если поле отсутствует/не список)."""

    if not isinstance(state, dict):
        return 0
    messages = state.get("messages")
    return len(messages) if isinstance(messages, list) else 0


def _initial_message_cursor(agent: Any, config: dict[str, Any]) -> int:
    """Возвращает стартовый курсор сообщений, чтобы печатать только новые."""

    get_state = getattr(agent, "get_state", None)
    if get_state is None:
        return 0
    try:
        snapshot = get_state(config)
    except Exception:
        return 0
    values = getattr(snapshot, "values", None)
    if isinstance(values, dict):
        return _message_count(values)
    return 0


def format_tool_call(tool_name: str, args: Any) -> str:
    """Форматирует вызов инструмента для вывода в терминал (план/subagent/аргументы)."""

    lines = [f"[Tool call] {tool_name}"]
    if tool_name == "write_todos" and isinstance(args, dict):
        lines.append("План:")
        lines.append(format_todos_for_user(args.get("todos", [])))
        return "\n".join(lines)

    if tool_name == "task" and isinstance(args, dict):
        description = str(args.get("description") or args.get("task") or "").strip()
        subagent = str(args.get("subagent_type") or args.get("name") or "").strip()
        if subagent:
            lines.append(f"Subagent: {subagent}")
        if description:
            lines.append("Задание:")
            lines.append(_indent_text(description, prefix="  "))
        return "\n".join(lines)

    lines.append("Аргументы:")
    lines.append(_indent_text(_format_json_preview(args), prefix="  "))
    return "\n".join(lines)


def format_tool_result(tool_name: str, content: str, *, status: str = "success") -> str:
    """Форматирует результат инструмента для терминала: заголовок + краткое содержание."""

    lines = [f"[Tool result] {tool_name} [{status}]"]
    lines.append(_summarize_tool_result(tool_name, content))
    return "\n".join(lines)


def _summarize_tool_result(tool_name: str, content: str) -> str:
    """Готовит компактное резюме результата tool под его тип (read_table/python/spill)."""

    text = str(content or "").strip()
    if not text:
        return "  (пустой результат)"

    parsed = _try_parse_json(text)
    if parsed is None:
        return _indent_text(_truncate_text(text, TOOL_RESULT_PREVIEW_CHARS), prefix="  ")

    if tool_name == "read_table" or _looks_like_read_table_payload(parsed):
        return _indent_text(_format_read_table_result_summary(parsed), prefix="  ")

    if tool_name == "execute_python_code" or (
        isinstance(parsed, dict) and "success" in parsed and "generated_code" in parsed
    ):
        return _indent_text(_format_execute_python_result_summary(parsed), prefix="  ")

    if tool_name == "write_todos" and isinstance(parsed, dict):
        return _indent_text(format_todos_for_user(parsed.get("todos", [])), prefix="  ")

    if isinstance(parsed, dict) and parsed.get("format") == "pkl":
        return _indent_text(
            "\n".join(
                [
                    f"Сохранен pickle: {parsed.get('saved_file', parsed.get('file_path', ''))}",
                    f"Строк: {parsed.get('rows', '?')}",
                ]
            ),
            prefix="  ",
        )

    if "saved_file" in parsed or "pickle" in text.lower():
        return _indent_text(_format_spill_file_summary(parsed, text), prefix="  ")

    return _indent_text(_format_json_preview(parsed), prefix="  ")


def _format_read_table_result_summary(payload: Any) -> str:
    """Резюмирует ответ read_table: статус, таблица, число строк, колонки, preview."""

    if not isinstance(payload, dict):
        return _truncate_text(str(payload), TOOL_RESULT_PREVIEW_CHARS)

    lines = [
        f"status: {payload.get('status', 'unknown')}",
        f"table_name: {payload.get('table_name', '')}",
        f"returned_rows: {payload.get('returned_rows', payload.get('matched_rows', payload.get('rows_count', '?')))}",
    ]
    columns = payload.get("columns")
    if isinstance(columns, list) and columns:
        lines.append(f"columns: {', '.join(map(str, columns[:20]))}")

    rows = payload.get("rows")
    if isinstance(rows, list) and rows:
        preview = rows[:3]
        lines.append("preview:")
        lines.append(_indent_text(_format_json_preview(preview), prefix="    "))

    if payload.get("schema"):
        lines.append("schema: присутствует")

    missing = payload.get("missing_columns") or payload.get("unknown_columns")
    if missing:
        lines.append(f"missing_columns: {missing}")

    limitations = payload.get("limitations")
    if limitations:
        lines.append(f"limitations: {limitations}")

    return "\n".join(lines)


def _format_execute_python_result_summary(payload: dict[str, Any]) -> str:
    """Резюмирует ответ execute_python_code: успех, сообщение, вывод/ошибка/traceback."""

    lines = [
        f"success: {payload.get('success')}",
        f"message: {payload.get('message', '')}",
    ]
    if payload.get("target_variable"):
        lines.append(f"target_variable: {payload.get('target_variable')}")
    if payload.get("error"):
        lines.append(f"error: {payload.get('error')}")
    if payload.get("traceback"):
        lines.append("traceback:")
        lines.append(_indent_text(_truncate_text(str(payload.get("traceback")), 1500), prefix="    "))
    if payload.get("execution_output"):
        lines.append("execution_output:")
        lines.append(
            _indent_text(_truncate_text(str(payload.get("execution_output")), 1200), prefix="    ")
        )
    if payload.get("variable_preview"):
        lines.append("variable_preview:")
        lines.append(
            _indent_text(_truncate_text(str(payload.get("variable_preview")), 1200), prefix="    ")
        )
    if payload.get("possible_causes"):
        lines.append(f"possible_causes: {payload.get('possible_causes')}")
    return "\n".join(lines)


def _format_spill_file_summary(parsed: dict[str, Any], raw_text: str) -> str:
    """Резюмирует offload-результат: путь к файлу, формат и число строк."""

    lines: list[str] = []
    for key in ("saved_file", "file_path", "format", "rows"):
        if key in parsed:
            lines.append(f"{key}: {parsed.get(key)}")
    if not lines:
        for line in raw_text.splitlines():
            if "Файл:" in line or "Путь:" in line or "Preview" in line:
                lines.append(line.strip())
    return "\n".join(lines) or _truncate_text(raw_text, TOOL_RESULT_PREVIEW_CHARS)


def _looks_like_read_table_payload(payload: Any) -> bool:
    """Эвристически определяет, что JSON-ответ похож на результат read_table."""

    return isinstance(payload, dict) and (
        "table_name" in payload or "returned_rows" in payload or "rows" in payload
    )


def _try_parse_json(text: str) -> Any | None:
    """Пробует распарсить текст как JSON; возвращает ``None`` при неудаче."""

    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _format_json_preview(value: Any) -> str:
    """Сериализует значение в JSON с обрезкой до лимита превью аргументов."""

    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        rendered = repr(value)
    return _truncate_text(rendered, TOOL_ARGS_PREVIEW_CHARS)


def _truncate_text(text: str, max_chars: int) -> str:
    """Обрезает текст до ``max_chars`` с пометкой о числе отброшенных символов."""

    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...[truncated {len(text) - max_chars} chars]"


def _indent_text(text: str, *, prefix: str) -> str:
    """Добавляет отступ ``prefix`` к каждой строке текста."""

    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines())


def main() -> int:
    """Делает один demo-``invoke`` на тестовых CSV и печатает ход выполнения."""

    settings = load_deep_agent_settings()
    test_data_tools = build_test_data_tools(TEST_DATA_DIR)
    agent = build_chat_agent(settings=settings, data_tools=test_data_tools)
    trace_handler = build_file_trace_handler(settings.trace_log_dir, label="demo")
    config = make_config(
        settings.thread_id,
        callbacks=[trace_handler],
        recursion_limit=settings.graph_recursion_limit,
    )

    user_message = DEFAULT_DEMO_QUERY
    print(f"Запрос: {user_message}", flush=True)
    print(f"Лог хода агента: {trace_handler.file_path}", flush=True)
    print("Агент: обрабатываю запрос...", flush=True)

    result = invoke_user_message(agent, config, user_message)
    state = resolve_agent_state(agent, config, result)
    print_loaded_skills_once(state, already_printed=False)
    print_messages(state, start_index=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Человекочитаемое логирование хода агента в один txt-файл.

Назначение: записать в один текстовый файл, в правильной последовательности и без
обрезаний, всё, что важно для отладки агента:
- system prompt, который поступает модели (с учётом preload skills);
- набор инструментов (имя, описание, схема аргументов);
- ответ агента (текст + вызовы инструментов с аргументами);
- ответ инструмента (content).

Метаданные (run_id, usage, response_metadata, id сообщений и т.п.) намеренно НЕ пишутся —
только аргументы, content и ответы.

Механизм: ``FileTraceCallbackHandler`` — это обычный LangChain callback handler. Он
передаётся в ``config={"callbacks": [handler]}`` при ``invoke``/``ainvoke`` и автоматически
получает события всех вложенных вызовов — supervisor, subagent-ов, critic и инструментов,
поэтому последовательность в файле отражает реальный порядок выполнения.

Чтобы файл оставался читаемым, повторяющийся без изменений system prompt и неизменный набор
инструментов не дублируются целиком — вместо этого пишется короткая отметка «без изменений».
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class FileTraceCallbackHandler(BaseCallbackHandler):
    """Пишет ход выполнения агента в один txt-файл в читаемом виде.

    Args:
        file_path: Путь к txt-файлу лога. Папка создаётся автоматически.
    """

    raise_error = False

    def __init__(self, file_path: str | Path) -> None:
        """Открывает файл лога, создаёт папку и пишет заголовок начала трассировки."""

        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index = 0
        self._last_system_prompt: str | None = None
        self._last_tools_signature: str | None = None
        self._logged_human_messages: set[str] = set()
        self._tool_names: dict[str, str] = {}
        self._raw_write(f"=== TRACE START {datetime.now().isoformat(timespec='seconds')} ===\n\n")

    # -- LangChain callback hooks -------------------------------------------------

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        """Логирует набор инструментов, system prompt и новые сообщения пользователя."""

        try:
            node = _node_label(kwargs)
            self._log_tools(serialized, kwargs, node=node)
            prompt_messages = messages[0] if messages else []
            self._log_system_prompt(prompt_messages, node=node)
            self._log_new_human_messages(prompt_messages, node=node)
        except Exception:  # noqa: BLE001 - логирование не должно ломать агента.
            pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Логирует ответ модели: текст и вызовы инструментов с аргументами."""

        try:
            node = _node_label(kwargs)
            for message in _iter_ai_messages(response):
                self._log_agent_response(message, node=node)
        except Exception:  # noqa: BLE001
            pass

    def on_tool_start(self, serialized: Any, input_str: Any, **kwargs: Any) -> None:
        """Логирует вызов инструмента: имя и аргументы."""

        try:
            run_id = str(kwargs.get("run_id") or "")
            name = _tool_name(serialized, kwargs)
            if run_id:
                self._tool_names[run_id] = name
            args = _tool_input_args(input_str, kwargs)
            body = f"name: {name}\narguments:\n{_indent(_to_text(args))}"
            self._section("TOOL CALL", body, node=_node_label(kwargs))
        except Exception:  # noqa: BLE001
            pass

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Логирует ответ инструмента: только content."""

        try:
            run_id = str(kwargs.get("run_id") or "")
            name = self._tool_names.pop(run_id, None) or _tool_message_name(output) or "tool"
            content = _tool_output_content(output)
            self._section("TOOL RESULT", f"name: {name}\ncontent:\n{_indent(content)}", node=_node_label(kwargs))
        except Exception:  # noqa: BLE001
            pass

    # -- internal helpers ---------------------------------------------------------

    def _log_tools(self, serialized: Any, kwargs: dict[str, Any], *, node: str | None) -> None:
        """Логирует набор инструментов; неизменный набор помечает «без изменений»."""

        tool_defs = _extract_tool_defs(serialized, kwargs)
        if not tool_defs:
            return
        signature = json.dumps([definition.get("name") for definition in tool_defs], ensure_ascii=False)
        if signature == self._last_tools_signature:
            self._section("TOOLS", "(без изменений, см. выше)", node=node)
            return
        self._last_tools_signature = signature
        blocks: list[str] = []
        for definition in tool_defs:
            name = definition.get("name", "")
            description = definition.get("description", "")
            parameters = definition.get("parameters")
            block = f"- {name}"
            if description:
                block += f"\n  описание: {description}"
            if parameters is not None:
                block += "\n  схема аргументов:\n" + _indent(_to_text(parameters), prefix="    ")
            blocks.append(block)
        self._section(f"TOOLS ({len(tool_defs)})", "\n".join(blocks), node=node)

    def _log_system_prompt(self, prompt_messages: Any, *, node: str | None) -> None:
        """Логирует system prompt; неизменный prompt помечает «без изменений»."""

        system_text = _collect_system_text(prompt_messages)
        if not system_text:
            return
        if system_text == self._last_system_prompt:
            self._section("SYSTEM PROMPT", "(без изменений, см. выше)", node=node)
            return
        self._last_system_prompt = system_text
        self._section("SYSTEM PROMPT", system_text, node=node)

    def _log_new_human_messages(self, prompt_messages: Any, *, node: str | None) -> None:
        """Логирует ещё не записанные сообщения пользователя (без повторов)."""

        for message in prompt_messages or []:
            if _message_type(message) != "human":
                continue
            text = _message_text(message).strip()
            if not text or text in self._logged_human_messages:
                continue
            self._logged_human_messages.add(text)
            self._section("USER MESSAGE", text, node=node)

    def _log_agent_response(self, message: Any, *, node: str | None) -> None:
        """Логирует ответ модели: текст и вызовы инструментов с аргументами."""

        parts: list[str] = []
        text = _message_text(message).strip()
        if text:
            parts.append(f"content:\n{_indent(text)}")
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            call_lines = ["tool_calls:"]
            for tool_call in tool_calls:
                name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                call_lines.append(f"  - {name}(arguments=")
                call_lines.append(_indent(_to_text(args), prefix="      "))
                call_lines.append("  )")
            parts.append("\n".join(call_lines))
        if not parts:
            return
        self._section("AGENT RESPONSE", "\n".join(parts), node=node)

    def _section(self, title: str, body: str, *, node: str | None = None) -> None:
        """Пишет нумерованную секцию лога с заголовком, телом и разделителем."""

        self._index += 1
        header = f"[{self._index}] {title}"
        if node:
            header += f"  (node: {node})"
        self._raw_write(f"{header}\n{body}\n{'-' * 80}\n")

    def _raw_write(self, text: str) -> None:
        """Потокобезопасно дописывает текст в файл лога."""

        with self._lock:
            with self.file_path.open("a", encoding="utf-8") as handle:
                handle.write(text)


def build_file_trace_handler(trace_log_dir: str | Path, *, label: str = "trace") -> FileTraceCallbackHandler:
    """Создаёт handler с уникальным по времени именем файла внутри ``trace_log_dir``."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
    file_path = Path(trace_log_dir) / f"{timestamp}_{safe_label}.txt"
    return FileTraceCallbackHandler(file_path)


# -- module-level extraction helpers ----------------------------------------------


def _node_label(kwargs: dict[str, Any]) -> str | None:
    """Возвращает имя узла LangGraph из metadata callback (если есть)."""

    metadata = kwargs.get("metadata") or {}
    if isinstance(metadata, dict):
        node = metadata.get("langgraph_node")
        if node:
            return str(node)
    return None


def _extract_tool_defs(serialized: Any, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Извлекает определения инструментов из invocation_params или serialized."""

    raw_tools = None
    invocation_params = kwargs.get("invocation_params")
    if isinstance(invocation_params, dict):
        raw_tools = invocation_params.get("tools")
    if not raw_tools and isinstance(serialized, dict):
        raw_tools = (serialized.get("kwargs") or {}).get("tools")
    if not isinstance(raw_tools, list):
        return []
    definitions: list[dict[str, Any]] = []
    for item in raw_tools:
        definitions.append(_normalize_tool_def(item))
    return [definition for definition in definitions if definition.get("name")]


def _normalize_tool_def(item: Any) -> dict[str, Any]:
    """Нормализует определение инструмента (OpenAI/Anthropic-формат) к единому виду."""

    if not isinstance(item, dict):
        return {"name": str(getattr(item, "name", "") or "")}
    # OpenAI-формат: {"type": "function", "function": {name, description, parameters}}
    if isinstance(item.get("function"), dict):
        function = item["function"]
        return {
            "name": function.get("name", ""),
            "description": function.get("description", ""),
            "parameters": function.get("parameters"),
        }
    # Anthropic-формат: {name, description, input_schema}
    return {
        "name": item.get("name", ""),
        "description": item.get("description", ""),
        "parameters": item.get("parameters") if "parameters" in item else item.get("input_schema"),
    }


def _collect_system_text(prompt_messages: Any) -> str:
    """Склеивает текст всех system-сообщений из набора сообщений запроса."""

    chunks = [
        _message_text(message)
        for message in (prompt_messages or [])
        if _message_type(message) == "system"
    ]
    return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def _iter_ai_messages(response: Any) -> list[Any]:
    """Собирает сообщения модели из всех generations ответа LLM."""

    messages: list[Any] = []
    generations = getattr(response, "generations", None) or []
    for generation_list in generations:
        for generation in generation_list or []:
            message = getattr(generation, "message", None)
            if message is not None:
                messages.append(message)
    return messages


def _message_type(message: Any) -> str:
    """Определяет тип сообщения (system/human/ai/tool) по полю type или имени класса."""

    message_type = getattr(message, "type", None)
    if message_type:
        return str(message_type)
    name = type(message).__name__.lower()
    if "system" in name:
        return "system"
    if "human" in name:
        return "human"
    if "ai" in name:
        return "ai"
    if "tool" in name:
        return "tool"
    return name


def _message_text(message: Any) -> str:
    """Возвращает текстовое содержимое сообщения (из объекта или dict)."""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return _content_to_text(content)


def _content_to_text(content: Any) -> str:
    """Приводит content (строку, список блоков или объект) к плоскому тексту."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _tool_name(serialized: Any, kwargs: dict[str, Any]) -> str:
    """Определяет имя инструмента из serialized или kwargs (fallback ``tool``)."""

    if isinstance(serialized, dict) and serialized.get("name"):
        return str(serialized["name"])
    name = kwargs.get("name")
    return str(name) if name else "tool"


def _tool_input_args(input_str: Any, kwargs: dict[str, Any]) -> Any:
    """Извлекает аргументы вызова инструмента из kwargs/input_str (с JSON-парсингом)."""

    inputs = kwargs.get("inputs")
    if isinstance(inputs, dict):
        return inputs
    if isinstance(input_str, dict):
        return input_str
    if isinstance(input_str, str):
        try:
            return json.loads(input_str)
        except (ValueError, TypeError):
            return input_str
    return input_str


def _tool_output_content(output: Any) -> str:
    """Извлекает content из результата инструмента (объект ToolMessage/dict/значение)."""

    content = getattr(output, "content", None)
    if content is not None:
        return _content_to_text(content)
    if isinstance(output, dict) and "content" in output:
        return _content_to_text(output["content"])
    return _content_to_text(output)


def _tool_message_name(output: Any) -> str | None:
    """Возвращает имя инструмента из объекта результата или ``None``."""

    name = getattr(output, "name", None)
    return str(name) if name else None


def _to_text(value: Any) -> str:
    """Сериализует значение в читаемый текст (строка как есть, иначе JSON)."""

    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        return str(value)


def _indent(text: str, *, prefix: str = "  ") -> str:
    """Добавляет отступ ``prefix`` к каждой строке текста."""

    return "\n".join(f"{prefix}{line}" if line else prefix.rstrip() for line in str(text).splitlines())


__all__ = ["FileTraceCallbackHandler", "build_file_trace_handler"]

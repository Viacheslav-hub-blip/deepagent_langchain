"""Логирование шагов DeepAgent в человекочитаемый txt-файл.

Содержит:
- FileTraceCallbackHandler: LangChain callback handler для записи prompt, ответов модели,
  tool calls и tool results без служебных metadata.
- build_trace_file_path: построение пути к trace-файлу запуска.
- _message_content: извлечение текстового содержимого сообщения.
- _message_role: определение роли сообщения для trace-лога.
- _estimate_token_count: грубая оценка числа токенов по длине текста.
- _extract_tool_specs: нормализация описаний tools из параметров вызова модели.
- _format_json: безопасное форматирование структур в JSON.
- _first_generation_message: извлечение первого сообщения из результата LLM.
- _tool_result_name: извлечение имени tool из результата.
- _tool_result_content: извлечение содержимого результата tool.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import LLMResult


class FileTraceCallbackHandler(BaseCallbackHandler):
    """Пишет ход работы DeepAgent в txt-файл.

    Args:
        file_path: Путь к файлу, в который нужно писать trace.

    Returns:
        Объект callback handler для передачи в ``config={"callbacks": [...]}``
        при вызове LangChain/LangGraph runnable.
    """

    def __init__(self, file_path: str | Path) -> None:
        """Инициализирует handler и очищает файл trace.

        Args:
            file_path: Абсолютный или относительный путь к txt-файлу trace.

        Returns:
            ``None``. Состояние handler хранит путь файла и последние сигнатуры
            system prompt/tools для компактной записи повторных шагов.
        """

        super().__init__()
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text("", encoding="utf-8")
        self._tool_names_by_run_id: dict[str, str] = {}
        self._model_call_index = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Логирует вход в chat model: tools, system prompt и пользовательские сообщения.

        Args:
            serialized: Описание модели от LangChain.
            messages: Батчи сообщений, переданные в модель.
            invocation_params: Параметры вызова модели; из них берется только список tools.
            **kwargs: Дополнительные callback-параметры LangChain, которые в trace не пишутся.

        Returns:
            ``None``. Метод дописывает секции в trace-файл.
        """

        del serialized, kwargs
        self._model_call_index += 1
        request_index = self._model_call_index
        batch = messages[0] if messages else []
        tools = _extract_tool_specs((invocation_params or {}).get("tools"))
        self._append_llm_request_header(request_index=request_index, messages=batch, tools=tools)
        self._append_tools(tools, request_index=request_index)
        self._append_prompt_messages(batch, request_index=request_index)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Логирует ответ модели и запрошенные ею вызовы инструментов.

        Args:
            response: Результат LLM-вызова LangChain.
            **kwargs: Служебные параметры callback. В trace не попадают.

        Returns:
            ``None``. Метод записывает только content и имена/аргументы tool calls.
        """

        del kwargs
        message = _first_generation_message(response)
        if message is None:
            return

        content = _message_content(message)
        self._append_section("AGENT RESPONSE", content or "(пустой ответ)")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        inputs: dict[str, Any] | None = None,
        run_id: Any = None,
        **kwargs: Any,
    ) -> None:
        """Логирует фактический старт tool-вызова.

        Args:
            serialized: Описание tool от LangChain; используется только поле ``name``.
            input_str: Строковый ввод tool, если ``inputs`` не передан.
            inputs: Структурированные аргументы tool.
            run_id: Идентификатор запуска tool для внутреннего сопоставления результата.
            **kwargs: Служебные параметры callback. В trace не попадают.

        Returns:
            ``None``. Метод дописывает имя tool и его входные аргументы.
        """

        del kwargs
        tool_name = str(serialized.get("name") or "(unknown)")
        if run_id is not None:
            self._tool_names_by_run_id[str(run_id)] = tool_name
        payload: Any = inputs if inputs is not None else input_str
        self._append_section("TOOL CALL", "\n".join([f"name: {tool_name}", "args:", _format_json(payload)]))

    def on_tool_end(self, output: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        """Логирует результат tool-вызова без служебных идентификаторов.

        Args:
            output: Результат tool. Для ``ToolMessage`` берется только ``content`` и ``name``.
            run_id: Идентификатор запуска tool для внутреннего поиска имени.
            **kwargs: Служебные параметры callback. В trace не попадают.

        Returns:
            ``None``. Метод дописывает секцию с результатом tool.
        """

        del kwargs
        tool_name = _tool_result_name(output) or self._tool_names_by_run_id.get(str(run_id), "(unknown)")
        content = _tool_result_content(output)
        self._append_section("TOOL RESULT", "\n".join([f"name: {tool_name}", "content:", content]))

    def _append_llm_request_header(
        self,
        *,
        request_index: int,
        messages: list[BaseMessage],
        tools: list[dict[str, Any]],
    ) -> None:
        """Записывает явную границу одного запроса к LLM и сводку объёма контекста.

        Args:
            request_index: Порядковый номер model call внутри текущего trace.
            messages: Полный список сообщений, переданный модели.
            tools: Нормализованный список tools, доступных модели в этом вызове.

        Returns:
            ``None``. Метод добавляет в trace заголовок запроса и таблицу размеров.
        """

        message_rows: list[str] = []
        message_chars_total = 0
        message_tokens_total = 0
        for index, message in enumerate(messages, start=1):
            content = _message_content(message)
            chars = len(content)
            tokens = _estimate_token_count(content)
            message_chars_total += chars
            message_tokens_total += tokens
            tool_calls = len(getattr(message, "tool_calls", []) or [])
            message_rows.append(
                " | ".join(
                    [
                        str(index),
                        _message_role(message),
                        message.__class__.__name__,
                        str(chars),
                        str(tokens),
                        str(tool_calls),
                        str(getattr(message, "name", "") or ""),
                    ]
                )
            )

        tools_text = _format_json(tools)
        tool_chars = len(tools_text)
        tool_tokens = _estimate_token_count(tools_text)
        body = "\n".join(
            [
                f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
                f"messages_count: {len(messages)}",
                f"tools_count: {len(tools)}",
                f"messages_chars: {message_chars_total}",
                f"messages_tokens_estimate: {message_tokens_total}",
                f"tools_chars: {tool_chars}",
                f"tools_tokens_estimate: {tool_tokens}",
                f"total_tokens_estimate: {message_tokens_total + tool_tokens}",
                "",
                "messages_table:",
                "index | role | class | chars | tokens_est | tool_calls | name",
                *message_rows,
            ]
        )
        self._append_section(f"LLM REQUEST #{request_index}", body)

    def _append_prompt_messages(self, messages: list[BaseMessage], *, request_index: int) -> None:
        """Записывает все сообщения, которые вошли в конкретный запрос к модели.

        Args:
            messages: Список сообщений, переданный модели.
            request_index: Порядковый номер model call внутри текущего trace.

        Returns:
            ``None``. Каждый model call логируется самодостаточно, без скрытия повторных
            system prompt и tool-сообщений.
        """

        for index, message in enumerate(messages, start=1):
            content = _message_content(message)
            header_lines = [
                f"request: {request_index}",
                f"index: {index}",
                f"role: {_message_role(message)}",
                f"class: {message.__class__.__name__}",
                f"chars: {len(content)}",
                f"tokens_estimate: {_estimate_token_count(content)}",
            ]
            name = str(getattr(message, "name", "") or "")
            if name:
                header_lines.append(f"name: {name}")
            tool_call_id = str(getattr(message, "tool_call_id", "") or "")
            if tool_call_id:
                header_lines.append(f"tool_call_id: {tool_call_id}")
            tool_calls = getattr(message, "tool_calls", []) or []
            if tool_calls:
                header_lines.extend(["tool_calls:", _format_json(tool_calls)])
            header_lines.extend(["content:", content or "(пустое сообщение)"])
            self._append_section(f"LLM REQUEST #{request_index} MESSAGE #{index}", "\n".join(header_lines))

    def _append_tools(self, tools: list[dict[str, Any]], *, request_index: int) -> None:
        """Записывает tools, доступные модели на текущем шаге.

        Args:
            tools: Нормализованный список описаний tools с именем, описанием и схемой.
            request_index: Порядковый номер model call внутри текущего trace.

        Returns:
            ``None``. Секция всегда пишет полный набор tools, потому что tools тоже
            входят в конкретный запрос к LLM.
        """

        title = f"LLM REQUEST #{request_index} TOOLS ({len(tools)})"
        lines: list[str] = []
        for index, tool in enumerate(tools, start=1):
            tool_text = _format_json(tool)
            lines.extend(
                [
                    f"{index}. {tool.get('name') or '(unknown)'}",
                    f"chars: {len(tool_text)}",
                    f"tokens_estimate: {_estimate_token_count(tool_text)}",
                    f"описание: {tool.get('description') or ''}",
                    "схема аргументов:",
                    _format_json(tool.get("parameters") or {}),
                    "",
                ]
            )
        self._append_section(title, "\n".join(lines).rstrip() or "(tools не переданы)")

    def _append_section(self, title: str, body: str) -> None:
        """Дописывает одну секцию в trace-файл.

        Args:
            title: Название секции.
            body: Текст секции.

        Returns:
            ``None``. Секция добавляется в конец файла UTF-8.
        """

        with self.file_path.open("a", encoding="utf-8") as file:
            file.write(f"\n\n===== {title} =====\n")
            file.write(str(body).rstrip())
            file.write("\n")


def build_trace_file_path(trace_log_dir: str | Path, *, prefix: str = "deep_agent_trace") -> Path:
    """Строит уникальный путь txt trace-файла в папке логов.

    Args:
        trace_log_dir: Папка для trace-файлов.
        prefix: Префикс имени файла без расширения.

    Returns:
        Путь вида ``<trace_log_dir>/<prefix>_<timestamp>.txt``.
    """

    log_dir = Path(trace_log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{prefix}_{timestamp}.txt"


def _message_content(message: BaseMessage) -> str:
    """Извлекает текст из LangChain message.

    Args:
        message: Сообщение LangChain.

    Returns:
        Текстовое содержимое сообщения. Списковые content-блоки сериализуются в JSON.
    """

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return _format_json(content)


def _message_role(message: BaseMessage) -> str:
    """Возвращает человекочитаемую роль LangChain-сообщения для trace-лога.

    Args:
        message: Сообщение LangChain.

    Returns:
        Роль сообщения: ``system``, ``user``, ``assistant``, ``tool`` или значение
        поля ``type``, если сообщение нестандартное.
    """

    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, ToolMessage):
        return "tool"
    message_type = str(getattr(message, "type", "") or "").strip()
    if message_type:
        if message_type == "ai":
            return "assistant"
        return message_type
    return message.__class__.__name__


def _estimate_token_count(text: str) -> int:
    """Грубо оценивает число токенов по длине текста.

    Args:
        text: Текст prompt-сообщения, tool schema или ответа.

    Returns:
        Приблизительное число токенов. Формула не заменяет tokenizer провайдера,
        но даёт стабильную оценку объёма контекста прямо в trace-файле.
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _extract_tool_specs(raw_tools: Any) -> list[dict[str, Any]]:
    """Нормализует описания tools из параметров вызова модели.

    Args:
        raw_tools: Значение ``invocation_params["tools"]`` от LangChain.

    Returns:
        Список словарей с ключами ``name``, ``description`` и ``parameters``.
    """

    if not isinstance(raw_tools, list):
        return []
    tools: list[dict[str, Any]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        function = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
        tools.append(
            {
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters") or function.get("args_schema") or {},
            }
        )
    return tools


def _format_json(value: Any) -> str:
    """Форматирует значение в JSON без падения на нестандартных объектах.

    Args:
        value: Любое значение Python.

    Returns:
        Строка JSON с ``ensure_ascii=False`` или строковое представление объекта.
    """

    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)


def _first_generation_message(response: LLMResult) -> BaseMessage | None:
    """Возвращает первое message из результата LLM.

    Args:
        response: Результат LangChain LLM-вызова.

    Returns:
        Первое сообщение генерации или ``None``, если структура пуста.
    """

    if not response.generations or not response.generations[0]:
        return None
    generation = response.generations[0][0]
    return getattr(generation, "message", None)


def _tool_result_name(output: Any) -> str:
    """Извлекает имя tool из результата.

    Args:
        output: Результат tool callback.

    Returns:
        Имя tool или пустую строку, если оно недоступно.
    """

    if isinstance(output, ToolMessage):
        return str(output.name or "")
    return str(getattr(output, "name", "") or "")


def _tool_result_content(output: Any) -> str:
    """Извлекает содержимое результата tool без metadata.

    Args:
        output: Результат tool callback.

    Returns:
        Текст результата tool.
    """

    if isinstance(output, ToolMessage):
        return _message_content(output)
    if isinstance(output, BaseMessage):
        return _message_content(output)
    if isinstance(output, (dict, list)):
        return _format_json(output)
    return str(output)


__all__ = ["FileTraceCallbackHandler", "build_trace_file_path"]

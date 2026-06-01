"""Логирование шагов DeepAgent в человекочитаемый txt-файл.

Содержит:
- FileTraceCallbackHandler: LangChain callback handler для записи prompt, ответов модели,
  tool calls и tool results без служебных metadata.
- build_trace_file_path: построение пути к trace-файлу запуска.
- _message_content: извлечение текстового содержимого сообщения.
- _extract_tool_specs: нормализация описаний tools из параметров вызова модели.
- _format_json: безопасное форматирование структур в JSON.
- _normalize_tool_signature: компактная сигнатура набора tools для дедупликации.
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
        self._last_system_prompt: str | None = None
        self._last_tools_signature: str | None = None
        self._tool_names_by_run_id: dict[str, str] = {}

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
        batch = messages[0] if messages else []
        tools = _extract_tool_specs((invocation_params or {}).get("tools"))
        self._append_section("MODEL STEP", datetime.now().isoformat(timespec="seconds"))
        self._append_tools(tools)
        self._append_prompt_messages(batch)

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
        for tool_call in getattr(message, "tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            self._append_section(
                "TOOL CALL",
                "\n".join(
                    [
                        f"name: {tool_call.get('name') or '(unknown)'}",
                        "args:",
                        _format_json(tool_call.get("args") or {}),
                    ]
                ),
            )

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

    def _append_prompt_messages(self, messages: list[BaseMessage]) -> None:
        """Записывает system/user сообщения из входа модели.

        Args:
            messages: Список сообщений, переданный модели.

        Returns:
            ``None``. Повторный неизменный system prompt заменяется короткой ссылкой
            на предыдущую запись.
        """

        system_prompt = "\n\n".join(
            _message_content(message) for message in messages if isinstance(message, SystemMessage)
        )
        if system_prompt:
            if system_prompt == self._last_system_prompt:
                self._append_section("SYSTEM PROMPT", "(без изменений, см. выше)")
            else:
                self._last_system_prompt = system_prompt
                self._append_section("SYSTEM PROMPT", system_prompt)

        for message in messages:
            if isinstance(message, HumanMessage):
                self._append_section("USER MESSAGE", _message_content(message))

    def _append_tools(self, tools: list[dict[str, Any]]) -> None:
        """Записывает tools, доступные модели на текущем шаге.

        Args:
            tools: Нормализованный список описаний tools с именем, описанием и схемой.

        Returns:
            ``None``. Если набор tools не изменился по именам, пишет компактную
            ссылку на предыдущую секцию.
        """

        signature = _normalize_tool_signature(tools)
        title = f"TOOLS ({len(tools)})"
        if signature and signature == self._last_tools_signature:
            self._append_section(title, "(без изменений, см. выше)")
            return

        self._last_tools_signature = signature
        lines: list[str] = []
        for index, tool in enumerate(tools, start=1):
            lines.extend(
                [
                    f"{index}. {tool.get('name') or '(unknown)'}",
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


def _normalize_tool_signature(tools: list[dict[str, Any]]) -> str:
    """Создает компактную сигнатуру набора tools для дедупликации.

    Args:
        tools: Нормализованный список tools.

    Returns:
        JSON-строка с отсортированными именами tools.
    """

    names = sorted(str(tool.get("name") or "") for tool in tools)
    return json.dumps(names, ensure_ascii=False)


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

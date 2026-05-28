"""Middleware сохранения больших табличных результатов tool-вызовов в CSV.

Содержит:
- ToolOutputFileMiddleware: middleware для замены больших табличных результатов ссылкой на файл.
- wrap_tool_call: синхронная обработка результата tool call.
- awrap_tool_call: асинхронная обработка результата tool call.
- _process_tool_message: сохранение табличного artifact в CSV и формирование компактного ToolMessage.
- _extract_tabular_payload: извлечение табличных строк из artifact или content.
- _write_rows_to_csv: запись строк в CSV-файл.
- _build_file_summary: формирование текста, который увидит модель вместо большого результата.
- _safe_filename_part: подготовка безопасного фрагмента имени файла.
- _row_to_mapping: преобразование строки результата в CSV-совместимый словарь.
"""

from __future__ import annotations

import ast
import csv
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deep_agent_test.agent_logging import DeepAgentEventLogger


@dataclass(frozen=True)
class ToolOutputFileMiddleware(AgentMiddleware):
    """Сохраняет большие табличные tool outputs в CSV и возвращает модели краткую ссылку.

    Args:
        output_dir: Папка для CSV-файлов с результатами инструментов.
        min_rows_to_save: Минимальное количество строк, при котором результат сохраняется в файл.
        min_content_chars_to_save: Минимальная длина текстового content, при которой tabular-result сохраняется в файл.
        preview_rows: Количество строк preview, которое остается в сообщении tool.
        inline_original_content_chars: Максимальная длина исходного content, который можно дублировать в summary.
        event_logger: Файловый логгер для записи факта сохранения большого результата.

    Returns:
        Middleware LangChain, который не меняет поведение tool, но заменяет большие
        табличные ответы компактным описанием файла.
    """

    output_dir: Path
    min_rows_to_save: int = 25
    min_content_chars_to_save: int = 4000
    preview_rows: int = 2
    inline_original_content_chars: int = 1000
    event_logger: DeepAgentEventLogger | None = None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Обрабатывает синхронный tool call и сохраняет большой результат в CSV.

        Args:
            request: Запрос на выполнение tool.
            handler: Функция фактического выполнения tool.

        Returns:
            Исходный Command или ToolMessage с компактным описанием сохраненного CSV.
        """

        result = handler(request)
        if isinstance(result, ToolMessage):
            return self._process_tool_message(result=result, tool_name=str(request.tool_call.get("name") or "tool"))
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Обрабатывает асинхронный tool call и сохраняет большой результат в CSV.

        Args:
            request: Запрос на асинхронное выполнение tool.
            handler: Асинхронная функция фактического выполнения tool.

        Returns:
            Исходный Command или ToolMessage с компактным описанием сохраненного CSV.
        """

        result = await handler(request)
        if isinstance(result, ToolMessage):
            return self._process_tool_message(result=result, tool_name=str(request.tool_call.get("name") or "tool"))
        return result

    def _process_tool_message(self, *, result: ToolMessage, tool_name: str) -> ToolMessage:
        """Сохраняет табличный результат ToolMessage в CSV, если он достаточно большой.

        Args:
            result: Сообщение с результатом выполнения инструмента.
            tool_name: Имя инструмента для имени файла и текста summary.

        Returns:
            Исходное ToolMessage или новое ToolMessage с кратким описанием CSV-файла.
        """

        rows = _extract_tabular_payload(result.artifact, result.content)
        content_text = str(result.content)
        if not rows:
            return result
        if len(rows) < self.min_rows_to_save and len(content_text) < self.min_content_chars_to_save:
            return result

        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_path = _write_rows_to_csv(
            rows=rows,
            output_dir=self.output_dir,
            tool_name=tool_name,
        )
        if self.event_logger is not None:
            self.event_logger.log_tool_event(
                "tool_output_saved",
                {
                    "tool_name": tool_name,
                    "file_path": str(file_path),
                    "rows": len(rows),
                    "columns": sorted({key for row in rows for key in row}),
                },
            )
        summary = _build_file_summary(
            tool_name=tool_name,
            file_path=file_path,
            rows=rows,
            preview_rows=self.preview_rows,
            original_content=content_text,
            inline_original_content_chars=self.inline_original_content_chars,
        )
        artifact = {
            "saved_file": str(file_path),
            "format": "csv",
            "rows": len(rows),
            "columns": sorted({key for row in rows for key in row}),
            "source_artifact_type": type(result.artifact).__name__,
        }
        return ToolMessage(
            content=summary,
            artifact=artifact,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            additional_kwargs=result.additional_kwargs,
            response_metadata=result.response_metadata,
        )


def _extract_tabular_payload(artifact: Any, content: Any) -> list[dict[str, Any]]:
    """Извлекает табличные строки из artifact или текстового content.

    Args:
        artifact: Artifact ToolMessage, если tool вернул структурированный результат.
        content: Текстовое содержимое ToolMessage.

    Returns:
        Список строк в формате ``list[dict]`` или пустой список.
    """

    rows = _extract_rows_from_value(artifact)
    if rows:
        return rows
    if not isinstance(content, str):
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(content)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        rows = _extract_rows_from_value(parsed)
        if rows:
            return rows
    return []


def _extract_rows_from_value(value: Any) -> list[dict[str, Any]]:
    """Извлекает строки ``list[dict]`` из распространенных структур результата.

    Args:
        value: Произвольное значение artifact или parsed content.

    Returns:
        Список строк в формате ``list[dict]`` или пустой список.
    """

    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            records = value.to_dict(orient="records")
        except TypeError:
            records = None
        if isinstance(records, list):
            return [_row_to_mapping(item) for item in records]
    if isinstance(value, list):
        return [_row_to_mapping(item) for item in value]
    if isinstance(value, dict):
        for key in ("rows", "records", "data", "result"):
            rows = _extract_rows_from_value(value.get(key))
            if rows:
                return rows
    return []


def _write_rows_to_csv(*, rows: list[dict[str, Any]], output_dir: Path, tool_name: str) -> Path:
    """Записывает табличные строки в CSV-файл.

    Args:
        rows: Строки таблицы в формате списка словарей.
        output_dir: Папка для сохранения CSV.
        tool_name: Имя инструмента для формирования имени файла.

    Returns:
        Абсолютный путь к созданному CSV-файлу.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = output_dir / f"{timestamp}_{_safe_filename_part(tool_name)}.csv"
    columns = sorted({key for row in rows for key in row})
    with file_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return file_path.resolve()


def _build_file_summary(
    *,
    tool_name: str,
    file_path: Path,
    rows: list[dict[str, Any]],
    preview_rows: int,
    original_content: str,
    inline_original_content_chars: int,
) -> str:
    """Формирует компактное описание сохраненного результата для модели.

    Args:
        tool_name: Имя инструмента.
        file_path: Абсолютный путь к CSV-файлу.
        rows: Табличные строки результата.
        preview_rows: Количество строк preview.
        original_content: Исходное текстовое содержимое tool output.
        inline_original_content_chars: Максимальная длина исходного content для дублирования в summary.

    Returns:
        Текст ToolMessage с путем к файлу, размером таблицы и preview.
    """

    preview = rows[: max(0, preview_rows)]
    columns = sorted({key for row in rows for key in row})
    preview_text = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
    original_note = ""
    if original_content and len(original_content) <= inline_original_content_chars:
        original_note = f"\n\nКраткий исходный вывод tool:\n{original_content}"
    return (
        f"Tool `{tool_name}` вернул большой табличный результат и он сохранен в CSV.\n"
        f"Файл: {file_path.resolve()}\n"
        f"Формат: csv; кодировка: utf-8-sig.\n"
        f"Строк: {len(rows)}; колонок: {len(columns)}.\n"
        f"Колонки: {', '.join(map(str, columns))}.\n"
        "Для дальнейшего анализа читай файл как DataFrame, например: "
        f"pd.read_csv(r\"{file_path.resolve()}\").\n"
        f"Preview первых {len(preview)} строк:\n{preview_text}"
        f"{original_note}"
    )


def _safe_filename_part(value: str) -> str:
    """Преобразует строку в безопасный фрагмент имени файла.

    Args:
        value: Исходная строка.

    Returns:
        Безопасный фрагмент имени файла из ASCII-символов.
    """

    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe[:80] or "tool"


def _row_to_mapping(value: Any) -> dict[str, Any]:
    """Преобразует строку табличного результата в CSV-совместимый словарь.

    Args:
        value: Строка результата в произвольном формате.

    Returns:
        Исходный словарь или словарь с полем ``value`` для скалярных элементов.
    """

    if isinstance(value, dict):
        return value
    return {"value": value}


__all__ = ["ToolOutputFileMiddleware"]

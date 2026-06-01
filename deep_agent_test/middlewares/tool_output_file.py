"""Middleware сохранения больших табличных результатов tool-вызовов в pickle."""

from __future__ import annotations

import ast
import json
import pickle
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@dataclass(frozen=True)
class ToolOutputFileMiddleware(AgentMiddleware):
    """Сохраняет табличные tool outputs в pickle и управляет тем, что попадает в контекст.

    Полный набор строк каждого табличного результата (например ``load_data``) всегда
    пишется в ``.pkl`` — для переиспользования без повторного Spark-запроса.

    В контекст модели:
    - маленький результат — исходный content без замены + пометка с путём к pkl;
    - большой результат — summary с preview (пороги ``min_rows_to_save`` /
      ``min_content_chars_to_save``), как раньше.
    """

    output_dir: Path
    min_rows_to_save: int = 10
    min_content_chars_to_save: int = 60000
    preview_rows: int = 3
    inline_original_content_chars: int = 1000

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Синхронно выполняет tool и офлоадит большой табличный результат в pickle."""

        result = handler(request)
        if isinstance(result, ToolMessage):
            return self._process_tool_message(result=result, tool_name=str(request.tool_call.get("name") or "tool"))
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Асинхронно выполняет tool и офлоадит большой табличный результат в pickle."""

        result = await handler(request)
        if isinstance(result, ToolMessage):
            return self._process_tool_message(result=result, tool_name=str(request.tool_call.get("name") or "tool"))
        return result

    def _process_tool_message(self, *, result: ToolMessage, tool_name: str) -> ToolMessage:
        """Всегда сохраняет табличные строки в pkl; в контекст — inline или summary по порогам."""

        rows = _extract_tabular_payload(result.artifact, result.content)
        if not rows:
            return result

        content_text = str(result.content)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        file_path = _write_rows_to_pkl(
            rows=rows,
            output_dir=self.output_dir,
            tool_name=tool_name,
        )
        virtual_path = _virtual_tool_output_path(file_path=file_path, output_dir=self.output_dir)
        query_metadata = _extract_query_metadata(result.artifact)
        artifact = {
            "saved_file": str(file_path),
            "virtual_file": virtual_path,
            "format": "pkl",
            "rows": len(rows),
            "columns": sorted({key for row in rows for key in row}),
            "source_artifact_type": type(result.artifact).__name__,
        }
        is_large = (
            len(rows) > self.min_rows_to_save
            or len(content_text) > self.min_content_chars_to_save
        )
        if is_large:
            content = _build_file_summary(
                tool_name=tool_name,
                file_path=file_path,
                virtual_path=virtual_path,
                rows=rows,
                preview_rows=self.preview_rows,
                original_content=content_text,
                inline_original_content_chars=self.inline_original_content_chars,
                query_metadata=query_metadata,
            )
        else:
            content = content_text + _build_inline_saved_file_note(
                file_path=file_path,
                virtual_path=virtual_path,
                rows=rows,
                query_metadata=query_metadata,
            )

        return ToolMessage(
            content=content,
            artifact=artifact,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            additional_kwargs=result.additional_kwargs,
            response_metadata=result.response_metadata,
        )


def _extract_tabular_payload(artifact: Any, content: Any) -> list[dict[str, Any]]:
    """Извлекает табличные строки из artifact или из content (JSON/literal-парсинг)."""

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
    """Приводит DataFrame/list/dict-обёртку к списку строк-словарей (или пустой список)."""

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


def _write_rows_to_pkl(*, rows: list[dict[str, Any]], output_dir: Path, tool_name: str) -> Path:
    """Пишет строки в pickle-файл с уникальным по времени именем и возвращает путь."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = output_dir / f"{timestamp}_{_safe_filename_part(tool_name)}.pkl"
    with file_path.open("wb") as file:
        pickle.dump(rows, file)
    return file_path.resolve()


def _virtual_tool_output_path(*, file_path: Path, output_dir: Path) -> str:
    """Строит виртуальный путь `/tool_outputs/...` для файловых инструментов DeepAgents.

    Args:
        file_path: Абсолютный путь к сохранённому pickle-файлу.
        output_dir: Локальная папка, смонтированная в backend как `/tool_outputs/`.

    Returns:
        Виртуальный путь, который можно передавать в filesystem tools (`read_file`, `ls`,
        `glob`) внутри DeepAgents.
    """

    try:
        relative_path = file_path.resolve().relative_to(output_dir.resolve()).as_posix()
    except ValueError:
        relative_path = file_path.name
    return f"/tool_outputs/{relative_path}"


def _build_file_summary(
    *,
    tool_name: str,
    file_path: Path,
    virtual_path: str,
    rows: list[dict[str, Any]],
    preview_rows: int,
    original_content: str,
    inline_original_content_chars: int,
    query_metadata: dict[str, Any] | None = None,
) -> str:
    """Строит текст summary офлоада: путь к pkl, объём, колонки, preview и как дочитать."""

    preview = rows[: max(0, preview_rows)]
    columns = sorted({key for row in rows for key in row})
    preview_text = json.dumps(preview, ensure_ascii=False, indent=2, default=str)
    original_note = ""
    if original_content and len(original_content) <= inline_original_content_chars:
        original_note = f"\n\nКраткий исходный вывод tool:\n{original_content}"
    query_note = _format_query_metadata(query_metadata)
    resolved_path = file_path.resolve()
    return (
        f"Tool `{tool_name}` вернул большой табличный результат, поэтому в контекст передан "
        f"только PREVIEW первых {len(preview)} строк, а ПОЛНЫЙ результат сохранён в файл.\n"
        f"{query_note}"
        f"ВАЖНО: всего в результате (в файле) {len(rows)} строк — это полный объём этого "
        f"запроса; в контексте сейчас лишь {len(preview)} строк для ознакомления.\n"
        f"Файл: {resolved_path.name}\n"
        f"saved_file: {resolved_path}\n"
        f"virtual_file: {virtual_path}\n"
        f"Формат: pickle (list[dict]).\n"
        f"Строк в файле: {len(rows)}; колонок: {len(columns)}.\n"
        f"Колонки: {', '.join(map(str, columns))}.\n"
        "Чтобы работать со ВСЕМИ строками или с урезанной выборкой из этого набора, используй "
        "`execute_python_code` (НЕ новый load_data). Helpers: read_pickle_file, "
        "describe_pickle_file, rows_to_dataframe, pd, np. Для Python используй `saved_file`; "
        "для filesystem tools (`read_file`, `ls`, `glob`) используй `virtual_file`. Пример:\n"
        f"rows = read_pickle_file(r\"{resolved_path}\")\n"
        "df = rows_to_dataframe(rows)\n"
        "При ошибке execute_python_code читай traceback из ответа tool и исправляй код.\n"
        f"Preview первых {len(preview)} строк:\n{preview_text}"
        f"{original_note}"
    )


def _extract_query_metadata(artifact: Any) -> dict[str, Any] | None:
    """Достаёт код запроса из artifact (если его положила обёртка tool).

    Счётчики исходной таблицы (всего в таблице / подошло под фильтры) намеренно НЕ
    переносятся: модели важно знать только объём самого результата (он указан в summary
    как число строк в файле) и что в контексте лишь preview.
    """

    if not isinstance(artifact, dict):
        return None
    keys = ("query_code", "is_aggregation")
    metadata = {key: artifact[key] for key in keys if key in artifact}
    return metadata or None


def _build_inline_saved_file_note(
    *,
    file_path: Path,
    virtual_path: str,
    rows: list[dict[str, Any]],
    query_metadata: dict[str, Any] | None = None,
) -> str:
    """Дополняет inline-ответ пометкой о pkl для переиспользования без повторного load_data."""

    columns = sorted({key for row in rows for key in row})
    resolved_path = file_path.resolve()
    query_note = _format_query_metadata(query_metadata)
    return (
        "\n\n[Полный результат сохранён в pickle для переиспользования без повторного load_data]\n"
        f"{query_note}"
        f"saved_file: {resolved_path}\n"
        f"virtual_file: {virtual_path}\n"
        f"Строк в файле: {len(rows)}; колонок: {len(columns)}.\n"
        f"Колонки: {', '.join(map(str, columns))}.\n"
        "Если следующий шаг — урезанная выборка из ЭТОГО же набора (другие фильтры, подмножество "
        "строк, агрегация, уникальные значения), НЕ запускай новый load_data: отфильтруй через "
        "`execute_python_code` (`read_pickle_file` → `rows_to_dataframe` / pandas).\n"
        f"Пример: rows = read_pickle_file(r\"{resolved_path}\")\n"
    )


def _format_query_metadata(query_metadata: dict[str, Any] | None) -> str:
    """Формирует блок с кодом запроса для summary офлоада (без счётчиков исходной таблицы)."""

    if not query_metadata:
        return ""
    query_code = query_metadata.get("query_code")
    if not query_code:
        return ""
    return f"Сгенерированный SQL-подобный запрос:\n{query_code}\n"


def _safe_filename_part(value: str) -> str:
    """Очищает строку до безопасной части имени файла (буквы/цифры/-/_, до 80 символов)."""

    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe[:80] or "tool"


def _row_to_mapping(value: Any) -> dict[str, Any]:
    """Приводит строку к словарю; не-словарь оборачивает в ``{"value": ...}``."""

    if isinstance(value, dict):
        return value
    return {"value": value}


__all__ = ["ToolOutputFileMiddleware"]

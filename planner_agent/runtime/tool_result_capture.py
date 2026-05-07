"""Перехват больших результатов LangChain tools и сохранение их как artifacts.

Содержит:
- CapturedToolResult: результат обработки tool output.
- capture_tool_result: основная функция перехвата результата инструмента.
- build_tool_trace_content: формирование безопасного tool trace без большого output.
- estimate_context_size: оценка размера результата в символах.
- format_artifact_reference: формат короткой ссылки для LLM.
- serialize_tool_result: сериализация результата tool.
- _inline_result_with_artifact: возврат маленького результата inline с artifact ref.
- _capture_existing_file: регистрация существующего файла как artifact.
- _capture_generated_artifact: сохранение большого результата как нового artifact.
- _looks_like_existing_file: проверка строки на путь к файлу.
- _result_kind_and_mime: определение типа artifact.
- _safe_filename_fragment: безопасный фрагмент имени файла.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from planner_agent.schemas.artifacts import Artifact
from planner_agent.services._json import to_jsonable
from planner_agent.services.artifact_service import ArtifactService

MAX_INLINE_TOOL_RESULT_CHARS = 10_000
TOOL_RESULT_PREVIEW_CHARS = 1_200
TOOL_TRACE_PREVIEW_CHARS = 2_000
TOOL_RESULT_SUMMARY_CHARS = 500


class CapturedToolResult(BaseModel):
    """Результат обработки вывода LangChain tool перед передачей в LLM."""

    content_for_llm: Any = Field(
        description=(
            "Значение, которое будет возвращено worker-агенту вместо сырого "
            "результата tool."
        ),
    )
    artifact_refs: list[str] = Field(
        default_factory=list,
        description="Идентификаторы artifacts, созданных из результата tool.",
    )
    artifact_index: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="JSON-совместимый индекс созданных artifacts.",
    )
    was_captured: bool = Field(
        default=False,
        description="Был ли результат заменен ссылкой на artifact.",
    )
    original_size_estimate: int = Field(
        default=0,
        description="Оценка размера исходного результата в символах.",
    )
    preview: str = Field(
        default="",
        description="Короткое preview результата без большого payload.",
    )
    result_kind: str = Field(
        default="tool_trace",
        description="Предполагаемый тип результата: dataset, source_excerpt или tool_trace.",
    )


def capture_tool_result(
        *,
        artifact_service: ArtifactService,
        run_id: str,
        node_id: str,
        task_id: str | None,
        tool_name: str,
        tool_input: Any,
        raw_result: Any,
        capture_id: str = "",
        max_inline_chars: int = MAX_INLINE_TOOL_RESULT_CHARS,
) -> CapturedToolResult:
    """Перехватывает результат tool и сохраняет большой output как artifact.

    Args:
        artifact_service: Сервис записи artifacts.
        run_id: Идентификатор ResearchRun.
        node_id: Идентификатор lineage node, внутри которого вызван tool.
        task_id: Идентификатор задачи worker.
        tool_name: Имя LangChain tool.
        tool_input: Аргументы вызова tool.
        raw_result: Сырой результат, возвращенный tool.
        capture_id: Опциональный идентификатор вызова для уникального имени artifact.
        max_inline_chars: Порог размера, выше которого результат сохраняется как artifact.

    Returns:
        CapturedToolResult с безопасным значением для LLM и artifact refs.
    """

    size_estimate = estimate_context_size(raw_result)
    preview = serialize_tool_result(raw_result, max_chars=TOOL_RESULT_PREVIEW_CHARS)
    existing_file = _capture_existing_file(
        artifact_service=artifact_service,
        run_id=run_id,
        node_id=node_id,
        task_id=task_id,
        tool_name=tool_name,
        tool_input=tool_input,
        raw_result=raw_result,
    )
    if existing_file is not None:
        return _captured_result_from_artifact(
            artifact=existing_file,
            raw_result=raw_result,
            preview=preview,
            size_estimate=size_estimate,
            reason="existing_file_reference",
        )

    if not _should_capture(raw_result, size_estimate, max_inline_chars):
        inline_artifact = _capture_inline_structured_result(
            artifact_service=artifact_service,
            run_id=run_id,
            node_id=node_id,
            task_id=task_id,
            tool_name=tool_name,
            tool_input=tool_input,
            raw_result=raw_result,
            preview=preview,
            size_estimate=size_estimate,
            capture_id=capture_id,
        )
        if inline_artifact is not None:
            return _inline_result_with_artifact(
                artifact=inline_artifact,
                raw_result=raw_result,
                preview=preview,
                size_estimate=size_estimate,
            )

        return CapturedToolResult(
            content_for_llm=raw_result,
            was_captured=False,
            original_size_estimate=size_estimate,
            preview=preview,
            result_kind=_result_kind_and_mime(raw_result)[0],
        )

    artifact = _capture_generated_artifact(
        artifact_service=artifact_service,
        run_id=run_id,
        node_id=node_id,
        task_id=task_id,
        tool_name=tool_name,
        tool_input=tool_input,
        raw_result=raw_result,
        preview=preview,
        size_estimate=size_estimate,
        capture_id=capture_id,
        capture_reason="context_budget_exceeded",
    )
    return _captured_result_from_artifact(
        artifact=artifact,
        raw_result=raw_result,
        preview=preview,
        size_estimate=size_estimate,
        reason="context_budget_exceeded",
    )


def build_tool_trace_content(
        *,
        tool_name: str,
        tool_input: Any,
        captured: CapturedToolResult,
) -> str:
    """Формирует tool trace без сохранения полного большого результата.

    Args:
        tool_name: Имя LangChain tool.
        tool_input: Аргументы вызова tool.
        captured: Результат обработки tool output.

    Returns:
        Текст tool trace с аргументами, preview, размером и artifact refs.
    """

    return (
        f"Tool: {tool_name}\n\n"
        f"Arguments:\n{serialize_tool_result(tool_input, max_chars=TOOL_TRACE_PREVIEW_CHARS)}"
        "\n\n"
        f"Captured: {captured.was_captured}\n"
        f"Original size estimate: {captured.original_size_estimate}\n"
        f"Artifact refs: {captured.artifact_refs}\n\n"
        f"Result preview:\n{captured.preview[:TOOL_TRACE_PREVIEW_CHARS]}"
    )


def estimate_context_size(value: Any) -> int:
    """Оценивает размер значения для передачи в контекст LLM.

    Args:
        value: Произвольное значение, возвращенное tool.

    Returns:
        Размер сериализованного значения в символах.
    """

    if isinstance(value, bytes):
        return len(value)
    return len(serialize_tool_result(value, max_chars=None))


def format_artifact_reference(
        *,
        artifact: Artifact,
        original_size_estimate: int,
        preview: str,
        raw_result: Any,
        reason: str,
) -> str:
    """Формирует короткую ссылку на artifact для передачи worker-у.

    Args:
        artifact: Artifact, созданный из результата tool.
        original_size_estimate: Оценка размера исходного результата.
        preview: Короткое preview результата.
        reason: Причина сохранения результата как artifact.

    Returns:
        Текстовое сообщение, которое можно безопасно передать в LLM context.
    """

    preview_is_truncated = "[truncated]" in preview
    preview_line = _single_line_preview(preview)
    result_meta = _build_result_metadata(raw_result)
    return (
        "Tool result was saved as an artifact because it is too large for the "
        "worker context.\n\n"
        "data_scope: partial_preview\n"
        "full_result_available_in_artifact: true\n"
        f"preview_is_truncated: {str(preview_is_truncated).lower()}\n"
        "worker_disclosure_required: true\n"
        f"reason: {reason}\n"
        f"artifact_id: {artifact.artifact_id}\n"
        f"kind: {artifact.kind}\n"
        f"uri: {artifact.uri}\n"
        f"mime_type: {artifact.mime_type}\n"
        f"original_size_estimate_chars: {original_size_estimate}\n"
        f"summary: {artifact.summary}\n\n"
        f"row_count: {result_meta['row_count']}\n"
        f"column_types: {serialize_tool_result(result_meta['column_types'], max_chars=2_000)}\n"
        f"has_nan_by_column: {serialize_tool_result(result_meta['has_nan_by_column'], max_chars=2_000)}\n\n"
        "important: The preview below is not the full tool result. "
        "Do not present conclusions based only on this preview as complete. "
        "Use artifact tools to inspect or profile the full artifact when "
        "full-data claims are required.\n\n"
        f"preview_line: {preview_line}"
    )


def serialize_tool_result(value: Any, *, max_chars: int | None = None) -> str:
    """Сериализует результат tool в текст для оценки размера или сохранения.

    Args:
        value: Произвольное значение tool output.
        max_chars: Опциональный лимит символов. ``None`` означает без лимита.

    Returns:
        Текстовое представление результата.
    """

    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes):
        text = f"<bytes length={len(value)}>"
    else:
        try:
            text = json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)
        except Exception:
            text = str(value)

    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated]"


def _should_capture(value: Any, size_estimate: int, max_inline_chars: int) -> bool:
    """Проверяет, нужно ли сохранять результат как artifact.

    Args:
        value: Сырой результат tool.
        size_estimate: Оценка размера результата.
        max_inline_chars: Максимальный inline-размер.

    Returns:
        ``True``, если результат нужно заменить artifact reference.
    """

    if isinstance(value, bytes):
        return True
    if _is_dataframe(value):
        return True
    return size_estimate > max_inline_chars


def _capture_existing_file(
        *,
        artifact_service: ArtifactService,
        run_id: str,
        node_id: str,
        task_id: str | None,
        tool_name: str,
        tool_input: Any,
        raw_result: Any,
) -> Artifact | None:
    """Регистрирует существующий файл, если tool вернул путь.

    Args:
        artifact_service: Сервис записи artifacts.
        run_id: Идентификатор ResearchRun.
        node_id: Идентификатор lineage node.
        task_id: Идентификатор задачи worker.
        tool_name: Имя tool.
        tool_input: Аргументы tool.
        raw_result: Сырой результат tool.

    Returns:
        Artifact для найденного файла или ``None``.
    """

    path = _looks_like_existing_file(raw_result)
    if path is None:
        return None

    kind, mime_type, _ = _result_kind_and_mime(raw_result, path=path)
    return artifact_service.register_file_artifact(
        run_id=run_id,
        node_id=node_id,
        kind=kind,
        path=path,
        mime_type=mime_type,
        summary=f"File returned by tool {tool_name}",
        metadata={
            "task_id": task_id,
            "tool_name": tool_name,
            "args_preview": serialize_tool_result(tool_input, max_chars=TOOL_RESULT_SUMMARY_CHARS),
            "artifact_role": "tool_returned_file",
            "capture_reason": "existing_file_reference",
            "reusable": True,
            "editable": True,
        },
    )


def _capture_generated_artifact(
        *,
        artifact_service: ArtifactService,
        run_id: str,
        node_id: str,
        task_id: str | None,
        tool_name: str,
        tool_input: Any,
        raw_result: Any,
        preview: str,
        size_estimate: int,
        capture_id: str,
        capture_reason: str,
) -> Artifact:
    """Сохраняет большой результат tool как новый artifact.

    Args:
        artifact_service: Сервис записи artifacts.
        run_id: Идентификатор ResearchRun.
        node_id: Идентификатор lineage node.
        task_id: Идентификатор задачи worker.
        tool_name: Имя tool.
        tool_input: Аргументы tool.
        raw_result: Сырой результат tool.
        preview: Короткое preview результата.
        size_estimate: Оценка размера результата.
        capture_id: Идентификатор вызова для уникального имени файла.
        capture_reason: Причина регистрации результата как artifact.

    Returns:
        Созданный Artifact.
    """

    kind, mime_type, extension = _result_kind_and_mime(raw_result)
    safe_tool_name = _safe_filename_fragment(tool_name)
    safe_task_id = _safe_filename_fragment(task_id or "unknown_task")
    safe_capture_id = _safe_filename_fragment(capture_id or "result")
    if isinstance(raw_result, bytes):
        content: str | bytes = raw_result
    elif _is_dataframe(raw_result):
        content = _dataframe_to_csv(raw_result)
        mime_type = "text/csv"
        extension = "csv"
        kind = "dataset"
    else:
        content = serialize_tool_result(raw_result, max_chars=None)

    return artifact_service.write_artifact(
        run_id=run_id,
        node_id=node_id,
        kind=kind,
        filename=(
            f"tasks/{safe_task_id}/tool_results/"
            f"{safe_tool_name}-{safe_capture_id}.{extension}"
        ),
        content=content,
        mime_type=mime_type,
        summary=preview[:TOOL_RESULT_SUMMARY_CHARS],
        metadata={
            "task_id": task_id,
            "tool_name": tool_name,
            "args_preview": serialize_tool_result(tool_input, max_chars=TOOL_RESULT_SUMMARY_CHARS),
            "artifact_role": "captured_tool_result",
            "capture_reason": capture_reason,
            "original_size_estimate": size_estimate,
            "max_inline_chars": MAX_INLINE_TOOL_RESULT_CHARS,
            "reusable": True,
            "editable": True,
        },
    )


def _capture_inline_structured_result(
        *,
        artifact_service: ArtifactService,
        run_id: str,
        node_id: str,
        task_id: str | None,
        tool_name: str,
        tool_input: Any,
        raw_result: Any,
        preview: str,
        size_estimate: int,
        capture_id: str,
) -> Artifact | None:
    """Регистрирует маленький структурированный tool result как reusable artifact.

    Args:
        artifact_service: Сервис записи artifacts.
        run_id: Идентификатор ResearchRun.
        node_id: Идентификатор lineage node.
        task_id: Идентификатор задачи worker.
        tool_name: Имя LangChain tool.
        tool_input: Аргументы вызова tool.
        raw_result: Исходный результат tool.
        preview: Preview результата.
        size_estimate: Оценка размера результата.
        capture_id: Идентификатор вызова для имени artifact.

    Returns:
        Artifact для структурированного результата или ``None`` для простых
        скалярных значений, которые не нужно хранить отдельно от tool trace.
    """

    if not _should_register_inline_artifact(raw_result):
        return None

    return _capture_generated_artifact(
        artifact_service=artifact_service,
        run_id=run_id,
        node_id=node_id,
        task_id=task_id,
        tool_name=tool_name,
        tool_input=tool_input,
        raw_result=raw_result,
        preview=preview,
        size_estimate=size_estimate,
        capture_id=capture_id,
        capture_reason="inline_structured_result",
    )


def _should_register_inline_artifact(value: Any) -> bool:
    """Проверяет, стоит ли сохранять маленький результат как artifact.

    Args:
        value: Результат LangChain tool.

    Returns:
        ``True`` для структурированных данных, которые могут понадобиться
        responder или UI: ``dict``, ``list`` и dataframe-like объекты.
    """

    return isinstance(value, (dict, list)) or _is_dataframe(value)


def _inline_result_with_artifact(
        *,
        artifact: Artifact,
        raw_result: Any,
        preview: str,
        size_estimate: int,
) -> CapturedToolResult:
    """Возвращает маленький tool result inline, сохранив его artifact ref.

    Args:
        artifact: Artifact с полным результатом tool.
        raw_result: Исходный результат, который можно передать worker inline.
        preview: Preview результата.
        size_estimate: Оценка размера результата.

    Returns:
        CapturedToolResult, где ``content_for_llm`` равен исходному результату,
        а artifact ref доступен для lineage/responder/UI.
    """

    return CapturedToolResult(
        content_for_llm=raw_result,
        artifact_refs=[artifact.artifact_id],
        artifact_index={artifact.artifact_id: artifact.model_dump(mode="json")},
        was_captured=False,
        original_size_estimate=size_estimate,
        preview=preview,
        result_kind=artifact.kind,
    )


def _captured_result_from_artifact(
        *,
        artifact: Artifact,
        raw_result: Any,
        preview: str,
        size_estimate: int,
        reason: str,
) -> CapturedToolResult:
    """Создает CapturedToolResult по зарегистрированному artifact.

    Args:
        artifact: Artifact результата.
        raw_result: Исходный результат tool.
        preview: Preview исходного результата.
        size_estimate: Оценка размера исходного результата.
        reason: Причина сохранения.

    Returns:
        CapturedToolResult со ссылкой вместо большого результата.
    """

    return CapturedToolResult(
        content_for_llm=format_artifact_reference(
            artifact=artifact,
            original_size_estimate=size_estimate,
            preview=preview,
            raw_result=raw_result,
            reason=reason,
        ),
        artifact_refs=[artifact.artifact_id],
        artifact_index={artifact.artifact_id: artifact.model_dump(mode="json")},
        was_captured=True,
        original_size_estimate=size_estimate,
        preview=preview,
        result_kind=_result_kind_and_mime(raw_result)[0],
    )


def _looks_like_existing_file(value: Any) -> Path | None:
    """Проверяет, является ли результат ссылкой на существующий файл.

    Args:
        value: Сырой результат tool.

    Returns:
        Path к файлу или ``None``.
    """

    if not isinstance(value, str):
        return None
    if "\n" in value or len(value) > 1_000:
        return None
    candidate = Path(value.strip().strip("\"'")).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()
    return None


def _result_kind_and_mime(
        value: Any,
        *,
        path: Path | None = None,
) -> tuple[str, str, str]:
    """Определяет kind, mime type и расширение artifact.

    Args:
        value: Сырой результат tool.
        path: Опциональный путь к существующему файлу.

    Returns:
        Кортеж ``(kind, mime_type, extension)``.
    """

    suffix = path.suffix.lower() if path else ""
    if suffix == ".csv":
        return "dataset", "text/csv", "csv"
    if suffix in {".json", ".jsonl"}:
        return "dataset", "application/json", suffix.lstrip(".")
    if suffix in {".parquet", ".pq"}:
        return "dataset", "application/vnd.apache.parquet", suffix.lstrip(".")
    if suffix in {".txt", ".log"}:
        return "source_excerpt", "text/plain", suffix.lstrip(".")
    if suffix == ".md":
        return "source_excerpt", "text/markdown", "md"

    if isinstance(value, bytes):
        return "dataset", "application/octet-stream", "bin"
    if _is_dataframe(value):
        return "dataset", "text/csv", "csv"
    if isinstance(value, list):
        return "dataset", "application/json", "json"
    if isinstance(value, dict):
        return "dataset", "application/json", "json"
    if isinstance(value, str):
        return "source_excerpt", "text/plain", "txt"
    return "tool_trace", "text/plain", "txt"


def _is_dataframe(value: Any) -> bool:
    """Проверяет, похож ли результат на pandas DataFrame.

    Args:
        value: Произвольное значение.

    Returns:
        ``True``, если объект похож на DataFrame.
    """

    return (
        hasattr(value, "to_csv")
        and hasattr(value, "shape")
        and hasattr(value, "columns")
    )


def _dataframe_to_csv(value: Any) -> str:
    """Сериализует DataFrame-подобный объект в CSV.

    Args:
        value: DataFrame-подобный объект.

    Returns:
        CSV-текст.
    """

    return value.to_csv(index=False)


def _safe_filename_fragment(value: str) -> str:
    """Преобразует строку в безопасный фрагмент имени файла.

    Args:
        value: Исходная строка.

    Returns:
        Безопасный фрагмент имени файла.
    """

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "unknown"


def _single_line_preview(preview: str) -> str:
    """Возвращает только первую непустую строку preview."""

    if not preview:
        return ""
    for line in preview.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _build_result_metadata(raw_result: Any) -> dict[str, Any]:
    """Строит компактные метаданные результата для prompt.

    Для табличных данных возвращает количество строк, типы колонок и наличие NaN.
    Для нетабличных данных оставляет безопасные дефолтные значения.
    """

    if _is_dataframe(raw_result):
        try:
            dtypes = {
                str(column): str(dtype)
                for column, dtype in raw_result.dtypes.items()
            }
            has_nan = {
                str(column): bool(raw_result[column].isna().any())
                for column in raw_result.columns
            }
            return {
                "row_count": int(getattr(raw_result, "shape", [0])[0]),
                "column_types": dtypes,
                "has_nan_by_column": has_nan,
            }
        except Exception:
            return {
                "row_count": 0,
                "column_types": {},
                "has_nan_by_column": {},
            }

    records = _to_records(raw_result)
    if not records:
        return {
            "row_count": 0,
            "column_types": {},
            "has_nan_by_column": {},
        }

    columns = sorted({str(key) for record in records for key in record.keys()})
    column_types: dict[str, str] = {}
    has_nan_by_column: dict[str, bool] = {}
    for column in columns:
        values = [record.get(column) for record in records]
        non_empty = [value for value in values if not _is_nan_like(value)]
        column_types[column] = _infer_column_type(non_empty)
        has_nan_by_column[column] = any(_is_nan_like(value) for value in values)
    return {
        "row_count": len(records),
        "column_types": column_types,
        "has_nan_by_column": has_nan_by_column,
    }


def _to_records(raw_result: Any) -> list[dict[str, Any]]:
    """Пытается привести результат инструмента к списку табличных записей."""

    if isinstance(raw_result, list):
        return [item for item in raw_result if isinstance(item, dict)]
    if isinstance(raw_result, dict):
        if all(not isinstance(value, (list, dict)) for value in raw_result.values()):
            return [raw_result]
        for value in raw_result.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    return []


def _infer_column_type(values: list[Any]) -> str:
    """Возвращает агрегированный тип колонки по непустым значениям."""

    if not values:
        return "unknown"
    kinds = {_value_kind(value) for value in values}
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


def _is_nan_like(value: Any) -> bool:
    """Определяет пустые/Nan-значения для унифицированной метрики."""

    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"", "nan", "null", "none", "na", "n/a"}
    if isinstance(value, float):
        # NaN != NaN в IEEE-754.
        return value != value
    return False


def _value_kind(value: Any) -> str:
    """Возвращает простой тип значения."""

    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


__all__ = [
    "CapturedToolResult",
    "MAX_INLINE_TOOL_RESULT_CHARS",
    "capture_tool_result",
    "build_tool_trace_content",
    "estimate_context_size",
    "format_artifact_reference",
    "serialize_tool_result",
]

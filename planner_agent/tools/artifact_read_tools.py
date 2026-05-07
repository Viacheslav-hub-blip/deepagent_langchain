"""LangChain tools для чтения artifacts без загрузки всего файла в контекст.

Содержит:
- ArtifactListInput: схема входа для artifact_list.
- ArtifactPreviewInput: схема входа для artifact_preview.
- ArtifactReadChunkInput: схема входа для artifact_read_chunk.
- ArtifactProfileInput: схема входа для artifact_profile.
- ArtifactSampleInput: схема входа для artifact_sample.
- ArtifactSearchInput: схема входа для artifact_search.
- ArtifactValueCountsInput: схема входа для artifact_value_counts.
- build_artifact_read_tools: фабрика LangChain tools для текущего ResearchRun.
- _artifact_to_reference: компактное описание artifact.
- _artifact_by_id: поиск artifact в текущем run.
- _load_tabular_records: загрузка CSV/JSON artifact как списка записей.
- _profile_records: расчет универсального профиля табличных записей.
- _sample_records: выборка записей по offset/limit.
- _search_records: поиск записей по текстовому запросу.
- _value_counts: расчет частот по одной или нескольким колонкам.
- _extract_json_records: извлечение списка записей из JSON payload.
- _record_matches_query: проверка совпадения записи с поисковым запросом.
- _read_text_chunk: чтение части текстового artifact.
- _read_text_preview: чтение preview текстового artifact.
- _is_text_mime_type: проверка текстового MIME-типа.
- _json_text: сериализация ответа tool в JSON.
- _clamp_int: ограничение числовых параметров.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from collections import Counter

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from planner_agent.schemas.artifacts import Artifact
from planner_agent.services.artifact_service import ArtifactService

DEFAULT_PREVIEW_CHARS = 2_000
DEFAULT_CHUNK_CHARS = 8_000
MAX_CHUNK_CHARS = 20_000
MAX_LIST_ITEMS = 50
MAX_PROFILE_COLUMNS = 200
MAX_PROFILE_TOP_VALUES = 20
MAX_SAMPLE_RECORDS = 100
MAX_SEARCH_RECORDS = 100
MAX_VALUE_COUNT_GROUPS = 200


class ArtifactListInput(BaseModel):
    """Параметры для получения списка artifacts текущего запуска."""

    kind: str = Field(
        default="",
        description="Опциональный фильтр по kind, например dataset или source_excerpt.",
    )
    reusable_only: bool = Field(
        default=False,
        description="Если True, вернуть только artifacts с metadata.reusable=True.",
    )
    editable_only: bool = Field(
        default=False,
        description="Если True, вернуть только artifacts с metadata.editable=True.",
    )
    limit: int = Field(
        default=50,
        description="Максимальное количество artifacts в ответе.",
    )


class ArtifactPreviewInput(BaseModel):
    """Параметры для чтения краткого preview artifact."""

    artifact_id: str = Field(description="Идентификатор artifact.")
    max_chars: int = Field(
        default=DEFAULT_PREVIEW_CHARS,
        description="Максимальное количество символов preview.",
    )


class ArtifactReadChunkInput(BaseModel):
    """Параметры для чтения части текстового artifact."""

    artifact_id: str = Field(description="Идентификатор artifact.")
    offset: int = Field(
        default=0,
        description="Смещение в символах от начала файла.",
    )
    limit: int = Field(
        default=DEFAULT_CHUNK_CHARS,
        description="Количество символов для чтения, максимум 20000.",
    )


class ArtifactProfileInput(BaseModel):
    """Параметры для построения универсального профиля табличного artifact."""

    artifact_id: str = Field(description="Идентификатор artifact.")
    top_values_limit: int = Field(
        default=10,
        description="Максимальное количество популярных значений по каждой колонке.",
    )


class ArtifactSampleInput(BaseModel):
    """Параметры для чтения выборки записей из табличного artifact."""

    artifact_id: str = Field(description="Идентификатор artifact.")
    offset: int = Field(default=0, description="Номер первой записи в выборке.")
    limit: int = Field(default=20, description="Максимальное количество записей в выборке.")


class ArtifactSearchInput(BaseModel):
    """Параметры для поиска записей в табличном artifact."""

    artifact_id: str = Field(description="Идентификатор artifact.")
    query: str = Field(description="Текст для поиска по значениям записей.")
    columns: list[str] = Field(
        default_factory=list,
        description="Опциональный список колонок для поиска. Если пусто, поиск идет по всем колонкам.",
    )
    limit: int = Field(default=20, description="Максимальное количество найденных записей.")


class ArtifactValueCountsInput(BaseModel):
    """Параметры для расчета частот значений по колонкам artifact."""

    artifact_id: str = Field(description="Идентификатор artifact.")
    columns: list[str] = Field(
        description="Список колонок для группировки. Можно передать одну или несколько колонок.",
    )
    limit: int = Field(default=20, description="Максимальное количество групп в ответе.")


def build_artifact_read_tools(
        *,
        artifact_service: ArtifactService,
        run_id: str,
) -> list[BaseTool]:
    """Создает LangChain tools для чтения artifacts текущего запуска.

    Args:
        artifact_service: Сервис доступа к artifact store.
        run_id: Идентификатор текущего ResearchRun.

    Returns:
        Список LangChain tools для чтения и анализа artifacts текущего запуска.
    """

    def artifact_list(
            kind: str = "",
            reusable_only: bool = False,
            editable_only: bool = False,
            limit: int = 50,
    ) -> str:
        """Вернуть компактный список artifacts текущего запуска."""

        safe_limit = _clamp_int(limit, minimum=1, maximum=MAX_LIST_ITEMS)
        artifacts = artifact_service.list_artifacts(run_id)
        filtered: list[Artifact] = []
        for artifact in artifacts:
            if kind and artifact.kind != kind:
                continue
            if reusable_only and not bool(artifact.metadata.get("reusable")):
                continue
            if editable_only and not bool(artifact.metadata.get("editable")):
                continue
            filtered.append(artifact)

        return _json_text(
            {
                "run_id": run_id,
                "total_matches": len(filtered),
                "shown": min(len(filtered), safe_limit),
                "artifacts": [
                    _artifact_to_reference(artifact)
                    for artifact in filtered[:safe_limit]
                ],
            }
        )

    def artifact_preview(artifact_id: str, max_chars: int = DEFAULT_PREVIEW_CHARS) -> str:
        """Вернуть metadata и короткое preview artifact."""

        artifact = _artifact_by_id(artifact_service, run_id, artifact_id)
        if artifact is None:
            return _json_text({"error": "artifact_not_found", "artifact_id": artifact_id})

        safe_max_chars = _clamp_int(max_chars, minimum=1, maximum=MAX_CHUNK_CHARS)
        preview_result = _read_text_chunk(artifact, offset=0, limit=safe_max_chars)
        preview = str(preview_result.get("content") or "")
        truncated = bool(preview_result.get("has_more"))
        return _json_text(
            {
                "artifact": _artifact_to_reference(artifact),
                "preview": preview,
                "preview_chars": len(preview),
                "total_chars": preview_result.get("total_chars"),
                "truncated": truncated,
                "data_scope": "partial" if truncated else "complete_text",
                "worker_disclosure_required": truncated,
            }
        )

    def artifact_read_chunk(
            artifact_id: str,
            offset: int = 0,
            limit: int = DEFAULT_CHUNK_CHARS,
    ) -> str:
        """Прочитать часть текстового artifact по offset/limit."""

        artifact = _artifact_by_id(artifact_service, run_id, artifact_id)
        if artifact is None:
            return _json_text({"error": "artifact_not_found", "artifact_id": artifact_id})

        safe_offset = _clamp_int(offset, minimum=0, maximum=10**12)
        safe_limit = _clamp_int(limit, minimum=1, maximum=MAX_CHUNK_CHARS)
        chunk_result = _read_text_chunk(
            artifact,
            offset=safe_offset,
            limit=safe_limit,
        )
        return _json_text(
            {
                "artifact": _artifact_to_reference(artifact),
                **chunk_result,
            }
        )

    def artifact_profile(artifact_id: str, top_values_limit: int = 10) -> str:
        """Построить универсальный профиль CSV/JSON artifact без доменных предположений."""

        artifact = _artifact_by_id(artifact_service, run_id, artifact_id)
        if artifact is None:
            return _json_text({"error": "artifact_not_found", "artifact_id": artifact_id})

        records_result = _load_tabular_records(artifact)
        if "error" in records_result:
            return _json_text({"artifact": _artifact_to_reference(artifact), **records_result})

        safe_top_values_limit = _clamp_int(
            top_values_limit,
            minimum=1,
            maximum=MAX_PROFILE_TOP_VALUES,
        )
        return _json_text(
            {
                "artifact": _artifact_to_reference(artifact),
                "profile": _profile_records(
                    records_result["records"],
                    top_values_limit=safe_top_values_limit,
                ),
            }
        )

    def artifact_sample(
            artifact_id: str,
            offset: int = 0,
            limit: int = 20,
    ) -> str:
        """Вернуть выборку записей из CSV/JSON artifact по offset/limit."""

        artifact = _artifact_by_id(artifact_service, run_id, artifact_id)
        if artifact is None:
            return _json_text({"error": "artifact_not_found", "artifact_id": artifact_id})

        records_result = _load_tabular_records(artifact)
        if "error" in records_result:
            return _json_text({"artifact": _artifact_to_reference(artifact), **records_result})

        safe_offset = _clamp_int(offset, minimum=0, maximum=10**12)
        safe_limit = _clamp_int(limit, minimum=1, maximum=MAX_SAMPLE_RECORDS)
        return _json_text(
            {
                "artifact": _artifact_to_reference(artifact),
                **_sample_records(
                    records_result["records"],
                    offset=safe_offset,
                    limit=safe_limit,
                ),
            }
        )

    def artifact_search(
            artifact_id: str,
            query: str,
            columns: list[str] | None = None,
            limit: int = 20,
    ) -> str:
        """Найти записи в CSV/JSON artifact по текстовому совпадению."""

        artifact = _artifact_by_id(artifact_service, run_id, artifact_id)
        if artifact is None:
            return _json_text({"error": "artifact_not_found", "artifact_id": artifact_id})

        records_result = _load_tabular_records(artifact)
        if "error" in records_result:
            return _json_text({"artifact": _artifact_to_reference(artifact), **records_result})

        safe_limit = _clamp_int(limit, minimum=1, maximum=MAX_SEARCH_RECORDS)
        return _json_text(
            {
                "artifact": _artifact_to_reference(artifact),
                **_search_records(
                    records_result["records"],
                    query=query,
                    columns=columns or [],
                    limit=safe_limit,
                ),
            }
        )

    def artifact_value_counts(
            artifact_id: str,
            columns: list[str],
            limit: int = 20,
    ) -> str:
        """Посчитать частоты значений или комбинаций значений в CSV/JSON artifact."""

        artifact = _artifact_by_id(artifact_service, run_id, artifact_id)
        if artifact is None:
            return _json_text({"error": "artifact_not_found", "artifact_id": artifact_id})

        records_result = _load_tabular_records(artifact)
        if "error" in records_result:
            return _json_text({"artifact": _artifact_to_reference(artifact), **records_result})

        safe_limit = _clamp_int(limit, minimum=1, maximum=MAX_VALUE_COUNT_GROUPS)
        return _json_text(
            {
                "artifact": _artifact_to_reference(artifact),
                **_value_counts(
                    records_result["records"],
                    columns=columns,
                    limit=safe_limit,
                ),
            }
        )

    return [
        StructuredTool.from_function(
            func=artifact_list,
            name="artifact_list",
            description=(
                "List artifacts available in the current run with optional filters "
                "by kind/reusable/editable. Use first to discover whether required "
                "datasets, reports, source excerpts or tool traces are already "
                "materialized, so you can reuse existing artifacts instead of "
                "triggering repeated data extraction."
            ),
            args_schema=ArtifactListInput,
        ),
        StructuredTool.from_function(
            func=artifact_preview,
            name="artifact_preview",
            description=(
                "Read artifact metadata and a bounded text preview without loading "
                "the full file into context. Use to confirm artifact provenance, "
                "purpose and content scope before deeper reads or downstream "
                "analytical steps."
            ),
            args_schema=ArtifactPreviewInput,
        ),
        StructuredTool.from_function(
            func=artifact_read_chunk,
            name="artifact_read_chunk",
            description=(
                "Read a bounded text slice from an artifact by offset and limit "
                "(max 20000 chars). Use for deterministic pagination over long "
                "reports, JSON exports, logs and traces while keeping LLM context "
                "size under control."
            ),
            args_schema=ArtifactReadChunkInput,
        ),
        StructuredTool.from_function(
            func=artifact_profile,
            name="artifact_profile",
            description=(
                "Build a schema-and-quality profile for CSV/JSON tabular artifacts: "
                "row count, columns, missingness, inferred value kinds and top "
                "values. Use as a first validation step after ingestion/materialization "
                "to understand structure before joins, metrics or rule checks."
            ),
            args_schema=ArtifactProfileInput,
        ),
        StructuredTool.from_function(
            func=artifact_sample,
            name="artifact_sample",
            description=(
                "Read a bounded record sample from CSV/JSON artifact by offset/limit. "
                "Use for spot-checks of row-level correctness after filters, joins, "
                "deduplication or enrichment, without loading full datasets to context."
            ),
            args_schema=ArtifactSampleInput,
        ),
        StructuredTool.from_function(
            func=artifact_search,
            name="artifact_search",
            description=(
                "Search CSV/JSON artifact records by text match across selected or "
                "all columns. Use for targeted lookups in already materialized exports "
                "when validating entities, event ids, rule names, counterparties or "
                "other investigation pivots."
            ),
            args_schema=ArtifactSearchInput,
        ),
        StructuredTool.from_function(
            func=artifact_value_counts,
            name="artifact_value_counts",
            description=(
                "Compute frequency counts for one or more columns in CSV/JSON artifact "
                "records. Use for fast aggregate diagnostics: distribution drift, "
                "dominant categories, repeated keys and sanity checks before formal "
                "metric calculation."
            ),
            args_schema=ArtifactValueCountsInput,
        ),
    ]


def _artifact_to_reference(artifact: Artifact) -> dict[str, Any]:
    """Преобразует Artifact в компактную ссылку для tool output.

    Args:
        artifact: Artifact из artifact store.

    Returns:
        JSON-совместимое компактное описание artifact.
    """

    return {
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind,
        "uri": artifact.uri,
        "mime_type": artifact.mime_type,
        "summary": artifact.summary,
        "checksum": artifact.checksum,
        "metadata": {
            key: artifact.metadata.get(key)
            for key in (
                "task_id",
                "tool_name",
                "artifact_role",
                "reusable",
                "editable",
                "capture_reason",
                "original_size_estimate",
            )
            if key in artifact.metadata
        },
    }


def _artifact_by_id(
        artifact_service: ArtifactService,
        run_id: str,
        artifact_id: str,
) -> Artifact | None:
    """Ищет artifact по id внутри текущего run.

    Args:
        artifact_service: Сервис artifacts.
        run_id: Идентификатор текущего ResearchRun.
        artifact_id: Идентификатор artifact.

    Returns:
        Artifact или ``None``.
    """

    return artifact_service.get_artifact(run_id, artifact_id)


def _load_tabular_records(artifact: Artifact) -> dict[str, Any]:
    """Загружает CSV/JSON artifact как список словарей без доменной логики.

    Args:
        artifact: Artifact, который нужно интерпретировать как табличные записи.

    Returns:
        Словарь с ключом ``records`` или словарь с ключом ``error``.
    """

    path = Path(artifact.uri)
    if not path.exists() or not path.is_file():
        return {"error": "artifact_file_not_found", "records": []}

    mime_type = (artifact.mime_type or "").lower()
    suffix = path.suffix.lower()
    try:
        if mime_type == "text/csv" or suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                return {"records": list(csv.DictReader(handle))}
        if mime_type in {"application/json", "application/x-jsonlines"} or suffix in {".json", ".jsonl"}:
            if suffix == ".jsonl" or mime_type == "application/x-jsonlines":
                records = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                return {"records": [item for item in records if isinstance(item, dict)]}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return {"records": _extract_json_records(payload)}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        return {"error": "artifact_parse_failed", "message": str(exc), "records": []}

    return {
        "error": "artifact_format_not_supported",
        "message": "Only CSV, JSON list/dict and JSONL artifacts are supported by this tool.",
        "records": [],
    }


def _extract_json_records(payload: Any) -> list[dict[str, Any]]:
    """Извлекает список записей из универсального JSON payload.

    Args:
        payload: JSON-совместимый объект из artifact.

    Returns:
        Список словарей. Если payload не содержит табличных записей, возвращается пустой список.
    """

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if all(not isinstance(value, (list, dict)) for value in payload.values()):
            return [payload]
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
    return []


def _profile_records(records: list[dict[str, Any]], *, top_values_limit: int) -> dict[str, Any]:
    """Рассчитывает универсальный профиль списка записей.

    Args:
        records: Табличные записи artifact.
        top_values_limit: Максимальное количество популярных значений по колонке.

    Returns:
        JSON-совместимый профиль: строки, колонки, пустые значения, типы и top values.
    """

    columns = sorted({str(key) for record in records for key in record.keys()})[:MAX_PROFILE_COLUMNS]
    profile_columns: dict[str, Any] = {}
    for column in columns:
        values = [record.get(column) for record in records]
        non_empty = [value for value in values if not _is_empty_value(value)]
        profile_columns[column] = {
            "missing_count": len(values) - len(non_empty),
            "non_missing_count": len(non_empty),
            "inferred_value_kinds": dict(Counter(_value_kind(value) for value in non_empty)),
            "top_values": [
                {"value": value, "count": count}
                for value, count in Counter(_stringify_value(value) for value in non_empty).most_common(top_values_limit)
            ],
        }

    return {
        "row_count": len(records),
        "column_count": len(columns),
        "columns": columns,
        "columns_profile": profile_columns,
        "truncated_columns": len({key for record in records for key in record.keys()}) > len(columns),
    }


def _sample_records(
        records: list[dict[str, Any]],
        *,
        offset: int,
        limit: int,
) -> dict[str, Any]:
    """Возвращает срез записей по offset/limit.

    Args:
        records: Табличные записи artifact.
        offset: Индекс первой записи.
        limit: Максимальное количество записей.

    Returns:
        JSON-совместимый результат выборки.
    """

    sample = records[offset:offset + limit]
    return {
        "row_count": len(records),
        "offset": offset,
        "limit": limit,
        "returned": len(sample),
        "has_more": offset + len(sample) < len(records),
        "next_offset": offset + len(sample),
        "records": sample,
    }


def _search_records(
        records: list[dict[str, Any]],
        *,
        query: str,
        columns: list[str],
        limit: int,
) -> dict[str, Any]:
    """Ищет записи по текстовому совпадению.

    Args:
        records: Табличные записи artifact.
        query: Поисковая строка.
        columns: Ограничение поиска конкретными колонками.
        limit: Максимальное количество результатов.

    Returns:
        JSON-совместимый результат поиска.
    """

    matches: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if _record_matches_query(record, query=query, columns=columns):
            matches.append({"row_index": index, "record": record})
            if len(matches) >= limit:
                break
    return {
        "query": query,
        "columns": columns,
        "row_count": len(records),
        "returned": len(matches),
        "records": matches,
    }


def _value_counts(
        records: list[dict[str, Any]],
        *,
        columns: list[str],
        limit: int,
) -> dict[str, Any]:
    """Считает частоты значений или комбинаций значений по колонкам.

    Args:
        records: Табличные записи artifact.
        columns: Колонки для группировки.
        limit: Максимальное количество групп.

    Returns:
        JSON-совместимый список частот.
    """

    selected_columns = [column for column in columns if column]
    if not selected_columns:
        return {"error": "columns_required", "counts": []}

    counter = Counter(
        tuple(_stringify_value(record.get(column)) for column in selected_columns)
        for record in records
    )
    return {
        "columns": selected_columns,
        "row_count": len(records),
        "returned": min(len(counter), limit),
        "counts": [
            {
                "values": dict(zip(selected_columns, values)),
                "count": count,
            }
            for values, count in counter.most_common(limit)
        ],
    }


def _record_matches_query(record: dict[str, Any], *, query: str, columns: list[str]) -> bool:
    """Проверяет, содержит ли запись поисковую строку.

    Args:
        record: Одна табличная запись.
        query: Поисковая строка.
        columns: Колонки, по которым нужно искать. Пустой список означает все колонки.

    Returns:
        ``True``, если query найден в одном из выбранных значений.
    """

    needle = str(query or "").casefold()
    if not needle:
        return False
    selected = columns or list(record.keys())
    return any(needle in _stringify_value(record.get(column)).casefold() for column in selected)


def _is_empty_value(value: Any) -> bool:
    """Проверяет, является ли значение пустым для целей профилирования.

    Args:
        value: Значение поля записи.

    Returns:
        ``True`` для None, пустой строки и пустых контейнеров.
    """

    return value is None or value == "" or value == [] or value == {}


def _value_kind(value: Any) -> str:
    """Возвращает простой тип значения для профиля artifact.

    Args:
        value: Значение поля записи.

    Returns:
        Название типа значения.
    """

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


def _stringify_value(value: Any) -> str:
    """Преобразует значение записи в стабильную строку для поиска и группировки.

    Args:
        value: Произвольное значение поля записи.

    Returns:
        Строковое представление значения.
    """

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _read_text_chunk(
        artifact: Artifact,
        *,
        offset: int,
        limit: int,
) -> dict[str, Any]:
    """Читает часть текстового artifact.

    Args:
        artifact: Artifact для чтения.
        offset: Смещение в символах.
        limit: Максимальное количество символов.

    Returns:
        JSON-совместимый результат чтения.
    """

    path = Path(artifact.uri)
    if not path.exists() or not path.is_file():
        return {
            "error": "artifact_file_not_found",
            "content": "",
            "data_scope": "unavailable",
            "worker_disclosure_required": True,
        }
    if not _is_text_mime_type(artifact.mime_type):
        return {
            "error": "artifact_is_not_text",
            "content": "",
            "data_scope": "unavailable",
            "worker_disclosure_required": True,
            "hint": "Use artifact metadata or a specialized reader for binary artifacts.",
        }

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "error": "artifact_decode_failed",
            "content": "",
            "data_scope": "unavailable",
            "worker_disclosure_required": True,
        }

    chunk = content[offset:offset + limit]
    has_more = offset + len(chunk) < len(content)
    partial = offset > 0 or has_more
    return {
        "offset": offset,
        "limit": limit,
        "content": chunk,
        "content_chars": len(chunk),
        "total_chars": len(content),
        "has_more": has_more,
        "next_offset": offset + len(chunk),
        "data_scope": "partial" if partial else "complete_text",
        "worker_disclosure_required": partial,
    }


def _read_text_preview(artifact: Artifact, *, max_chars: int) -> str:
    """Читает preview текстового artifact.

    Args:
        artifact: Artifact для чтения.
        max_chars: Максимальное количество символов.

    Returns:
        Preview или пустая строка, если файл не текстовый/недоступен.
    """

    result = _read_text_chunk(artifact, offset=0, limit=max_chars)
    return str(result.get("content") or "")


def _is_text_mime_type(mime_type: str) -> bool:
    """Проверяет, можно ли читать artifact как UTF-8 текст.

    Args:
        mime_type: MIME-тип artifact.

    Returns:
        ``True`` для текстовых и JSON/CSV artifacts.
    """

    normalized = (mime_type or "").lower()
    return (
        normalized.startswith("text/")
        or normalized in {"application/json", "application/x-jsonlines"}
    )


def _json_text(payload: dict[str, Any]) -> str:
    """Сериализует payload tool output в JSON.

    Args:
        payload: JSON-совместимый словарь.

    Returns:
        JSON-строка.
    """

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    """Ограничивает целое число диапазоном.

    Args:
        value: Исходное значение.
        minimum: Минимально допустимое значение.
        maximum: Максимально допустимое значение.

    Returns:
        Значение внутри диапазона.
    """

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


__all__ = [
    "ArtifactListInput",
    "ArtifactPreviewInput",
    "ArtifactReadChunkInput",
    "ArtifactProfileInput",
    "ArtifactSampleInput",
    "ArtifactSearchInput",
    "ArtifactValueCountsInput",
    "build_artifact_read_tools",
]

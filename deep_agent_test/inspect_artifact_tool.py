"""Фактическая проверка локальных артефактов для data-retrieval-critic (без вердикта)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from deep_agent_test.settings import DeepAgentSettings, PROJECT_ROOT


class InspectArtifactPathInput(BaseModel):
    """Аргументы tool ``inspect_artifact_path``: путь к файлу и число строк preview."""

    path: str = Field(description="Абсолютный или относительный путь к файлу для проверки.")
    pickle_preview_rows: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Сколько строк показать из pickle (0 — только метаданные).",
    )


def _allowed_roots(settings: DeepAgentSettings) -> list[Path]:
    """Возвращает корни, в пределах которых критику разрешено проверять файлы."""

    return [settings.tool_outputs_dir.resolve(), PROJECT_ROOT.resolve()]


def _is_allowed_path(path: Path, settings: DeepAgentSettings) -> bool:
    """Проверяет, что путь находится внутри одного из разрешённых корней."""

    resolved = path.resolve()
    return any(resolved == root or root in resolved.parents for root in _allowed_roots(settings))


def inspect_artifact_path(
    path: str,
    pickle_preview_rows: int = 3,
    *,
    settings: DeepAgentSettings | None = None,
) -> str:
    """Возвращает только наблюдаемые факты о пути; интерпретацию оставляет LLM-critic."""

    from deep_agent_test.settings import load_deep_agent_settings

    settings = settings or load_deep_agent_settings()
    target = Path(path.strip())
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()

    if not _is_allowed_path(target, settings):
        return json.dumps(
            {
                "path": str(target),
                "allowed": False,
                "error": "path_outside_allowed_roots",
                "allowed_roots": [str(root) for root in _allowed_roots(settings)],
            },
            ensure_ascii=False,
        )

    payload: dict[str, Any] = {
        "path": str(target),
        "allowed": True,
        "exists": target.exists(),
        "is_file": target.is_file() if target.exists() else False,
        "is_dir": target.is_dir() if target.exists() else False,
    }
    if not target.exists():
        return json.dumps(payload, ensure_ascii=False)

    if target.is_file():
        payload["size_bytes"] = target.stat().st_size
        payload["suffix"] = target.suffix.lower()

    if target.is_file() and target.suffix.lower() == ".pkl":
        try:
            with target.open("rb") as handle:
                loaded = pickle.load(handle)
        except Exception as exc:
            payload["pickle_readable"] = False
            payload["pickle_error"] = str(exc)
        else:
            payload["pickle_readable"] = True
            payload["pickle_type"] = type(loaded).__name__
            if isinstance(loaded, list):
                payload["pickle_row_count"] = len(loaded)
                if loaded and isinstance(loaded[0], dict):
                    payload["pickle_columns"] = sorted({key for row in loaded for key in row})
                    limit = max(0, min(pickle_preview_rows, len(loaded)))
                    payload["pickle_preview"] = loaded[:limit]
            else:
                payload["pickle_preview"] = str(loaded)[:2000]

    return json.dumps(payload, ensure_ascii=False, default=str)


def build_inspect_artifact_tool(settings: DeepAgentSettings | None = None) -> StructuredTool:
    """Собирает StructuredTool ``inspect_artifact_path`` с замыканием на settings.

    Args:
        settings: Настройки агента; если ``None`` — загружаются из JSON-конфига.

    Returns:
        ``StructuredTool``, возвращающий факты о файле для суждения critic-а.
    """

    from deep_agent_test.settings import load_deep_agent_settings

    settings = settings or load_deep_agent_settings()

    def _run(path: str, pickle_preview_rows: int = 3) -> str:
        """Делегирует проверку пути ``inspect_artifact_path`` с замкнутыми settings."""

        return inspect_artifact_path(path, pickle_preview_rows, settings=settings)

    return StructuredTool.from_function(
        func=_run,
        name="inspect_artifact_path",
        description=(
            "Возвращает наблюдаемые факты о файле на диске: существует ли, размер, "
            "для .pkl — читается ли и сколько строк/колонок в preview. "
            "Не принимает решений за тебя — только факты для суждения critic-а."
        ),
        args_schema=InspectArtifactPathInput,
    )


__all__ = [
    "InspectArtifactPathInput",
    "build_inspect_artifact_tool",
    "inspect_artifact_path",
]

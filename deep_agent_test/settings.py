"""Конфигурация аналитического DeepAgent."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "defaults.json"
CONFIG_ENV_VAR = "DEEP_AGENT_CONFIG_PATH"
REQUIRED_CONFIG_KEYS = (
    "thread_id",
    "skills_virtual_dir",
    "skills_root",
    "data_tools_factory",
    "data_tools_factory_kwargs",
    "tool_outputs_dir",
    "max_chars_per_skill",
    "tool_output_min_rows_to_save",
    "tool_output_min_content_chars_to_save",
    "tool_output_preview_rows",
    "tool_output_inline_original_chars",
    "context_edit_trigger_tokens",
    "context_edit_keep_tool_results",
    "file_search_use_ripgrep",
    "max_consecutive_tool_calls",
    "max_subagent_model_calls",
    "max_critic_iterations",
    "graph_recursion_limit",
    "trace_log_dir",
)


@dataclass(frozen=True)
class DeepAgentSettings:
    """Настройки сборки и запуска аналитического DeepAgent."""

    thread_id: str
    skills_virtual_dir: str
    skills_root: Path
    data_tools_factory: str | None
    data_tools_factory_kwargs: dict[str, Any]
    tool_outputs_dir: Path
    max_chars_per_skill: int
    tool_output_min_rows_to_save: int
    tool_output_min_content_chars_to_save: int
    tool_output_preview_rows: int
    tool_output_inline_original_chars: int
    context_edit_trigger_tokens: int
    context_edit_keep_tool_results: int
    file_search_use_ripgrep: bool
    max_consecutive_tool_calls: int
    max_subagent_model_calls: int
    max_critic_iterations: int
    graph_recursion_limit: int
    trace_log_dir: Path
    enable_retrieval_critic: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], project_root: Path = PROJECT_ROOT) -> "DeepAgentSettings":
        """Собирает типизированные настройки из словаря конфигурации.

        Args:
            payload: Сырой словарь из JSON-конфига (defaults + override).
            project_root: Корень проекта для разрешения относительных путей.

        Returns:
            Готовый ``DeepAgentSettings``.

        Raises:
            ValueError: Не хватает обязательного ключа или ключ имеет неверный тип.
        """

        _validate_required_config_keys(payload)
        return cls(
            thread_id=str(payload["thread_id"]),
            skills_virtual_dir=str(payload["skills_virtual_dir"]),
            skills_root=_resolve_project_path(payload["skills_root"], project_root),
            data_tools_factory=_optional_str_from_config(payload, "data_tools_factory"),
            data_tools_factory_kwargs=_dict_from_config(payload, "data_tools_factory_kwargs"),
            tool_outputs_dir=_resolve_project_path(payload["tool_outputs_dir"], project_root),
            max_chars_per_skill=_int_from_config(payload, "max_chars_per_skill"),
            tool_output_min_rows_to_save=_int_from_config(payload, "tool_output_min_rows_to_save"),
            tool_output_min_content_chars_to_save=_int_from_config(
                payload,
                "tool_output_min_content_chars_to_save",
            ),
            tool_output_preview_rows=_int_from_config(payload, "tool_output_preview_rows"),
            tool_output_inline_original_chars=_int_from_config(payload, "tool_output_inline_original_chars"),
            context_edit_trigger_tokens=_int_from_config(payload, "context_edit_trigger_tokens"),
            context_edit_keep_tool_results=_int_from_config(payload, "context_edit_keep_tool_results"),
            file_search_use_ripgrep=_bool_from_config(payload, "file_search_use_ripgrep"),
            max_consecutive_tool_calls=_int_from_config(payload, "max_consecutive_tool_calls"),
            max_subagent_model_calls=_int_from_config(payload, "max_subagent_model_calls"),
            max_critic_iterations=_int_from_config(payload, "max_critic_iterations"),
            graph_recursion_limit=_int_from_config(payload, "graph_recursion_limit"),
            trace_log_dir=_resolve_project_path(payload["trace_log_dir"], project_root),
            enable_retrieval_critic=(
                _bool_from_config(payload, "enable_retrieval_critic")
                if "enable_retrieval_critic" in payload
                else True
            ),
        )


def load_deep_agent_settings(config_path: str | Path | None = None) -> DeepAgentSettings:
    """Загружает настройки агента из JSON-конфига (defaults + опциональный override)."""

    payload = _load_config_payload(config_path)
    return DeepAgentSettings.from_mapping(payload)


def _load_config_payload(config_path: str | Path | None = None) -> dict[str, Any]:
    """Читает defaults-конфиг и мёржит поверх него override (аргумент или env)."""

    default_payload = _read_json_file(DEFAULT_CONFIG_PATH)
    raw_path = config_path or os.environ.get(CONFIG_ENV_VAR)
    if raw_path is None:
        return default_payload

    custom_path = Path(raw_path)
    if custom_path.resolve() == DEFAULT_CONFIG_PATH.resolve():
        return default_payload

    custom_payload = _read_json_file(custom_path)
    return {**default_payload, **custom_payload}


def _read_json_file(path: Path) -> dict[str, Any]:
    """Читает JSON-файл и проверяет, что он содержит объект (dict)."""

    resolved_path = path.resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain JSON object: {resolved_path}")
    return payload


def _validate_required_config_keys(payload: dict[str, Any]) -> None:
    """Проверяет наличие всех обязательных ключей конфига, иначе бросает ValueError."""

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in payload]
    if missing_keys:
        raise ValueError(f"DeepAgent config missing required keys: {', '.join(missing_keys)}")


def _resolve_project_path(value: Any, project_root: Path) -> Path:
    """Приводит значение к абсолютному пути относительно корня проекта."""

    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _int_from_config(payload: dict[str, Any], key: str) -> int:
    """Читает int-ключ конфига, иначе бросает ValueError с именем ключа."""

    try:
        return int(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"Config key '{key}' must be an integer.") from None


def _bool_from_config(payload: dict[str, Any], key: str) -> bool:
    """Читает bool-ключ конфига, принимая bool или строковые true/false/1/0/yes/no."""

    value = payload[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"Config key '{key}' must be a boolean.")


def _dict_from_config(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Читает dict-ключ конфига, иначе бросает ValueError с именем ключа."""

    value = payload[key]
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"Config key '{key}' must be an object.")


def _optional_str_from_config(payload: dict[str, Any], key: str) -> str | None:
    """Читает строковый ключ конфига, допускающий ``null`` (возвращает ``None``)."""

    value = payload[key]
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    raise ValueError(f"Config key '{key}' must be a string or null.")


__all__ = [
    "CONFIG_ENV_VAR",
    "DEFAULT_CONFIG_PATH",
    "PROJECT_ROOT",
    "REQUIRED_CONFIG_KEYS",
    "DeepAgentSettings",
    "load_deep_agent_settings",
]

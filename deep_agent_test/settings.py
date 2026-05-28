"""Конфигурация аналитического DeepAgent.

Содержит:
- PROJECT_ROOT: корневая папка проекта.
- DEFAULT_CONFIG_PATH: путь к стандартному JSON-конфигу.
- CONFIG_ENV_VAR: имя переменной окружения с путем к альтернативному конфигу.
- REQUIRED_CONFIG_KEYS: обязательные ключи итогового JSON-конфига.
- DeepAgentSettings: типизированные настройки сборки агента.
- DeepAgentSettings.from_mapping: создание настроек из словаря.
- load_deep_agent_settings: загрузка настроек из JSON-файла.
- _load_config_payload: загрузка стандартного конфига и пользовательских переопределений.
- _read_json_file: чтение JSON-конфига.
- _validate_required_config_keys: проверка обязательных ключей конфига.
- _resolve_project_path: преобразование относительного пути в абсолютный путь проекта.
- _int_from_config: безопасное чтение целого числа из конфига.
- _bool_from_config: безопасное чтение булева значения из конфига.
- _dict_from_config: безопасное чтение словаря из конфига.
- _optional_str_from_config: безопасное чтение необязательной строки из конфига.
"""

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
    "few_shot_examples_dir",
    "few_shot_index_dir",
    "logs_dir",
    "tool_outputs_dir",
    "max_chars_per_skill",
    "few_shot_top_k",
    "few_shot_max_examples",
    "trace_preview_chars",
    "log_available_tools",
    "log_model_tool_calls",
    "log_tool_execution",
    "log_tool_result",
    "print_tool_calls",
    "print_tool_results",
    "print_plan_prompts",
    "tool_output_min_rows_to_save",
    "tool_output_min_content_chars_to_save",
    "tool_output_preview_rows",
    "tool_output_inline_original_chars",
)


@dataclass(frozen=True)
class DeepAgentSettings:
    """Настройки сборки и запуска аналитического DeepAgent.

    Args:
        thread_id: Идентификатор диалога LangGraph по умолчанию.
        skills_virtual_dir: Виртуальный путь skills внутри backend DeepAgents.
        skills_root: Абсолютный путь к локальной папке skills.
        data_tools_factory: Import path callable-фабрики, которая возвращает LangChain tools чтения данных.
        data_tools_factory_kwargs: Именованные аргументы для callable-фабрики tools чтения данных.
        few_shot_examples_dir: Абсолютный путь к markdown-примерам few-shot.
        few_shot_index_dir: Абсолютный путь к генерируемому индексу few-shot.
        logs_dir: Абсолютный путь к папке файловых логов агента.
        tool_outputs_dir: Абсолютный путь к папке сохраненных больших tool outputs.
        max_chars_per_skill: Максимальная длина одного markdown-файла skills в prompt context.
        few_shot_top_k: Количество кандидатов после векторного поиска few-shot.
        few_shot_max_examples: Максимальное количество few-shot примеров в prompt.
        trace_preview_chars: Максимальная длина preview в trace-логах.
        log_available_tools: Нужно ли писать список tools, доступных модели.
        log_model_tool_calls: Нужно ли писать tool calls из ответа модели.
        log_tool_execution: Нужно ли писать старт выполнения tool.
        log_tool_result: Нужно ли писать результат выполнения tool.
        print_tool_calls: Нужно ли печатать в консоль фактические вызовы tools.
        print_tool_results: Нужно ли печатать в консоль ответы tools.
        print_plan_prompts: Нужно ли печатать system/user prompt перед первичным планом.
        tool_output_min_rows_to_save: Минимум строк для сохранения tool output в CSV.
        tool_output_min_content_chars_to_save: Минимум символов content для сохранения tool output.
        tool_output_preview_rows: Количество строк preview в summary сохраненного tool output.
        tool_output_inline_original_chars: Максимальная длина исходного tool content, дублируемого в summary.

    Returns:
        Неизменяемый объект настроек, готовый к передаче в сборку агента.
    """

    thread_id: str
    skills_virtual_dir: str
    skills_root: Path
    data_tools_factory: str | None
    data_tools_factory_kwargs: dict[str, Any]
    few_shot_examples_dir: Path
    few_shot_index_dir: Path
    logs_dir: Path
    tool_outputs_dir: Path
    max_chars_per_skill: int
    few_shot_top_k: int
    few_shot_max_examples: int
    trace_preview_chars: int
    log_available_tools: bool
    log_model_tool_calls: bool
    log_tool_execution: bool
    log_tool_result: bool
    print_tool_calls: bool
    print_tool_results: bool
    print_plan_prompts: bool
    tool_output_min_rows_to_save: int
    tool_output_min_content_chars_to_save: int
    tool_output_preview_rows: int
    tool_output_inline_original_chars: int

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], project_root: Path = PROJECT_ROOT) -> "DeepAgentSettings":
        """Создает настройки из словаря JSON-конфига.

        Args:
            payload: Словарь с настройками из JSON-файла.
            project_root: Абсолютный путь к корню проекта для относительных путей.

        Returns:
            Заполненный объект ``DeepAgentSettings``.
        """

        _validate_required_config_keys(payload)
        return cls(
            thread_id=str(payload["thread_id"]),
            skills_virtual_dir=str(payload["skills_virtual_dir"]),
            skills_root=_resolve_project_path(payload["skills_root"], project_root),
            data_tools_factory=_optional_str_from_config(payload, "data_tools_factory"),
            data_tools_factory_kwargs=_dict_from_config(payload, "data_tools_factory_kwargs"),
            few_shot_examples_dir=_resolve_project_path(
                payload["few_shot_examples_dir"],
                project_root,
            ),
            few_shot_index_dir=_resolve_project_path(
                payload["few_shot_index_dir"],
                project_root,
            ),
            logs_dir=_resolve_project_path(payload["logs_dir"], project_root),
            tool_outputs_dir=_resolve_project_path(
                payload["tool_outputs_dir"],
                project_root,
            ),
            max_chars_per_skill=_int_from_config(payload, "max_chars_per_skill"),
            few_shot_top_k=_int_from_config(payload, "few_shot_top_k"),
            few_shot_max_examples=_int_from_config(payload, "few_shot_max_examples"),
            trace_preview_chars=_int_from_config(payload, "trace_preview_chars"),
            log_available_tools=_bool_from_config(payload, "log_available_tools"),
            log_model_tool_calls=_bool_from_config(payload, "log_model_tool_calls"),
            log_tool_execution=_bool_from_config(payload, "log_tool_execution"),
            log_tool_result=_bool_from_config(payload, "log_tool_result"),
            print_tool_calls=_bool_from_config(payload, "print_tool_calls"),
            print_tool_results=_bool_from_config(payload, "print_tool_results"),
            print_plan_prompts=_bool_from_config(payload, "print_plan_prompts"),
            tool_output_min_rows_to_save=_int_from_config(payload, "tool_output_min_rows_to_save"),
            tool_output_min_content_chars_to_save=_int_from_config(
                payload,
                "tool_output_min_content_chars_to_save",
            ),
            tool_output_preview_rows=_int_from_config(payload, "tool_output_preview_rows"),
            tool_output_inline_original_chars=_int_from_config(payload, "tool_output_inline_original_chars"),
        )


def load_deep_agent_settings(config_path: str | Path | None = None) -> DeepAgentSettings:
    """Загружает настройки DeepAgent из JSON-файла.

    Args:
        config_path: Необязательный путь к JSON-конфигу. Если путь не передан,
            используется переменная окружения ``DEEP_AGENT_CONFIG_PATH`` или
            стандартный ``deep_agent_test/config/defaults.json``.

    Returns:
        Объект ``DeepAgentSettings`` с абсолютными путями.
    """

    payload = _load_config_payload(config_path)
    return DeepAgentSettings.from_mapping(payload)


def _load_config_payload(config_path: str | Path | None = None) -> dict[str, Any]:
    """Загружает итоговый payload настроек из стандартного JSON и пользовательского override.

    Args:
        config_path: Необязательный путь к пользовательскому JSON-конфигу.

    Returns:
        Словарь настроек. Пользовательский конфиг переопределяет ключи из
        ``defaults.json``, поэтому дефолты живут в JSON, а не в Python-коде.
    """

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
    """Читает JSON-файл настроек.

    Args:
        path: Путь к JSON-файлу.

    Returns:
        Словарь настроек.

    Raises:
        FileNotFoundError: Если файл настроек не найден.
        ValueError: Если JSON содержит не объект.
    """

    resolved_path = path.resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain JSON object: {resolved_path}")
    return payload


def _validate_required_config_keys(payload: dict[str, Any]) -> None:
    """Проверяет, что итоговый конфиг содержит все обязательные ключи.

    Args:
        payload: Итоговый словарь настроек после применения пользовательских переопределений.

    Returns:
        None.

    Raises:
        ValueError: Если отсутствует хотя бы один обязательный ключ.
    """

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in payload]
    if missing_keys:
        raise ValueError(f"DeepAgent config missing required keys: {', '.join(missing_keys)}")


def _resolve_project_path(value: Any, project_root: Path) -> Path:
    """Преобразует путь из конфига в абсолютный путь.

    Args:
        value: Значение пути из JSON-конфига.
        project_root: Абсолютный путь к корню проекта.

    Returns:
        Абсолютный путь. Относительные пути считаются относительно ``project_root``.
    """

    path = Path(str(value))
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _int_from_config(payload: dict[str, Any], key: str) -> int:
    """Читает целое число из JSON-конфига.

    Args:
        payload: Словарь настроек.
        key: Имя ключа с числом.

    Returns:
        Целое число из конфига.

    Raises:
        ValueError: Если значение отсутствует или не приводится к целому числу.
    """

    try:
        return int(payload[key])
    except (TypeError, ValueError):
        raise ValueError(f"Config key '{key}' must be an integer.") from None


def _bool_from_config(payload: dict[str, Any], key: str) -> bool:
    """Читает булево значение из JSON-конфига.

    Args:
        payload: Словарь настроек.
        key: Имя ключа с булевым значением.

    Returns:
        Булево значение из конфига.

    Raises:
        ValueError: Если значение не похоже на булево.
    """

    value = payload[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "да"}:
            return True
        if normalized in {"0", "false", "no", "n", "нет"}:
            return False
    raise ValueError(f"Config key '{key}' must be a boolean.")


def _dict_from_config(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Читает словарь из JSON-конфига.

    Args:
        payload: Словарь настроек.
        key: Имя ключа со словарем.

    Returns:
        Копия словаря из конфига.

    Raises:
        ValueError: Если значение не является JSON-объектом.
    """

    value = payload[key]
    if isinstance(value, dict):
        return dict(value)
    raise ValueError(f"Config key '{key}' must be an object.")


def _optional_str_from_config(payload: dict[str, Any], key: str) -> str | None:
    """Читает необязательную строку из JSON-конфига.

    Args:
        payload: Словарь настроек.
        key: Имя ключа со строкой или null.

    Returns:
        Непустая строка или ``None``.

    Raises:
        ValueError: Если значение не является строкой или null.
    """

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

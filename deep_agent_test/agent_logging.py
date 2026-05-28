"""Файловое логирование событий аналитического DeepAgent.

Содержит:
- DeepAgentLogFiles: набор файлов логирования.
- DeepAgentEventLogger: простой JSONL-логгер событий агента.
- DeepAgentEventLogger.__init__: инициализация путей JSONL-логов.
- DeepAgentEventLogger.log_event: запись события в общий и отдельный лог.
- DeepAgentEventLogger.log_tool_event: запись события tool call.
- DeepAgentEventLogger.log_loaded_skills: запись загруженных skills.
- DeepAgentEventLogger.log_few_shot_selection: запись выбранных few-shot примеров.
- DeepAgentEventLogger._append_jsonl: добавление JSON-строки в файл.
- build_deep_agent_logger: создание логгера из настроек агента.
- _utc_now_iso: формирование UTC timestamp.
- _json_default: сериализация нестандартных объектов в JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deep_agent_test.settings import DeepAgentSettings


@dataclass(frozen=True)
class DeepAgentLogFiles:
    """Пути к файлам логирования DeepAgent.

    Args:
        events: Общий JSONL-файл со всеми событиями.
        tool_calls: Отдельный JSONL-файл с вызовами инструментов.
        loaded_skills: Отдельный JSONL-файл с загруженными skills.
        selected_few_shot: Отдельный JSONL-файл с выбранными few-shot примерами.

    Returns:
        Набор абсолютных путей к log-файлам.
    """

    events: Path
    tool_calls: Path
    loaded_skills: Path
    selected_few_shot: Path


class DeepAgentEventLogger:
    """Пишет события агента в общий JSONL-файл и в отдельные файлы по категориям.

    Args:
        log_dir: Папка, в которой будут созданы log-файлы.
        enabled: Нужно ли фактически записывать события.

    Returns:
        Логгер без вывода в консоль.
    """

    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        """Инициализирует файловый логгер агента.

        Args:
            log_dir: Папка для JSONL-логов.
            enabled: Флаг включения записи логов.

        Returns:
            None.
        """

        self.log_dir = log_dir.resolve()
        self.enabled = enabled
        self.files = DeepAgentLogFiles(
            events=self.log_dir / "events.jsonl",
            tool_calls=self.log_dir / "tool_calls.jsonl",
            loaded_skills=self.log_dir / "loaded_skills.jsonl",
            selected_few_shot=self.log_dir / "selected_few_shot.jsonl",
        )

    def log_event(self, event_type: str, payload: dict[str, Any], category_path: Path | None = None) -> None:
        """Записывает событие в общий лог и, при необходимости, в отдельный файл.

        Args:
            event_type: Тип события, например ``tool_start`` или ``skills_loaded``.
            payload: JSON-сериализуемые данные события.
            category_path: Необязательный путь к отдельному JSONL-файлу категории.

        Returns:
            None.
        """

        if not self.enabled:
            return

        record = {
            "timestamp": _utc_now_iso(),
            "event_type": event_type,
            **payload,
        }
        self._append_jsonl(self.files.events, record)
        if category_path is not None:
            self._append_jsonl(category_path, record)

    def log_tool_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Записывает событие tool call в общий и отдельный tool log.

        Args:
            event_type: Тип события tool call.
            payload: Данные события tool call.

        Returns:
            None.
        """

        self.log_event(event_type, payload, self.files.tool_calls)

    def log_loaded_skills(self, payload: dict[str, Any]) -> None:
        """Записывает сведения о предварительно загруженных skills.

        Args:
            payload: Данные о путях skills и размере загруженного context.

        Returns:
            None.
        """

        self.log_event("skills_loaded", payload, self.files.loaded_skills)

    def log_few_shot_selection(self, payload: dict[str, Any]) -> None:
        """Записывает сведения о выбранных few-shot примерах.

        Args:
            payload: Пользовательский запрос, кандидаты и выбранные примеры.

        Returns:
            None.
        """

        self.log_event("few_shot_selected", payload, self.files.selected_few_shot)

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        """Добавляет одну JSON-строку в файл.

        Args:
            path: Целевой JSONL-файл.
            record: Словарь события.

        Returns:
            None.
        """

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            file.write("\n")


def build_deep_agent_logger(settings: DeepAgentSettings) -> DeepAgentEventLogger:
    """Создает файловый логгер из настроек DeepAgent.

    Args:
        settings: Настройки агента с папкой ``logs_dir``.

    Returns:
        Объект ``DeepAgentEventLogger``.
    """

    return DeepAgentEventLogger(log_dir=settings.logs_dir)


def _utc_now_iso() -> str:
    """Возвращает текущий UTC timestamp в ISO-формате.

    Args:
        Отсутствуют.

    Returns:
        Строка timestamp с timezone UTC.
    """

    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Сериализует нестандартные объекты для JSON-логов.

    Args:
        value: Значение, которое стандартный JSON encoder не смог сериализовать.

    Returns:
        JSON-совместимое представление значения.
    """

    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


__all__ = [
    "DeepAgentEventLogger",
    "DeepAgentLogFiles",
    "build_deep_agent_logger",
]

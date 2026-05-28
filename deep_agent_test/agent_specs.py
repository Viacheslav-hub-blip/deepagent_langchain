"""Контракт supervisor-а, subagents и HITL-действий аналитического DeepAgent.

Содержит:
- CLARIFY_ANALYSIS_REQUEST_TOOL: имя tool уточнения пользовательского запроса.
- DATA_RETRIEVAL_AGENT_NAME: имя subagent-а чтения данных.
- PYTHON_ANALYSIS_AGENT_NAME: имя subagent-а Python-анализа.
- build_clarify_interrupt_config: сборка Human-in-the-loop правила для уточняющего вопроса.
- build_analytics_subagent_specs: сборка спецификаций subagents для ``create_deep_agent``.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from deep_agent_test.prompts import DATA_RETRIEVAL_PROMPT, PYTHON_ANALYSIS_PROMPT
from deep_agent_test.settings import DeepAgentSettings

CLARIFY_ANALYSIS_REQUEST_TOOL = "clarify_analysis_request"
DATA_RETRIEVAL_AGENT_NAME = "data-retrieval-agent"
PYTHON_ANALYSIS_AGENT_NAME = "python-analysis-agent"

DATA_RETRIEVAL_AGENT_DESCRIPTION = (
    "Читает фактические строки данных через read_table по таблицам, которые описаны в skills. "
    "Используй этого subagent-а для выборок по конкретным полям, фильтрам, ключам и периоду. "
    "Не используй его для расчетов, join, отчетов или создания файлов."
)
PYTHON_ANALYSIS_AGENT_DESCRIPTION = (
    "Выполняет Python-анализ по уже полученным данным: объединение источников, сортировку, "
    "нормализацию, расчеты, подготовку CSV/отчетов и сохранение готового файла."
)

SubagentStatus = Literal["success", "partial", "empty", "needs_more_input", "schema_error", "error"]


class DataRetrievalResponse(BaseModel):
    """Структурированный ответ data-retrieval-agent для supervisor-а."""

    status: SubagentStatus = Field(description="Статус выполнения шага чтения данных.")
    rows_count: int = Field(default=0, description="Количество найденных строк.")
    tables_used: list[str] = Field(default_factory=list, description="Список реально использованных таблиц.")
    fields_used: list[str] = Field(default_factory=list, description="Список реально использованных полей.")
    filters_used: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Список реально примененных фильтров.",
    )
    period_used: str = Field(default="", description="Период или формулировка точечного поиска.")
    key_values_for_next_step: dict[str, Any] = Field(
        default_factory=dict,
        description="Ключевые значения для следующего шага, выбранные по skills или фактическому результату чтения.",
    )
    missing_required_inputs: list[str] = Field(
        default_factory=list,
        description="Какие обязательные входные значения отсутствуют.",
    )
    limitations: list[str] = Field(default_factory=list, description="Ограничения результата.")
    preview_rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Небольшой preview строк для supervisor-а.",
    )
    summary: str = Field(default="", description="Краткий итог шага на русском языке.")
    target_field_found: bool | None = Field(
        default=None,
        description="Найдено ли целевое поле текущего пользовательского шага в выбранной таблице.",
    )
    routing_keys_found: bool | None = Field(
        default=None,
        description="Удалось ли получить ключи маршрутизации для следующего шага.",
    )

    @model_validator(mode="after")
    def normalize_success_payload(self) -> "DataRetrievalResponse":
        """Нормализует success-ответ без бесконечного retry structured output."""

        if self.status == "success" and self.rows_count > 0:
            has_non_empty_key_values = _has_non_empty_dict(self.key_values_for_next_step)
            has_non_empty_preview = any(_has_non_empty_dict(row) for row in self.preview_rows)
            if not has_non_empty_key_values and has_non_empty_preview:
                self.key_values_for_next_step = _extract_routing_keys(self.preview_rows)
                has_non_empty_key_values = _has_non_empty_dict(self.key_values_for_next_step)
            if not has_non_empty_key_values or not has_non_empty_preview:
                self.status = "partial"
                if self.routing_keys_found is True and not has_non_empty_key_values:
                    self.routing_keys_found = False
                self.limitations.append(
                    "Structured output subagent-а неполный: отсутствуют фактические "
                    "key_values_for_next_step или preview_rows. Проверь полный read_table output в tool trace."
                )
        return self


def _extract_routing_keys(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Извлекает ключи маршрутизации из первой непустой preview-строки.

    Args:
        rows: Preview-строки, возвращенные ``data-retrieval-agent``.

    Returns:
        Словарь ключей, которые можно использовать для следующего шага.
    """

    for row in rows:
        if not _has_non_empty_dict(row):
            continue
        return {key: value for key, value in row.items() if value not in (None, "", [], {}, ())}
    return {}


def _has_non_empty_dict(value: Any) -> bool:
    """Проверяет, что значение является словарем с хотя бы одним непустым значением.

    Args:
        value: Проверяемое значение произвольного типа.

    Returns:
        ``True``, если значение является словарем и содержит хотя бы одно поле
        с непустым значением; иначе ``False``.
    """

    if not isinstance(value, dict):
        return False
    return any(item not in (None, "", [], {}, ()) for item in value.values())


class PythonAnalysisResponse(BaseModel):
    """Структурированный ответ python-analysis-agent для supervisor-а."""

    status: SubagentStatus = Field(description="Статус аналитического шага.")
    summary: str = Field(default="", description="Краткий итог анализа.")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Ключевые рассчитанные метрики.")
    tables: dict[str, Any] = Field(default_factory=dict, description="Сведения об итоговых таблицах/shape/preview.")
    files: list[dict[str, Any]] = Field(default_factory=list, description="Сохраненные файлы и их метаданные.")
    artifacts: list[dict[str, Any]] = Field(default_factory=list, description="Промежуточные artifacts, если есть.")
    limitations: list[str] = Field(default_factory=list, description="Ограничения результата.")
    errors: list[str] = Field(default_factory=list, description="Ошибки и условия retry.")


def build_clarify_interrupt_config() -> dict[str, dict[str, Any]]:
    """Собирает конфигурацию interrupt для уточняющего вопроса supervisor-а.

    Args:
        Отсутствуют.

    Returns:
        Словарь ``interrupt_on`` для ``create_deep_agent``. Пользователь отвечает
        текстом через решение ``respond``.
    """

    return {
        CLARIFY_ANALYSIS_REQUEST_TOOL: {
            "allowed_decisions": ["respond"],
            "description": "Опишите задачу анализа, ожидаемый результат, формат ответа и обязательные элементы.",
        },
    }


def build_analytics_subagent_specs(
    *,
    settings: DeepAgentSettings,
    data_tools: list[Any],
    analysis_tools: list[Any],
    common_middleware: list[Any],
) -> list[dict[str, Any]]:
    """Собирает спецификации subagents аналитического DeepAgent.

    Args:
        settings: Настройки агента с виртуальной папкой skills.
        data_tools: Tools чтения данных для ``data-retrieval-agent``.
        analysis_tools: Tools расчетов и сохранения файлов для ``python-analysis-agent``.
        common_middleware: Middleware, которые должны работать внутри обоих subagents.

    Returns:
        Список спецификаций subagents в формате ``create_deep_agent``.
    """

    return [
        {
            "name": DATA_RETRIEVAL_AGENT_NAME,
            "description": DATA_RETRIEVAL_AGENT_DESCRIPTION,
            "system_prompt": DATA_RETRIEVAL_PROMPT,
            "tools": data_tools,
            # Для чтения данных response_format отключен намеренно: DeepAgent возвращает supervisor-у
            # только финальный structured output subagent-а, и при пустом preview_rows теряются строки read_table.
            "skills": [settings.skills_virtual_dir],
            "middleware": common_middleware,
        },
        {
            "name": PYTHON_ANALYSIS_AGENT_NAME,
            "description": PYTHON_ANALYSIS_AGENT_DESCRIPTION,
            "system_prompt": PYTHON_ANALYSIS_PROMPT,
            "tools": analysis_tools,
            "response_format": PythonAnalysisResponse,
            "skills": [settings.skills_virtual_dir],
            "middleware": common_middleware,
        },
    ]


__all__ = [
    "CLARIFY_ANALYSIS_REQUEST_TOOL",
    "DATA_RETRIEVAL_AGENT_NAME",
    "PYTHON_ANALYSIS_AGENT_NAME",
    "build_analytics_subagent_specs",
    "build_clarify_interrupt_config",
]

"""HTTP-схемы запросов и ответов для ``planner_agent.http_api``.

Содержит:
- ApiHealth: ответ health-check endpoint.
- ArtifactTextResponse: ответ чтения текстового artifact.
- AgentInvokeRequest: запрос запуска агента из UI.
- AgentRunResponse: ответ запуска агента из UI.
- BranchCreatedResponse: ответ создания branch metadata.
- DialogContextRequest: запрос сборки dialog context поверх существующих runs.
- DialogContextPreviewResponse: ответ preview для dialog context.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from planner_agent.schemas.lineage import ResearchRun
from planner_agent.services.dialog_context_service import ContextRunRef, DialogContext
from planner_agent.services.run_inspection_service import RunResult
from planner_agent.schemas.skills import SkillRecord


class ApiHealth(BaseModel):
    """Состояние API приложения.

    Args:
        status: Технический статус API.
        service: Имя сервиса.

    Returns:
        Краткий health-check ответ.
    """

    status: str = Field(description="Технический статус API.")
    service: str = Field(description="Имя сервиса.")


class ArtifactTextResponse(BaseModel):
    """Текстовое содержимое artifact для UI.

    Args:
        run_id: Идентификатор ResearchRun.
        artifact_id: Идентификатор artifact.
        content: Прочитанный текст или ``None``, если artifact недоступен как текст.
        max_chars: Запрошенный лимит символов или ``None``.
        truncated: Признак возможного обрезания по ``max_chars``.

    Returns:
        Текстовый payload artifact для UI.
    """

    run_id: str = Field(description="Идентификатор ResearchRun.")
    artifact_id: str = Field(description="Идентификатор artifact.")
    content: str | None = Field(
        default=None,
        description="Текстовое содержимое artifact или ``None``.",
    )
    max_chars: int | None = Field(
        default=None,
        description="Запрошенный лимит символов.",
    )
    truncated: bool = Field(description="Был ли ответ потенциально обрезан.")


class AgentInvokeRequest(BaseModel):
    """Запрос запуска агента через UI API.

    Args:
        user_query: Основной пользовательский запрос.
        session_id: Идентификатор внешней сессии.
        user_id: Идентификатор пользователя.
        filesystem_context: Дополнительный контекст рабочих директорий и файлов.
        context_runs: Необязательные прошлые ResearchRun для follow-up диалога.

    Returns:
        Валидированный запрос запуска агента.
    """

    user_query: str = Field(description="Основной пользовательский запрос.")
    session_id: str = Field(default="", description="Идентификатор внешней сессии.")
    user_id: str | None = Field(default=None, description="Идентификатор пользователя.")
    filesystem_context: dict[str, str] = Field(
        default_factory=dict,
        description="Дополнительный контекст рабочих файлов и директорий.",
    )
    context_runs: list[ContextRunRef] = Field(
        default_factory=list,
        description="Существующие runs, доступные follow-up запуску.",
    )


class AgentRunResponse(BaseModel):
    """Ответ API после запуска агента.

    Args:
        run_id: Идентификатор созданного или продолженного ResearchRun.
        messages: Сериализованные LangChain messages финального ответа.
        result: Read-only результат запуска, если он доступен в хранилище.

    Returns:
        Данные, достаточные UI для перехода к run graph и отображения ответа.
    """

    run_id: str = Field(description="Идентификатор ResearchRun.")
    messages: list[dict[str, object]] = Field(
        default_factory=list,
        description="Сериализованные LangChain messages.",
    )
    result: RunResult | None = Field(
        default=None,
        description="Полный read-only результат запуска или ``None``.",
    )


class AgentLiveRunResponse(BaseModel):
    """Ответ мгновенного запуска агента для live UI.

    Args:
        run_id: Идентификатор ResearchRun, который уже можно читать через graph endpoints.
        run: Созданная запись ResearchRun.

    Returns:
        Данные, достаточные UI для polling `/runs/{run_id}/graph`.
    """

    run_id: str = Field(description="Идентификатор ResearchRun.")
    run: ResearchRun = Field(description="Созданная запись ResearchRun.")


class BranchCreatedResponse(BaseModel):
    """Ответ API после создания branch metadata.

    Args:
        run: Созданный branch ResearchRun.
        branch_started_node_id: Идентификатор первого node ветки.

    Returns:
        Информация, достаточная UI для перехода к новой ветке.
    """

    run: ResearchRun = Field(description="Созданный branch ResearchRun.")
    branch_started_node_id: str | None = Field(
        default=None,
        description="Идентификатор branch_started node, если он найден.",
    )


class DialogContextRequest(BaseModel):
    """Запрос сборки context для чатового follow-up поверх существующих runs.

    Args:
        user_query: Текущий пользовательский follow-up запрос. Поле не используется
            для поиска, но возвращается в ответе, чтобы UI мог проверить полный
            пакет будущего запуска.
        context_runs: Список существующих ResearchRun, которые нужно сделать
            доступными агенту.

    Returns:
        Запрос на preview dialog context.
    """

    user_query: str = Field(
        default="",
        description="Текущий follow-up запрос пользователя.",
    )
    context_runs: list[ContextRunRef] = Field(
        default_factory=list,
        description="Существующие runs, доступные follow-up запуску.",
    )


class DialogContextPreviewResponse(BaseModel):
    """Ответ preview собранного dialog context.

    Args:
        user_query: Пользовательский follow-up запрос из входного payload.
        context: Собранный DialogContext.

    Returns:
        Preview context, который будет добавлен агенту при запуске с context_runs.
    """

    user_query: str = Field(description="Текущий follow-up запрос пользователя.")
    context: DialogContext = Field(description="Собранный dialog context.")


class SkillListView(BaseModel):
    """Ответ списка доступных skills.

    Args:
        skills: Список записей доступных skills с метаданными.

    Returns:
        Набор данных для UI отображения библиотеки skills.
    """

    skills: list[SkillRecord] = Field(
        default_factory=list,
        description="Список доступных skills.",
    )


class SkillViewResponse(BaseModel):
    """Полное содержимое skill по имени.

    Args:
        name: Имя skill.
        content: Полный текст SKILL.md.

    Returns:
        Минимальный payload для просмотра/редактирования skill.
    """

    name: str = Field(description="Имя skill.")
    content: str = Field(description="Полное содержимое SKILL.md.")


class SkillCreateRequest(BaseModel):
    """Запрос создания нового skill.

    Args:
        name: Имя skill (имя директории).
        content: Полное содержимое SKILL.md.

    Returns:
        Валидированный запрос на создание skill.
    """

    name: str = Field(description="Имя skill (имя директории).")
    content: str = Field(description="Полное содержимое SKILL.md.")


class SkillCreateResponse(BaseModel):
    """Ответ на успешное создание skill.

    Args:
        name: Имя созданного skill.
        message: Текстовое подтверждение.

    Returns:
        Подтверждение создания skill.
    """

    name: str = Field(description="Имя созданного skill.")
    message: str = Field(default="Skill created successfully.")


__all__ = [
    "AgentInvokeRequest",
    "AgentLiveRunResponse",
    "AgentRunResponse",
    "ApiHealth",
    "ArtifactTextResponse",
    "BranchCreatedResponse",
    "DialogContextPreviewResponse",
    "DialogContextRequest",
    "SkillCreateRequest",
    "SkillCreateResponse",
    "SkillListView",
    "SkillViewResponse",
]

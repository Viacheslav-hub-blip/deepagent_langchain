"""FastAPI приложение для HTTP-доступа к research-agent (runs, invoke, branch).

Содержит:
- create_app: factory приложения с read-only endpoints для runs, nodes и artifacts.
- invoke_agent_run: endpoint запуска нового или follow-up ResearchRun.
- invoke_branch_run: endpoint создания и запуска branch ResearchRun.
- build_dialog_context: endpoint preview dialog context для follow-up запусков.
- _services: извлечение контейнера сервисов из состояния приложения.
- _require_agent: проверка наличия агента в API services.
- _build_run_response: сбор ответа запуска агента.
- _serialize_messages: сериализация LangChain messages.
- _not_found: единый HTTP 404 ответ.
- _service_unavailable: единый HTTP 503 ответ.
- _branch_started_node_id: поиск первого branch_started node созданной ветки.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from langchain_core.messages import BaseMessage

from planner_agent.chat_runner import build_chat_initial_state
from planner_agent.models import AgentState
from planner_agent.schemas.artifacts import Artifact
from planner_agent.schemas.lineage import BranchRequest, ResearchRun, StateNode
from planner_agent.services.dialog_context_service import DialogContextService
from planner_agent.services.run_inspection_service import (
    ArtifactContentPreview,
    ArtifactDetails,
    NodeDetails,
    NodeInspectorView,
    RunGraph,
    RunResult,
    RunSummary,
)

from planner_agent.services.skills_service import SkillsService

from .config import ApiServices, ApiSettings, build_api_services
from .schemas import (
    AgentInvokeRequest,
    AgentLiveRunResponse,
    AgentRunResponse,
    ApiHealth,
    ArtifactTextResponse,
    BranchCreatedResponse,
    DialogContextPreviewResponse,
    DialogContextRequest,
    SkillCreateRequest,
    SkillCreateResponse,
    SkillListView,
    SkillViewResponse,
)


def create_app(
        *,
        settings: ApiSettings | None = None,
        services: ApiServices | None = None,
) -> FastAPI:
    """Создает FastAPI приложение для research-agent SDK.

    Args:
        settings: Настройки путей и API-префикса. Если ``None``, используются
            значения по умолчанию.
        services: Готовый контейнер сервисов. Используется в тестах и внешних
            приложениях, которым нужно переиспользовать уже созданные сервисы.

    Returns:
        Сконфигурированное FastAPI приложение.
    """

    resolved_settings = settings or ApiSettings()
    resolved_services = services or build_api_services(resolved_settings)
    if resolved_services.dialog_context_service is None:
        resolved_services.dialog_context_service = DialogContextService(
            resolved_services.inspection_service
        )
    app = FastAPI(
        title="research-agent API",
        version="0.1.0",
        description="Backend API для repository-like UI поверх ResearchRun.",
    )
    app.state.api_services = resolved_services

    prefix = resolved_settings.api_prefix.rstrip("/")

    def get_services() -> ApiServices:
        """Возвращает сервисы API из состояния FastAPI приложения.

        Returns:
            ApiServices, созданные при инициализации приложения.
        """

        return _services(app)

    @app.get(f"{prefix}/health", response_model=ApiHealth)
    def health() -> ApiHealth:
        """Возвращает технический статус API.

        Returns:
            ApiHealth со статусом ``ok``.
        """

        return ApiHealth(status="ok", service="research-agent-api")

    @app.get(f"{prefix}/runs", response_model=list[RunSummary])
    def list_runs(
            api: ApiServices = Depends(get_services),
    ) -> list[RunSummary]:
        """Возвращает список сохраненных ResearchRun .

        Args:
            api: Контейнер сервисов API.

        Returns:
            Список кратких сводок запусков.
        """

        return api.inspection_service.list_run_summaries()

    @app.post(f"{prefix}/runs/invoke", response_model=AgentRunResponse)
    async def invoke_agent_run(
            request: AgentInvokeRequest,
            api: ApiServices = Depends(get_services),
    ) -> AgentRunResponse:
        """Запускает новый или follow-up ResearchRun через переданный ResearchAgent.

        Args:
            request: Пользовательский запрос и необязательный context_runs.
            api: Контейнер сервисов API.

        Returns:
            AgentRunResponse с messages, run_id и RunResult.
        """

        agent = _require_agent(api)
        messages = await agent.ainvoke(
            {
                "user_query": request.user_query,
                "session_id": request.session_id,
                "user_id": request.user_id,
                "filesystem_context": request.filesystem_context,
                "context_runs": [
                    item.model_dump(mode="json") for item in request.context_runs
                ],
            }
        )
        return _build_run_response(api=api, messages=messages)

    @app.post(f"{prefix}/runs/live", response_model=AgentLiveRunResponse)
    async def start_live_agent_run(
            request: AgentInvokeRequest,
            api: ApiServices = Depends(get_services),
    ) -> AgentLiveRunResponse:
        """Создает ResearchRun сразу и запускает агента в background task.

        UI может немедленно читать ``/runs/{run_id}/graph`` и видеть новые
        lineage nodes по мере того, как ``agent.ainvoke`` записывает их в storage.

        Args:
            request: Пользовательский запрос и необязательный context_runs.
            api: Контейнер сервисов API.

        Returns:
            AgentLiveRunResponse с run_id для polling graph endpoints.
        """

        agent = _require_agent(api)
        run, state = _build_live_run_state(api=api, request=request)
        asyncio.create_task(
            _run_live_agent_task(
                api=api,
                agent=agent,
                state=state,
                run_id=run.run_id,
            )
        )
        return AgentLiveRunResponse(run_id=run.run_id, run=run)

    @app.get(f"{prefix}/runs/{{run_id}}", response_model=ResearchRun)
    def get_run(
            run_id: str,
            api: ApiServices = Depends(get_services),
    ) -> ResearchRun:
        """Возвращает одну запись ResearchRun.

        Args:
            run_id: Идентификатор запуска.
            api: Контейнер сервисов API.

        Returns:
            ResearchRun.
        """

        run = api.inspection_service.get_run(run_id)
        if run is None:
            raise _not_found("run_not_found")
        return run

    @app.get(f"{prefix}/runs/{{run_id}}/result", response_model=RunResult)
    def get_run_result(
            run_id: str,
            api: ApiServices = Depends(get_services),
            include_nodes: bool = True,
            include_artifacts: bool = True,
            include_final_state: bool = True,
    ) -> RunResult:
        """Возвращает полный read-only результат запуска.

        Args:
            run_id: Идентификатор запуска.
            api: Контейнер сервисов API.
            include_nodes: Включать ли lineage nodes.
            include_artifacts: Включать ли artifacts.
            include_final_state: Включать ли snapshot финального или последнего node.

        Returns:
            RunResult.
        """

        result = api.inspection_service.get_run_result(
            run_id,
            include_nodes=include_nodes,
            include_artifacts=include_artifacts,
            include_final_state=include_final_state,
        )
        if result is None:
            raise _not_found("run_not_found")
        return result

    @app.get(f"{prefix}/runs/{{run_id}}/graph", response_model=RunGraph)
    def get_run_graph(
            run_id: str,
            api: ApiServices = Depends(get_services),
    ) -> RunGraph:
        """Возвращает lineage graph запуска для экрана Run Graph.

        Args:
            run_id: Идентификатор запуска.
            api: Контейнер сервисов API.

        Returns:
            RunGraph.
        """

        graph = api.inspection_service.get_run_graph(run_id)
        if graph is None:
            raise _not_found("run_not_found")
        return graph

    @app.get(f"{prefix}/runs/{{run_id}}/nodes", response_model=list[StateNode])
    def list_nodes(
            run_id: str,
            api: ApiServices = Depends(get_services),
    ) -> list[StateNode]:
        """Возвращает nodes запуска.

        Args:
            run_id: Идентификатор запуска.
            api: Контейнер сервисов API.

        Returns:
            Список StateNode.
        """

        if api.inspection_service.get_run(run_id) is None:
            raise _not_found("run_not_found")
        return api.inspection_service.list_nodes(run_id)

    @app.get(f"{prefix}/runs/{{run_id}}/nodes/{{node_id}}", response_model=NodeDetails)
    def get_node_details(
            run_id: str,
            node_id: str,
            api: ApiServices = Depends(get_services),
            include_snapshot: bool = True,
    ) -> NodeDetails:
        """Возвращает базовые детали node.

        Args:
            run_id: Идентификатор запуска.
            node_id: Идентификатор lineage node.
            api: Контейнер сервисов API.
            include_snapshot: Загружать ли snapshot node.

        Returns:
            NodeDetails.
        """

        details = api.inspection_service.get_node_details(
            run_id,
            node_id,
            include_snapshot=include_snapshot,
        )
        if details is None:
            raise _not_found("node_not_found")
        return details

    @app.get(
        f"{prefix}/runs/{{run_id}}/nodes/{{node_id}}/inspector",
        response_model=NodeInspectorView,
    )
    def get_node_inspector(
            run_id: str,
            node_id: str,
            api: ApiServices = Depends(get_services),
            include_snapshot: bool = True,
            preview_chars: int = Query(default=4_000, ge=0),
            snapshot_preview_chars: int = Query(default=1_000, ge=0),
    ) -> NodeInspectorView:
        """Возвращает модель Node Inspector для UI.

        Args:
            run_id: Идентификатор запуска.
            node_id: Идентификатор lineage node.
            api: Контейнер сервисов API.
            include_snapshot: Загружать ли полный snapshot.
            preview_chars: Лимит preview для artifacts.
            snapshot_preview_chars: Лимит preview для секций snapshot.

        Returns:
            NodeInspectorView.
        """

        view = api.inspection_service.get_node_inspector_view(
            run_id,
            node_id,
            include_snapshot=include_snapshot,
            preview_chars=preview_chars,
            snapshot_preview_chars=snapshot_preview_chars,
        )
        if view is None:
            raise _not_found("node_not_found")
        return view

    @app.get(f"{prefix}/runs/{{run_id}}/artifacts", response_model=list[Artifact])
    def list_artifacts(
            run_id: str,
            api: ApiServices = Depends(get_services),
    ) -> list[Artifact]:
        """Возвращает artifacts запуска.

        Args:
            run_id: Идентификатор запуска.
            api: Контейнер сервисов API.

        Returns:
            Список Artifact.
        """

        if api.inspection_service.get_run(run_id) is None:
            raise _not_found("run_not_found")
        return api.inspection_service.list_artifacts(run_id)

    @app.get(
        f"{prefix}/runs/{{run_id}}/artifacts/{{artifact_id}}",
        response_model=ArtifactDetails,
    )
    def get_artifact_details(
            run_id: str,
            artifact_id: str,
            api: ApiServices = Depends(get_services),
            preview_chars: int = Query(default=4_000, ge=0),
    ) -> ArtifactDetails:
        """Возвращает детали artifact с безопасным preview.

        Args:
            run_id: Идентификатор запуска.
            artifact_id: Идентификатор artifact.
            api: Контейнер сервисов API.
            preview_chars: Лимит preview.

        Returns:
            ArtifactDetails.
        """

        details = api.inspection_service.get_artifact_details(
            run_id,
            artifact_id,
            preview_chars=preview_chars,
        )
        if details is None:
            raise _not_found("artifact_not_found")
        return details

    @app.get(
        f"{prefix}/runs/{{run_id}}/artifacts/{{artifact_id}}/preview",
        response_model=ArtifactContentPreview,
    )
    def preview_artifact(
            run_id: str,
            artifact_id: str,
            api: ApiServices = Depends(get_services),
            preview_chars: int = Query(default=4_000, ge=0),
    ) -> ArtifactContentPreview:
        """Возвращает только preview artifact.

        Args:
            run_id: Идентификатор запуска.
            artifact_id: Идентификатор artifact.
            api: Контейнер сервисов API.
            preview_chars: Лимит preview.

        Returns:
            ArtifactContentPreview.
        """

        preview = api.inspection_service.preview_artifact(
            run_id,
            artifact_id,
            preview_chars=preview_chars,
        )
        if preview is None:
            raise _not_found("artifact_not_found")
        return preview

    @app.get(
        f"{prefix}/runs/{{run_id}}/artifacts/{{artifact_id}}/text",
        response_model=ArtifactTextResponse,
    )
    def read_artifact_text(
            run_id: str,
            artifact_id: str,
            api: ApiServices = Depends(get_services),
            max_chars: Annotated[int | None, Query(ge=0)] = None,
    ) -> ArtifactTextResponse:
        """Возвращает текст artifact, если artifact доступен как UTF-8 файл.

        Args:
            run_id: Идентификатор запуска.
            artifact_id: Идентификатор artifact.
            api: Контейнер сервисов API.
            max_chars: Опциональный лимит символов.

        Returns:
            ArtifactTextResponse.
        """

        artifact = api.inspection_service.get_artifact(run_id, artifact_id)
        if artifact is None:
            raise _not_found("artifact_not_found")
        content = api.inspection_service.read_artifact_text(
            run_id,
            artifact_id,
            max_chars=max_chars,
        )
        return ArtifactTextResponse(
            run_id=run_id,
            artifact_id=artifact_id,
            content=content,
            max_chars=max_chars,
            truncated=bool(max_chars is not None and content is not None and len(content) >= max_chars),
        )

    @app.get(f"{prefix}/runs/{{run_id}}/artifacts/{{artifact_id}}/file")
    def download_artifact_file(
            run_id: str,
            artifact_id: str,
            api: ApiServices = Depends(get_services),
    ) -> FileResponse:
        """Отдаёт файл artifact для скачивания (выгрузки данных и др.).

        Args:
            run_id: Идентификатор запуска.
            artifact_id: Идентификатор artifact.
            api: Контейнер сервисов API.

        Returns:
            FileResponse с ``Content-Disposition: attachment``.
        """

        path = api.inspection_service.artifact_download_path(run_id, artifact_id)
        if path is None:
            raise _not_found("artifact_not_found")
        artifact = api.inspection_service.get_artifact(run_id, artifact_id)
        filename = path.name
        media_type = "application/octet-stream"
        if artifact is not None:
            if artifact.mime_type:
                media_type = artifact.mime_type
            for key in ("original_filename", "filename", "export_filename"):
                meta = artifact.metadata or {}
                if isinstance(meta.get(key), str) and meta[key].strip():
                    filename = Path(meta[key]).name
                    break
        return FileResponse(
            path=str(path),
            filename=filename,
            media_type=media_type,
        )

    @app.post(f"{prefix}/branches", response_model=BranchCreatedResponse)
    def create_branch(
            request: BranchRequest,
            api: ApiServices = Depends(get_services),
    ) -> BranchCreatedResponse:
        """Создает branch metadata от выбранного node без запуска модели.

        Args:
            request: Описание source node, новой задачи и режима ветки.
            api: Контейнер сервисов API.

        Returns:
            BranchCreatedResponse с новым ResearchRun.
        """

        try:
            branch = api.lineage_service.branch_from(request)
        except FileNotFoundError as exc:
            raise _not_found(str(exc)) from exc
        return BranchCreatedResponse(
            run=branch,
            branch_started_node_id=_branch_started_node_id(api, branch.run_id),
        )

    @app.post(f"{prefix}/branches/invoke", response_model=AgentRunResponse)
    async def invoke_branch_run(
            request: BranchRequest,
            api: ApiServices = Depends(get_services),
    ) -> AgentRunResponse:
        """Создает branch metadata и сразу запускает ветку через переданный агент.

        Args:
            request: Описание source node, новой задачи и режима ветки.
            api: Контейнер сервисов API.

        Returns:
            AgentRunResponse с результатом branch run.
        """

        agent = _require_agent(api)
        try:
            messages = await agent.ainvoke_branch(request)
        except FileNotFoundError as exc:
            raise _not_found(str(exc)) from exc
        return _build_run_response(api=api, messages=messages)

    @app.post(f"{prefix}/dialog-context", response_model=DialogContextPreviewResponse)
    def build_dialog_context(
            request: DialogContextRequest,
            api: ApiServices = Depends(get_services),
    ) -> DialogContextPreviewResponse:
        """Собирает preview context для follow-up диалога без запуска агента.

        Args:
            request: Текущий запрос пользователя и список context runs.
            api: Контейнер сервисов API.

        Returns:
            DialogContextPreviewResponse с текстом, который будет доступен агенту.
        """

        if api.dialog_context_service is None:
            api.dialog_context_service = DialogContextService(api.inspection_service)
        context = api.dialog_context_service.build_context(request.context_runs)
        return DialogContextPreviewResponse(
            user_query=request.user_query,
            context=context,
        )

    # ---- Skills endpoints ----

    @app.get(f"{prefix}/skills", response_model=SkillListView)
    def list_skills(
            api: ApiServices = Depends(get_services),
    ) -> SkillListView:
        """Возвращает список всех доступных skills.

        Args:
            api: Контейнер сервисов API.

        Returns:
            SkillListView со списком записей skills.
        """

        skills_service = _require_skills_service(api)
        return SkillListView(skills=skills_service.skills_list())

    @app.get(f"{prefix}/skills/{{skill_name}}", response_model=SkillViewResponse)
    def get_skill(
            skill_name: str,
            api: ApiServices = Depends(get_services),
    ) -> SkillViewResponse:
        """Возвращает полное содержимое skill по имени.

        Args:
            skill_name: Имя skill.
            api: Контейнер сервисов API.

        Returns:
            SkillViewResponse с именем и текстом SKILL.md.

        Raises:
            HTTPException 404: Если skill не найден.
        """

        skills_service = _require_skills_service(api)
        result = skills_service.skill_view(skill_name)
        if not result.get("success"):
            raise _not_found("skill_not_found")
        return SkillViewResponse(
            name=skill_name,
            content=str(result.get("content", "")),
        )

    @app.post(f"{prefix}/skills", response_model=SkillCreateResponse, status_code=201)
    def create_skill(
            request: SkillCreateRequest,
            api: ApiServices = Depends(get_services),
    ) -> SkillCreateResponse:
        """Создает новый skill из переданного содержимого.

        Args:
            request: SkillCreateRequest с именем и содержимым.
            api: Контейнер сервисов API.

        Returns:
            SkillCreateResponse с именем созданного skill.

        Raises:
            HTTPException 409: Если skill с таким именем уже существует.
        """

        skills_service = _require_skills_service(api)

        # Проверка на дубликат
        existing = next(
            (s for s in skills_service.skills_list() if s.name == request.name),
            None,
        )
        if existing is not None:
            raise HTTPException(status_code=409, detail="skill_already_exists")

        frontmatter = (
            f"---\n"
            f"name: {request.name}\n"
            f"description: {request.name}\n"
            f"---\n\n"
        )
        skills_service.skill_create(
            name=request.name,
            content=frontmatter + request.content,
        )
        return SkillCreateResponse(name=request.name)

    @app.delete(f"{prefix}/skills/{{skill_name}}", status_code=204)
    def delete_skill(
            skill_name: str,
            api: ApiServices = Depends(get_services),
    ) -> None:
        """Удаляет skill по имени.

        Args:
            skill_name: Имя skill для удаления.
            api: Контейнер сервисов API.

        Raises:
            HTTPException 404: Если skill не найден.
        """

        skills_service = _require_skills_service(api)
        try:
            skills_service.skill_delete(skill_name)
        except FileNotFoundError as exc:
            raise _not_found(str(exc)) from exc

    return app


def _services(app: FastAPI) -> ApiServices:
    """Возвращает контейнер сервисов из состояния приложения.

    Args:
        app: FastAPI приложение.

    Returns:
        ApiServices.
    """

    return app.state.api_services


def _require_agent(api: ApiServices) -> Any:
    """Возвращает настроенный агент или выбрасывает HTTP 503.

    Args:
        api: Контейнер сервисов API.

    Returns:
        Объект агента с методами ``ainvoke`` и ``ainvoke_branch``.

    Raises:
        HTTPException: Если agent не был передан в ApiServices.
    """

    if api.agent is None:
        raise _service_unavailable("agent_not_configured")
    return api.agent


def _require_skills_service(api: ApiServices) -> SkillsService:
    """Возвращает настроенный skills service или выбрасывает HTTP 503.

    Args:
        api: Контейнер сервисов API.

    Returns:
        SkillsService.

    Raises:
        HTTPException: Если skills_service не был передан в ApiServices.
    """

    if api.skills_service is None:
        raise _service_unavailable("skills_service_not_configured")
    return api.skills_service


def _build_run_response(
        *,
        api: ApiServices,
        messages: list[BaseMessage],
) -> AgentRunResponse:
    """Собирает единый ответ после запуска агента.

    Args:
        api: Контейнер сервисов API.
        messages: Финальные LangChain messages, возвращенные агентом.

    Returns:
        AgentRunResponse с run_id, сериализованными messages и RunResult.
    """

    run_id = str(getattr(api.agent, "last_run_id", "") or "")
    result = api.inspection_service.get_run_result(run_id) if run_id else None
    if result is None and api.agent is not None and callable(getattr(api.agent, "get_run_result", None)):
        result = api.agent.get_run_result(run_id or None)
    return AgentRunResponse(
        run_id=run_id or (result.run.run_id if result is not None else ""),
        messages=_serialize_messages(messages),
        result=result,
    )


def _build_live_run_state(
        *,
        api: ApiServices,
        request: AgentInvokeRequest,
) -> tuple[ResearchRun, AgentState]:
    """Создает run до запуска агента и готовит state для live polling.

    Args:
        api: Контейнер сервисов API.
        request: Payload запуска агента.

    Returns:
        Пара ``(ResearchRun, AgentState)`` для background ``ainvoke``.
    """

    run = api.lineage_service.create_run(
        initial_user_query=request.user_query,
        session_id=request.session_id,
        user_id=request.user_id,
    )
    state = build_chat_initial_state(
        request.user_query,
        session_id=request.session_id,
        user_id=request.user_id,
        filesystem_context=request.filesystem_context,
    )
    if request.context_runs:
        if api.dialog_context_service is None:
            api.dialog_context_service = DialogContextService(api.inspection_service)
        dialog_context = api.dialog_context_service.build_context(request.context_runs)
        if dialog_context.rendered_context:
            state.ephemeral_recalls["dialog_context"] = dialog_context.rendered_context
            state.filesystem_context["dialog_context"] = dialog_context.rendered_context

    state = state.model_copy(
        update={
            "run_id": run.run_id,
            "initial_user_query": request.user_query,
        },
        deep=True,
    )
    run_started_node = api.lineage_service.create_state_node(
        run_id=run.run_id,
        node_type="run_started",
        title="Run started",
        status="succeeded",
        summary=request.user_query[:500],
        state=state,
        created_by="system",
        metadata={
            "session_id": request.session_id,
            "user_id": request.user_id,
            "started_by": "live_api",
        },
    )
    return run, state.model_copy(
        update={
            "current_node_id": run_started_node.node_id,
            "parent_node_ids": [run_started_node.node_id],
            "lineage_events": [run_started_node.model_dump(mode="json")],
        },
        deep=True,
    )


async def _run_live_agent_task(
        *,
        api: ApiServices,
        agent: Any,
        state: AgentState,
        run_id: str,
) -> None:
    """Выполняет background ainvoke и проставляет итоговый статус run."""

    try:
        await agent.ainvoke(state)
    except Exception as exc:  # pragma: no cover - background safety net
        _mark_live_run_finished(api=api, run_id=run_id, status="failed", error=str(exc))
        return
    _mark_live_run_finished(api=api, run_id=run_id, status="succeeded")


def _mark_live_run_finished(
        *,
        api: ApiServices,
        run_id: str,
        status: str,
        error: str | None = None,
) -> None:
    """Обновляет статус run после background выполнения."""

    run = api.lineage_service.get_run(run_id)
    if run is None:
        return
    run.status = status  # type: ignore[assignment]
    if error:
        run.metadata = {**run.metadata, "live_error": error}
    api.lineage_service.update_run(run)


def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, object]]:
    """Сериализует LangChain messages для HTTP ответа.

    Args:
        messages: Список LangChain BaseMessage.

    Returns:
        Список JSON-совместимых словарей.
    """

    serialized: list[dict[str, object]] = []
    for message in messages:
        if isinstance(message, BaseMessage):
            payload = message.model_dump(mode="json")
            payload.setdefault("type", message.type)
            serialized.append(payload)
            continue
        serialized.append({"type": "unknown", "content": str(message)})
    return serialized


def _not_found(detail: str) -> HTTPException:
    """Создает единый HTTP 404 ответ.

    Args:
        detail: Машиночитаемый код или краткое описание ошибки.

    Returns:
        HTTPException со статусом 404.
    """

    return HTTPException(status_code=404, detail=detail)


def _service_unavailable(detail: str) -> HTTPException:
    """Создает единый HTTP 503 ответ.

    Args:
        detail: Машиночитаемый код или краткое описание ошибки.

    Returns:
        HTTPException со статусом 503.
    """

    return HTTPException(status_code=503, detail=detail)


def _branch_started_node_id(api: ApiServices, run_id: str) -> str | None:
    """Находит branch_started node новой ветки.

    Args:
        api: Контейнер сервисов API.
        run_id: Идентификатор branch run.

    Returns:
        Идентификатор branch_started node или ``None``.
    """

    node = next(
        (
            candidate
            for candidate in api.inspection_service.list_nodes(run_id)
            if candidate.node_type == "branch_started"
        ),
        None,
    )
    return node.node_id if node is not None else None


__all__ = ["create_app"]

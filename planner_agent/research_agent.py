"""Backend facade для запуска research-agent как LangChain Runnable.

Содержит:
- ResearchAgentInput: входная схема для LangChain Runnable.
- ResearchAgent: Runnable-совместимый объект агента.
- invoke_branch: синхронное создание ветки и продолжение исследования.
- ainvoke_branch: асинхронное создание ветки и продолжение исследования.
- ContextRunRef: необязательная ссылка на существующий run для follow-up контекста.
- get_node_inspector_view: read API для экрана Node Inspector через ResearchAgent.
- get_artifact_details: read API для подробностей artifact через ResearchAgent.
- preview_artifact: read API для preview artifact через ResearchAgent.
- _resolve_directory: нормализация директорий относительно workspace.
- _selected_run_id: выбор run_id для read API.
- _coerce_input_payload: нормализация входа Runnable.
- _coerce_agent_state: нормализация результата graph в AgentState.
- _run_coro_sync: запуск coroutine из синхронного invoke.
- _normalize_batch_config: нормализация config для batch/abatch.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .chat_runner import build_chat_initial_state, run_agent_from_state
from .factory import planner_agent
from .models import AgentState
from .prompts import AnalysisAgentPrompts
from .schemas.artifacts import Artifact
from .schemas.lineage import BranchRequest, ResearchRun
from .services.artifact_service import ArtifactService
from .services.branch_resume_service import BranchResumeService
from .services.dialog_context_service import ContextRunRef, DialogContextService
from .services.lineage_service import LineageService
from .services.memory_service import MemoryService
from .services.run_inspection_service import (
    ArtifactContentPreview,
    ArtifactDetails,
    NodeDetails,
    NodeInspectorView,
    RunGraph,
    RunInspectionService,
    RunResult,
    RunSummary,
)
from .services.skills_service import SkillsService
from .tools.registry import ToolRegistry


class ResearchAgentInput(BaseModel):
    """Входные данные research-agent в стиле LangChain Runnable."""

    user_query: str = Field(
        default="",
        description="Исходный пользовательский запрос для нового запуска.",
    )
    session_id: str = Field(
        default="",
        description="Идентификатор внешней сессии, если он известен.",
    )
    user_id: str | None = Field(
        default=None,
        description="Идентификатор пользователя, если он известен.",
    )
    filesystem_context: dict[str, str] = Field(
        default_factory=dict,
        description="Дополнительные сведения о рабочих файлах и директориях.",
    )
    context_runs: list[ContextRunRef] = Field(
        default_factory=list,
        description="Необязательные существующие ResearchRun для follow-up диалога.",
    )
    state: AgentState | None = Field(
        default=None,
        description="Готовое состояние агента для запуска из snapshot/branch.",
    )


class ResearchAgent(Runnable[Any, list[BaseMessage]]):
    """LangChain Runnable facade для planner-first research-agent.

    Args:
        model: ChatModel для planner/worker/validator/responder.
        sandbox: Объект песочницы или виртуального окружения для workspace tools.
        tools: Список внешних LangChain tools, доступных worker.
        prompts: Опциональный набор промптов агента.
        workspace_root: Корневая директория рабочего пространства.
        sources_dir: Директория исходных файлов относительно workspace или absolute path.
        contexts_dir: Директория контекстных файлов относительно workspace или absolute path.
        runs_dir: Директория хранения ResearchRun.
        memory_dir: Директория memory-файлов.
        skills_dir: Директория skills.
        lineage_service: Готовый сервис lineage, если его нужно переиспользовать.
        artifact_service: Готовый сервис artifacts, если его нужно переиспользовать.
        memory_service: Готовый сервис memory, если его нужно переиспользовать.
        skills_service: Готовый сервис skills, если его нужно переиспользовать.
        tool_registry: Опциональный registry LangChain tools.
        enabled_tool_names: Подмножество tool names, которое будет доступно worker.
        graph: Готовый скомпилированный graph для тестов или внешней сборки.
        stream_console: Включает потоковый диагностический вывод graph в консоль.

    Returns:
        Экземпляр ResearchAgent, совместимый с LangChain Runnable API.
    """

    def __init__(
            self,
            *,
            model: BaseChatModel | None = None,
            sandbox: Any | None = None,
            tools: list[BaseTool] | None = None,
            prompts: AnalysisAgentPrompts | None = None,
            workspace_root: str = ".",
            sources_dir: str | None = None,
            contexts_dir: str | None = None,
            runs_dir: str | None = None,
            memory_dir: str | None = None,
            skills_dir: str | None = None,
            lineage_service: LineageService | None = None,
            artifact_service: ArtifactService | None = None,
            memory_service: MemoryService | None = None,
            skills_service: SkillsService | None = None,
            tool_registry: ToolRegistry | None = None,
            enabled_tool_names: set[str] | None = None,
            graph: Any | None = None,
            stream_console: bool = False,
    ) -> None:
        workspace_path = Path(workspace_root).resolve()
        resolved_runs_dir = _resolve_directory(workspace_path, runs_dir, "runs")
        resolved_memory_dir = _resolve_directory(workspace_path, memory_dir, "memory")
        resolved_skills_dir = _resolve_directory(workspace_path, skills_dir, "skills")

        self.lineage_service = lineage_service or LineageService(resolved_runs_dir)
        self.artifact_service = artifact_service or ArtifactService(resolved_runs_dir)
        self.memory_service = memory_service or MemoryService(resolved_memory_dir)
        self.skills_service = skills_service or SkillsService(resolved_skills_dir)
        self.inspection_service = RunInspectionService(
            lineage_service=self.lineage_service,
            artifact_service=self.artifact_service,
        )
        self.dialog_context_service = DialogContextService(self.inspection_service)
        self._last_state: AgentState | None = None
        self.stream_console = stream_console

        if graph is not None:
            self.graph = graph
            return

        if model is None:
            raise ValueError("model is required when graph is not provided")
        if sandbox is None:
            raise ValueError("sandbox is required when graph is not provided")

        self.graph = planner_agent(
            model=model,
            sandbox=sandbox,
            tools=tools or [],
            prompts=prompts,
            workspace_root=str(workspace_path),
            sources_dir=sources_dir,
            contexts_dir=contexts_dir,
            runs_dir=str(resolved_runs_dir),
            memory_dir=str(resolved_memory_dir),
            skills_dir=str(resolved_skills_dir),
            lineage_service=self.lineage_service,
            artifact_service=self.artifact_service,
            memory_service=self.memory_service,
            skills_service=self.skills_service,
            tool_registry=tool_registry,
            enabled_tool_names=enabled_tool_names,
        )

    def invoke(
            self,
            input: Any,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> list[BaseMessage]:
        """Синхронно запускает агента через стандартный LangChain ``invoke``.

        Args:
            input: Строка запроса, словарь, ResearchAgentInput или AgentState.
            config: Опциональный LangChain/LangGraph config.
            **kwargs: Дополнительные поля запуска: ``session_id``, ``user_id``,
                ``filesystem_context``.

        Returns:
            Список LangChain messages из финального состояния агента.
        """

        state = self._coerce_input_to_state(input, **kwargs)
        if callable(getattr(self.graph, "invoke", None)):
            raw_result = self.graph.invoke(state, config=config)
            final_state = _coerce_agent_state(raw_result, fallback=state)
            return self._finalize_output(final_state)

        return _run_coro_sync(self.ainvoke(input, config=config, **kwargs))

    async def ainvoke(
            self,
            input: Any,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> list[BaseMessage]:
        """Асинхронно запускает агента через стандартный LangChain ``ainvoke``.

        Args:
            input: Строка запроса, словарь, ResearchAgentInput или AgentState.
            config: Опциональный LangChain/LangGraph config.
            **kwargs: Дополнительные поля запуска: ``session_id``, ``user_id``,
                ``filesystem_context``.

        Returns:
            Список LangChain messages из финального состояния агента.
        """

        state = self._coerce_input_to_state(input, **kwargs)
        result = await run_agent_from_state(
            self.graph,
            state,
            config=config,
            stream_console=self.stream_console,
        )
        return self._finalize_output(result.state)

    def batch(
            self,
            inputs: list[Any],
            config: RunnableConfig | list[RunnableConfig] | None = None,
            *,
            return_exceptions: bool = False,
            **kwargs: Any,
    ) -> list[list[BaseMessage] | Exception]:
        """Синхронно запускает несколько независимых исследований.

        Args:
            inputs: Список входов, совместимых с ``invoke``.
            config: Один config для всех запусков или список config по входам.
            return_exceptions: Если ``True``, ошибки возвращаются в списке.
            **kwargs: Общие дополнительные поля запуска.

        Returns:
            Список ответов ``list[BaseMessage]`` или Exception при ``return_exceptions=True``.
        """

        configs = _normalize_batch_config(config, len(inputs))
        results: list[list[BaseMessage] | Exception] = []
        for item, item_config in zip(inputs, configs):
            try:
                results.append(self.invoke(item, config=item_config, **kwargs))
            except Exception as exc:
                if not return_exceptions:
                    raise
                results.append(exc)
        return results

    async def abatch(
            self,
            inputs: list[Any],
            config: RunnableConfig | list[RunnableConfig] | None = None,
            *,
            return_exceptions: bool = False,
            **kwargs: Any,
    ) -> list[list[BaseMessage] | Exception]:
        """Асинхронно запускает несколько независимых исследований.

        Args:
            inputs: Список входов, совместимых с ``ainvoke``.
            config: Один config для всех запусков или список config по входам.
            return_exceptions: Если ``True``, ошибки возвращаются в списке.
            **kwargs: Общие дополнительные поля запуска.

        Returns:
            Список ответов ``list[BaseMessage]`` или Exception при ``return_exceptions=True``.
        """

        configs = _normalize_batch_config(config, len(inputs))

        async def _call(item: Any, item_config: RunnableConfig | None) -> list[BaseMessage]:
            """Выполняет один элемент async batch.

            Args:
                item: Один вход агента.
                item_config: Config для этого входа.

            Returns:
                Список LangChain messages одного запуска.
            """

            return await self.ainvoke(item, config=item_config, **kwargs)

        calls = [_call(item, item_config) for item, item_config in zip(inputs, configs)]
        raw_results = await asyncio.gather(*calls, return_exceptions=return_exceptions)
        return list(raw_results)

    def stream(
            self,
            input: Any,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> Any:
        """Возвращает single-result stream, совместимый с LangChain ``stream``.

        Args:
            input: Вход агента, совместимый с ``invoke``.
            config: Опциональный LangChain/LangGraph config.
            **kwargs: Дополнительные поля запуска.

        Yields:
            Единственный список LangChain messages после завершения graph.
        """

        yield self.invoke(input, config=config, **kwargs)

    async def astream(
            self,
            input: Any,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> Any:
        """Возвращает single-result async stream, совместимый с LangChain ``astream``.

        Args:
            input: Вход агента, совместимый с ``ainvoke``.
            config: Опциональный LangChain/LangGraph config.
            **kwargs: Дополнительные поля запуска.

        Yields:
            Единственный список LangChain messages после завершения graph.
        """

        yield await self.ainvoke(input, config=config, **kwargs)

    def build_branch_state(
            self,
            *,
            branch_run_id: str,
            branch_node_id: str | None = None,
    ) -> AgentState:
        """Строит AgentState для продолжения заранее созданной ветки.

        Args:
            branch_run_id: Идентификатор branch ResearchRun.
            branch_node_id: Опциональный node ветки, из которого нужно восстановиться.

        Returns:
            AgentState, восстановленный из branch snapshot.
        """

        return BranchResumeService(self.lineage_service).build_initial_state(
            branch_run_id=branch_run_id,
            branch_node_id=branch_node_id,
        )

    def branch_from(self, request: BranchRequest) -> ResearchRun:
        """Создает новую ветку от выбранного lineage node.

        Args:
            request: BranchRequest с source run/node и новой задачей.

        Returns:
            Новый ResearchRun ветки.
        """

        return self.lineage_service.branch_from(request)

    def invoke_branch(
            self,
            request: BranchRequest,
            config: RunnableConfig | None = None,
    ) -> list[BaseMessage]:
        """Создает ветку от lineage node и синхронно продолжает исследование из snapshot.

        Args:
            request: BranchRequest с source run/node, новой задачей и режимом ветки.
            config: Опциональный LangChain/LangGraph config для запуска ветки.

        Returns:
            Список LangChain messages из финального состояния ветки.
        """

        branch = self.branch_from(request)
        branch_state = self.build_branch_state(branch_run_id=branch.run_id)
        return self.invoke(branch_state, config=config)

    async def ainvoke_branch(
            self,
            request: BranchRequest,
            config: RunnableConfig | None = None,
    ) -> list[BaseMessage]:
        """Создает ветку от lineage node и асинхронно продолжает исследование из snapshot.

        Args:
            request: BranchRequest с source run/node, новой задачей и режимом ветки.
            config: Опциональный LangChain/LangGraph config для запуска ветки.

        Returns:
            Список LangChain messages из финального состояния ветки.
        """

        branch = self.branch_from(request)
        branch_state = self.build_branch_state(branch_run_id=branch.run_id)
        return await self.ainvoke(branch_state, config=config)

    def inspector(self) -> RunInspectionService:
        """Возвращает сервис чтения runs, snapshots и artifacts.

        Returns:
            RunInspectionService, связанный с текущими сервисами агента.
        """

        return self.inspection_service

    def list_run_summaries(self) -> list[RunSummary]:
        """Возвращает краткие сводки сохраненных запусков агента.

        Returns:
            Список RunSummary, отсортированный от новых запусков к старым.
        """

        return self.inspection_service.list_run_summaries()

    def get_run_result(
            self,
            run_id: str | None = None,
            *,
            include_nodes: bool = True,
            include_artifacts: bool = True,
            include_final_state: bool = True,
    ) -> RunResult | None:
        """Возвращает полный read-only результат запуска.

        Args:
            run_id: Идентификатор запуска. Если ``None``, используется последний
                запуск текущего объекта ResearchAgent.
            include_nodes: Включать lineage nodes.
            include_artifacts: Включать artifacts.
            include_final_state: Включать snapshot финального или последнего node.

        Returns:
            RunResult или ``None``, если run_id неизвестен или запуск не найден.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return None
        return self.inspection_service.get_run_result(
            selected_run_id,
            include_nodes=include_nodes,
            include_artifacts=include_artifacts,
            include_final_state=include_final_state,
        )

    def get_run_graph(self, run_id: str | None = None) -> RunGraph | None:
        """Возвращает lineage graph запуска.

        Args:
            run_id: Идентификатор запуска. Если ``None``, используется последний
                запуск текущего объекта ResearchAgent.

        Returns:
            RunGraph или ``None``, если запуск не найден.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return None
        return self.inspection_service.get_run_graph(selected_run_id)

    def get_node_details(
            self,
            node_id: str,
            run_id: str | None = None,
            *,
            include_snapshot: bool = True,
    ) -> NodeDetails | None:
        """Возвращает подробности lineage node.

        Args:
            node_id: Идентификатор lineage node.
            run_id: Идентификатор запуска. Если ``None``, используется последний
                запуск текущего объекта ResearchAgent.
            include_snapshot: Загружать ли snapshot node.

        Returns:
            NodeDetails или ``None``, если запуск или node не найдены.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return None
        return self.inspection_service.get_node_details(
            selected_run_id,
            node_id,
            include_snapshot=include_snapshot,
        )

    def get_node_inspector_view(
            self,
            node_id: str,
            run_id: str | None = None,
            *,
            include_snapshot: bool = True,
            preview_chars: int = 4_000,
            snapshot_preview_chars: int = 1_000,
    ) -> NodeInspectorView | None:
        """Возвращает полную read-only модель Node Inspector для UI.

        Args:
            node_id: Идентификатор lineage node, который нужно открыть в UI.
            run_id: Идентификатор запуска. Если ``None``, используется последний
                запуск текущего объекта ResearchAgent.
            include_snapshot: Загружать ли полный snapshot node.
            preview_chars: Максимальное количество символов preview для artifacts.
            snapshot_preview_chars: Максимальное количество символов preview для
                каждой top-level секции snapshot.

        Returns:
            NodeInspectorView или ``None``, если run_id неизвестен либо node не найден.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return None
        return self.inspection_service.get_node_inspector_view(
            selected_run_id,
            node_id,
            include_snapshot=include_snapshot,
            preview_chars=preview_chars,
            snapshot_preview_chars=snapshot_preview_chars,
        )

    def get_final_report(self, run_id: str | None = None) -> str | None:
        """Возвращает финальный markdown-отчет запуска.

        Args:
            run_id: Идентификатор запуска. Если ``None``, используется последний
                запуск текущего объекта ResearchAgent.

        Returns:
            Текст финального отчета или ``None``.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return None
        return self.inspection_service.get_final_report(selected_run_id)

    def list_artifacts(self, run_id: str | None = None) -> list[Artifact]:
        """Возвращает artifacts запуска.

        Args:
            run_id: Идентификатор запуска. Если ``None``, используется последний
                запуск текущего объекта ResearchAgent.

        Returns:
            Список Artifact. Если run_id неизвестен, возвращается пустой список.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return []
        return self.inspection_service.list_artifacts(selected_run_id)

    def get_artifact_details(
            self,
            artifact_id: str,
            run_id: str | None = None,
            *,
            preview_chars: int = 4_000,
    ) -> ArtifactDetails | None:
        """Возвращает подробности artifact и безопасное preview содержимого.

        Args:
            artifact_id: Идентификатор artifact.
            run_id: Идентификатор запуска. Если ``None``, используется последний запуск
                текущего объекта ResearchAgent.
            preview_chars: Максимальное количество символов preview для текстового artifact.

        Returns:
            ArtifactDetails или ``None``, если run_id неизвестен или artifact не найден.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return None
        return self.inspection_service.get_artifact_details(
            selected_run_id,
            artifact_id,
            preview_chars=preview_chars,
        )

    def preview_artifact(
            self,
            artifact_id: str,
            run_id: str | None = None,
            *,
            preview_chars: int = 4_000,
    ) -> ArtifactContentPreview | None:
        """Возвращает только preview содержимого artifact.

        Args:
            artifact_id: Идентификатор artifact.
            run_id: Идентификатор запуска. Если ``None``, используется последний запуск
                текущего объекта ResearchAgent.
            preview_chars: Максимальное количество символов preview для текстового artifact.

        Returns:
            ArtifactContentPreview или ``None``, если run_id неизвестен или artifact не найден.
        """

        selected_run_id = _selected_run_id(run_id, self.last_run_id)
        if not selected_run_id:
            return None
        return self.inspection_service.preview_artifact(
            selected_run_id,
            artifact_id,
            preview_chars=preview_chars,
        )

    @property
    def last_state(self) -> AgentState | None:
        """Возвращает последнее финальное состояние агента.

        Returns:
            AgentState последнего запуска в этом объекте или ``None``.
        """

        return self._last_state

    @property
    def last_run_id(self) -> str:
        """Возвращает run_id последнего запуска агента.

        Returns:
            Идентификатор последнего ResearchRun или пустую строку.
        """

        return self._last_state.run_id if self._last_state else ""

    def _coerce_input_to_state(self, input: Any, **kwargs: Any) -> AgentState:
        """Преобразует LangChain Runnable input в AgentState.

        Args:
            input: Строка, словарь, ResearchAgentInput или AgentState.
            **kwargs: Переопределения ``session_id``, ``user_id`` и
                ``filesystem_context``.

        Returns:
            AgentState, готовый для передачи в graph.

        Raises:
            TypeError: Если вход имеет неподдерживаемый тип.
            ValueError: Если вход не содержит ни ``user_query``, ни ``state``.
        """

        if isinstance(input, AgentState):
            return input

        payload = _coerce_input_payload(input)
        if payload.state is not None:
            return payload.state

        user_query = kwargs.pop("user_query", None) or payload.user_query
        if not user_query:
            raise ValueError("ResearchAgent input must include user_query or state")

        session_id = kwargs.pop("session_id", payload.session_id)
        user_id = kwargs.pop("user_id", payload.user_id)
        filesystem_context = kwargs.pop(
            "filesystem_context",
            payload.filesystem_context or None,
        )
        state = build_chat_initial_state(
            str(user_query),
            session_id=session_id,
            user_id=user_id,
            filesystem_context=filesystem_context,
        )
        if payload.context_runs:
            dialog_context = self.dialog_context_service.build_context(payload.context_runs)
            if dialog_context.rendered_context:
                state.ephemeral_recalls["dialog_context"] = dialog_context.rendered_context
                state.filesystem_context["dialog_context"] = dialog_context.rendered_context
        return state

    def _finalize_output(self, state: AgentState) -> list[BaseMessage]:
        """Сохраняет последнее состояние и возвращает LangChain messages.

        Args:
            state: Финальное состояние после выполнения graph.

        Returns:
            Список LangChain messages из финального состояния.
        """

        self._last_state = state
        return list(state.messages or [])


def _resolve_directory(
        workspace_root: Path,
        directory: str | None,
        default_subdir: str,
) -> Path:
    """Нормализует директорию относительно workspace.

    Args:
        workspace_root: Абсолютный путь к workspace.
        directory: Пользовательский путь или ``None``.
        default_subdir: Поддиректория по умолчанию.

    Returns:
        Абсолютный путь к директории.
    """

    if directory is None or not str(directory).strip():
        return (workspace_root / default_subdir).resolve()

    candidate = Path(directory)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _selected_run_id(explicit_run_id: str | None, last_run_id: str) -> str:
    """Выбирает run_id для read API методов ResearchAgent.

    Args:
        explicit_run_id: Явно переданный идентификатор запуска или ``None``.
        last_run_id: Последний run_id текущего объекта ResearchAgent.

    Returns:
        Явный run_id, если он передан, иначе последний run_id или пустая строка.
    """

    if explicit_run_id is not None and str(explicit_run_id).strip():
        return str(explicit_run_id)
    return last_run_id or ""


def _coerce_input_payload(input: Any) -> ResearchAgentInput:
    """Преобразует произвольный Runnable input в ResearchAgentInput.

    Args:
        input: Строка запроса, dict или ResearchAgentInput.

    Returns:
        Нормализованный ResearchAgentInput.

    Raises:
        TypeError: Если тип входа не поддерживается.
    """

    if isinstance(input, ResearchAgentInput):
        return input
    if isinstance(input, str):
        return ResearchAgentInput(user_query=input)
    if isinstance(input, dict):
        return ResearchAgentInput.model_validate(input)
    raise TypeError(
        "ResearchAgent input must be str, dict, ResearchAgentInput, or AgentState"
    )


def _coerce_agent_state(raw_result: Any, *, fallback: AgentState) -> AgentState:
    """Преобразует результат graph.invoke в AgentState.

    Args:
        raw_result: Результат graph.invoke/ainvoke.
        fallback: Исходное состояние на случай нестандартного результата.

    Returns:
        AgentState после нормализации результата.
    """

    if isinstance(raw_result, AgentState):
        return raw_result
    if isinstance(raw_result, dict):
        merged = fallback.model_dump()
        merged.update(raw_result)
        return AgentState.model_validate(merged)
    return fallback


def _run_coro_sync(coro: Any) -> Any:
    """Выполняет coroutine из синхронного ``invoke``.

    Если event loop уже запущен, coroutine выполняется в отдельном потоке,
    чтобы не ломать notebook/backend окружения.

    Args:
        coro: Coroutine для выполнения.

    Returns:
        Результат coroutine.

    Raises:
        BaseException: Пробрасывает ошибку coroutine без обертки.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_box: dict[str, Any] = {}
    error_box: dict[str, BaseException] = {}

    def _runner() -> None:
        """Запускает coroutine в отдельном потоке.

        Returns:
            None.
        """

        try:
            result_box["result"] = asyncio.run(coro)
        except BaseException as exc:
            error_box["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("result")


def _normalize_batch_config(
        config: RunnableConfig | list[RunnableConfig] | None,
        input_count: int,
) -> Sequence[RunnableConfig | None]:
    """Нормализует config для batch/abatch.

    Args:
        config: Один config, список config или ``None``.
        input_count: Количество входов batch.

    Returns:
        Последовательность config той же длины, что и входы.

    Raises:
        ValueError: Если длина списка config не совпадает с количеством входов.
    """

    if isinstance(config, list):
        if len(config) != input_count:
            raise ValueError("config list length must match inputs length")
        return config
    return [config] * input_count


__all__ = ["ContextRunRef", "ResearchAgent", "ResearchAgentInput"]

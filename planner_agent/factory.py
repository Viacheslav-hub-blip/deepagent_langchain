"""Factory для сборки LangGraph research-agent.

Содержит:
- _resolve_directory: нормализация директорий относительно workspace.
- _prepare_worker_tools: подготовка LangChain tools и оборачивание генераторов кода.
- _filter_worker_tools: ограничение набора worker-инструментов публичным контрактом.
- _configure_sandbox_working_directory: настройка cwd sandbox для execute_python_code.
- planner_agent: сборка LangGraph workflow с сервисами, tools и nodes.
"""

import functools
from pathlib import Path
from typing import Any, List, Optional, Set

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph

from .agent_nodes.context_builder_node import context_builder_node
from .agent_nodes.critic_node import critic_node
from .agent_nodes.initialize_node import initializer_node
from .agent_nodes.planner_node import planner_node
from .agent_nodes.replanner_node import replanner_node
from .agent_nodes.responder_node import responder_node
from .agent_nodes.scheduler_node import scheduler_node
from .agent_nodes.validator_node import validator_node
from .agent_nodes.worker_node import worker_node
from .models import AgentState
from .prompts import AnalysisAgentPrompts
from .services.artifact_service import ArtifactService
from .services.lineage_service import LineageService
from .services.memory_service import MemoryService
from .services.skills_service import SkillsService
from .tools.python_analysis_tool import (
    PYTHON_ANALYSIS_TOOL_NAME,
    build_python_analysis_tool,
)
from .tools.registry import ToolRegistry
from .tools.skill_tools import build_skill_read_tools

PUBLIC_WORKER_TOOL_NAMES: frozenset[str] = frozenset(
    {"execute_python_code", "read_table", "list_skills", "load_skill"}
)

LEGACY_TOOL_NAME_ALIASES: dict[str, str] = {
    "python_analysis": PYTHON_ANALYSIS_TOOL_NAME,
    "spark_query_table": "read_table",
    "skill_list": "list_skills",
    "skill_view": "load_skill",
}


def _resolve_directory(
        workspace_root: Path,
        directory: Optional[str],
        default_subdir: str,
) -> Path:
    """Возвращает абсолютный путь директории.

    Args:
        workspace_root: Корневая директория рабочего пространства.
        directory: Пользовательский путь, абсолютный или относительный.
        default_subdir: Поддиректория по умолчанию, если путь не задан.

    Returns:
        Абсолютный нормализованный путь.
    """
    if directory is None or not str(directory).strip():
        return (workspace_root / default_subdir).resolve()

    candidate = Path(directory)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _prepare_worker_tools(
        tools: list[BaseTool],
        sandbox: Any,
        code_generator_tool_names: set[str],
) -> list[BaseTool]:
    """Подготавливает tools для worker-узлов.

    Args:
        tools: Исходный список LangChain tools.
        sandbox: Песочница, в которой должны выполняться сгенерированные Python-коды.
        code_generator_tool_names: Имена tools, которые генерируют код и должны
            быть обернуты в BaseCodeExecutorTool.

    Returns:
        Список tools, где генераторы кода заменены на tools-исполнители.
    """

    prepared = [
        tool
        for tool in tools
        if tool.name not in code_generator_tool_names
        and tool.name != PYTHON_ANALYSIS_TOOL_NAME
    ]
    prepared.append(build_python_analysis_tool(sandbox))
    return prepared


def _filter_worker_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Оставляет только публично разрешенные worker-инструменты.

    Args:
        tools: Полный список собранных LangChain tools.

    Returns:
        Список tools с именами из ``PUBLIC_WORKER_TOOL_NAMES`` без дублей.
    """

    selected: list[BaseTool] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name not in PUBLIC_WORKER_TOOL_NAMES or tool.name in seen:
            continue
        selected.append(tool)
        seen.add(tool.name)
    return selected


def _normalize_enabled_tool_names(
        enabled_tool_names: Optional[Set[str]],
        code_generator_tool_names: set[str],
) -> Optional[Set[str]]:
    """Нормализует список разрешенных tools после замены генератора кода.

    Args:
        enabled_tool_names: Явный набор разрешенных tools или ``None``.
        code_generator_tool_names: Legacy-имена внешних генераторов кода.

    Returns:
        Набор имен, где legacy-генераторы заменены на ``execute_python_code``,
        или ``None`` при отсутствии явного фильтра.
    """

    if enabled_tool_names is None:
        return None
    normalized = {
        LEGACY_TOOL_NAME_ALIASES.get(name, name)
        for name in enabled_tool_names
    }
    if normalized.intersection(code_generator_tool_names):
        normalized.difference_update(code_generator_tool_names)
        normalized.add(PYTHON_ANALYSIS_TOOL_NAME)
    return normalized


def _configure_sandbox_working_directory(sandbox: Any, workspace_path: Path) -> None:
    """Настраивает рабочую директорию sandbox для относительных путей в коде.

    Args:
        sandbox: Объект песочницы, переданный в агент.
        workspace_path: Абсолютный путь workspace, относительно которого должны
            резолвиться имена файлов внутри ``execute_python_code``.

    Returns:
        ``None``. Функция изменяет sandbox на месте, если это поддерживается.
    """

    setter = getattr(sandbox, "set_working_directory", None)
    if callable(setter):
        setter(workspace_path)
        return
    try:
        setattr(sandbox, "working_directory", workspace_path)
    except Exception:
        return


def planner_agent(
        model: BaseChatModel,
        sandbox: Any,
        tools: List[BaseTool],
        prompts: Optional[AnalysisAgentPrompts] = None,
        code_generator_tool_names: Set[str] = {"generate_python_code"},
        enable_workspace_tools: bool = True,
        workspace_root: str = ".",
        sources_dir: Optional[str] = None,
        contexts_dir: Optional[str] = None,
        lineage_service: Optional[LineageService] = None,
        artifact_service: Optional[ArtifactService] = None,
        memory_service: Optional[MemoryService] = None,
        skills_service: Optional[SkillsService] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enabled_tool_names: Optional[Set[str]] = None,
        runs_dir: Optional[str] = None,
        memory_dir: Optional[str] = None,
        skills_dir: Optional[str] = None,
):
    """Собирает LangGraph workflow research-agent.

    Args:
        model: LangChain chat model для planner, worker, validator и responder.
        sandbox: Песочница или runtime-объект с переменными и методами превью.
        tools: Внешние LangChain tools, доступные worker.
        prompts: Набор системных prompt-шаблонов.
        code_generator_tool_names: Имена tools, которые генерируют Python-код.
        enable_workspace_tools: Добавлять ли tools чтения/записи workspace.
        workspace_root: Корень рабочего пространства.
        sources_dir: Директория исходных файлов.
        contexts_dir: Директория контекстных файлов.
        lineage_service: Готовый сервис lineage или ``None``.
        artifact_service: Готовый сервис artifacts или ``None``.
        memory_service: Готовый сервис memory или ``None``.
        skills_service: Готовый сервис skills или ``None``.
        tool_registry: Готовый registry tools или ``None``.
        enabled_tool_names: Подмножество разрешенных tools.
        runs_dir: Директория сохранения runs.
        memory_dir: Директория memory-файлов.
        skills_dir: Директория skills.

    Returns:
        Скомпилированный LangGraph workflow.
    """

    if prompts is None:
        prompts = AnalysisAgentPrompts()

    workspace_path = Path(workspace_root).resolve()
    _configure_sandbox_working_directory(sandbox, workspace_path)
    resolved_sources_dir = _resolve_directory(
        workspace_root=workspace_path,
        directory=sources_dir,
        default_subdir="sources",
    )
    resolved_contexts_dir = _resolve_directory(
        workspace_root=workspace_path,
        directory=contexts_dir,
        default_subdir="contexts",
    )
    resolved_runs_dir = _resolve_directory(
        workspace_root=workspace_path,
        directory=runs_dir,
        default_subdir="runs",
    )
    resolved_memory_dir = _resolve_directory(
        workspace_root=workspace_path,
        directory=memory_dir,
        default_subdir="memory",
    )
    resolved_skills_dir = _resolve_directory(
        workspace_root=workspace_path,
        directory=skills_dir,
        default_subdir="skills",
    )
    final_lineage_service = lineage_service or LineageService(resolved_runs_dir)
    final_artifact_service = artifact_service or ArtifactService(resolved_runs_dir)
    final_memory_service = memory_service or MemoryService(resolved_memory_dir)
    final_skills_service = skills_service or SkillsService(resolved_skills_dir)

    final_worker_tools = _prepare_worker_tools(
        tools=tools,
        sandbox=sandbox,
        code_generator_tool_names=set(code_generator_tool_names),
    )

    del enable_workspace_tools

    final_worker_tools.extend(build_skill_read_tools(final_skills_service))
    final_worker_tools = _filter_worker_tools(final_worker_tools)

    final_tool_registry = tool_registry or ToolRegistry()
    final_tool_registry.register_many(final_worker_tools)
    final_worker_tools = final_tool_registry.enabled(
        _normalize_enabled_tool_names(
            enabled_tool_names,
            set(code_generator_tool_names),
        ),
        strict=False,
    )

    fs_context = {
        "workspace_root": str(workspace_path),
        "sources_dir": str(resolved_sources_dir),
        "contexts_dir": str(resolved_contexts_dir),
        "skills_dir": str(resolved_skills_dir),
    }

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "initializer",
        functools.partial(
            initializer_node,
            sandbox=sandbox,
            filesystem_context=fs_context,
            lineage_service=final_lineage_service,
            skills_service=final_skills_service,
        ),
    )
    workflow.add_node(
        "context_builder",
        functools.partial(
            context_builder_node,
            memory_service=final_memory_service,
            skills_service=final_skills_service,
            lineage_service=final_lineage_service,
        ),
    )
    workflow.add_node(
        "planner",
        functools.partial(
            planner_node,
            llm=model,
            tools=final_worker_tools,
            prompt=prompts.planner_system,
            plan_review_prompt=prompts.plan_reviewer_system,
            force_replan=False,
            lineage_service=final_lineage_service,
            artifact_service=final_artifact_service,
            skills_service=final_skills_service,
        ),
    )
    workflow.add_node(
        "replanner",
        functools.partial(
            replanner_node,
            llm=model,
            tools=final_worker_tools,
            prompt=prompts.replanner_system,
            plan_review_prompt=prompts.plan_reviewer_system,
            lineage_service=final_lineage_service,
            artifact_service=final_artifact_service,
            skills_service=final_skills_service,
        ),
    )
    workflow.add_node(
        "worker",
        functools.partial(
            worker_node,
            llm=model,
            tools=final_worker_tools,
            sandbox=sandbox,
            prompt=prompts.worker_system,
            lineage_service=final_lineage_service,
            artifact_service=final_artifact_service,
            skills_service=final_skills_service,
        ),
    )
    workflow.add_node(
        "validator",
        functools.partial(
            validator_node,
            llm=model,
            prompt=prompts.validator_system,
            artifact_service=final_artifact_service,
            lineage_service=final_lineage_service,
        ),
    )
    workflow.add_node(
        "critic",
        functools.partial(
            critic_node,
            llm=model,
            prompt=prompts.critic_system,
            tools=final_worker_tools,
            artifact_service=final_artifact_service,
            lineage_service=final_lineage_service,
        ),
    )
    workflow.add_node(
        "responder",
        functools.partial(
            responder_node,
            llm=model,
            prompt=prompts.responder_system,
            lineage_service=final_lineage_service,
            artifact_service=final_artifact_service,
        ),
    )
    workflow.add_node(
        "scheduler",
        functools.partial(
            scheduler_node,
            lineage_service=final_lineage_service,
        ),
    )

    workflow.add_edge(START, "initializer")

    return workflow.compile()

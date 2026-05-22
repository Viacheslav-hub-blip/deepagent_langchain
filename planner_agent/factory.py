"""Factory для сборки LangGraph research-agent."""

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
from .tools.execute_python_code_tool import (
    EXECUTE_PYTHON_CODE_TOOL_NAME,
    build_execute_python_code_tool,
)
from .tools.registry import ToolRegistry
from .tools.skill_tools import build_skill_read_tools

PUBLIC_WORKER_TOOL_NAMES: frozenset[str] = frozenset(
    {"execute_python_code", "read_table", "list_skills", "load_skill"}
)


def _resolve_directory(
        workspace_root: Path,
        directory: Optional[str],
        default_subdir: str,
) -> Path:
    if directory is None or not str(directory).strip():
        return (workspace_root / default_subdir).resolve()

    candidate = Path(directory)
    if candidate.is_absolute():
        return candidate.resolve()
    return (workspace_root / candidate).resolve()


def _prepare_worker_tools(tools: list[BaseTool], sandbox: Any) -> list[BaseTool]:
    """Добавляет ``execute_python_code`` и исключает дубликат из внешних tools."""

    prepared = [
        tool for tool in tools if tool.name != EXECUTE_PYTHON_CODE_TOOL_NAME
    ]
    prepared.append(build_execute_python_code_tool(sandbox))
    return prepared


def _filter_worker_tools(tools: list[BaseTool]) -> list[BaseTool]:
    selected: list[BaseTool] = []
    seen: set[str] = set()
    for tool in tools:
        if tool.name not in PUBLIC_WORKER_TOOL_NAMES or tool.name in seen:
            continue
        selected.append(tool)
        seen.add(tool.name)
    return selected


def _configure_sandbox_working_directory(sandbox: Any, workspace_path: Path) -> None:
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
    """Собирает скомпилированный LangGraph workflow research-agent."""

    if prompts is None:
        prompts = AnalysisAgentPrompts()

    workspace_path = Path(workspace_root).resolve()
    _configure_sandbox_working_directory(sandbox, workspace_path)
    resolved_sources_dir = _resolve_directory(workspace_path, sources_dir, "sources")
    resolved_contexts_dir = _resolve_directory(workspace_path, contexts_dir, "contexts")
    resolved_runs_dir = _resolve_directory(workspace_path, runs_dir, "runs")
    resolved_memory_dir = _resolve_directory(workspace_path, memory_dir, "memory")
    resolved_skills_dir = _resolve_directory(workspace_path, skills_dir, "skills")

    final_lineage_service = lineage_service or LineageService(resolved_runs_dir)
    final_artifact_service = artifact_service or ArtifactService(resolved_runs_dir)
    final_memory_service = memory_service or MemoryService(resolved_memory_dir)
    final_skills_service = skills_service or SkillsService(resolved_skills_dir)

    final_worker_tools = _filter_worker_tools(
        _prepare_worker_tools(tools, sandbox) + build_skill_read_tools(final_skills_service),
    )

    final_tool_registry = tool_registry or ToolRegistry()
    final_tool_registry.register_many(final_worker_tools)
    final_worker_tools = final_tool_registry.enabled(enabled_tool_names, strict=False)

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

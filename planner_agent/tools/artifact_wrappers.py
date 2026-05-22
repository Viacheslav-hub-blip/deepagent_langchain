"""Обертки LangChain tools для записи tool traces и больших outputs как artifacts.

Содержит:
- ArtifactToolWrapper: wrapper над обычным LangChain tool.
- wrap_tools_for_artifacts: массовое оборачивание tools для worker.
- _clean_runtime_kwargs: удаление runtime-only kwargs.
- _format_tool_success_response: единый текстовый ответ LLM для успешного вызова.
- _format_inline_tool_result_text: преобразование inline-результата tool в текст без JSON-обертки.
- _format_structured_tool_error_text: преобразование структурированной ошибки tool в текст.
- _format_tool_exception_response: текстовая диагностика исключения инструмента.
- _build_tool_error_payload: диагностический JSON для ошибки инструмента.
- _tool_error_possible_causes: вероятные причины ошибки инструмента.
- _tool_error_solution_options: варианты исправления ошибки инструмента.
- _format_tool_error_trace_content: текст trace artifact для ошибки.
- _limited_serialized: компактная сериализация аргументов/результатов.
- _limit_tool_text: обрезка диагностического текста инструмента.
- _tool_input_from_call: восстановление входа tool из args/kwargs.
- _safe_filename_fragment: безопасный фрагмент имени файла.
- _build_artifact_label: формирование человеко-читаемого id artifact-а вызова.
- _build_variable_name: формирование имени sandbox-переменной для DataFrame.
- _is_dataframe: проверка DataFrame-подобного результата.
- _is_meta_tool: проверка имени мета-инструмента чтения существующих данных.
- _is_significant_for_task_refs: фильтр artifacts для добавления в task.artifact_refs.
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Any
from uuid import uuid4

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from ..models import Task
from ..runtime.tool_result_capture import (
    build_tool_trace_content,
    capture_tool_result,
    serialize_tool_result,
)
from ..runtime.tool_errors import tool_error_possible_causes, tool_error_solution_options
from ..runtime.tool_text import ToolTextResult, is_tool_error_result
from ..runtime.sandbox import PythonSandboxProtocol
from ..schemas.artifacts import Artifact
from ..services.artifact_service import ArtifactService
from .execute_python_code_tool import EXECUTE_PYTHON_CODE_TOOL_NAME

TOOL_ARTIFACT_SUMMARY_MAX_LEN = 500
TOOL_ERROR_TRACEBACK_MAX_LEN = 8_000
INLINE_TOOL_RESULT_MAX_CHARS = 8_000

# Префиксы инструментов, которые читают существующие artifacts/skills и не должны
# создавать собственные artifact-записи. Их вызовы покрываются tool-calls trace
# на уровне worker_node, а сами они не приносят новых данных.
META_TOOL_PREFIXES: tuple[str, ...] = ()
META_TOOL_NAMES: frozenset[str] = frozenset({"list_skills", "load_skill"})

# Капчи tool результата, которые попадают в task.artifact_refs. Это только
# существенные (большие или ссылочные) данные, на которые могут опираться
# responder/critic/replanner. Маленькие inline-результаты остаются в
# state.artifact_index, но не загромождают контекст следующих узлов.
SIGNIFICANT_CAPTURE_REASONS: frozenset[str] = frozenset(
    {"context_budget_exceeded", "existing_file_reference"}
)


class ArtifactToolWrapper(BaseTool):
    """Обертка над LangChain tool, которая защищает контекст worker от больших outputs."""

    _wrapped_tool: BaseTool = PrivateAttr()
    _artifact_service: ArtifactService = PrivateAttr()
    _run_id: str = PrivateAttr()
    _node_id: str = PrivateAttr()
    _task: Task = PrivateAttr()
    _artifact_index: dict[str, Any] = PrivateAttr()
    _tool_traces: list[dict[str, Any]] = PrivateAttr()
    _sandbox: PythonSandboxProtocol | None = PrivateAttr(default=None)
    _call_counter: int = PrivateAttr(default=0)

    def __init__(
            self,
            *,
            wrapped_tool: BaseTool,
            artifact_service: ArtifactService,
            run_id: str,
            node_id: str,
            task: Task,
            artifact_index: dict[str, Any],
            tool_traces: list[dict[str, Any]],
            sandbox: PythonSandboxProtocol | None = None,
    ) -> None:
        """Создает wrapper над обычным LangChain tool.

        Args:
            wrapped_tool: Исходный LangChain tool.
            artifact_service: Сервис записи artifacts.
            run_id: Идентификатор ResearchRun.
            node_id: Идентификатор worker_started node.
            task: Текущая задача worker.
            artifact_index: Общий индекс artifacts для обновления state.
            tool_traces: Список trace-событий для обновления state.
            sandbox: Песочница, куда DataFrame-результаты добавляются как переменные.

        Returns:
            None.
        """

        super().__init__(
            name=wrapped_tool.name,
            description=(
                f"{wrapped_tool.description} "
                "Runtime note: large outputs are automatically captured into run "
                "artifacts and replaced with compact references in LLM context. "
                "Large outputs are saved as artifacts for lineage and UI inspection."
            ).strip(),
            args_schema=wrapped_tool.args_schema,
            return_direct=wrapped_tool.return_direct,
            response_format=wrapped_tool.response_format,
        )
        self._wrapped_tool = wrapped_tool
        self._artifact_service = artifact_service
        self._run_id = run_id
        self._node_id = node_id
        self._task = task
        self._artifact_index = artifact_index
        self._tool_traces = tool_traces
        self._sandbox = sandbox

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Синхронно вызывает tool и возвращает безопасный для LLM результат.

        Args:
            *args: Позиционные аргументы tool.
            **kwargs: Именованные аргументы tool.

        Returns:
            Исходный маленький результат или artifact reference для большого результата.
        """

        clean_kwargs = _clean_runtime_kwargs(kwargs)
        tool_input = _tool_input_from_call(args, clean_kwargs)
        try:
            result = self._wrapped_tool.invoke(tool_input)
        except Exception as exc:
            return self._record_tool_exception(tool_input=tool_input, exc=exc)
        return self._record_tool_result(tool_input=tool_input, result=result)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Асинхронно вызывает tool и возвращает безопасный для LLM результат.

        Args:
            *args: Позиционные аргументы tool.
            **kwargs: Именованные аргументы tool.

        Returns:
            Исходный маленький результат или artifact reference для большого результата.
        """

        clean_kwargs = _clean_runtime_kwargs(kwargs)
        tool_input = _tool_input_from_call(args, clean_kwargs)
        try:
            result = await self._wrapped_tool.ainvoke(tool_input)
        except Exception as exc:
            return self._record_tool_exception(tool_input=tool_input, exc=exc)
        return await self._record_tool_result_async(tool_input=tool_input, result=result)

    def _record_tool_result(self, *, tool_input: Any, result: Any) -> Any:
        """Записывает tool trace и artifacts, затем возвращает результат для LLM.

        Args:
            tool_input: Аргументы вызова tool.
            result: Сырой результат tool.

        Returns:
            Значение, которое будет передано worker-агенту.
        """

        trace_id = uuid4().hex
        self._call_counter += 1
        result_label = _build_artifact_label(
            task_id=self._task.task_id,
            retry_count=self._task.retry_count,
            tool_name=self.name,
            sequence=self._call_counter,
            suffix="",
        )
        trace_label = _build_artifact_label(
            task_id=self._task.task_id,
            retry_count=self._task.retry_count,
            tool_name=self.name,
            sequence=self._call_counter,
            suffix="trace",
        )
        captured = capture_tool_result(
            artifact_service=self._artifact_service,
            run_id=self._run_id,
            node_id=self._node_id,
            task_id=self._task.task_id,
            tool_name=self.name,
            tool_input=tool_input,
            raw_result=result,
            capture_id=trace_id,
            artifact_label=result_label,
        )
        self._record_dataframe_variable_sync(
            result=result,
            result_label=result_label,
            captured=captured,
        )
        self._write_trace_artifact(
            trace_id=trace_id,
            trace_label=trace_label,
            tool_input=tool_input,
            captured=captured,
        )
        return _format_tool_success_response(
            tool_name=self.name,
            tool_input=tool_input,
            captured=captured,
        )

    async def _record_tool_result_async(self, *, tool_input: Any, result: Any) -> Any:
        """Асинхронно записывает tool trace и добавляет DataFrame в sandbox.

        Args:
            tool_input: Аргументы вызова tool.
            result: Сырой результат tool.

        Returns:
            Значение, которое будет передано worker-агенту.
        """

        trace_id = uuid4().hex
        self._call_counter += 1
        result_label = _build_artifact_label(
            task_id=self._task.task_id,
            retry_count=self._task.retry_count,
            tool_name=self.name,
            sequence=self._call_counter,
            suffix="",
        )
        trace_label = _build_artifact_label(
            task_id=self._task.task_id,
            retry_count=self._task.retry_count,
            tool_name=self.name,
            sequence=self._call_counter,
            suffix="trace",
        )
        captured = capture_tool_result(
            artifact_service=self._artifact_service,
            run_id=self._run_id,
            node_id=self._node_id,
            task_id=self._task.task_id,
            tool_name=self.name,
            tool_input=tool_input,
            raw_result=result,
            capture_id=trace_id,
            artifact_label=result_label,
        )
        await self._record_dataframe_variable_async(
            result=result,
            result_label=result_label,
            captured=captured,
        )
        self._write_trace_artifact(
            trace_id=trace_id,
            trace_label=trace_label,
            tool_input=tool_input,
            captured=captured,
        )
        return _format_tool_success_response(
            tool_name=self.name,
            tool_input=tool_input,
            captured=captured,
        )

    def _record_tool_exception(self, *, tool_input: Any, exc: Exception) -> str:
        """Записывает ошибку инструмента и возвращает понятный ответ для LLM.

        Args:
            tool_input: Аргументы вызова инструмента.
            exc: Исключение, возникшее внутри исходного инструмента.

        Returns:
            Текст с описанием ошибки, контекстом и вариантами исправления.
        """

        trace_id = uuid4().hex
        self._call_counter += 1
        trace_label = _build_artifact_label(
            task_id=self._task.task_id,
            retry_count=self._task.retry_count,
            tool_name=self.name,
            sequence=self._call_counter,
            suffix="error_trace",
        )
        error_payload = _build_tool_error_payload(
            tool_name=self.name,
            tool_input=tool_input,
            exc=exc,
        )
        content = _format_tool_error_trace_content(
            tool_name=self.name,
            tool_input=tool_input,
            error_payload=error_payload,
        )
        artifact = self._artifact_service.write_artifact(
            run_id=self._run_id,
            node_id=self._node_id,
            kind="tool_trace",
            filename=(
                f"tasks/{_safe_filename_fragment(self._task.task_id or 'unknown_task')}"
                f"/tool_calls/{_safe_filename_fragment(self.name)}-{trace_id}-error.txt"
            ),
            content=content,
            mime_type="text/plain",
            summary=str(error_payload["error"])[:TOOL_ARTIFACT_SUMMARY_MAX_LEN],
            metadata={
                "trace_id": trace_id,
                "task_id": self._task.task_id,
                "tool_name": self.name,
                "args_preview": serialize_tool_result(
                    tool_input,
                    max_chars=TOOL_ARTIFACT_SUMMARY_MAX_LEN,
                ),
                "artifact_role": "tool_call_trace",
                "captured": False,
                "tool_error": True,
                "reusable": False,
            },
            artifact_id=trace_label,
        )
        self._artifact_index[artifact.artifact_id] = artifact.model_dump(mode="json")
        self._tool_traces.append(
            {
                "trace_id": trace_id,
                "run_id": self._run_id,
                "node_id": self._node_id,
                "task_id": self._task.task_id,
                "tool_name": self.name,
                "args_preview": serialize_tool_result(
                    tool_input,
                    max_chars=TOOL_ARTIFACT_SUMMARY_MAX_LEN,
                ),
                "result_preview": str(error_payload["error"])[:TOOL_ARTIFACT_SUMMARY_MAX_LEN],
                "artifact_id": artifact.artifact_id,
                "artifact_uri": artifact.uri,
                "captured": False,
                "tool_error": True,
            }
        )
        error_payload["tool_trace_artifact_id"] = artifact.artifact_id
        error_payload["tool_trace_uri"] = artifact.uri
        return _format_tool_exception_response(error_payload=error_payload)

    def _write_trace_artifact(
            self,
            *,
            trace_id: str,
            trace_label: str,
            tool_input: Any,
            captured: Any,
    ) -> None:
        """Записывает trace вызова tool и обновляет индексы состояния.

        Args:
            trace_id: Уникальный идентификатор вызова.
            trace_label: Человекочитаемый artifact_id trace-файла.
            tool_input: Аргументы вызова tool.
            captured: Результат обработки output инструмента.

        Returns:
            None.
        """

        content = build_tool_trace_content(
            tool_name=self.name,
            tool_input=tool_input,
            captured=captured,
        )
        artifact = self._artifact_service.write_artifact(
            run_id=self._run_id,
            node_id=self._node_id,
            kind="tool_trace",
            filename=(
                f"tasks/{_safe_filename_fragment(self._task.task_id or 'unknown_task')}"
                f"/tool_calls/{_safe_filename_fragment(self.name)}-{trace_id}.txt"
            ),
            content=content,
            mime_type="text/plain",
            summary=captured.preview[:TOOL_ARTIFACT_SUMMARY_MAX_LEN],
            metadata={
                "trace_id": trace_id,
                "task_id": self._task.task_id,
                "tool_name": self.name,
                "args_preview": serialize_tool_result(
                    tool_input,
                    max_chars=TOOL_ARTIFACT_SUMMARY_MAX_LEN,
                ),
                "artifact_role": "tool_call_trace",
                "captured": captured.was_captured,
                "captured_artifact_refs": captured.artifact_refs,
                "original_size_estimate": captured.original_size_estimate,
                "reusable": True,
            },
            artifact_id=trace_label,
        )
        self._artifact_index.update(captured.artifact_index)
        self._artifact_index[artifact.artifact_id] = artifact.model_dump(mode="json")
        self._update_task_artifact_refs(captured_index=captured.artifact_index)
        self._tool_traces.append(
            {
                "trace_id": trace_id,
                "run_id": self._run_id,
                "node_id": self._node_id,
                "task_id": self._task.task_id,
                "tool_name": self.name,
                "args_preview": serialize_tool_result(
                    tool_input,
                    max_chars=TOOL_ARTIFACT_SUMMARY_MAX_LEN,
                ),
                "result_preview": captured.preview[:TOOL_ARTIFACT_SUMMARY_MAX_LEN],
                "artifact_id": artifact.artifact_id,
                "artifact_uri": artifact.uri,
                "captured": captured.was_captured,
                "captured_artifact_refs": captured.artifact_refs,
                "original_size_estimate": captured.original_size_estimate,
            }
        )

    async def _record_dataframe_variable_async(
            self,
            *,
            result: Any,
            result_label: str,
            captured: Any,
    ) -> str | None:
        """Добавляет DataFrame-результат в sandbox и дополняет ссылку для LLM.

        Args:
            result: Сырой результат tool.
            result_label: Стабильный label результата.
            captured: Результат обработки output инструмента.

        Returns:
            Имя созданной переменной или None.
        """

        variable_name = _build_variable_name(result_label)
        if not _is_dataframe(result) or self._sandbox is None:
            return None
        await self._sandbox.add_variable(variable_name, result)
        self._sandbox.last_dataframe_variable = variable_name
        self._append_dataframe_variable_reference(captured=captured, variable_name=variable_name)
        return variable_name

    def _record_dataframe_variable_sync(
            self,
            *,
            result: Any,
            result_label: str,
            captured: Any,
    ) -> str | None:
        """Синхронно добавляет DataFrame-результат в sandbox, если это возможно.

        Args:
            result: Сырой результат tool.
            result_label: Стабильный label результата.
            captured: Результат обработки output инструмента.

        Returns:
            Имя созданной переменной или None.
        """

        variable_name = _build_variable_name(result_label)
        if not _is_dataframe(result) or self._sandbox is None:
            return None
        self._sandbox.globals[variable_name] = result
        self._sandbox.last_dataframe_variable = variable_name
        self._append_dataframe_variable_reference(captured=captured, variable_name=variable_name)
        return variable_name

    def _append_dataframe_variable_reference(self, *, captured: Any, variable_name: str) -> None:
        """Добавляет имя sandbox-переменной в metadata artifact и ответ LLM.

        Args:
            captured: Результат обработки output инструмента.
            variable_name: Имя переменной с DataFrame.

        Returns:
            None.
        """

        for artifact_id, payload in captured.artifact_index.items():
            metadata = payload.get("metadata") if isinstance(payload, dict) else None
            if isinstance(metadata, dict):
                metadata["variable_name"] = variable_name
                metadata["sandbox_variable_name"] = variable_name
            if (
                    isinstance(captured.content_for_llm, str)
                    and "<tool_result>" not in captured.content_for_llm
                    and "sandbox_variable_name" not in captured.content_for_llm
            ):
                captured.content_for_llm = (
                    f"{captured.content_for_llm}\n"
                    f"variable_name: {variable_name}\n"
                    f"sandbox_variable_name: {variable_name}"
                )
            return

    def _update_task_artifact_refs(
            self,
            *,
            captured_index: dict[str, Any],
    ) -> None:
        """Добавляет в ``task.artifact_refs`` только значимые artifacts вызова.

        Поведение:
            - tool_trace artifact самой обертки никогда не попадает в task.artifact_refs;
              он остается доступен только через state.artifact_index и tool_calls trace.
            - результат, попавший в state inline (без захвата), не считается значимым.
            - артефакты, созданные мета-инструментами (list_skills/load_skill), исключаются —
              это просто чтение уже существующих данных.
            - inline-структурированные результаты (capture_reason=inline_structured_result)
              исключаются — они уже попали в контекст worker-а как обычные значения.

        Args:
            captured_index: Артефакты, созданные ``capture_tool_result`` (id -> payload).
        """

        if _is_meta_tool(self.name):
            return
        for artifact_id, payload in captured_index.items():
            if _is_significant_for_task_refs(payload):
                if artifact_id and artifact_id not in self._task.artifact_refs:
                    self._task.artifact_refs.append(artifact_id)


def wrap_tools_for_artifacts(
        *,
        tools: list[BaseTool],
        artifact_service: ArtifactService | None,
        run_id: str,
        node_id: str | None,
        task: Task,
        artifact_index: dict[str, Any],
        tool_traces: list[dict[str, Any]],
        sandbox: PythonSandboxProtocol | None = None,
) -> list[BaseTool]:
    """Оборачивает tools в ArtifactToolWrapper при наличии artifact service.

    Args:
        tools: Исходные LangChain tools.
        artifact_service: Сервис artifacts или ``None``.
        run_id: Идентификатор ResearchRun.
        node_id: Идентификатор worker node.
        task: Текущая задача worker.
        artifact_index: Индекс artifacts для обновления state.
        tool_traces: Список trace-событий для обновления state.
        sandbox: Песочница для добавления DataFrame-результатов как переменных.

    Returns:
        Список исходных или обернутых tools.
    """

    if artifact_service is None or not run_id or not node_id:
        return tools

    return [
        ArtifactToolWrapper(
            wrapped_tool=tool,
            artifact_service=artifact_service,
            run_id=run_id,
            node_id=node_id,
            task=task,
            artifact_index=artifact_index,
            tool_traces=tool_traces,
            sandbox=sandbox,
        )
        for tool in tools
    ]


def _clean_runtime_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Удаляет kwargs, которые LangChain передает в runtime, но не в tool input.

    Args:
        kwargs: Именованные аргументы wrapper-вызова.

    Returns:
        Очищенный словарь аргументов.
    """

    return {
        key: value
        for key, value in kwargs.items()
        if key not in {"run_manager", "callbacks", "config"}
    }


def _format_tool_success_response(
        *,
        tool_name: str,
        tool_input: Any,
        captured: Any,
) -> str:
    """Формирует понятный ответ инструмента для LLM.

    Args:
        tool_name: Имя вызванного инструмента.
        tool_input: Аргументы инструмента.
        captured: Результат перехвата output, созданный ``capture_tool_result``.

    Returns:
        Текстовый ответ инструмента без JSON-обертки.
    """

    if tool_name == EXECUTE_PYTHON_CODE_TOOL_NAME:
        formatted = _format_execute_python_code_tool_response(
            tool_input=tool_input,
            raw_result=captured.content_for_llm,
        )
        if formatted is not None:
            return formatted

    if captured.was_captured:
        return str(captured.content_for_llm)
    if is_tool_error_result(captured.content_for_llm):
        return captured.content_for_llm
    if isinstance(captured.content_for_llm, dict) and captured.content_for_llm.get("ok") is False:
        return _format_structured_tool_error_text(
            tool_name=tool_name,
            tool_input=tool_input,
            payload=captured.content_for_llm,
        )

    result_text = _format_inline_tool_result_text(captured.content_for_llm)
    if captured.artifact_refs:
        return (
            f"Инструмент {tool_name} выполнил запрос.\n"
            "Структурированный результат также сохранен как artifact для последующего чтения.\n"
            f"Artifact: {', '.join(captured.artifact_refs)}.\n"
            f"Тип результата: {captured.result_kind}.\n\n"
            f"Данные:\n{result_text}\n\n"
            "Дальше: используй текст выше, если его достаточно. Если нужен полный файл или проверка данных, "
            "используй artifact_id в выводе."
        )
    return (
        f"Инструмент {tool_name} выполнил запрос.\n"
        f"Тип результата: {captured.result_kind}.\n\n"
        f"Данные:\n{result_text}\n\n"
        "Дальше: используй этот результат как фактический вывод инструмента. "
        "Если данных недостаточно, повтори вызов с более точными аргументами или используй подходящий инструмент анализа."
    )


def _format_execute_python_code_tool_response(
        *,
        tool_input: Any,
        raw_result: Any,
) -> str | ToolTextResult | None:
    """Формирует ответ execute_python_code с акцентом на stdout и preview.

    Args:
        tool_input: Аргументы вызова инструмента.
        raw_result: JSON-строка или словарь результата ``PythonExecutionResult``.

    Returns:
        Текст для LLM, ``ToolTextResult`` при ошибке выполнения или ``None``.
    """

    payload = _parse_execute_python_code_payload(raw_result)
    if payload is None:
        return None

    if payload.get("success") is True:
        lines = [
            "Python-код выполнен успешно.",
            str(payload.get("message") or "").strip(),
        ]
        target_variable = str(payload.get("target_variable") or "").strip()
        if target_variable:
            lines.append(f"target_variable: {target_variable}")
        variable_preview = str(payload.get("variable_preview") or "").strip()
        if variable_preview:
            lines.append("preview результата:")
            lines.append(variable_preview)
        execution_output = str(payload.get("execution_output") or "").strip()
        if execution_output:
            lines.append("вывод в консоль (execution_output):")
            lines.append(execution_output)
        available_variables = payload.get("available_variables")
        if isinstance(available_variables, list) and available_variables:
            preview_names = ", ".join(str(name) for name in available_variables[:40])
            suffix = ""
            if len(available_variables) > 40:
                suffix = f" ... еще {len(available_variables) - 40}"
            lines.append(f"available_variables: {preview_names}{suffix}")
        lines.append(
            "Дальше: используй execution_output и preview как фактический результат. "
            "Если нужна именованная переменная для следующего шага, повтори вызов с target_variable."
        )
        return _limit_tool_text("\n".join(line for line in lines if line), max_chars=INLINE_TOOL_RESULT_MAX_CHARS)

    lines = [
        "Ошибка execute_python_code: код не выполнен или не дал ожидаемый результат.",
        str(payload.get("message") or payload.get("error") or "Неизвестная ошибка."),
    ]
    if payload.get("error"):
        lines.append(f"error: {payload['error']}")
    execution_output = str(payload.get("execution_output") or "").strip()
    if execution_output:
        lines.append("вывод в консоль (execution_output):")
        lines.append(execution_output)
    if payload.get("traceback"):
        lines.append("traceback:")
        lines.append(str(payload["traceback"]))
    if payload.get("possible_causes"):
        lines.append("possible_causes:")
        lines.extend(f"- {item}" for item in payload["possible_causes"])
    if payload.get("solution_options"):
        lines.append("solution_options:")
        lines.extend(f"- {item}" for item in payload["solution_options"])
    if payload.get("retry_guidance"):
        lines.append(f"retry_guidance: {payload['retry_guidance']}")
    if payload.get("available_variables"):
        lines.append(
            "available_variables: "
            + ", ".join(str(name) for name in payload["available_variables"][:40])
        )
    lines.append(f"Аргументы вызова: {_limited_serialized(tool_input)}.")
    return ToolTextResult(_limit_tool_text("\n".join(lines), max_chars=INLINE_TOOL_RESULT_MAX_CHARS), is_error=True)


def _parse_execute_python_code_payload(raw_result: Any) -> dict[str, Any] | None:
    """Извлекает JSON-ответ execute_python_code из результата tool.

    Args:
        raw_result: Сырой результат tool.

    Returns:
        Словарь с полями ``success``, ``execution_output`` и т.д. или ``None``.
    """

    if isinstance(raw_result, dict) and "success" in raw_result:
        return raw_result
    if not isinstance(raw_result, str):
        return None
    text = raw_result.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if isinstance(parsed, dict) and "success" in parsed:
        return parsed
    return None


def _format_inline_tool_result_text(value: Any) -> str:
    """Преобразует inline-результат инструмента в человекочитаемый текст.

    Args:
        value: Результат tool, который можно передать worker inline.

    Returns:
        Текст без JSON-обертки ``ok/tool_name/result``.
    """

    if isinstance(value, str):
        return _limit_tool_text(value, max_chars=INLINE_TOOL_RESULT_MAX_CHARS)
    if isinstance(value, list):
        if not value:
            return "Пустой список."
        rows = [f"Получено элементов: {len(value)}."]
        for index, item in enumerate(value[:10], start=1):
            rows.append(f"{index}. {_format_inline_item_text(item)}")
        if len(value) > 10:
            rows.append(f"...еще {len(value) - 10} элементов не показаны в кратком выводе.")
        return "\n".join(rows)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            lines.append(f"- {key}: {_format_inline_item_text(item)}")
        return "\n".join(lines) if lines else "Пустой объект."
    return _limit_tool_text(str(value), max_chars=INLINE_TOOL_RESULT_MAX_CHARS)


def _format_inline_item_text(value: Any) -> str:
    """Формирует короткий текст для элемента inline-результата.

    Args:
        value: Элемент списка или значение словаря.

    Returns:
        Однострочное человекочитаемое представление значения.
    """

    if isinstance(value, dict):
        parts = [f"{key}={_format_inline_item_text(item)}" for key, item in value.items()]
        return "; ".join(parts)
    if isinstance(value, list):
        if not value:
            return "пустой список"
        preview = "; ".join(_format_inline_item_text(item) for item in value[:3])
        suffix = f"; еще {len(value) - 3} элементов" if len(value) > 3 else ""
        return f"список из {len(value)} элементов: {preview}{suffix}"
    return _limit_tool_text(str(value), max_chars=500)


def _format_structured_tool_error_text(
        *,
        tool_name: str,
        tool_input: Any,
        payload: dict[str, Any],
) -> str:
    """Преобразует старый структурированный payload ошибки в текст.

    Args:
        tool_name: Имя инструмента.
        tool_input: Аргументы вызова инструмента.
        payload: Словарь ошибки, который вернул исходный tool.

    Returns:
        Человекочитаемый текст ошибки без JSON-обертки.
    """

    error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    code = error.get("code") or payload.get("code") or "unknown_error"
    message = error.get("message") or payload.get("message") or "Инструмент вернул ошибку."
    missing_columns = error.get("missing_columns") or payload.get("missing_columns")
    available_tables = error.get("available_tables") or payload.get("available_tables")
    lines = [
        f"Ошибка инструмента {tool_name}: запрос не выполнен.",
        f"Код ошибки: {code}.",
        f"Причина: {message}",
    ]
    if missing_columns:
        lines.append(f"Отсутствующие поля: {', '.join(map(str, missing_columns))}.")
    if available_tables:
        lines.append(f"Доступные таблицы и алиасы: {', '.join(map(str, available_tables))}.")
    if payload.get("schema"):
        lines.append(f"Схема таблицы: {_limited_serialized(payload['schema'])}.")
    lines.append(f"Аргументы вызова: {_limited_serialized(tool_input)}.")
    lines.append("Как исправить: не повторяй тот же вызов без изменений; исправь table_name, select_columns или filters по тексту ошибки.")
    return ToolTextResult("\n".join(lines), is_error=True)


def _format_tool_exception_response(*, error_payload: dict[str, Any]) -> str:
    """Формирует текстовый ответ об исключении инструмента для worker-а.

    Args:
        error_payload: Диагностический payload, собранный ``_build_tool_error_payload``.

    Returns:
        Текст с типом ошибки, причиной, вариантами исправления и ссылкой на trace artifact.
    """

    error = error_payload["error"]
    lines = [
        f"Ошибка инструмента {error_payload['tool_name']}: инструмент не смог выполнить запрос.",
        f"Тип ошибки: {error['type']}.",
        f"Сообщение: {error['message']}",
        f"Аргументы вызова: {error_payload['tool_input']}.",
    ]
    if error_payload.get("possible_causes"):
        lines.append("Возможные причины:")
        lines.extend(f"- {cause}" for cause in error_payload["possible_causes"])
    if error_payload.get("solution_options"):
        lines.append("Как исправить:")
        lines.extend(f"- {option}" for option in error_payload["solution_options"])
    lines.append(f"Повтор: {error_payload['retry_guidance']}")
    if error_payload.get("tool_trace_artifact_id"):
        lines.append(f"Trace artifact: {error_payload['tool_trace_artifact_id']}.")
    return ToolTextResult("\n".join(lines), is_error=True)


def _build_tool_error_payload(
        *,
        tool_name: str,
        tool_input: Any,
        exc: Exception,
) -> dict[str, Any]:
    """Собирает диагностический payload ошибки инструмента.

    Args:
        tool_name: Имя инструмента, который завершился ошибкой.
        tool_input: Аргументы вызова инструмента.
        exc: Исключение исходного инструмента.

    Returns:
        JSON-совместимый словарь с причиной ошибки и вариантами исправления.
    """

    error_type = exc.__class__.__name__
    error_message = str(exc)
    return {
        "ok": False,
        "tool_name": tool_name,
        "what_happened": "Инструмент не смог выполнить запрос и вернул ошибку.",
        "error": {
            "type": error_type,
            "message": error_message,
        },
        "tool_input": _limited_serialized(tool_input),
        "possible_causes": tool_error_possible_causes(error_type),
        "solution_options": tool_error_solution_options(tool_name, error_type),
        "retry_guidance": (
            "Не повторяй тот же вызов без изменений. Сначала исправь аргументы, "
            "проверь доступные файлы/переменные/artifacts или выбери другой инструмент."
        ),
        "traceback": _limit_tool_text(traceback.format_exc(), max_chars=TOOL_ERROR_TRACEBACK_MAX_LEN),
    }


def _format_tool_error_trace_content(
        *,
        tool_name: str,
        tool_input: Any,
        error_payload: dict[str, Any],
) -> str:
    """Формирует текст trace artifact для ошибки инструмента.

    Args:
        tool_name: Имя инструмента.
        tool_input: Аргументы инструмента.
        error_payload: JSON-совместимое описание ошибки.

    Returns:
        Текстовый trace с аргументами и диагностикой.
    """

    return (
        f"Tool: {tool_name}\n\n"
        f"Arguments:\n{serialize_tool_result(tool_input, max_chars=TOOL_ARTIFACT_SUMMARY_MAX_LEN)}\n\n"
        f"Error response:\n{_format_tool_exception_response(error_payload=error_payload)}\n\n"
        f"Traceback:\n{error_payload.get('traceback', '')}"
    )


def _limited_serialized(value: Any) -> str:
    """Сериализует значение с ограничением длины для диагностического ответа.

    Args:
        value: Произвольное значение.

    Returns:
        Короткая JSON/text строка.
    """

    return serialize_tool_result(value, max_chars=INLINE_TOOL_RESULT_MAX_CHARS)


def _limit_tool_text(value: str, *, max_chars: int) -> str:
    """Обрезает диагностический текст инструмента.

    Args:
        value: Исходный текст.
        max_chars: Максимальная длина результата.

    Returns:
        Текст с пометкой об обрезании при превышении лимита.
    """

    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _tool_input_from_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """Восстанавливает фактический input tool из args/kwargs.

    Args:
        args: Позиционные аргументы wrapper-вызова.
        kwargs: Очищенные именованные аргументы wrapper-вызова.

    Returns:
        Значение, которое нужно передать в исходный LangChain tool.
    """

    if kwargs:
        return kwargs
    if len(args) == 1:
        return args[0]
    return list(args)


def _safe_filename_fragment(value: str) -> str:
    """Преобразует строку в безопасный фрагмент имени файла.

    Args:
        value: Исходная строка.

    Returns:
        Безопасный фрагмент имени файла.
    """

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "unknown"


def _build_artifact_label(
        *,
        task_id: str | None,
        retry_count: int,
        tool_name: str,
        sequence: int,
        suffix: str = "",
) -> str:
    """Строит человеко-читаемый artifact_id вида ``t{task}_{tool}_{n}``.

    Args:
        task_id: Идентификатор задачи плана.
        retry_count: Номер повтора задачи (0 для первого запуска).
        tool_name: Имя LangChain tool.
        sequence: Порядковый номер вызова инструмента в текущей задаче.
        suffix: Опциональный текстовый суффикс (например, ``trace``).

    Returns:
        Строка вида ``t{task_id}_{tool}_{seq}`` или ``t{task_id}_r{n}_{tool}_{seq}_{suffix}``.
    """

    safe_task = _safe_filename_fragment(task_id or "task")
    safe_tool = _safe_filename_fragment(tool_name or "tool")
    retry_part = f"_r{retry_count}" if retry_count else ""
    suffix_part = f"_{suffix}" if suffix else ""
    return f"t{safe_task}{retry_part}_{safe_tool}_{sequence}{suffix_part}"


def _build_variable_name(label: str) -> str:
    """Преобразует artifact label в корректное имя Python-переменной.

    Args:
        label: Человекочитаемый label результата tool.

    Returns:
        Имя переменной, пригодное для sandbox globals.
    """

    safe = re.sub(r"\W+", "_", label).strip("_")
    if not safe:
        safe = "df_tool_result"
    if safe[0].isdigit():
        safe = f"df_{safe}"
    return safe


def _is_dataframe(value: Any) -> bool:
    """Проверяет, похож ли результат на DataFrame.

    Args:
        value: Произвольный результат tool.

    Returns:
        True, если объект поддерживает базовый DataFrame-интерфейс.
    """

    return (
        hasattr(value, "to_csv")
        and hasattr(value, "shape")
        and hasattr(value, "columns")
    )


def _is_meta_tool(tool_name: str) -> bool:
    """Проверяет, относится ли инструмент к мета-tools чтения существующих данных."""

    name = (tool_name or "").lower()
    return name in META_TOOL_NAMES or any(name.startswith(prefix) for prefix in META_TOOL_PREFIXES)


def _is_significant_for_task_refs(payload: dict[str, Any] | Artifact) -> bool:
    """Решает, следует ли добавить artifact в ``task.artifact_refs``.

    Args:
        payload: JSON-представление artifact из capture index либо сам Artifact.

    Returns:
        ``True`` для больших захваченных результатов и ссылок на существующие файлы;
        ``False`` для inline-структурированных результатов и иных служебных artifacts.
    """

    if isinstance(payload, Artifact):
        metadata = payload.metadata
    elif isinstance(payload, dict):
        metadata = payload.get("metadata")
    else:
        return False

    if not isinstance(metadata, dict):
        return False
    capture_reason = str(metadata.get("capture_reason") or "")
    return capture_reason in SIGNIFICANT_CAPTURE_REASONS

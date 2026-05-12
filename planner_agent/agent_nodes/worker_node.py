"""
Модуль содержит логику узла-воркера для мультиагентной системы на базе LangGraph.

Воркер получает задачу из планировщика, подготавливает инструменты,
формирует системный промпт, запускает ReAct-агента и возвращает
результат выполнения задачи вместе с командой перехода к critic-узлу.
"""

import asyncio
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, Send

from ..models import CriticPayload, Task, TaskStatus, WorkerPayload
from ..runtime.sandbox import PythonSandboxProtocol
from ..schemas.lineage import StateNode
from ..services.artifact_service import ArtifactService
from ..services.lineage_service import LineageService
from ..services.prompt_trace_service import write_prompt_trace, write_tool_calls_trace
from ..services.skills_service import SkillsService
from ..tools.artifact_read_tools import build_artifact_read_tools
from ..tools.artifact_wrappers import wrap_tools_for_artifacts
from ..tools.python_analysis_tool import PYTHON_ANALYSIS_TOOL_NAME
from ..tools.skill_tools import build_skill_read_tools

# Ключи LangGraph Pregel для чтения актуального AgentState из worker-узла (см. CONFIG_KEY_*).
_CONFIGURABLE_KEY = "configurable"
_PREGEL_READ_KEY = "__pregel_read"
LEGACY_CODE_TOOL_ALIASES: dict[str, str] = {
    "generate_python_code": PYTHON_ANALYSIS_TOOL_NAME,
}


def _initial_user_query_from_graph_state(config: RunnableConfig | None) -> str:
    """Достаёт исходный запрос из состояния графа на каждом вызове worker.

    Не полагается на поля ``WorkerPayload``: значение читается через ``__pregel_read``,
    чтобы совпадать с ``AgentState.initial_user_query`` / первым human-сообщением.
    """

    if config is None:
        return ""
    read_fn = config.get(_CONFIGURABLE_KEY, {}).get(_PREGEL_READ_KEY)
    if read_fn is None or not callable(read_fn):
        return ""
    try:
        chunk = read_fn(["initial_user_query", "messages"], fresh=False)
    except Exception:
        return ""
    if not isinstance(chunk, dict):
        return ""
    stored = chunk.get("initial_user_query")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()
    for msg in chunk.get("messages") or []:
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            text = str(content).strip() if content else ""
            if text:
                return text
    return ""


def _worker_human_invoke_message(
        task: Task,
        initial_user_query: str = "",
) -> str:
    """Формирует human-сообщение для ReAct worker-а.

    Args:
        task: Текущая задача worker-а.
        initial_user_query: Исходный запрос пользователя на весь запуск.

    Returns:
        Текст с исходным запросом и конкретной задачей шага (тег ``TASK``).
    """

    query = (initial_user_query or "").strip()
    desc = (task.description or "").strip()
    task_id = task.task_id or DEFAULT_TASK_ID
    retry_context = _format_worker_retry_context(task)

    parts: list[str] = []
    if query:
        parts.append(f"Исходный запрос пользователя - {query}")
    if desc:
        parts.append(
            "Твоя конкретная задача для ответа на вопрос пользователя -\n"
            f"<TASK {task_id}>\n{desc}\n</TASK {task_id}>"
        )
    else:
        parts.append(
            "Твоя конкретная задача для ответа на вопрос пользователя - "
            "Выполни поставленную задачу"
        )
    message = "\n\n".join(parts)
    if retry_context:
        return f"{message}\n\n{retry_context}"
    return message


def _format_worker_retry_context(task: Task) -> str:
    """Формирует контекст прошлой неуспешной попытки для worker-а.

    Args:
        task: Текущая задача с сохраненными ``generated_code`` и ``error_log``.

    Returns:
        XML-подобный блок с кодом и ошибкой прошлой попытки или пустую строку.
    """

    blocks: list[str] = []
    if task.generated_code:
        blocks.append(
            "<previous_generated_code>\n"
            f"{task.generated_code}\n"
            "</previous_generated_code>"
        )
    if task.error_log:
        blocks.append(
            "<previous_execution_error>\n"
            f"{task.error_log}\n"
            "</previous_execution_error>"
        )
    if not blocks:
        return ""
    return "\n".join(
        [
            "<previous_worker_attempt>",
            "Используй этот код и ошибку как контекст для исправленного повторного выполнения.",
            *blocks,
            "</previous_worker_attempt>",
        ]
    )

# Заглушка при отсутствии загруженных переменных
NO_VARIABLES_PLACEHOLDER: str = "Переменные не загружены"

# Идентификатор задачи по умолчанию, если task_id не задан
DEFAULT_TASK_ID: str = "unknown_task"

# Имя critic-узла в графе
CRITIC_NODE_NAME: str = "critic"

# Максимальная длина превью результата (символов)
RESULT_PREVIEW_MAX_LEN: int = 20_000

# Максимальная длина полного ответа worker-а, сохраняемого в plan для следующих узлов.
FULL_RESULT_STATE_MAX_LEN: int = 200_000

# Максимальная длина вывода в лог воркера (символов)
WORKER_LOG_MAX_LEN: int = 50_000

# Максимальная длина краткого описания artifact в индексе
ARTIFACT_SUMMARY_MAX_LEN: int = 500

# Максимальное количество внутренних шагов ReAct worker-а.
REACT_AGENT_RECURSION_LIMIT: int = 36

# Максимальная длина блока, который выводится в терминал.
CONSOLE_BLOCK_MAX_LENGTH: int = 10_000

# Максимальное число skills, автоматически загружаемых worker-ом из previews.
MAX_AUTO_LOADED_SKILLS: int = 5

# Максимальная длина preview одного skill в prompt worker-а.
SKILL_PREVIEW_MAX_LEN: int = 800

# Максимальная длина полного текста одного skill в prompt worker-а.
LOADED_SKILL_MAX_LEN: int = 8_000

# Максимальное число пакетов sandbox, показываемых worker-у.
MAX_INSTALLED_PACKAGES_IN_PROMPT: int = 60

# Приоритетные библиотеки, которые важнее показать при сжатии списка пакетов.
PREFERRED_INSTALLED_PACKAGES: tuple[str, ...] = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
    "pyarrow",
    "openpyxl",
    "sqlalchemy",
    "requests",
)

async def _print_content_block(title: str, content: str) -> None:
    """Асинхронно выводит читаемый блок worker-диагностики в терминал.

    Args:
        title: Заголовок блока.
        content: Текст блока. Длинный текст обрезается для читаемости.

    Returns:
        ``None``. Функция выполняет только консольный вывод.
    """

    text = (content or "").strip() or "(empty)"
    if len(text) > CONSOLE_BLOCK_MAX_LENGTH:
        text = f"{text[:CONSOLE_BLOCK_MAX_LENGTH]}\n...[truncated for console]..."
    border = "=" * min(max(len(title), 16), 80)
    await asyncio.to_thread(
        print,
        f"\n{border}\n{title}\n{border}\n{text}\n",
        flush=True,
    )


def _format_worker_console_result(
        *,
        task: Task,
        selected_tools: list[BaseTool],
        prepared_tools: list[BaseTool],
) -> str:
    """Форматирует результат worker-а для вывода в терминал.

    Args:
        task: Выполненная задача worker-а.
        selected_tools: Domain/source tools, выбранные для задачи.
        prepared_tools: Итоговый список tools, включая runtime artifact tools.

    Returns:
        Многострочная строка с задачей, статусом, tools, результатом и ошибкой.
    """

    lines = [
        f"Task ID: {task.task_id or DEFAULT_TASK_ID}",
        f"Status: {task.status.value}",
        f"Description: {task.description}",
        f"Dependencies: {task.dependencies or []}",
        f"Selected tools: {[tool.name for tool in selected_tools]}",
        f"Runtime tools: {[tool.name for tool in prepared_tools]}",
    ]
    if task.config:
        lines.append(f"Config: {task.config}")
    if task.output_variable_name:
        lines.append(f"Output variable: {task.output_variable_name}")
    if task.generated_code:
        lines.append("Generated code:")
        lines.append(task.generated_code)
    if task.full_result:
        lines.append("Model answer:")
        lines.append(task.full_result)
    elif task.result_preview:
        lines.append("Result preview:")
        lines.append(task.result_preview)
    if task.error_log:
        lines.append("Error:")
        lines.append(task.error_log)
    if task.artifact_refs:
        lines.append(f"Artifact refs: {task.artifact_refs}")
    return "\n".join(lines)


def _supports_with_context(tool: BaseTool) -> bool:
    """
    Проверяет, поддерживает ли инструмент метод ``with_context``.

    :param tool: Инструмент LangChain.
    :return: ``True``, если метод ``with_context`` доступен и является callable.
    """
    return callable(getattr(tool, "with_context", None))


def _select_task_tools(tools: list[BaseTool], task: Task) -> list[BaseTool]:
    """Выбирает основные инструменты, явно назначенные текущей задаче.

    Args:
        tools: Полный список domain/source/tools, доступных агенту.
        task: Текущая задача воркера с полем ``suggested_tools``.

    Returns:
        Список инструментов, имена которых указаны в ``task.suggested_tools``.
        Если список ``suggested_tools`` пуст, возвращается пустой список: worker
        сможет использовать только runtime artifact tools, добавляемые отдельно.
    """

    requested_names = {
        str(tool_name).strip()
        for tool_name in task.suggested_tools or []
        if str(tool_name).strip()
    }
    if not requested_names:
        task_text = " ".join(
            [
                task.description or "",
                json.dumps(task.config or {}, ensure_ascii=False),
            ]
        )
        requested_names = {
            tool.name
            for tool in tools
            if tool.name and re.search(rf"\b{re.escape(tool.name)}\b", task_text)
        }

    if not requested_names:
        return []

    requested_names = {
        LEGACY_CODE_TOOL_ALIASES.get(tool_name, tool_name)
        for tool_name in requested_names
    }
    return [tool for tool in tools if tool.name in requested_names]


async def _prepare_tools(tools: list[BaseTool], task: Task) -> list[BaseTool]:
    """
    Подготавливает список инструментов, передавая в совместимые из них
    контекст текущей задачи (сгенерированный код и лог ошибок).

    :param tools: Исходный список инструментов.
    :param task: Текущая задача воркера.
    :return: Список подготовленных инструментов.
    """
    prepared_tools: list[BaseTool] = []

    for tool in tools:
        if not _supports_with_context(tool):
            prepared_tools.append(tool)
            continue

        try:
            prepared_tools.append(
                tool.with_context(
                    previous_code=task.generated_code,
                    error_context=task.error_log,
                )
            )
        except TypeError:
            # Инструмент не принимает error_context — передаём только код
            prepared_tools.append(
                tool.with_context(previous_code=task.generated_code)
            )

    return prepared_tools


def _format_installed_packages(packages: dict[str, str]) -> str:
    """Форматирует список установленных pip-пакетов для системного промпта worker-а.

    Args:
        packages: Словарь {имя_пакета: версия}.

    Returns:
        XML-подобный блок с перечнем пакетов или пустую строку.
    """
    if not packages:
        return ""
    lines = [
        "<available_python_packages>",
        "Ниже перечислены библиотеки, доступные для импорта в sandbox-окружении. "
        "Используй их при выполнении Python-кода через инструмент python_analysis.",
        "",
    ]
    package_names = _select_installed_package_names(packages)
    for name in package_names:
        version = packages[name]
        lines.append(f"  - {name}=={version}")
    hidden_count = max(0, len(packages) - len(package_names))
    if hidden_count:
        lines.append(f"  - ... еще {hidden_count} пакетов скрыто для экономии контекста")
    lines.append("</available_python_packages>")
    return "\n".join(lines)


def _select_installed_package_names(packages: dict[str, str]) -> list[str]:
    """Выбирает компактный список установленных пакетов для prompt worker-а.

    Args:
        packages: Словарь ``{имя_пакета: версия}`` из sandbox.

    Returns:
        Список имен пакетов: сначала приоритетные аналитические библиотеки,
        затем остальные по алфавиту в пределах бюджета prompt.
    """

    available = set(packages)
    preferred = [name for name in PREFERRED_INSTALLED_PACKAGES if name in available]
    remaining = [name for name in sorted(available) if name not in preferred]
    return [*preferred, *remaining][:MAX_INSTALLED_PACKAGES_IN_PROMPT]


async def _create_worker_system_prompt(
        payload: WorkerPayload,
        prompt: str,
        loaded_skills: dict[str, str] | None = None,
        installed_packages: dict[str, str] | None = None,
) -> str:
    """
    Формирует системный промпт для воркера на основе шаблона и данных задачи.

    :param payload: Полезная нагрузка воркера (задача + схемы контекста).
    :param prompt: Шаблон промпта с плейсхолдерами.
    :param loaded_skills: Загруженные skills (опционально).
    :param installed_packages: Словарь установленных pip-пакетов (опционально).
    :return: Готовый системный промпт.
    """
    task, schemas = payload.task, payload.context_schemas
    schema_text = "\n".join(
        f"{name}: {desc}" for name, desc in schemas.items()
    )
    if not schema_text:
        schema_text = NO_VARIABLES_PLACEHOLDER

    base_prompt = prompt.format(
        task_description=task.description,
        schema_text=schema_text,
        task_config=str(task.config) if task.config else "",
        previous_results=payload.previous_results,
    )
    context_blocks = "\n\n".join(
        block for block in (
            _format_task_contract(task),
            _format_resolved_inputs(payload.resolved_inputs),
            _format_dependency_context(payload.dependency_context),
            _format_filesystem_context(payload.filesystem_context),
            _format_artifact_context(payload.artifact_context),
            _format_skill_previews(payload.skill_previews),
            _format_loaded_skills(loaded_skills or {}),
            _format_installed_packages(installed_packages or {}),
        )
        if block
    )
    if not context_blocks:
        return base_prompt
    return f"{base_prompt}\n\n{context_blocks}"


def _format_task_contract(task: Task) -> str:
    """Форматирует контракт текущей задачи для worker prompt.

    Args:
        task: Текущая задача worker-а с ожидаемым результатом, artifacts,
            tools и skills.

    Returns:
        XML-подобный блок контракта задачи или пустую строку.
    """

    blocks: list[str] = []
    if task.expected_output:
        blocks.append(f"expected_output: {task.expected_output}")
    if task.required_artifacts:
        blocks.append("required_artifacts:")
        blocks.extend(f"- {item}" for item in task.required_artifacts)
    if task.suggested_tools:
        blocks.append("suggested_tools:")
        blocks.extend(f"- {item}" for item in task.suggested_tools)
    if task.suggested_skills:
        blocks.append("suggested_skills:")
        blocks.extend(f"- {item}" for item in task.suggested_skills)

    if not blocks:
        return ""
    task_id = task.task_id or DEFAULT_TASK_ID
    return "\n".join(
        [
            f"<TASK {task_id}>",
            "<task_contract>",
            *blocks,
            "</task_contract>",
            f"</TASK {task_id}>",
        ]
    )


def _format_resolved_inputs(resolved_inputs: dict[str, Any]) -> str:
    """Форматирует уже разрешенные значения параметров для worker prompt.

    Args:
        resolved_inputs: Словарь скалярных значений, извлеченных из зависимостей
            и config задачи.

    Returns:
        XML-подобный блок с параметрами или пустую строку.
    """

    if not resolved_inputs:
        return ""
    lines = [
        "<resolved_inputs>",
        "Используй эти значения как фактические параметры инструментов. "
        "Не передавай строковые имена ключей вместо значений.",
    ]
    for key, value in resolved_inputs.items():
        lines.append(f"- {key}: {value!r}")
    lines.append("</resolved_inputs>")
    return "\n".join(lines)


def _format_dependency_context(dependency_context: dict[str, Any]) -> str:
    """Форматирует структурированный контекст зависимостей для worker prompt.

    Args:
        dependency_context: Context package с транзитивными зависимостями,
            результатами, ошибками и artifacts.

    Returns:
        XML-подобный блок с компактным описанием зависимостей или пустую строку.
    """

    dependencies = dependency_context.get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        return ""

    lines = [
        "<dependency_context>",
        "Ниже перечислены все транзитивные зависимости текущей задачи. "
        "Используй их как фактический контекст, особенно для параметров инструментов.",
        f"dependency_ids: {dependency_context.get('dependency_ids', [])}",
    ]
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        task_id = str(dependency.get("task_id") or DEFAULT_TASK_ID)
        lines.append(f"<TASK {task_id}>")
        lines.append(
            (
                f"task_id: {dependency.get('task_id')}; "
                f"status: {dependency.get('status')}; "
                f"output_variable_name: {dependency.get('output_variable_name')}; "
                f"artifact_refs: {dependency.get('artifact_refs')}; "
                f"description: {dependency.get('description')}; "
                f"result: {dependency.get('result') or dependency.get('result_preview')}"
            )
        )
        if dependency.get("error_log"):
            lines.append(f"  error_log: {dependency.get('error_log')}")
        if dependency.get("validation_reason"):
            lines.append(f"  validation_reason: {dependency.get('validation_reason')}")
        lines.append(f"</TASK {task_id}>")
    lines.append("</dependency_context>")
    return "\n".join(lines)


def _format_filesystem_context(filesystem_context: dict[str, str]) -> str:
    if not filesystem_context:
        return ""
    lines = ["<workspace_context>"]
    for name, value in filesystem_context.items():
        lines.append(f"{name}: {value}")
    lines.append("</workspace_context>")
    return "\n".join(lines)


def _format_artifact_context(artifact_context: dict[str, Any]) -> str:
    """Форматирует artifact context для worker prompt.

    Args:
        artifact_context: Словарь с выбранными artifacts и метаданными лимитов.

    Returns:
        XML-подобный блок, где каждый artifact выделен парным тегом
        ``ARTIFACT``, или пустая строка.
    """

    artifacts = artifact_context.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return ""

    lines = ["<artifact_context>"]
    for artifact_id, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            continue
        artifact_name = str(artifact.get("artifact_name") or artifact_id).strip() or "artifact"
        schema_line = str(artifact.get("schema") or "").strip()
        lines.append(f"<ARTIFACT {artifact_name}>")
        if schema_line:
            dataframe_file_name = str(artifact.get("dataframe_file_name") or "").strip()
            tool_name = str(artifact.get("tool_name") or "").strip()
            preview_row = str(artifact.get("preview_row") or "").strip()
            lines.append(
                "Схема загруженного из инстурмента dataframe: "
                f"{schema_line}. "
                f"название файла с datafrme  - {dataframe_file_name or artifact_name}. "
                "Для того чтобы загрузить dataframe вы должны использовать "
                f"инструмент - {tool_name or 'artifact_read_tools'}"
            )
            lines.append(f"schema: {schema_line}")
            if preview_row:
                lines.append(f"preview_row: {preview_row}")
        lines.append(f"</ARTIFACT {artifact_name}>")
    lines.append("</artifact_context>")
    return "\n".join(lines)


def _format_skill_previews(skill_previews: dict[str, str]) -> str:
    """Форматирует preview skills для worker prompt.

    Args:
        skill_previews: Словарь ``{skill_name: preview}`` из состояния агента.

    Returns:
        XML-подобный блок со списком skills или пустую строку.
    """

    if not skill_previews:
        return ""

    lines = [
        "<available_skill_previews>",
        "Если skill подходит к задаче, загрузи полный текст через tool skill_view.",
    ]
    for skill_name, preview in skill_previews.items():
        preview_text = _limit_text(
            str(preview),
            max_chars=SKILL_PREVIEW_MAX_LEN,
        )
        lines.append(f"- {skill_name}: {preview_text}")
    lines.append("</available_skill_previews>")
    return "\n".join(lines)


def _load_task_skills(
        task: Task,
        skills_service: SkillsService | None,
        available_skill_names: list[str] | None = None,
) -> dict[str, str]:
    """Загружает полный текст skills для текущей worker-задачи.

    Args:
        task: Задача worker-а с возможным списком ``suggested_skills``.
        skills_service: Сервис чтения skills или ``None``.
        available_skill_names: Имена skills, найденные на этапе initialize.
            Используются как мягкий fallback, если planner не указал
            ``suggested_skills`` явно.

    Returns:
        Словарь ``{skill_name: skill_content}`` с загруженными skills.
    """

    if skills_service is None:
        return {}

    requested_skills = [
        str(skill_name).strip()
        for skill_name in task.suggested_skills
        if str(skill_name).strip()
    ]
    if not requested_skills and available_skill_names:
        requested_skills = [
            str(skill_name).strip()
            for skill_name in available_skill_names[:MAX_AUTO_LOADED_SKILLS]
            if str(skill_name).strip()
        ]

    loaded: dict[str, str] = {}
    for skill_name in dict.fromkeys(requested_skills):
        if not skill_name:
            continue
        result = skills_service.skill_view(skill_name)
        if not result.get("success"):
            continue
        content = result.get("content")
        if isinstance(content, str) and content.strip():
            loaded[skill_name] = content
    return loaded


def _format_loaded_skills(loaded_skills: dict[str, str]) -> str:
    if not loaded_skills:
        return ""

    blocks = ["<loaded_skills>"]
    for skill_name, content in loaded_skills.items():
        limited_content = _limit_text(content.strip(), max_chars=LOADED_SKILL_MAX_LEN)
        blocks.append(f"<skill name=\"{skill_name}\">\n{limited_content}\n</skill>")
    blocks.append("</loaded_skills>")
    return "\n".join(blocks)


async def _apply_tool_output(task: Task, raw_output: str) -> bool:
    """
    Разбирает вывод инструмента и обновляет состояние задачи.

    Если вывод является валидным JSON со структурой ``{"success": bool, ...}``,
    заполняет соответствующие поля задачи. В противном случае сохраняет
    сырой вывод как превью результата.

    :param task: Задача, состояние которой необходимо обновить.
    :param raw_output: Строковый вывод инструмента.
    :return: ``True`` при успешном выполнении инструмента, ``False`` при ошибке.
    """
    try:
        parsed = json.loads(raw_output)
    except Exception:
        # Не JSON — сохраняем как есть
        task.result_preview = _limit_text(raw_output, max_chars=RESULT_PREVIEW_MAX_LEN)
        task.error_log = None
        return True

    if isinstance(parsed, dict) and "success" in parsed:
        if parsed.get("generated_code"):
            task.generated_code = parsed.get("generated_code")
        if parsed.get("target_variable"):
            task.output_variable_name = parsed.get("target_variable")
        if parsed["success"]:
            task.result_preview = (
                    parsed.get("variable_preview")
                    or parsed.get("message")
                    or "OK"
            )
            task.error_log = None
            return True

        task.error_log = _format_tool_error_log(parsed)
        task.result_preview = parsed.get("message")
        return False

    if isinstance(parsed, dict) and "ok" in parsed:
        task.result_preview = _limit_text(
            json.dumps(parsed, ensure_ascii=False, default=str),
            max_chars=RESULT_PREVIEW_MAX_LEN,
        )
        if parsed.get("ok") is True:
            task.error_log = None
            return True

        task.error_log = _format_tool_error_log(parsed)
        return False

    # JSON, но без флага success — сохраняем строковое представление
    task.result_preview = _limit_text(str(parsed), max_chars=RESULT_PREVIEW_MAX_LEN)
    task.error_log = None
    return True


def _format_tool_error_log(parsed: dict[str, Any]) -> str:
    """Формирует лог ошибки инструмента для сохранения в задаче и retry-контексте.

    Args:
        parsed: Распарсенный JSON-ответ инструмента с полями ошибки выполнения.

    Returns:
        Многострочный текст ошибки, который передается следующей попытке worker-а.
    """

    lines = [
        str(
            parsed.get("error")
            or parsed.get("message")
            or "Неизвестная ошибка инструмента"
        )
    ]
    if parsed.get("ok") is False:
        lines.insert(0, "Tool returned ok=false")
    if parsed.get("execution_output"):
        lines.append("stdout/stderr:")
        lines.append(str(parsed.get("execution_output")))
    if parsed.get("traceback"):
        lines.append("traceback:")
        lines.append(str(parsed.get("traceback")))
    if parsed.get("possible_causes"):
        lines.append("possible_causes:")
        lines.append(json.dumps(parsed.get("possible_causes"), ensure_ascii=False, default=str))
    if parsed.get("solution_options"):
        lines.append("solution_options:")
        lines.append(json.dumps(parsed.get("solution_options"), ensure_ascii=False, default=str))
    if parsed.get("retry_guidance"):
        lines.append("retry_guidance:")
        lines.append(str(parsed.get("retry_guidance")))
    if parsed.get("schema"):
        lines.append("schema:")
        lines.append(json.dumps(parsed.get("schema"), ensure_ascii=False, default=str))
    return "\n".join(line for line in lines if line)


def _limit_text(text: str, *, max_chars: int) -> str:
    """Обрезает текст для хранения в состоянии или prompt.

    Args:
        text: Исходный текст.
        max_chars: Максимальное количество символов.

    Returns:
        Текст в пределах лимита с явной пометкой об обрезании.
    """

    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _get_last_message(messages: list[Any], message_type: type) -> Any | None:
    """
    Возвращает последнее сообщение указанного типа из списка сообщений.

    :param messages: Список сообщений агента.
    :param message_type: Тип искомого сообщения.
    :return: Последнее сообщение нужного типа или ``None``.
    """
    return next(
        (msg for msg in reversed(messages) if isinstance(msg, message_type)),
        None,
    )


def _format_react_message_for_console(message: Any) -> tuple[str, str] | None:
    """Форматирует только входные и выходные данные инструментов worker-а.

    Args:
        message: Сообщение LangChain из внутреннего ReAct-агента worker-а.

    Returns:
        Пара ``(title, content)`` для консольного блока или ``None`` для
        сообщений без tool input/output.
    """

    if isinstance(message, AIMessage):
        tool_calls = getattr(message, "tool_calls", None) or []
        lines: list[str] = []
        for raw_call in tool_calls:
            call = _tool_call_record_to_dict(raw_call)
            name = call.get("name") or "unknown_tool"
            args = _normalize_ai_tool_arguments(call)
            args_text = (
                json.dumps(args, ensure_ascii=False, indent=2)
                if isinstance(args, (dict, list))
                else str(args)
            )
            lines.extend(
                [
                    f"tool: {name}",
                    f"tool_call_id: {call.get('id') or call.get('tool_call_id') or ''}",
                    "args:",
                    args_text,
                ]
            )
        if not lines:
            return None
        return "WORKER TOOL INPUT", "\n".join(lines)

    if isinstance(message, ToolMessage):
        tool_name = getattr(message, "name", None) or "unknown_tool"
        tool_call_id = getattr(message, "tool_call_id", None) or ""
        content = str(message.content or "")
        return (
            f"WORKER TOOL OUTPUT {tool_name}",
            f"tool_call_id: {tool_call_id}\nresult:\n{content}",
        )

    return None


async def _astream_react_agent_to_console(
        agent: Any,
        invoke_payload: dict[str, Any],
        *,
        config: dict[str, Any],
        task_id: str,
) -> dict[str, Any]:
    """Запускает ReAct worker-а в stream-режиме и печатает новые сообщения.

    Args:
        agent: ReAct-agent, созданный через ``create_react_agent``.
        invoke_payload: Payload вызова с начальными сообщениями.
        config: Config ReAct-агента, включая recursion limit.
        task_id: Идентификатор задачи worker-а для заголовков консоли.

    Returns:
        Финальное состояние ReAct-агента в том же формате, что возвращает
        ``ainvoke``: словарь с ключом ``messages``.
    """

    if not callable(getattr(agent, "astream", None)):
        return await agent.ainvoke(invoke_payload, config=config)

    final_state: dict[str, Any] | None = None
    printed_count = 0
    try:
        async for chunk in agent.astream(
                invoke_payload,
                config=config,
                stream_mode="values",
        ):
            if not isinstance(chunk, dict):
                continue
            final_state = chunk
            messages = list(chunk.get("messages", []) or [])
            for message in messages[printed_count:]:
                printed_count += 1
                if isinstance(message, HumanMessage):
                    continue
                block = _format_react_message_for_console(message)
                if block is None:
                    continue
                title, content = block
                await _print_content_block(f"{title}: task {task_id}", content)
    except TypeError:
        return await agent.ainvoke(invoke_payload, config=config)

    return final_state or {"messages": []}


def _tool_call_record_to_dict(tool_call: Any) -> dict[str, Any]:
    """Приводит элемент ``tool_calls`` AIMessage к словарю."""

    if isinstance(tool_call, dict):
        return tool_call
    return {
        "name": getattr(tool_call, "name", None),
        "args": getattr(tool_call, "args", None),
        "id": getattr(tool_call, "id", None),
        "arguments": getattr(tool_call, "arguments", None),
    }


def _normalize_ai_tool_arguments(tc_dict: dict[str, Any]) -> Any:
    """Возвращает аргументы вызова инструмента из записи tool_call."""

    args = tc_dict.get("args")
    if args is not None:
        return args
    raw = tc_dict.get("arguments")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def extract_react_tool_calls_from_messages(
        messages: list[Any],
        *,
        result_preview_max_chars: int = 200_000,
) -> list[dict[str, Any]]:
    """Собирает последовательность вызовов инструментов из истории ReAct-агента.

    Использует пары AIMessage.tool_calls и следующие ToolMessage с тем же
    ``tool_call_id``, чтобы критик видел те же имена, аргументы и превью
    ответов инструментов, что и в реальном сообщении графа агента.

    Args:
        messages: Список сообщений из ответа ``create_react_agent().ainvoke``.
        result_preview_max_chars: Максимальная длина превью содержимого ToolMessage.

    Returns:
        Упорядоченный список словарей с полями ``tool_call_id``, ``tool_name``,
        ``arguments``, ``tool_result_preview``.
    """

    pending_meta: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, AIMessage):
            raw_calls = getattr(msg, "tool_calls", None) or []
            for tc in raw_calls:
                tc_dict = _tool_call_record_to_dict(tc)
                tc_id = tc_dict.get("id") or tc_dict.get("tool_call_id")
                name = tc_dict.get("name")
                args = _normalize_ai_tool_arguments(tc_dict)
                if tc_id:
                    pending_meta[str(tc_id)] = {
                        "tool_name": name,
                        "arguments": args,
                    }
            continue

        if isinstance(msg, ToolMessage):
            tc_id_raw = getattr(msg, "tool_call_id", None)
            tc_key = str(tc_id_raw) if tc_id_raw is not None else ""
            meta = pending_meta.pop(tc_key, {}) if tc_key else {}
            tool_name = getattr(msg, "name", None) or meta.get("tool_name")
            content = str(msg.content)
            preview = content
            if len(preview) > result_preview_max_chars:
                preview = f"{preview[:result_preview_max_chars]}...[truncated]"
            records.append(
                {
                    "tool_call_id": tc_id_raw,
                    "tool_name": tool_name,
                    "arguments": meta.get("arguments"),
                    "tool_result_preview": preview,
                }
            )

    return records


async def worker_node(
        payload: WorkerPayload,
        sandbox: PythonSandboxProtocol,
        tools: list[BaseTool],
        llm: BaseChatModel,
        prompt: str,
        lineage_service: LineageService | None = None,
        artifact_service: ArtifactService | None = None,
        skills_service: SkillsService | None = None,
        *,
        config: RunnableConfig | None = None,
) -> Command:
    """
    Асинхронный узел-воркер графа LangGraph.

    Выполняет следующие шаги:
    1. Устанавливает статус задачи в ``RUNNING``.
    2. Подготавливает инструменты с контекстом задачи.
    3. Формирует системный промпт.
    4. Создаёт и запускает ReAct-агента.
    5. Разбирает результат и обновляет состояние задачи.
    6. Возвращает ``Command`` с обновлением плана и переходом к critic-узлу.

    :param payload: Полезная нагрузка воркера (задача + схемы контекста).
    :param sandbox: Клиент Python-песочницы для получения превью переменных.
    :param tools: Список инструментов, доступных агенту.
    :param llm: Языковая модель (BaseChatModel).
    :param prompt: Шаблон системного промпта.
    :param lineage_service: Опциональный сервис записи worker/task nodes.
    :param artifact_service: Опциональный сервис записи результатов worker как artifacts.
    :return: Команда LangGraph с обновлённым планом и маршрутом к валидатору.
    """
    task = payload.task
    task_id = task.task_id or DEFAULT_TASK_ID
    task.status = TaskStatus.RUNNING
    lineage_events: list[dict[str, Any]] = []
    artifact_index: dict[str, Any] = {}
    tool_traces: list[dict[str, Any]] = []
    react_message_tool_calls: list[dict[str, Any]] = []
    loaded_skills = _load_task_skills(
        task,
        skills_service,
        available_skill_names=list(payload.skill_previews.keys()),
    )
    worker_started_node_id = _create_worker_started_lineage(
        payload=payload,
        task=task,
        loaded_skills=loaded_skills,
        lineage_service=lineage_service,
        lineage_events=lineage_events,
    )

    selected_tools = _select_task_tools(tools, task)
    runtime_tools = _with_artifact_read_tools(
        tools=selected_tools,
        artifact_service=artifact_service,
        run_id=payload.run_id,
    )
    runtime_tools = _with_skill_read_tools(
        tools=runtime_tools,
        skills_service=skills_service,
    )

    # Подготовка инструментов с контекстом текущей задачи
    prepared_tools = await _prepare_tools(runtime_tools, task)
    prepared_tools = wrap_tools_for_artifacts(
        tools=prepared_tools,
        artifact_service=artifact_service,
        run_id=payload.run_id,
        node_id=worker_started_node_id,
        task=task,
        artifact_index=artifact_index,
        tool_traces=tool_traces,
        sandbox=sandbox,
    )

    # Собираем список установленных пакетов для информирования модели
    installed_packages = sandbox.get_installed_packages()

    # Формирование системного промпта
    system_prompt = await _create_worker_system_prompt(
        payload, prompt,
        loaded_skills=loaded_skills,
        installed_packages=installed_packages,
    )
    initial_user_query = _initial_user_query_from_graph_state(config) or payload.initial_user_query
    human_invoke = _worker_human_invoke_message(task, initial_user_query)
    prompt_trace_artifacts = write_prompt_trace(
        artifact_service=artifact_service,
        run_id=payload.run_id,
        node_id=worker_started_node_id,
        stage="worker",
        system_prompt=system_prompt,
        human_prompt=human_invoke,
        payload={
            "task": task.model_dump(mode="json"),
            "initial_user_query": initial_user_query,
            "resolved_inputs": payload.resolved_inputs,
            "dependency_context": payload.dependency_context,
            "artifact_context": payload.artifact_context,
            "loaded_skill_names": list(loaded_skills.keys()),
        },
    )
    if prompt_trace_artifacts:
        artifact_index.update(prompt_trace_artifacts)

    # Создание ReAct-агента
    agent = create_react_agent(
        model=llm,
        tools=prepared_tools,
        prompt=system_prompt,
    )

    try:
        response = await _astream_react_agent_to_console(
            agent,
            {"messages": [HumanMessage(content=human_invoke)]},
            config={"recursion_limit": REACT_AGENT_RECURSION_LIMIT},
            task_id=task_id,
        )
        messages = response.get("messages", [])
        react_message_tool_calls = extract_react_tool_calls_from_messages(messages)
        tool_call_artifacts = write_tool_calls_trace(
            artifact_service=artifact_service,
            run_id=payload.run_id,
            node_id=worker_started_node_id,
            stage="worker",
            tool_calls=react_message_tool_calls,
            task_id=task_id,
        )
        if tool_call_artifacts:
            artifact_index.update(tool_call_artifacts)

        last_tool_msg = _get_last_message(messages, ToolMessage)
        last_ai_msg = _get_last_message(messages, AIMessage)

        if last_tool_msg is not None:
            # Разбираем структурированный вывод инструмента
            success = await _apply_tool_output(task, str(last_tool_msg.content))
        else:
            # Инструмент не вызывался — используем текст ответа модели
            fallback_text = (
                str(last_ai_msg.content)
                if last_ai_msg
                else str(messages[-1].content)
            )
            task.result_preview = fallback_text[:RESULT_PREVIEW_MAX_LEN]
            task.error_log = None
            success = True

        # Сохраняем полный текстовый ответ модели
        if last_ai_msg is not None:
            task.full_result = str(last_ai_msg.content)
        elif messages:
            task.full_result = str(messages[-1].content)

        task.status = (
            TaskStatus.NEEDS_VALIDATION if success else TaskStatus.FAILED
        )

    except Exception as exc:
        task.status = TaskStatus.FAILED
        task.error_log = str(exc)
        react_message_tool_calls = []

    await _print_content_block(
        f"WORKER MODEL RESPONSE: task {task_id}",
        _format_worker_console_result(
            task=task,
            selected_tools=selected_tools,
            prepared_tools=prepared_tools,
        ),
    )

    # Получаем актуальные превью всех переменных из песочницы
    current_data_schemas = await sandbox.get_all_variable_previews()
    finished_node_id = _create_worker_finished_lineage(
        payload=payload,
        task=task,
        data_schemas=current_data_schemas,
        lineage_service=lineage_service,
        artifact_service=artifact_service,
        parent_node_id=worker_started_node_id,
        lineage_events=lineage_events,
        artifact_index=artifact_index,
        tool_traces=tool_traces,
    )
    _shrink_task_for_state(task)

    update_payload: dict[str, Any] = {
        "plan": {task_id: task},
        "data_schemas": current_data_schemas,
    }
    if artifact_index:
        update_payload["artifact_index"] = artifact_index
    if loaded_skills:
        update_payload["loaded_skills"] = loaded_skills
    if tool_traces:
        update_payload["tool_traces"] = tool_traces
    if lineage_events:
        update_payload["lineage_events"] = lineage_events

    return Command(
        update=update_payload,
        goto=[
            Send(
                CRITIC_NODE_NAME,
                CriticPayload(
                    worker_payload=payload,
                    run_id=payload.run_id,
                    parent_node_ids=[finished_node_id] if finished_node_id else [],
                    artifact_index=artifact_index,
                    tool_traces=tool_traces,
                    react_message_tool_calls=react_message_tool_calls,
                ),
            )
        ],
    )


def _shrink_task_for_state(task: Task) -> None:
    """Сжимает тяжелые поля задачи перед сохранением в runtime state.

    Args:
        task: Задача worker-а, которую нужно сохранить в план.

    Returns:
        ``None``. Функция изменяет поля задачи на месте, оставляя полные данные
        доступными через artifacts, записанные до сжатия.
    """

    if task.full_result:
        task.full_result = _limit_text(
            task.full_result,
            max_chars=FULL_RESULT_STATE_MAX_LEN,
        )
    if task.result_preview:
        task.result_preview = _limit_text(
            task.result_preview,
            max_chars=RESULT_PREVIEW_MAX_LEN,
        )
    if task.error_log:
        task.error_log = _limit_text(task.error_log, max_chars=WORKER_LOG_MAX_LEN)


def _with_artifact_read_tools(
        *,
        tools: list[BaseTool],
        artifact_service: ArtifactService | None,
        run_id: str,
) -> list[BaseTool]:
    """Добавляет runtime tools для чтения artifacts в текущем ResearchRun.

    Args:
        tools: Базовый список tools worker.
        artifact_service: Сервис artifacts или ``None``.
        run_id: Идентификатор текущего ResearchRun.

    Returns:
        Список tools с добавленными artifact_list/artifact_preview/artifact_read_chunk.
    """

    if artifact_service is None or not run_id:
        return tools

    existing_names = {tool.name for tool in tools}
    artifact_tools = [
        tool
        for tool in build_artifact_read_tools(
            artifact_service=artifact_service,
            run_id=run_id,
        )
        if tool.name not in existing_names
    ]
    return [*tools, *artifact_tools]


def _with_skill_read_tools(
        *,
        tools: list[BaseTool],
        skills_service: SkillsService | None,
) -> list[BaseTool]:
    """Добавляет runtime tools для самостоятельной загрузки skills worker-ом.

    Args:
        tools: Базовый список инструментов worker-а.
        skills_service: Сервис чтения skills или ``None``.

    Returns:
        Список tools с добавленными ``skill_list`` и ``skill_view`` без дублей.
    """

    existing_names = {tool.name for tool in tools}
    skill_tools = [
        tool
        for tool in build_skill_read_tools(skills_service)
        if tool.name not in existing_names
    ]
    return [*tools, *skill_tools]


def _create_worker_started_lineage(
        *,
        payload: WorkerPayload,
        task: Task,
        lineage_service: LineageService | None,
        lineage_events: list[dict[str, Any]],
        loaded_skills: dict[str, str] | None = None,
) -> str | None:
    if lineage_service is None or not payload.run_id:
        return None

    task_id = task.task_id or DEFAULT_TASK_ID
    node = lineage_service.create_state_node(
        run_id=payload.run_id,
        node_type="worker_started",
        title=f"Worker started: task {task_id}",
        parent_ids=payload.parent_node_ids,
        status="running",
        summary=task.description[:500],
        state={
            "run_id": payload.run_id,
            "task": task.model_dump(mode="json"),
            "context_schemas": payload.context_schemas,
            "previous_results": payload.previous_results,
            "resolved_inputs": payload.resolved_inputs,
            "dependency_context": payload.dependency_context,
            "filesystem_context": payload.filesystem_context,
            "skill_previews": payload.skill_previews,
            "artifact_context": payload.artifact_context,
            "loaded_skills": loaded_skills or {},
        },
        created_by="agent",
        metadata={
            "task_id": task_id,
            "dependencies": task.dependencies,
            "loaded_skill_names": list((loaded_skills or {}).keys()),
            "artifact_context_count": payload.artifact_context.get("artifact_count", 0),
            "has_filesystem_context": bool(payload.filesystem_context),
        },
    )
    lineage_events.append(node.model_dump(mode="json"))
    return node.node_id


def _invoked_tool_names(tool_traces: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for trace in tool_traces or []:
        name = trace.get("tool_name")
        if not name or name in seen:
            continue
        seen.add(str(name))
        names.append(str(name))
    return names


def _create_worker_finished_lineage(
        *,
        payload: WorkerPayload,
        task: Task,
        data_schemas: dict[str, str],
        lineage_service: LineageService | None,
        parent_node_id: str | None,
        lineage_events: list[dict[str, Any]],
        artifact_service: ArtifactService | None = None,
        artifact_index: dict[str, Any] | None = None,
        tool_traces: list[dict[str, Any]] | None = None,
) -> str | None:
    if lineage_service is None or not payload.run_id:
        return None

    task_id = task.task_id or DEFAULT_TASK_ID
    succeeded = task.status == TaskStatus.NEEDS_VALIDATION
    node_type = "task_completed" if succeeded else "task_failed"
    invoked_tools = _invoked_tool_names(tool_traces)
    node = StateNode(
        run_id=payload.run_id,
        node_type=node_type,
        title=f"Task finished: {task_id}",
        parent_ids=[parent_node_id] if parent_node_id else payload.parent_node_ids,
        status="succeeded" if succeeded else "failed",
        summary=(task.result_preview or task.error_log or task.description)[:500],
        created_by="agent",
        metadata={
            "task_id": task_id,
            "task_status": task.status.value,
            "validation_required": task.status == TaskStatus.NEEDS_VALIDATION,
            "invoked_tool_names": invoked_tools,
        },
    )

    written_artifacts = _write_worker_artifacts(
        artifact_service=artifact_service,
        run_id=payload.run_id,
        node_id=node.node_id,
        task=task,
    )
    if written_artifacts:
        if artifact_index is not None:
            artifact_index.update(written_artifacts)
    node.artifact_refs = list(dict.fromkeys(task.artifact_refs))

    node = lineage_service.append_node(
        node,
        state={
            "run_id": payload.run_id,
            "task": task.model_dump(mode="json"),
            "data_schemas": data_schemas,
            "artifact_index": artifact_index or written_artifacts,
        },
    )
    lineage_events.append(node.model_dump(mode="json"))
    return node.node_id


def _write_worker_artifacts(
        *,
        artifact_service: ArtifactService | None,
        run_id: str,
        node_id: str,
        task: Task,
) -> dict[str, Any]:
    if artifact_service is None:
        return {}

    task_id = task.task_id or DEFAULT_TASK_ID
    safe_task_id = _safe_filename_fragment(task_id)
    retry_suffix = f"_r{task.retry_count}" if task.retry_count else ""
    artifacts: dict[str, Any] = {}

    result_content = _task_result_content(task)
    if result_content:
        result_artifact = artifact_service.write_artifact(
            run_id=run_id,
            node_id=node_id,
            kind="model_output",
            filename=f"tasks/{safe_task_id}/result.md",
            content=result_content,
            mime_type="text/markdown",
            summary=result_content[:ARTIFACT_SUMMARY_MAX_LEN],
            metadata={
                "task_id": task_id,
                "task_status": task.status.value,
                "artifact_role": "worker_result",
            },
            artifact_id=f"t{safe_task_id}{retry_suffix}_result",
        )
        task.artifact_refs.append(result_artifact.artifact_id)
        artifacts[result_artifact.artifact_id] = result_artifact.model_dump(mode="json")

    if task.generated_code:
        code_artifact = artifact_service.write_artifact(
            run_id=run_id,
            node_id=node_id,
            kind="code_trace",
            filename=f"tasks/{safe_task_id}/code_trace.txt",
            content=task.generated_code,
            mime_type="text/plain",
            summary=task.generated_code[:ARTIFACT_SUMMARY_MAX_LEN],
            metadata={
                "task_id": task_id,
                "task_status": task.status.value,
                "artifact_role": "generated_code_trace",
                "persistent_executable": False,
            },
            artifact_id=f"t{safe_task_id}{retry_suffix}_code",
        )
        artifacts[code_artifact.artifact_id] = code_artifact.model_dump(mode="json")
        # generated_code уже доступен напрямую через task.generated_code и
        # tool_calls trace, поэтому в task.artifact_refs его не дублируем.

    return artifacts


def _task_result_content(task: Task) -> str:
    return (
            task.full_result
            or task.result_preview
            or task.error_log
            or ""
    )


def _safe_filename_fragment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or DEFAULT_TASK_ID

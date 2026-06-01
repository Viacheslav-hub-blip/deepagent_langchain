"""Сборка native DeepAgents агента для аналитики данных.

Главный файл сборки. Точка входа — :func:`build_analytics_deep_agent`: она собирает
supervisor по нумерованным шагам (settings -> data tools -> middleware -> backend ->
subagents -> custom tools -> ``create_deep_agent``).

Как редактировать/кастомизировать (подробности — в ``README.md``):
- Данные: передай свои ``data_tools=[...]`` в :func:`build_analytics_deep_agent` или укажи
  ``data_tools_factory`` в конфиге (``resources/config/defaults.json`` / override через
  ``DEEP_AGENT_CONFIG_PATH``).
- Конфиг и пороги: ключи в ``resources/config/defaults.json`` (offload, skills, лимиты).
- Поведение supervisor/critic: правь общие prompts в ``core/prompts.py`` (без доменной логики).
- Доменные знания: добавляй/редактируй ``resources/skills/<name>/SKILL.md`` — менять код не нужно.
- Внутренний critic: включается флагом ``enable_retrieval_critic`` (см. шаг 5). При
  ``false`` critic не подключается и не влияет на сборку.
- Новые subagents: расширь ``build_analytics_subagent_specs`` в ``retrieval_subagents.py``.

Служебные функции:
- build_data_tools: сборка инструментов чтения данных через фабрику из настроек.
- _load_callable_from_path: импорт callable по строковому пути.
- _normalize_data_tools: проверка и нормализация списка инструментов.
- build_analytics_deep_agent: сборка supervisor и subagents.
- build_skills_backend: сборка backend для skills и tool outputs.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    FilesystemFileSearchMiddleware,
    ModelCallLimitMiddleware,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from deep_agent_test.core.agent_specs import DATA_RETRIEVAL_CRITIC_AGENT_NAME
from deep_agent_test.middlewares.critic_loop_cap import CriticLoopCapMiddleware
from deep_agent_test.core.retrieval_subagents import build_analytics_subagent_specs
from deep_agent_test.tools.data_tools_wrapper import wrap_data_tools_with_query_code
from deep_agent_test.tools.execute_python_code import build_execute_python_code_tool
from deep_agent_test.tools.load_skills import build_load_skills_tool
from deep_agent_test.core.prompts import (
    BUILTIN_TOOLS_PROMPT_APPEND_RU,
    RUSSIAN_TOOL_DESCRIPTION_OVERRIDES,
    SYSTEM_PROMPT,
)
from deep_agent_test.core.python_sandbox import build_python_sandbox
from deep_agent_test.core.settings import DeepAgentSettings, load_deep_agent_settings
from deep_agent_test.middlewares.skills_context import PreloadedSkillsContextMiddleware
from deep_agent_test.middlewares.tool_loop_guard import ToolLoopGuardMiddleware
from deep_agent_test.middlewares.tool_output_file import ToolOutputFileMiddleware
from deep_agent_test.middlewares.tool_descriptions import PromptToolDescriptionsMiddleware


def build_data_tools(settings: DeepAgentSettings | None = None) -> list[BaseTool]:
    """Собирает инструменты чтения данных через фабрику из JSON-конфига."""

    settings = settings or load_deep_agent_settings()
    if not settings.data_tools_factory:
        raise ValueError(
            "Не настроена фабрика tools чтения данных. "
            "Передайте data_tools в build_analytics_deep_agent или укажите "
            "data_tools_factory в deep_agent_test/resources/config/defaults.json или override-конфиге."
        )
    factory = _load_callable_from_path(settings.data_tools_factory)
    return _normalize_data_tools(factory(**settings.data_tools_factory_kwargs))


def _load_callable_from_path(import_path: str) -> Callable[..., Any]:
    """Импортирует callable по строке вида ``module:attr`` или ``module.attr``."""

    if ":" in import_path:
        module_name, attribute_name = import_path.split(":", 1)
    else:
        try:
            module_name, attribute_name = import_path.rsplit(".", 1)
        except ValueError:
            raise ValueError(f"Некорректный import path фабрики tools: {import_path}") from None
    if not module_name or not attribute_name:
        raise ValueError(f"Некорректный import path фабрики tools: {import_path}")

    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError(f"Объект {import_path} не является callable.")
    return factory


def _normalize_data_tools(raw_tools: Any) -> list[BaseTool]:
    """Приводит результат фабрики data-tools к списку ``BaseTool`` с проверкой типов."""

    if isinstance(raw_tools, BaseTool):
        return [raw_tools]
    if not isinstance(raw_tools, (list, tuple)):
        raise TypeError("Фабрика tools должна вернуть BaseTool или список BaseTool.")

    tools: list[BaseTool] = []
    for item in raw_tools:
        if not isinstance(item, BaseTool):
            raise TypeError(f"Фабрика tools вернула объект не BaseTool: {type(item).__name__}")
        tools.append(item)
    return tools


def build_analytics_deep_agent(
    model: Any,
    settings: DeepAgentSettings | None = None,
    data_tools: list[BaseTool] | None = None,
) -> Any:
    """Собирает аналитический DeepAgent supervisor по шагам инициализации.

    Это главная точка сборки агента. Сборка нативная для DeepAgents: supervisor получает
    встроенные tools (`write_todos`, filesystem, `task`), два custom middleware, custom
    tools `execute_python_code` и `load_skills`, и один subagent с внутренним critic.

    Шаги инициализации (см. нумерацию в теле функции) и точки кастомизации:

    1. Settings — все пороги и пути. Кастомизация: `resources/config/defaults.json`, override-файл
       через env `DEEP_AGENT_CONFIG_PATH`, либо готовый ``settings`` в аргументе.
    2. Data tools — инструменты чтения данных (`load_data`). Кастомизация: передай свои
       ``BaseTool`` в ``data_tools`` или укажи фабрику в ``data_tools_factory`` конфига.
    3. Middleware — два custom-механизма (принудительная загрузка skills, offload больших
       tool outputs в pickle) плюс нативные/кастомные middleware:
       ContextEditingMiddleware (очистка старых tool-результатов при лимите токенов),
       FilesystemFileSearchMiddleware (glob/grep поиск по spill-файлам),
       ToolLoopGuardMiddleware (защита от зацикливания на повторных вызовах tool),
       CriticLoopCapMiddleware (лимит циклов critic — только при включённом critic) и
       нативный ModelCallLimitMiddleware (бюджет ходов одного запуска субагента).
       Кастомизация: пороги в settings; модель выбора skills.
    4. Backend — где DeepAgents видит skills и spill-файлы.
    5. Subagents — `data-retrieval-agent`; внутренний `data-retrieval-critic` подключается
       по флагу ``settings.enable_retrieval_critic`` (при ``false`` субагент отдаёт отчёт
       supervisor-у напрямую). Кастомизация: ``build_analytics_subagent_specs``.
    6. Custom tools supervisor — `execute_python_code` (расчёты, чтение `.pkl`) и
       `load_skills` (пакетная загрузка skills).
    7. Сборка `create_deep_agent(...)` со всеми частями.

    Args:
        model: Chat model LangChain для supervisor, subagent и critic.
        settings: Готовые настройки; если ``None`` — загружаются из JSON-конфига.
        data_tools: Готовые tools чтения данных; если ``None`` — берутся из фабрики конфига.

    Returns:
        Скомпилированный DeepAgents граф (supervisor), готовый к ``invoke``/``stream``.
    """

    # Шаг 1. Настройки: пути skills, папка spill-файлов, пороги offload, thread_id.
    settings = settings or load_deep_agent_settings()

    # Шаг 2. Инструменты чтения данных. Аргумент имеет приоритет над фабрикой из конфига.
    # Оборачиваем их в слой прозрачности: агент получает сгенерированный код запроса и
    # счётчики строк, а большие таблицы корректно уходят в offload (artifact с rows).
    if data_tools is None:
        data_tools = build_data_tools(settings)
    data_tools = wrap_data_tools_with_query_code(data_tools)

    # Шаг 3. Два custom middleware.
    # 3a. Принудительная загрузка skills. Supervisor через LLM выбирает релевантные skills
    #     по запросу и кладёт их контент в system prompt до первого хода модели, кэшируя
    #     выбор в общий словарь. Субагенты переиспользуют этот же выбор (тот же набор
    #     skills, без повторного LLM-вызова и дублей в контексте). Critic skills не получает.
    shared_skills_selection: dict[str, Any] = {}
    supervisor_skills_middleware = PreloadedSkillsContextMiddleware(
        skills_root=settings.skills_root,
        skills_virtual_dir=settings.skills_virtual_dir,
        max_chars_per_file=settings.max_chars_per_skill,
        model=model,
        select_skills=True,
        shared_selection=shared_skills_selection,
    )
    subagent_skills_middleware = PreloadedSkillsContextMiddleware(
        skills_root=settings.skills_root,
        skills_virtual_dir=settings.skills_virtual_dir,
        max_chars_per_file=settings.max_chars_per_skill,
        model=model,
        select_skills=False,
        shared_selection=shared_skills_selection,
    )
    # 3b. Offload больших табличных tool outputs в pickle, чтобы не раздувать контекст.
    tool_output_file_middleware = ToolOutputFileMiddleware(
        output_dir=settings.tool_outputs_dir,
        min_rows_to_save=settings.tool_output_min_rows_to_save,
        min_content_chars_to_save=settings.tool_output_min_content_chars_to_save,
        preview_rows=settings.tool_output_preview_rows,
        inline_original_content_chars=settings.tool_output_inline_original_chars,
    )
    # 3c. Context editing: очищает старые tool-результаты при достижении лимита токенов,
    #     оставляя последние N (дополняет offload, держит контекст компактным).
    context_editing_middleware = ContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=settings.context_edit_trigger_tokens,
                keep=settings.context_edit_keep_tool_results,
            )
        ]
    )
    # 3d. File search: glob/grep поиск по папке spill-файлов (.pkl). Папку создаём заранее,
    #     чтобы root_path существовал. use_ripgrep=False не требует бинарника rg на хосте.
    settings.tool_outputs_dir.mkdir(parents=True, exist_ok=True)
    file_search_middleware = FilesystemFileSearchMiddleware(
        root_path=str(settings.tool_outputs_dir),
        use_ripgrep=settings.file_search_use_ripgrep,
    )
    # 3e. Loop guard: блокирует серию подряд идущих вызовов одного tool после N повторов
    #     и просит модель сменить подход или завершить шаг (защита от зацикливания).
    tool_loop_guard_middleware = ToolLoopGuardMiddleware(
        max_consecutive_tool_calls=settings.max_consecutive_tool_calls,
    )
    # 3f. Prompt-only переопределение descriptions встроенных tools и финальный
    #     русский блок с приоритетом над англоязычными examples из DeepAgents.
    tool_descriptions_middleware = PromptToolDescriptionsMiddleware(
        tool_descriptions=RUSSIAN_TOOL_DESCRIPTION_OVERRIDES,
        system_prompt_append=BUILTIN_TOOLS_PROMPT_APPEND_RU,
    )
    # 3g. Critic loop cap: ограничивает число циклов task(data-retrieval-critic) внутри
    #     data-retrieval-agent, чтобы он не зацикливался на проверках до лимита рекурсии.
    critic_loop_cap_middleware = CriticLoopCapMiddleware(
        critic_subagent_type=DATA_RETRIEVAL_CRITIC_AGENT_NAME,
        max_critic_iterations=settings.max_critic_iterations,
    )
    # 3h. Бюджет шагов субагента: нативный ModelCallLimitMiddleware ограничивает число
    #     ходов модели внутри одного запуска data-retrieval-agent. По исчерпании лимита
    #     (exit_behavior="end") субагент мягко завершается и возвращает supervisor-у то,
    #     что уже собрал, вместо упора в recursion limit графа.
    subagent_step_limit_middleware = ModelCallLimitMiddleware(
        run_limit=settings.max_subagent_model_calls,
        exit_behavior="end",
    )
    # Базовые middleware без skills — общие для supervisor, data-retrieval-agent и critic.
    base_middleware = [
        tool_output_file_middleware,
        context_editing_middleware,
        file_search_middleware,
        tool_loop_guard_middleware,
        tool_descriptions_middleware,
    ]
    # Supervisor выбирает skills; data-retrieval-agent переиспользует тот же выбор, ограничен
    # лимитом циклов critic и бюджетом шагов на запуск; critic — без skills, без cap и без
    # step-бюджета (сам критика не вызывает и работает в отдельном вложенном запуске).
    # Critic loop cap нужен только когда critic включён; при отключённом critic он бесполезен.
    supervisor_middleware = [supervisor_skills_middleware, *base_middleware]
    subagent_middleware = [
        subagent_skills_middleware,
        *base_middleware,
        *([critic_loop_cap_middleware] if settings.enable_retrieval_critic else []),
        subagent_step_limit_middleware,
    ]
    critic_middleware = list(base_middleware)

    # Шаг 4. Backend skills/spill-файлов.
    backend = build_skills_backend(settings)

    # Шаг 5. Subagent чтения данных с внутренним critic (critic отключается флагом
    # settings.enable_retrieval_critic — тогда subagent отдаёт отчёт supervisor-у напрямую).
    subagents = build_analytics_subagent_specs(
        settings=settings,
        data_tools=data_tools,
        common_middleware=subagent_middleware,
        critic_middleware=critic_middleware,
        model=model,
        backend=backend,
        enable_critic=settings.enable_retrieval_critic,
    )

    # Шаг 6. Custom tools supervisor: выполнение Python-кода и пакетная загрузка skills.
    python_sandbox = build_python_sandbox(settings)
    python_tool = build_execute_python_code_tool(python_sandbox)
    load_skills_tool = build_load_skills_tool(settings)

    # Шаг 7. Финальная сборка DeepAgents supervisor.
    return create_deep_agent(
        model=model,
        tools=[python_tool, load_skills_tool],
        system_prompt=SYSTEM_PROMPT,
        subagents=subagents,
        skills=[settings.skills_virtual_dir],
        backend=backend,
        middleware=supervisor_middleware,
        checkpointer=MemorySaver(),
    )


def build_skills_backend(settings: DeepAgentSettings | None = None) -> Any:
    """Собирает CompositeBackend: state по умолчанию + read-only skills и tool_outputs.

    Args:
        settings: Настройки агента; если ``None`` — загружаются из JSON-конфига.

    Returns:
        ``CompositeBackend`` с маршрутами на локальные папки skills и spill-файлов.
    """

    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend

    settings = settings or load_deep_agent_settings()
    tool_outputs_virtual = "/tool_outputs/"
    return CompositeBackend(
        default=StateBackend(),
        routes={
            settings.skills_virtual_dir: FilesystemBackend(
                root_dir=settings.skills_root,
                virtual_mode=True,
            ),
            tool_outputs_virtual: FilesystemBackend(
                root_dir=settings.tool_outputs_dir,
                virtual_mode=True,
            ),
        },
    )


__all__ = [
    "build_analytics_deep_agent",
    "build_data_tools",
    "build_skills_backend",
]

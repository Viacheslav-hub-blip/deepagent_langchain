# AnaliticAgenticPlatform

`AnaliticAgenticPlatform` - planner-first аналитический агент на LangGraph и
LangChain. Пакет `planner_agent` строит план исследования, выполняет шаги через
подключенные LangChain tools, сохраняет lineage, snapshots и artifacts, а затем
формирует финальный markdown-отчет.

Проект рассчитан на рабочие аналитические сценарии: источник данных остается
внутри ваших инструментов, LLM получает только управляемые выгрузки, краткие
превью и ссылки на artifacts.

## Возможности

- Планирование задачи в `FullPlan`: цель, шаги, зависимости, ожидаемые outputs,
  рекомендуемые tools и skills.
- Параллельное выполнение независимых задач через LangGraph `Send`.
- Подключение любых LangChain `BaseTool` как source tools.
- Встроенный `execute_python_code` для расчетов в `ClientPythonSandbox`.
- Контроль качества через validator, critic и replanner.
- Сохранение lineage, prompt traces, tool calls, snapshots и artifacts на диск.
- Follow-up и branch-запуски от выбранного узла графа.
- Опциональный FastAPI слой для UI и внешних интеграций.
- Skills и memory как локальная процедурная память проекта.

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Опционально для HTTP API:

```powershell
pip install -e ".[api]"
```

Для разработки:

```powershell
pip install -e ".[dev]"
```

## Быстрая проверка без API-ключей

Демо использует fake LLM и локальные CSV из `examples/data`, поэтому внешние
API не вызываются.

```powershell
python examples\chat_run.py
```

Тесты:

```powershell
$env:PYTHONUTF8=1
python -m unittest discover -s tests
```

Lint:

```powershell
uvx ruff check planner_agent sandbox examples tests main_deepseek_test.py main_e2e_branch_dialog.py main_sandbox_code_example.py main_ui_agent_server.py
```

## Минимальный запуск агента

```python
import os

from langchain_openai import ChatOpenAI

from examples.fake_spark_tools import build_fake_spark_tools
from planner_agent import ResearchAgent
from sandbox import ClientPythonSandbox

llm = ChatOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    model=os.environ.get("AGENT_MODEL", "gpt-4o-mini"),
    temperature=0.1,
)

agent = ResearchAgent(
    model=llm,
    sandbox=ClientPythonSandbox(),
    tools=build_fake_spark_tools(data_dir="examples/data"),
    workspace_root=".",
    sources_dir="examples/data",
    skills_dir="skills",
    memory_dir="memory",
    runs_dir="runs",
)

messages = agent.invoke(
    "Разбери сработку event_id=evt-c42-2025-01-03-block-001",
    config={"recursion_limit": 60},
)
print(messages[-1].content)
```

API-ключи не хранятся в репозитории. Перед запуском реальной модели задайте
переменные окружения, например `OPENROUTER_API_KEY` или `OPENAI_API_KEY`.

## UI API

`main_ui_agent_server.py` больше не зависит от локального `model.py`. Модель
создается из переменных окружения:

- `OPENROUTER_API_KEY` или `OPENAI_API_KEY` - обязательный ключ.
- `AGENT_MODEL` - имя модели, по умолчанию `gpt-4o-mini`.
- `OPENAI_BASE_URL` или `OPENROUTER_BASE_URL` - опциональный endpoint.

Запуск:

```powershell
pip install -e ".[api]"
$env:OPENROUTER_API_KEY="..."
$env:AGENT_MODEL="..."
python main_ui_agent_server.py
```

API будет доступен на `http://127.0.0.1:8000/api/v1`.

## Структура проекта

```text
planner_agent/
  agent_nodes/          # LangGraph nodes: planner, worker, validator, critic, responder
  http_api/             # FastAPI application factory and request schemas
  insight_pipeline/     # Дополнительный pipeline пакетной обработки инсайтов
  runtime/              # Sandbox protocol, workspace helpers, tool result capture
  schemas/              # Публичные Pydantic-схемы и re-export модулей
  services/             # Lineage, artifacts, memory, skills, policy, inspection
  tools/                # Tool registry, skill tools, execute_python_code, wrappers
  factory.py            # Сборка LangGraph workflow
  research_agent.py     # LangChain Runnable facade
sandbox/                # Host-side ClientPythonSandbox
examples/               # Локальные demo tools, CSV и offline examples
tests/                  # Unit tests без внешних API
skills/                 # SKILL.md методики
memory/                 # Локальная память проекта
runs/                   # Локальные результаты запусков, не коммитятся
```

## Интеграция рабочих источников

Агент не подключается к Spark, БД или внутренним сервисам сам. Для рабочего
использования передайте собственные LangChain tools в `ResearchAgent(tools=...)`.
Хороший source tool должен:

- принимать строгую Pydantic-схему входа;
- требовать явный список колонок вместо `*`;
- возвращать `pandas.DataFrame`, компактный `dict/list` или понятную текстовую
  ошибку;
- описывать доступные таблицы, поля и фильтры в `description`;
- не раскрывать секреты и лишние персональные данные.

Большие результаты автоматически сохраняются как artifacts, а в контекст LLM
попадает компактная ссылка с `artifact_id`, summary и метаданными.

## Skills и memory

- `skills/**/SKILL.md` - процедурные методики, чеклисты, glossary и схемы.
- `memory/*.md` - устойчивый контекст проекта или пользователя.
- Runtime tools `list_skills` и `load_skill` позволяют worker-у читать только
  релевантные skills во время выполнения плана.

## Artifacts и GitHub

В репозиторий не должны попадать runtime-данные и локальное окружение:

- `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`;
- `.venv/`, `.uv-cache/`;
- `.idea/`, `.vscode/`;
- `runs/`, `examples/runs/`, `examples/runs_mvp_e2e/`;
- `.env`, `.env.*`, локальный `model.py`.

Для пустых runtime-каталогов можно держать только `.gitkeep`.

## Проверка перед экспортом

```powershell
uvx ruff check planner_agent sandbox examples tests main_deepseek_test.py main_e2e_branch_dialog.py main_sandbox_code_example.py main_ui_agent_server.py
$env:PYTHONUTF8=1
python -m unittest discover -s tests
git status --short
```

Если запускаете реальные LLM-интеграции, используйте только переменные
окружения или secret storage вашей инфраструктуры.

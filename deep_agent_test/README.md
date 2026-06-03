# Аналитический DeepAgent

Этот пакет содержит готовую надстройку над базовым `deepagents`. Цель пакета простая:
дать агенту доменные инструкции, безопасные инструменты чтения данных, контроль больших
ответов инструментов и понятную точку запуска в других проектах.

Локальный тестовый запуск проекта рассчитан на один файл:

```bash
python run.py
```

В `run.py` нет параметров командной строки. В нем собирается fake-инструмент `load_data`
поверх CSV из папки `data`, собирается агент и выполняется один `stream_events`.
Production-инструмент поверх Spark использует тот же `query`-интерфейс и то же описание
tool, но создается через `build_spark_data_tools`.

## Что добавлено к базовому DeepAgent

Базовый `deepagents` уже умеет вызывать инструменты, запускать subagent-ов, читать файлы
и вести список задач. В этом проекте поверх него добавлены конкретные вещи для
аналитики таблиц.

1. Предзагрузка skills.

   Перед первым ответом агент выбирает нужные файлы `SKILL.md` из
   `deep_agent_test/resources/skills` и добавляет их в system prompt. `SKILL.md`
   сделаны короткими: они содержат карточки источников и workflow, а полные справочники
   полей лежат рядом в `fields.md`/`joins.md` и читаются только при необходимости.
   Эти же skills передаются в `data-retrieval-agent`, поэтому supervisor и subagent
   работают с одним набором доменных правил.

2. Data-retrieval subagent.

   Основной агент не читает таблицы напрямую. Для чтения данных он вызывает
   `data-retrieval-agent`. Этот subagent получает задачу, вызывает `load_data` и
   возвращает supervisor-у короткий структурированный отчет.

3. Опциональный critic для чтения данных.

   Внутри `data-retrieval-agent` можно включить `data-retrieval-critic`. Он проверяет,
   что ответ действительно основан на результатах инструментов и что заявленные файлы
   существуют. Флаг включения находится в конфиге: `enable_retrieval_critic`.

4. Инструменты чтения данных.

   `build_spark_data_tools(spark, query_parser_model=model)` создает настоящий
   `load_data` поверх Spark session. `build_fake_spark_data_tools(query_parser_model=model)`
   создает fake `load_data` поверх локальных CSV. Оба инструмента имеют одинаковые
   `name`, `description` и `args_schema`: один аргумент `query` с SQL-подобным запросом,
   alias таблицы, обязательным периодом, явными колонками результата, фильтрами,
   агрегациями и сортировкой. Разбор `query` выполняется LLM-нормализатором, а не
   regex-парсером.

5. Прозрачный ответ `load_data`.

   Обертка над data-tools добавляет к результату SQL-подобное описание запроса:
   какие поля читались, из какой таблицы, какие фильтры применялись. Это снижает риск,
   что агент перепутает пример строк с полным результатом.

6. Offload больших таблиц.

   Если tool возвращает много строк или слишком большой текст, результат сохраняется в
   pickle в `runs/deep_agent_tool_outputs`. В контекст агента попадает короткое описание,
   путь к файлу и preview. Полный файл можно читать через `execute_python_code`.

7. Безопасный Python sandbox.

   Tool `execute_python_code` нужен для расчетов по выгруженным данным, чтения pickle,
   join, фильтрации и подготовки итоговых таблиц. Перед выполнением код проходит
   простую проверку: запрещены `eval`, `exec`, shell-вызовы и удаление файлов.

8. Защита от циклов инструментов.

   `ToolLoopGuardMiddleware` останавливает серию одинаковых tool-вызовов, если агент
   зациклился на одном инструменте.

## Структура пакета

```text
deep_agent_test/
  core/
    analytics_deep_agent.py   # сборка агента
    retrieval_subagents.py    # data-retrieval-agent и critic
    settings.py               # загрузка настроек
    prompts.py                # общие промпты supervisor/subagent/critic
    state.py                  # дополнительные поля state
    python_sandbox.py         # persistent Python namespace
    agent_specs.py            # имена агентов и structured output critic-а
    trace_logging.py          # подробный trace одного запуска агента

  middlewares/
    skills_context.py         # предзагрузка skills
    tool_output_file.py       # сохранение больших результатов tool в pickle
    tool_loop_guard.py        # защита от повторяющихся tool-вызовов
    critic_loop_cap.py        # лимит проверок critic-а

  tools/
    spark_data.py             # load_data поверх Spark session
    fake_spark_data.py        # load_data поверх локальных CSV для тестов без Spark
    data_query_schema.py      # pydantic-схемы query и LLM-разбора
    data_tools_wrapper.py     # прозрачное описание запросов к data-tools
    execute_python_code.py    # безопасное выполнение Python-кода
    load_skills.py            # ручная дозагрузка skills
    inspect_artifact.py       # проверка файлов для critic-а

  resources/
    config/defaults.json      # настройки по умолчанию
    skills/**/SKILL.md        # короткие карточки источников и workflow
    skills/**/fields.md       # подробные поля, читаются по необходимости
    skills/**/joins.md        # подробные правила связи источников
```

Целевой запуск идет через `python run.py`. Trace-логгер подключается как LangChain
callback и пишет подробный txt-файл по каждому запросу к LLM.

## Минимальный fake-запуск

Файл `run.py` находится в корне проекта и по умолчанию использует fake-данные:

```python
from deep_agent_test import build_analytics_deep_agent, load_deep_agent_settings
from deep_agent_test.core.trace_logging import FileTraceCallbackHandler, build_trace_file_path
from deep_agent_test.tools.fake_spark_data import build_fake_spark_data_tools
from model import model

USER_MESSAGE = "текст запроса пользователя"

settings = load_deep_agent_settings()
data_tools = build_fake_spark_data_tools(query_parser_model=model)
agent = build_analytics_deep_agent(model=model, settings=settings, data_tools=data_tools)
trace_file_path = build_trace_file_path(settings.trace_log_dir)
trace_handler = FileTraceCallbackHandler(trace_file_path)
result = agent.invoke(
    {"messages": [{"role": "user", "content": USER_MESSAGE}]},
    config={
        "callbacks": [trace_handler],
        "configurable": {"thread_id": settings.thread_id},
        "recursion_limit": settings.graph_recursion_limit,
    },
)
print(f"Trace log: {trace_file_path}")
```

Чтобы проверить настоящий Spark-инструмент, замените только сборку data-tools:

```python
from pyspark.sql import SparkSession

from deep_agent_test import build_spark_data_tools

spark = SparkSession.builder.appName("analytics-deep-agent").getOrCreate()
data_tools = build_spark_data_tools(spark, query_parser_model=model)
```

Остальная сборка агента не меняется. Чтобы задать другой запрос, измените константу
`USER_MESSAGE`.

## Конфигурация

Основной конфиг лежит здесь:

```text
deep_agent_test/resources/config/defaults.json
```

Главные параметры:

- `skills_root` - локальная папка со skills.
- `skills_virtual_dir` - виртуальный путь, который видит DeepAgent.
- `tool_outputs_dir` - папка для pickle-файлов с большими результатами.
- `max_chars_per_skill` - максимальный размер одного skill в prompt.
- `tool_output_min_rows_to_save` - после какого числа строк сохранять результат в файл.
- `context_edit_trigger_tokens` - когда чистить старые tool results из контекста.
- `max_consecutive_tool_calls` - сколько одинаковых вызовов tool подряд разрешено.
- `max_subagent_model_calls` - лимит шагов модели внутри data-retrieval-agent.
- `max_critic_iterations` - лимит проверок critic-а.
- `enable_retrieval_critic` - включать ли внутренний critic.
- `trace_log_dir` - папка для txt-логов с содержимым запросов к LLM.

Если нужен отдельный конфиг для другого проекта, укажите путь в переменной окружения
`DEEP_AGENT_CONFIG_PATH`. Значения из этого файла переопределят defaults.

## Trace-лог

Каждый вызов модели записывается отдельным блоком `LLM REQUEST #N`. В начале блока
есть сводка:

- `messages_count` и `tools_count` - сколько сообщений и tools ушло в этот запрос;
- `messages_chars`, `tools_chars` и `total_tokens_estimate` - грубая оценка объема
  контекста;
- `messages_table` - таблица всех сообщений с ролью, классом, размером и числом
  tool calls.

После сводки идут секции `LLM REQUEST #N TOOLS` и `LLM REQUEST #N MESSAGE #M`.
Они содержат полный набор tools и полный content каждого сообщения, которое попало
в конкретный запрос к LLM.

## Формат `load_data`

`load_data` принимает один параметр `query`. Внутри `query` передается SQL-подобный
запрос. Для каждой выборки обязателен `PERIOD`; `SELECT *` и `SELECT all` запрещены.

Пример обычной выборки:

```text
query:
LOAD uko
PERIOD event_dt FROM '20260123' TO '20260124'
SELECT event_id, event_dt, event_dttm_readable, epk_id, event_description, transaction_amount
WHERE epk_id = '2099007770421989000001'
ORDER BY event_dt ASC, event_dttm_readable ASC
```

Пример агрегации:

```text
query:
LOAD cards
PERIOD event_dt FROM '20260101' TO '20260131'
SELECT event_description, COUNT(*) AS events_count, sum(transaction_amount_in_rub) AS amount_rub
GROUP BY event_description
ORDER BY events_count DESC
```

Пример вычисляемой колонки:

```text
query:
LOAD cards
PERIOD event_dt FROM '20260101' TO '20260131'
DERIVE event_month = year_month(event_dt)
SELECT event_month, count(event_id) AS events_count
WHERE event_month = '202601'
GROUP BY event_month
```

Поддерживаемые операторы фильтра:

```text
=, !=, <>, >, >=, <, <=, LIKE, CONTAINS, IN (...), BETWEEN, AND, OR
```

Поддерживаемые операции для `DERIVE`:

```text
year, month, year_month, date, lower, upper, length, abs
```

Поддерживаемые агрегаты:

```text
count, count_distinct, min, max, sum, mean. `COUNT(*)` разрешён.
```

## Skills

Skills лежат в:

```text
deep_agent_test/resources/skills
```

Каждый skill - это папка с коротким файлом `SKILL.md`. Он должен описывать один
понятный участок домена: таблицу, правило поиска или тип аналитического запроса.
`SKILL.md` попадает в preload context, поэтому держите его компактным.

Подробный контекст выносится в соседние файлы:

- `fields.md` - полный список полей и описания редких колонок;
- `joins.md` - правила связи таблиц и fallback-маршруты;
- другие файлы - только если они читаются по явному триггеру из `SKILL.md`.

Пример структуры:

```text
resources/skills/hit-table/SKILL.md
resources/skills/hit-table/fields.md
resources/skills/hit-table/joins.md
resources/skills/cards-event-table/SKILL.md
resources/skills/cards-event-table/fields.md
resources/skills/uko-event-table/SKILL.md
resources/skills/uko-event-table/fields.md
```

Когда добавлять новый skill:

- появилась новая таблица;
- появились новые поля с важными правилами интерпретации;
- агент часто ошибается в одном и том же типе запроса;
- нужно зафиксировать правила связи между источниками.

Как добавлять:

- в `SKILL.md` добавляйте только назначение источника, alias, зерно, ключи, главные
  поля, критические ограничения и ссылки на дополнительные файлы;
- полный список полей добавляйте в `fields.md`;
- в `SKILL.md` явно пишите, когда читать `fields.md` или `joins.md`, например:
  schema error, редкое поле, вопрос про смысл поля, маршрут связи.

Когда не добавлять новый skill:

- правило нужно только для одного конкретного запуска;
- это временная подсказка;
- это можно выразить в пользовательском запросе.

## Как переиспользовать в другом проекте

1. Скопируйте пакет `deep_agent_test` и корневой `run.py`.
2. Подключите свою модель в `model.py`.
3. Настройте Spark session в `run.py`.
4. Проверьте, что `spark.table(table_name)` видит нужные таблицы.
5. Обновите `resources/skills` под свой домен.
6. При необходимости переопределите `resources/config/defaults.json` через
   `DEEP_AGENT_CONFIG_PATH`.

Код агента не должен знать бизнес-смысл таблиц. Этот смысл должен жить в skills:
короткая маршрутизация в `SKILL.md`, подробности в `fields.md` и `joins.md`.
Так пакет проще переносить между проектами: код отвечает за механику, skills отвечают
за домен.

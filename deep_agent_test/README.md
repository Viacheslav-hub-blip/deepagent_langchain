# Аналитический DeepAgent

Этот пакет содержит готовую надстройку над базовым `deepagents`. Цель пакета простая:
дать агенту доменные инструкции, безопасные инструменты чтения данных, контроль больших
ответов инструментов и понятную точку запуска в других проектах.

Запуск проекта рассчитан на один файл:

```bash
python run.py
```

В `run.py` нет параметров командной строки. В нем создается Spark session, собирается
инструмент `load_data`, собирается агент и выполняется один `invoke`.

## Что добавлено к базовому DeepAgent

Базовый `deepagents` уже умеет вызывать инструменты, запускать subagent-ов, читать файлы
и вести список задач. В этом проекте поверх него добавлены конкретные вещи для
аналитики таблиц.

1. Предзагрузка skills.

   Перед первым ответом агент выбирает нужные файлы `SKILL.md` из
   `deep_agent_test/resources/skills` и добавляет их в system prompt. Эти же skills
   передаются в `data-retrieval-agent`, поэтому supervisor и subagent работают с одним
   набором доменных правил.

2. Data-retrieval subagent.

   Основной агент не читает таблицы напрямую. Для чтения данных он вызывает
   `data-retrieval-agent`. Этот subagent получает задачу, вызывает `load_data` и
   возвращает supervisor-у короткий структурированный отчет.

3. Опциональный critic для чтения данных.

   Внутри `data-retrieval-agent` можно включить `data-retrieval-critic`. Он проверяет,
   что ответ действительно основан на результатах инструментов и что заявленные файлы
   существуют. Флаг включения находится в конфиге: `enable_retrieval_critic`.

4. Инструмент чтения Spark-таблиц.

   `build_spark_data_tools(spark)` создает tool `load_data`. Tool принимает простые
   строковые аргументы. В схемах инструмента нет входных `list[]`: списки колонок,
   фильтров, агрегаций и сортировок передаются строкой и разбираются внутри инструмента.

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

  middlewares/
    skills_context.py         # предзагрузка skills
    tool_output_file.py       # сохранение больших результатов tool в pickle
    tool_loop_guard.py        # защита от повторяющихся tool-вызовов
    critic_loop_cap.py        # лимит проверок critic-а

  tools/
    spark_data.py             # load_data поверх Spark session
    data_tools_wrapper.py     # прозрачное описание запросов к data-tools
    execute_python_code.py    # безопасное выполнение Python-кода
    load_skills.py            # ручная дозагрузка skills
    inspect_artifact.py       # проверка файлов для critic-а

  resources/
    config/defaults.json      # настройки по умолчанию
    skills/**/SKILL.md        # доменные инструкции для агента
```

В старой версии были отдельный терминальный demo-runner, trace-логгер и локальные CSV.
Они удалены из пакета, потому что целевой запуск теперь идет через `python run.py` и
реальную Spark session.

## Минимальный запуск

Файл `run.py` находится в корне проекта. В нем должен быть только код запуска:

```python
from pyspark.sql import SparkSession

from deep_agent_test import build_analytics_deep_agent, build_spark_data_tools, load_deep_agent_settings
from model import model

USER_MESSAGE = "текст запроса пользователя"

spark = SparkSession.builder.appName("analytics-deep-agent").getOrCreate()
settings = load_deep_agent_settings()
data_tools = build_spark_data_tools(spark)
agent = build_analytics_deep_agent(model=model, settings=settings, data_tools=data_tools)
result = agent.invoke(
    {"messages": [{"role": "user", "content": USER_MESSAGE}]},
    config={
        "configurable": {"thread_id": settings.thread_id},
        "recursion_limit": settings.graph_recursion_limit,
    },
)
```

В репозитории уже есть готовый `run.py` с таким сценарием. Чтобы задать другой запрос,
измените константу `USER_MESSAGE`.

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

Если нужен отдельный конфиг для другого проекта, укажите путь в переменной окружения
`DEEP_AGENT_CONFIG_PATH`. Значения из этого файла переопределят defaults.

## Формат `load_data`

Все сложные параметры передаются строками.

Пример обычной выборки:

```text
table_name: csp_afpc_sss_inc.uko_event
select_columns: event_id, event_dt, event_dttm_readable, epk_id, event_description, transaction_amount
filters: epk_id eq 2099007770421989000001; event_dt in 20260123,20260124
order_by: event_dt asc; event_dttm_readable asc
```

Пример агрегации:

```text
table_name: csp_afpc_sss_inc.cards_event
select_columns:
filters: event_dt between 20260101,20260131
group_by: event_description
aggregations: count(event_id) as events_count; sum(transaction_amount_in_rub) as amount_rub
order_by: events_count desc
```

Пример вычисляемой колонки:

```text
derived_columns: event_month = year_month(event_dt)
filters: event_month eq 202601
```

Поддерживаемые операторы фильтра:

```text
eq, ne, gt, gte, lt, lte, contains, in, between, is_null, not_null
```

Поддерживаемые операции для `derived_columns`:

```text
year, month, year_month, date, lower, upper, length, abs
```

Поддерживаемые агрегаты:

```text
count, count_distinct, min, max, sum, mean
```

## Skills

Skills лежат в:

```text
deep_agent_test/resources/skills
```

Каждый skill - это папка с файлом `SKILL.md`. Skill должен описывать один понятный
участок домена: таблицу, группу полей, правило поиска или тип аналитического запроса.

Пример структуры:

```text
resources/skills/uko-event-table/SKILL.md
resources/skills/cards-event-table/SKILL.md
resources/skills/hit-table/SKILL.md
```

Когда добавлять новый skill:

- появилась новая таблица;
- появились новые поля с важными правилами интерпретации;
- агент часто ошибается в одном и том же типе запроса;
- нужно зафиксировать правила связи между источниками.

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

Код агента не должен знать бизнес-смысл таблиц. Этот смысл должен жить в `SKILL.md`.
Так пакет проще переносить между проектами: код отвечает за механику, skills отвечают
за домен.

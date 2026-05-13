# Insight Pipeline

`planner_agent.insight_pipeline` - библиотечный слой для поиска повторяющихся проблем в потоке сработок.

Pipeline не заменяет текущий `ResearchAgent`. Он использует его как исполнителя для разбора одной записи, а сам отвечает за batch-логику: загрузку файла, обогащение, кластеризацию, выбор групп, запуск агента по строкам и суммаризацию проблем внутри групп.

## Общая Логика

Текущая целевая схема такая:

```text
source file / DataFrame
  ↓
loaded_df
  ↓
enriched_df
  ↓
clustered_df = enriched_df + колонка названия группы
  ↓
group_selection_decisions = бинарное решение selector-а по каждой группе
  ↓
filtered_df = clustered_df только с выбранными группами
  ↓
analyzed_df = filtered_df + описание записи агентом
  ↓
group_summaries = суммаризация проблемы в каждой выбранной группе
```

Главная точка входа:

```python
run = pipeline.run(source, read_kwargs=None)
```

Файл с orchestrator-ом:

```text
planner_agent/insight_pipeline/pipeline.py
```

## Быстрый Старт

```python
from planner_agent.insight_pipeline import (
    CaseAgentProcessor,
    InsightPipeline,
    InsightPipelineConfig,
)

config = InsightPipelineConfig(
    text_column="comment_text",
    group_column="problem_group",
    case_id_column="case_id",
    event_id_column="event_id",
    min_group_size=5,
    max_cases_per_group=None,
)

pipeline = InsightPipeline(
    config=config,
    enricher=my_enricher,
    clusterer=my_clusterer,
    group_selector=my_group_selector,
    case_processor=CaseAgentProcessor(agent=research_agent),
)

run = pipeline.run("cases.xlsx", read_kwargs={"sheet_name": "Sheet1"})

filtered_df = run.filtered_df
analyzed_df = run.analyzed_df
group_summaries_df = run.group_summaries_df()
```

Если `agent` не передан, pipeline не падает. Он сформирует prompt-ы и вернет статус `skipped`. Это удобно для проверки загрузки, обогащения, кластеризации и отбора групп без API-ключей.

## Последовательность Вызовов

Внутри `InsightPipeline.run(...)` шаги идут так:

```text
1. _load_source(source)
   Input: str | Path | pandas.DataFrame
   Output: pandas.DataFrame loaded_df

2. enricher.enrich(loaded_df)
   Input: pandas.DataFrame
   Output: pandas.DataFrame enriched_df

3. _cluster(enriched_df)
   Input: pandas.DataFrame
   Output: pandas.DataFrame clustered_df

4. _classify_groups(clustered_df)
   Input: pandas.DataFrame
   Output: list[GroupSelectionDecision]

5. _filter_selected_groups(clustered_df, group_selection_decisions)
   Input: pandas.DataFrame + list[GroupSelectionDecision]
   Output: pandas.DataFrame filtered_df

6. _process_filtered_cases(filtered_df)
   Input: pandas.DataFrame
   Output: tuple[pandas.DataFrame analyzed_df, list[CaseAnalysisRecord]]

7. group_summarizer.summarize(analyzed_df, case_results, selected_groups, config)
   Input: pandas.DataFrame + list[CaseAnalysisRecord] + list[str] + InsightPipelineConfig
   Output: list[GroupProblemSummary]

8. InsightPipelineRun(...)
   Output: контейнер со всеми промежуточными и итоговыми результатами
```

## Вход Pipeline

Метод:

```python
InsightPipeline.run(
    source: str | pathlib.Path | pandas.DataFrame,
    *,
    read_kwargs: dict[str, Any] | None = None,
) -> InsightPipelineRun
```

`source` может быть:

```text
str | Path       путь к файлу
pandas.DataFrame готовый DataFrame
```

Поддерживаемые форматы файлов:

```text
.csv
.xlsx
.xls
.parquet
.json
.jsonl
```

Чтение файла реализовано здесь:

```text
planner_agent/insight_pipeline/io.py
```

Функция:

```python
load_dataframe(path: str | Path, **read_kwargs: Any) -> pandas.DataFrame
```

## Выход Pipeline

`pipeline.run(...)` возвращает:

```python
InsightPipelineRun
```

Поля `InsightPipelineRun`:

```python
loaded_df: pandas.DataFrame
enriched_df: pandas.DataFrame
clustered_df: pandas.DataFrame
filtered_df: pandas.DataFrame
analyzed_df: pandas.DataFrame
group_selection_decisions: list[GroupSelectionDecision]
selected_groups: list[str]
case_results: list[CaseAnalysisRecord]
group_summaries: list[GroupProblemSummary]
metadata: dict[str, Any]
```

Удобные методы:

```python
run.case_results_df() -> pandas.DataFrame
run.group_summaries_df() -> pandas.DataFrame
run.group_selection_decisions_df() -> pandas.DataFrame
```

## DataFrame-Стадии

### loaded_df

Тип:

```python
pandas.DataFrame
```

Содержит исходные колонки из файла или переданного DataFrame. Pipeline не требует жесткой схемы на этом этапе.

Минимально полезные колонки:

```text
case_id       опционально, id записи
event_id      опционально, id сработки
comment_text  колонка с текстом для кластеризации
```

Имена колонок задаются в `InsightPipelineConfig`.

### enriched_df

Тип:

```python
pandas.DataFrame
```

Это `loaded_df` после `enricher.enrich(...)`.

Ожидается:

```text
исходные колонки +
любые дополнительные поля, которые нужны агенту
```

Примеры дополнительных полей:

```text
epk_id
event_dt
channel
amount
mcc_code
mcc_name
recipient_info
hits_extra_facts
events_day
previous_events
posterious_events
```

### clustered_df

Тип:

```python
pandas.DataFrame
```

Это `enriched_df` после `clusterer.cluster(...)`.

Обязательная добавленная колонка:

```python
config.group_column
```

По умолчанию:

```text
problem_group
```

Пример:

```text
comment_text                              problem_group
"Заблокировали перевод самому себе"       "Перевод себе"
"Не прошла регулярная оплата"             "Регулярная оплата"
```

### filtered_df

Тип:

```python
pandas.DataFrame
```

Это `clustered_df`, отфильтрованный по группам, которые selector признал значимыми.

Добавленные служебные колонки:

```python
config.selected_group_column: bool
config.group_selection_reason_column: str
```

По умолчанию:

```text
is_significant_group: bool
group_selection_reason: str
```

В `filtered_df` остаются только строки, где:

```python
is_significant_group == True
```

### analyzed_df

Тип:

```python
pandas.DataFrame
```

Это `filtered_df` после обработки строк агентом.

Добавленные колонки:

```python
config.agent_report_column: str
config.agent_status_column: str
config.agent_error_column: str
config.agent_structured_result_column: dict | None
config.agent_missing_data_requests_column: list[dict] | None
```

По умолчанию:

```text
agent_record_description
agent_processing_status
agent_processing_error
agent_structured_result
agent_missing_data_requests
```

Статусы:

```text
processed  агент успешно обработал запись
skipped    агент не передан, сформирован только prompt
failed     вызов агента завершился ошибкой
```

## Конфигурация

Схема:

```python
InsightPipelineConfig
```

Файл:

```text
planner_agent/insight_pipeline/schemas.py
```

Поля:

```python
text_column: str = "comment_text"
group_column: str = "problem_group"
case_id_column: str = "case_id"
event_id_column: str = "event_id"
selected_group_column: str = "is_significant_group"
group_selection_reason_column: str = "group_selection_reason"
agent_report_column: str = "agent_record_description"
agent_status_column: str = "agent_processing_status"
agent_error_column: str = "agent_processing_error"
agent_structured_result_column: str = "agent_structured_result"
agent_missing_data_requests_column: str = "agent_missing_data_requests"
max_cases_per_group: int | None = None
min_group_size: int = 3
max_groups: int | None = 10
agent_recursion_limit: int = 60
include_full_row_prompt: bool = True
```

Важные параметры:

```text
max_cases_per_group=None
```

означает, что агент обработает все строки из `filtered_df`.

```text
max_cases_per_group=20
```

означает, что агент обработает не больше 20 строк в каждой выбранной группе.

## Контракт Enricher

Назначение:

```text
исходный df -> исходный df + дополнительные колонки
```

Ожидаемый тип:

```python
class MyEnricher:
    def enrich(self, df: pandas.DataFrame) -> pandas.DataFrame:
        ...
```

Input:

```python
df: pandas.DataFrame
```

Output:

```python
enriched_df: pandas.DataFrame
```

Требования:

```text
1. Не удалять исходные колонки без необходимости.
2. Сохранять индекс строк, если дальше важно сопоставление с исходным df.
3. Добавлять новые поля как обычные колонки.
4. Не вызывать агента внутри enricher.
```

Пример:

```python
class MyEnricher:
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["source_loaded"] = True
        result["events_day"] = load_events_for_cases(result)
        return result
```

## Контракт Clusterer

Назначение:

```text
enriched_df -> enriched_df + колонка названия группы
```

Ожидаемый тип:

```python
class MyClusterer:
    def cluster(
        self,
        df: pandas.DataFrame,
        *,
        text_column: str,
        group_column: str,
    ) -> pandas.DataFrame:
        ...
```

Input:

```python
df: pandas.DataFrame
text_column: str
group_column: str
```

Output:

```python
clustered_df: pandas.DataFrame
```

Обязательное условие:

```python
group_column in clustered_df.columns
```

Пример:

```python
class MyClusterer:
    def cluster(self, df, *, text_column, group_column):
        result = df.copy()
        result[group_column] = my_cluster_library.predict(result[text_column])
        return result
```

## Контракт Group Selector

Назначение:

```text
одна группа -> бинарное решение: имеет смысл или нет
```

Предпочтительный интерфейс:

```python
class MyGroupSelector:
    def classify_group(
        self,
        group_name: str,
        group_df: pandas.DataFrame,
        config: InsightPipelineConfig,
    ) -> GroupSelectionDecision:
        ...
```

Input:

```python
group_name: str
group_df: pandas.DataFrame
config: InsightPipelineConfig
```

Output:

```python
GroupSelectionDecision
```

Схема `GroupSelectionDecision`:

```python
group_name: str
is_meaningful: bool
reason: str
confidence: float
total_records: int
```

Пример LLM-selector-а:

```python
class MyLLMGroupSelector:
    def classify_group(self, group_name, group_df, config):
        examples = group_df[config.text_column].dropna().head(10).tolist()
        decision = llm_binary_classify(group_name=group_name, examples=examples)
        return GroupSelectionDecision(
            group_name=group_name,
            is_meaningful=decision["is_meaningful"],
            reason=decision["reason"],
            confidence=decision["confidence"],
            total_records=len(group_df),
        )
```

Совместимость со старым интерфейсом сохранена:

```python
def select_groups(df, config) -> list[str]
```

Но для новой логики лучше использовать `classify_group(...)`.

## Контракт Case Processor

Компонент:

```python
CaseAgentProcessor
```

Назначение:

```text
одна строка filtered_df -> CaseAnalysisRecord
```

Метод:

```python
process_row(
    row: pandas.Series,
    *,
    row_index: int,
    group_name: str,
    config: InsightPipelineConfig,
) -> CaseAnalysisRecord
```

Внутри вызывается:

```python
agent.invoke(prompt, config={"recursion_limit": config.agent_recursion_limit})
```

Ожидаемый агент:

```python
class AgentLike:
    def invoke(self, prompt: str, config: dict | None = None) -> Any:
        ...
```

Текущий `ResearchAgent` подходит под этот контракт.

## Prompt Для Агента

Файл:

```text
planner_agent/insight_pipeline/formatting.py
```

Функция:

```python
build_case_prompt(
    row: Mapping[str, Any],
    *,
    group_name: str,
    row_index: int,
    max_string_length: int | None = None,
) -> str
```

Input:

```python
row: Mapping[str, Any]
group_name: str
row_index: int
max_string_length: int | None
```

Output:

```python
prompt: str
```

Ожидаемый блок в ответе агента:

```text
<structured_result>
{
  "case_id": "...",
  "event_id": "...",
  "group_name": "...",
  "facts": [],
  "hypotheses": [],
  "missing_data_requests": [],
  "limitations": [],
  "final_summary": "..."
}
</structured_result>
```

Если JSON найден, он попадет в:

```python
CaseAnalysisRecord.structured_result
analyzed_df[config.agent_structured_result_column]
```

Если агент запросил дозагрузку данных, она попадет в:

```python
CaseAnalysisRecord.missing_data_requests
analyzed_df[config.agent_missing_data_requests_column]
```

## Схема CaseAnalysisRecord

```python
row_index: int
case_id: str
group_name: str
status: Literal["processed", "skipped", "failed"]
agent_prompt: str
report_markdown: str
structured_result: dict[str, Any]
missing_data_requests: list[MissingDataRequest]
error: str
```

`report_markdown` записывается в:

```python
analyzed_df["agent_record_description"]
```

## Схема MissingDataRequest

```python
source_name: str
reason: str
lookup_keys: dict[str, Any]
period: str | None
priority: Literal["low", "medium", "high"]
```

Пример:

```json
{
  "source_name": "client_transaction_history",
  "reason": "Нужно проверить похожие операции клиента за 180 дней.",
  "lookup_keys": {
    "epk_id": "123",
    "event_dt": "2026-04-10"
  },
  "period": "180 дней до event_dt",
  "priority": "high"
}
```

## Контракт Group Summarizer

Компонент:

```python
GroupProblemSummarizer
```

Метод:

```python
summarize(
    df: pandas.DataFrame,
    records: list[CaseAnalysisRecord],
    *,
    selected_groups: list[str],
    config: InsightPipelineConfig,
) -> list[GroupProblemSummary]
```

Input:

```python
df: analyzed_df
records: list[CaseAnalysisRecord]
selected_groups: list[str]
config: InsightPipelineConfig
```

Output:

```python
list[GroupProblemSummary]
```

Схема `GroupProblemSummary`:

```python
group_name: str
total_records: int
processed_records: int
failed_records: int
missing_data_requests_count: int
problem_summary: str
evidence_case_ids: list[str]
limitations: list[str]
```

## Как Подключить Реальные Компоненты

```python
pipeline = InsightPipeline(
    config=InsightPipelineConfig(
        text_column="comment_text",
        group_column="problem_group",
        max_cases_per_group=None,
    ),
    enricher=MySparkEnricher(...),
    clusterer=MyCommentClusterer(...),
    group_selector=MyLLMGroupSelector(...),
    case_processor=CaseAgentProcessor(agent=research_agent),
    group_summarizer=GroupProblemSummarizer(summarizer=my_group_llm_summarizer),
)

run = pipeline.run("input.xlsx")
```

## Что Менять Чаще Всего

### Добавить новые данные из таблиц

Меняй `enricher`.

```python
class MyEnricher:
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        extra = load_extra_table(...)
        return result.merge(extra, on="event_id", how="left")
```

### Подключить готовую библиотеку кластеризации

Меняй `clusterer`.

```python
class MyClusterer:
    def cluster(self, df, *, text_column, group_column):
        result = df.copy()
        result[group_column] = cluster_library.add_group(result, text_column)
        return result
```

### Сделать LLM бинарную классификацию групп

Меняй `group_selector`.

```python
class MySelector:
    def classify_group(self, group_name, group_df, config):
        examples = group_df[config.text_column].head(20).tolist()
        llm_result = classify_group_with_llm(group_name, examples)
        return GroupSelectionDecision(
            group_name=group_name,
            is_meaningful=llm_result.is_meaningful,
            reason=llm_result.reason,
            confidence=llm_result.confidence,
            total_records=len(group_df),
        )
```

### Изменить текст, который получает агент

Меняй `build_case_prompt(...)` в:

```text
planner_agent/insight_pipeline/formatting.py
```

Или передай свой builder:

```python
processor = CaseAgentProcessor(
    agent=research_agent,
    prompt_builder=my_prompt_builder,
)
```

## Пример Без API-Ключей

Файл:

```text
examples/insight_pipeline_usage.py
```

Запуск:

```bash
python examples/insight_pipeline_usage.py
```

В примере:

```text
DemoClusterer  имитирует кластеризацию
DemoAgent      имитирует агента
```

Этот пример нужен только для проверки wiring pipeline без внешних LLM и без API-ключей.

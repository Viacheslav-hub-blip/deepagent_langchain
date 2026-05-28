# Native Analytics DeepAgent

Production-oriented реализация аналитического DeepAgent на native LangChain/DeepAgents. В консоль выводятся сообщения агента, загруженные skills, вызовы инструментов, ответы инструментов и, при включенном `print_plan_prompts`, prompt-ы перед первичным планом. Трассировка tools, загруженных skills и выбранных few-shot примеров дополнительно пишется в JSONL-файлы.

## Структура

- `analytics_deep_agent.py` - сборка supervisor, subagents, tools, skills backend, HITL и Python-analysis tool.
- `agent_specs.py` - имена subagents и сборка спецификаций subagents/HITL без смешивания с основной фабрикой агента.
- `prompts.py` - все prompt-шаблоны и описание, где каждый prompt используется.
- `run_native_analytics_chat.py` - интерактивный терминальный runner.
- `settings.py` и `config/defaults.json` - типизированная загрузка настроек и значения по умолчанию.
- `skills/` - локальные skills агента, которые автоматически сканируются при старте.
- `data/` - тестовые CSV-таблицы для локального примера: `hits`, `cards_event`, `uko_event`.
- `agent_logging.py` - простой файловый JSONL-логгер без вывода в консоль.
- `prompt_debug_middleware.py` - диагностический вывод system/user prompt перед первичным планом.
- `trace_logging_middleware.py` - middleware трассировки model tool calls и фактических tool executions.
- `skills_context_middleware.py` - автосканирование папки skills и preload markdown context в state и system prompt.
- `few_shot_examples_index.py` - построение и чтение локального vector index few-shot примеров.
- `few_shot_examples_middleware.py` - поиск, LLM-реранжирование и подстановка few-shot примеров.
- `tool_output_file_middleware.py` - сохранение больших табличных tool outputs в CSV.
- `plan_approval_middleware.py` - одноразовое подтверждение первого `write_todos`.
- `few_shot_examples/` - markdown-примеры для few-shot поиска.

## Запуск

Файл `run_native_analytics_chat.py` запускает пример агента на тестовых CSV из `deep_agent_test/data`
через `examples.fake_spark_tools.build_fake_spark_tools`. Модель и embeddings берутся из
корневого `model.py`. Я не запускаю этот сценарий при проверках, потому что он может
использовать ваши API-ключи.

```powershell
uv run --extra deep-agent-test python deep_agent_test/run_native_analytics_chat.py
```

Альтернативный конфиг можно передать через переменную окружения. В `main()` все равно будет
явно подставлен тестовый `read_table`, а остальные настройки возьмутся из конфига:

```powershell
$env:DEEP_AGENT_CONFIG_PATH="C:\path\to\deep_agent_config.json"
uv run --extra deep-agent-test python deep_agent_test/run_native_analytics_chat.py
```

## Конфигурация

Основной конфиг: `deep_agent_test/config/defaults.json`.

Ключевые настройки:

- `thread_id` - thread id LangGraph для терминального чата.
- `skills_virtual_dir` - виртуальный путь skills в DeepAgents backend.
- `skills_root` - локальная папка skills. По умолчанию используется `deep_agent_test/skills`. Это единственная настройка для preload skills: middleware рекурсивно найдет файлы `SKILL.md` внутри skill-папок.
- `data_tools_factory` - import path callable-фабрики production tools чтения данных. Может быть `null`, если tools передаются через `data_tools`.
- `data_tools_factory_kwargs` - kwargs для фабрики tools.
- `few_shot_examples_dir` - markdown-примеры few-shot.
- `few_shot_index_dir` - генерируемый индекс few-shot. По умолчанию лежит в `runs/` и не идет в коммит.
- `logs_dir` - папка файловых логов.
- `tool_outputs_dir` - папка CSV-файлов с большими результатами tools.
- `max_chars_per_skill` - максимальная длина одного найденного markdown-файла skills в prompt context.
- `few_shot_top_k` и `few_shot_max_examples` - параметры поиска и выбора few-shot.
- `trace_preview_chars` - длина preview в JSONL-логах.
- `print_tool_calls` - печать фактических вызовов инструментов в консоль.
- `print_tool_results` - печать кратких ответов инструментов в консоль.
- `print_plan_prompts` - печать system prompt и user prompt перед первичной генерацией плана.
- `tool_output_*` - пороги сохранения больших tool outputs в CSV, число preview-строк и лимит дублирования исходного content.

## Логи

По умолчанию логи пишутся в `runs/deep_agent_logs/`:

- `events.jsonl` - общий поток всех событий.
- `tool_calls.jsonl` - все model tool calls, старты и завершения tools, сохраненные tool outputs.
- `loaded_skills.jsonl` - какие skills были загружены в prompt context.
- `selected_few_shot.jsonl` - кандидаты и выбранные few-shot примеры.

Большие табличные результаты tools сохраняются отдельно в `runs/deep_agent_tool_outputs/`.

## Production Data Tools

`build_analytics_deep_agent` принимает параметр `data_tools`. Для production можно передать реальные LangChain tools чтения данных напрямую:

```python
from deep_agent_test.analytics_deep_agent import build_analytics_deep_agent
from deep_agent_test.settings import load_deep_agent_settings

settings = load_deep_agent_settings()
agent = build_analytics_deep_agent(
    model=chat_model,
    embeddings_model=embeddings_model,
    settings=settings,
    data_tools=[production_read_table_tool],
)
```

Если `data_tools` не передан, `build_data_tools(settings)` загружает фабрику из `data_tools_factory`.

Пример override-конфига:

```json
{
  "skills_root": "skills",
  "data_tools_factory": "my_project.analytics_tools:build_data_tools",
  "data_tools_factory_kwargs": {
    "profile": "prod"
  }
}
```

Фабрика должна вернуть один LangChain `BaseTool` или список `BaseTool`. Supervisor напрямую не получает `read_table`; чтение данных доступно только `data-retrieval-agent`. Если ни `data_tools`, ни `data_tools_factory` не заданы, сборка агента падает с явной ошибкой вместо использования тестового адаптера.

## Проверка Без API-Ключей

Синтаксис можно проверить без запуска модели:

```powershell
uv run --extra deep-agent-test python -m py_compile `
  deep_agent_test/analytics_deep_agent.py `
  deep_agent_test/agent_specs.py `
  deep_agent_test/prompts.py `
  deep_agent_test/run_native_analytics_chat.py `
  deep_agent_test/settings.py `
  deep_agent_test/agent_logging.py `
  deep_agent_test/prompt_debug_middleware.py `
  deep_agent_test/trace_logging_middleware.py `
  deep_agent_test/skills_context_middleware.py `
  deep_agent_test/few_shot_examples_middleware.py `
  deep_agent_test/few_shot_examples_index.py `
  deep_agent_test/tool_output_file_middleware.py `
  deep_agent_test/plan_approval_middleware.py
```

Юнит-тест middleware подтверждения плана:

```powershell
uv run --extra deep-agent-test --extra dev pytest tests/test_deep_agent_plan_approval.py
```

## Справочник Функций

### `prompts.py`

- `SYSTEM_PROMPT` - основной prompt supervisor.
- `DATA_RETRIEVAL_PROMPT` - prompt subagent-а чтения данных.
- `PYTHON_ANALYSIS_PROMPT` - prompt subagent-а Python-анализа.
- `READ_TABLE_DESCRIPTION` - описание tool `read_table`.
- `PRELOADED_SKILLS_CONTEXT_PROMPT_TEMPLATE` - шаблон добавления preloaded skills context.
- `FEW_SHOT_RERANK_SYSTEM_PROMPT` - prompt выбора few-shot примеров.
- `FEW_SHOT_PROMPT_BLOCK_TEMPLATE` - шаблон добавления выбранных few-shot примеров.
- `RUNNER_CONTINUE_INSTRUCTION_TEMPLATE` - служебная инструкция продолжения runner-а.

### `analytics_deep_agent.py`

- `clarify_analysis_request(question)` - возвращает уточняющий вопрос; фактический ответ пользователя проходит через HITL `respond`.
- `build_read_table_description()` - возвращает подробное описание `read_table` для LLM.
- `build_data_tools(settings)` - загружает production tools чтения данных через фабрику из конфига.
- `_load_callable_from_path(import_path)` - загружает callable по import path.
- `_normalize_data_tools(raw_tools)` - проверяет и нормализует результат фабрики tools.
- `run_python_analysis(task, code, input_data)` - выполняет аналитический Python-код с обычными imports/builtins текущего виртуального окружения; блокирует только операции удаления файлов и директорий.
- `DeleteOperationError` - ошибка, которая возвращается при попытке удалить файл или директорию.
- `_validate_python_code_without_delete_operations(code)` - проверяет AST кода на операции удаления перед выполнением.
- `_call_uses_delete_api(node)` - определяет Python API удаления вроде `os.remove`, `Path.unlink`, `shutil.rmtree`.
- `_attribute_root_name(node)` - извлекает корневое имя attribute chain для точечной проверки API.
- `_call_uses_delete_shell_command(node)` - определяет shell-команды удаления в `os.system` и `subprocess`.
- `_literal_command_text(node)` - извлекает literal shell-команду из AST-вызова.
- `_command_text_contains_delete(command_text)` - ищет команды `rm`, `del`, `rmdir`, `rd`, `Remove-Item`.
- `_temporary_delete_operation_guard()` - временно блокирует runtime-вызовы удаления во время `exec`.
- `_install_temporary_patch(target, attribute_name, replacement)` - временно подменяет функцию или метод.
- `_blocked_delete_callable(function_name)` - создает callable, который всегда запрещает удаление.
- `_guarded_shell_callable(function_name, original)` - проверяет runtime shell/subprocess команды перед выполнением.
- `_command_invocation_text(args, kwargs)` - извлекает текст команды из runtime-аргументов.
- `_format_python_analysis_tool_result(payload)` - форматирует результат Python-анализа для модели и middleware.
- `_extract_rows_for_artifact(value)` - извлекает строки из DataFrame/list/dict результата без строгой проверки типа элемента.
- `_row_to_mapping(value)` - преобразует элемент результата в словарь без отбрасывания значения.
- `build_analysis_artifact(file_name, content)` - формирует текстовый artifact без записи файла.
- `save_analysis_file(file_name, content, output_dir)` - сохраняет файл внутри проекта и возвращает абсолютный путь.
- `_is_relative_to(path, parent)` - проверяет, что путь не выходит за пределы проекта.
- `build_analytics_deep_agent(model, embeddings_model, settings, event_logger, data_tools)` - собирает DeepAgent supervisor с subagents, middleware, HITL и checkpointer.
- `get_skills_root(settings)` - возвращает абсолютный путь к локальной папке skills.
- `build_skills_backend(settings)` - создает DeepAgents backend для `/skills/` и state scratch-файлов.
- `build_skills_permissions(settings)` - запрещает запись в `/skills/**`.
- `invoke_agent(agent, message, thread_id)` - отправляет пользовательское сообщение агенту.
- `resume_with_decision(agent, thread_id, decision)` - продолжает выполнение после HITL interrupt.
- `resume_with_user_answer(agent, thread_id, answer)` - отправляет текстовый ответ пользователя на уточняющий вопрос.

### `agent_specs.py`

- `build_clarify_interrupt_config()` - собирает HITL-config для уточняющего вопроса.
- `build_analytics_subagent_specs(settings, data_tools, analysis_tools, common_middleware)` - собирает спецификации `data-retrieval-agent` и `python-analysis-agent`.

### `run_native_analytics_chat.py`

- `build_chat_agent(settings, data_tools)` - собирает агента для терминального runner-а с моделью из `model.py`.
- `make_config(thread_id)` - создает LangGraph config с `thread_id`.
- `run_chat(settings, data_tools)` - запускает интерактивный цикл терминального чата.
- `invoke_user_message(agent, config, message)` - отправляет новое сообщение пользователя.
- `resume_with_decisions(agent, config, decisions)` - продолжает graph после списка HITL-решений.
- `collect_human_decisions(interrupts)` - собирает решения пользователя для всех interrupt payload.
- `collect_single_decision(action_request, review_config)` - запрашивает одно решение `approve`, `edit`, `reject` или `respond`.
- `collect_edit_feedback(action_name)` - превращает текстовую правку пользователя в HITL decision.
- `continue_until_agent_boundary(agent, config, result, require_progress)` - автоматически продолжает graph до interrupt или финального состояния.
- `should_continue_agent_loop(result, require_progress)` - решает, нужно ли вернуть управление агенту.
- `build_continue_instruction(result, require_progress)` - формирует служебное сообщение runner-а для продолжения.
- `requires_progress_after_decisions(decisions)` - проверяет, ожидается ли действие после HITL-решения.
- `print_loaded_skills_once(result, already_printed)` - один раз печатает загруженные skills в формате ответа агента.
- `print_turn_result(result)` - печатает только содержательный ответ агента.
- `extract_interrupt_values(result)` - извлекает значения `Interrupt.value`.
- `last_agent_response_text(result)` - возвращает текст последнего нового AIMessage без tool calls.
- `last_message_has_tool_calls(result)` - проверяет tool calls у последнего сообщения.
- `has_unfinished_todos(result)` - проверяет незавершенные todo в state.
- `has_completed_todos(result)` - проверяет наличие завершенных todo.
- `format_todos_for_user(todos)` - форматирует план для вывода пользователю.
- `message_to_text(message)` - преобразует LangChain message или dict в текст.
- `TEST_DATA_DIR` - папка `deep_agent_test/data` с тестовыми CSV для примера.
- `build_test_data_tools(data_dir)` - создает sync/async `read_table` для тестовых CSV.
- `main()` - точка входа примера: загружает настройки, создает test `read_table` на CSV из `TEST_DATA_DIR` и запускает чат.

### `settings.py`

- `DeepAgentSettings.from_mapping(payload, project_root)` - создает типизированные настройки из словаря.
- `load_deep_agent_settings(config_path)` - загружает настройки из JSON-конфига или `DEEP_AGENT_CONFIG_PATH`.
- `_load_config_payload(config_path)` - загружает `defaults.json` и накладывает пользовательские переопределения.
- `_read_json_file(path)` - читает JSON-файл и проверяет, что в нем объект.
- `_validate_required_config_keys(payload)` - проверяет обязательные ключи итогового конфига.
- `_resolve_project_path(value, project_root)` - приводит путь из конфига к абсолютному.
- `_int_from_config(payload, key)` - строго читает целое число из конфига.
- `_bool_from_config(payload, key)` - строго читает булево значение из конфига.
- `_dict_from_config(payload, key)` - строго читает JSON-объект из конфига.
- `_optional_str_from_config(payload, key)` - читает строку или `null`.

### `agent_logging.py`

- `DeepAgentEventLogger.__init__(log_dir, enabled)` - инициализирует пути JSONL-логов.
- `DeepAgentEventLogger.log_event(event_type, payload, category_path)` - пишет событие в общий лог и optional category log.
- `DeepAgentEventLogger.log_tool_event(event_type, payload)` - пишет событие tools в общий и отдельный tool log.
- `DeepAgentEventLogger.log_loaded_skills(payload)` - пишет сведения о загруженных skills.
- `DeepAgentEventLogger.log_few_shot_selection(payload)` - пишет сведения о выбранных few-shot примерах.
- `DeepAgentEventLogger._append_jsonl(path, record)` - добавляет одну JSON-строку в файл.
- `build_deep_agent_logger(settings)` - создает логгер из настроек.
- `_utc_now_iso()` - возвращает UTC timestamp.
- `_json_default(value)` - сериализует нестандартные объекты для JSON.

### `prompt_debug_middleware.py`

- `PromptDebugConsoleMiddleware.wrap_model_call(request, handler)` - печатает system prompt и user prompt перед первым вызовом supervisor-модели.
- `_last_user_prompt(messages)` - извлекает последний настоящий пользовательский prompt.
- `_message_content_to_text(content)` - преобразует содержимое LangChain message в текст.
- `_system_prompt_to_text(system_message)` - преобразует system prompt из `ModelRequest` в текст.

### `trace_logging_middleware.py`

- `ToolTraceLoggingMiddleware.wrap_model_call(request, handler)` - логирует доступные tools перед вызовом модели, если включено.
- `ToolTraceLoggingMiddleware.after_model(state, runtime)` - логирует tool calls из ответа модели.
- `ToolTraceLoggingMiddleware.wrap_tool_call(request, handler)` - логирует sync tool start/end.
- `ToolTraceLoggingMiddleware.awrap_tool_call(request, handler)` - логирует async tool start/end.
- `_agent_name(runtime)` - извлекает имя агента из runtime metadata.
- `_format_json(value, max_chars)` - форматирует значение как короткий JSON.
- `_format_tool_result(value, max_chars)` - форматирует результат tool call.
- `_print_tool_call(agent, tool_name, args_preview)` - печатает вызов tool в консоль.
- `_print_tool_result(agent, tool_name, result_preview)` - печатает ответ tool в консоль.
- `_preview_text(text, max_chars)` - обрезает длинный текст для логов.
- `_tool_names(tools)` - извлекает имена tools.

### `skills_context_middleware.py`

- `PreloadedSkillsContextMiddleware.before_agent(state, runtime)` - сканирует папку `skills_root`, читает markdown-файлы skills и пишет context в state.
- `PreloadedSkillsContextMiddleware.wrap_model_call(request, handler)` - добавляет загруженный skills context в system message.
- `build_preloaded_skills_context(skills_root, skills_virtual_dir, max_chars_per_file)` - автоматически собирает compact context из найденных markdown-файлов skills.
- `discover_skill_context_files(skills_root)` - находит файлы `SKILL.md` внутри папки skills.
- `_read_context_file(path, max_chars)` - читает файл context и ограничивает длину.
- `_virtual_skill_path(skills_root, path, skills_virtual_dir)` - строит виртуальный путь для prompt context и логов.
- `_normalize_virtual_dir(value)` - нормализует виртуальную папку skills.
- `_truncate_text(text, max_chars)` - обрезает длинный текст.

### `few_shot_examples_index.py`

- `FewShotExamplesStore.__init__(index_dir, embeddings)` - загружает документы и векторы индекса.
- `FewShotExamplesStore.search(query, top_k)` - выполняет cosine search по few-shot векторам.
- `update_few_shot_examples_index(examples_dir, index_dir, embeddings)` - инкрементально обновляет индекс.
- `parse_few_shot_example_file(path, examples_dir)` - парсит markdown-пример.
- `collect_example_file_hashes(examples_dir)` - собирает sha256 markdown-файлов.
- `compute_file_sha256(path)` - вычисляет sha256 файла.
- `load_few_shot_documents(path)` - читает `documents.jsonl`.
- `load_index_manifest(path)` - читает manifest индекса.
- `save_index_manifest(path, manifest)` - сохраняет manifest индекса.
- `_split_example_header(text)` - берет заголовок markdown до `---`.
- `_load_previous_index(documents_path, vectors_path)` - загружает прежний индекс.
- `_load_vectors(path)` - загружает `vectors.npy`.
- `_build_vectors_array(vectors)` - собирает список векторов в матрицу.
- `_cosine_scores(query_vector, vectors)` - считает cosine similarity.
- `_write_json_atomic(path, payload)` - атомарно пишет JSON.
- `_write_jsonl_atomic(path, rows)` - атомарно пишет JSONL.
- `_write_vectors_atomic(path, vectors)` - атомарно пишет numpy vectors.

### `few_shot_examples_middleware.py`

- `FewShotExamplesMiddleware.before_agent(state, runtime)` - ищет и кеширует few-shot примеры для последнего запроса.
- `FewShotExamplesMiddleware.wrap_model_call(request, handler)` - добавляет selected examples в system prompt.
- `select_few_shot_examples_with_llm(model, user_query, candidates, max_examples)` - выбирает подходящие примеры через structured output.
- `load_full_example_markdown(selected)` - читает полный markdown выбранных примеров.
- `build_few_shot_prompt_block(examples_markdown)` - формирует prompt-блок few-shot.
- `extract_last_user_query(messages)` - извлекает последний настоящий пользовательский запрос.
- `build_user_query_key(user_query)` - строит sha256 cache key запроса.
- `_format_candidates_for_rerank(candidates)` - форматирует кандидатов для LLM-реранжирования.

### `tool_output_file_middleware.py`

- `ToolOutputFileMiddleware.wrap_tool_call(request, handler)` - обрабатывает sync tool output.
- `ToolOutputFileMiddleware.awrap_tool_call(request, handler)` - обрабатывает async tool output.
- `ToolOutputFileMiddleware._process_tool_message(result, tool_name)` - сохраняет большой табличный результат в CSV.
- `_extract_tabular_payload(artifact, content)` - извлекает строки из artifact или content.
- `_extract_rows_from_value(value)` - извлекает строки из распространенных структур и оборачивает скаляры в `value`.
- `_write_rows_to_csv(rows, output_dir, tool_name)` - пишет строки в CSV.
- `_build_file_summary(tool_name, file_path, rows, preview_rows, original_content, inline_original_content_chars)` - формирует summary для модели.
- `_safe_filename_part(value)` - делает безопасный фрагмент имени файла.
- `_row_to_mapping(value)` - преобразует строку результата в CSV-совместимый словарь.

### `plan_approval_middleware.py`

- `FirstPlanApprovalMiddleware.after_model(state, runtime)` - прерывает выполнение перед первым `write_todos`.
- `_context_is_loaded(state)` - проверяет, загружен ли skills context.
- `_find_last_ai_message(messages)` - находит последнее AIMessage.
- `_find_plan_tool_call(message)` - находит tool call `write_todos`.
- `_is_internal_runner_message(message)` - определяет служебное сообщение runner-а.
- `_last_user_key(messages)` - вычисляет ключ последнего пользовательского сообщения.

### `__init__.py`

- `build_analytics_deep_agent` - публичный импорт сборки агента.
- `load_deep_agent_settings` - публичный импорт загрузки настроек.

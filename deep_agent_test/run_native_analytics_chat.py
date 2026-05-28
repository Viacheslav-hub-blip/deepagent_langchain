"""Терминальный чат для проверки native DeepAgents аналитического агента.

Содержит:
- EXIT_COMMANDS: команды выхода из терминального чата.
- HITL_DECISION_COMMANDS: команды, которые нельзя отправлять как обычный вопрос.
- TEST_DATA_DIR: папка с тестовыми CSV-данными внутри пакета агента.
- build_chat_agent: сборка агента с моделью из корневого model.py.
- build_test_data_tools: сборка sync/async tools чтения тестовых CSV.
- make_config: создание config с thread_id для LangGraph.
- run_chat: основной цикл терминального чата.
- invoke_user_message: отправка пользовательского сообщения агенту.
- resume_with_decisions: продолжение     выполнения после human-in-the-loop interrupt.
- collect_human_decisions: сбор решений пользователя по interrupt payload.
- collect_single_decision: сбор одного решения approve/edit/reject/respond.
- collect_edit_feedback: сбор текстовых правок на естественном языке.
- continue_until_agent_boundary: автоматическое продолжение до interrupt или финального state.
- should_continue_agent_loop: проверка необходимости автоматического продолжения.
- build_continue_instruction: создание служебной инструкции runner-а.
- requires_progress_after_decisions: проверка необходимости продолжения после HITL.
- print_loaded_skills_once: вывод списка загруженных skills один раз за сессию runner-а.
- print_turn_result: вывод ответа агента.
- extract_interrupt_values: извлечение interrupt payload из результата LangGraph.
- last_agent_response_text: извлечение последнего содержательного ответа агента.
- last_message_has_tool_calls: проверка tool calls у последнего сообщения.
- has_unfinished_todos: проверка незавершенных todo в state.
- has_completed_todos: проверка завершенных todo в state.
- has_only_final_response_todo_in_progress: проверка, что остался только финальный ответ.
- needs_final_response_after_completed_todos: проверка необходимости финального ответа после закрытия todo.
- last_message_is_tool_message: проверка, что последний message является ToolMessage.
- format_todos_for_user: форматирование плана анализа для пользователя.
- message_to_text: преобразование сообщения LangChain в текст.
- main: точка входа для запуска файла.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.tools.structured import StructuredTool
from langgraph.types import Command

from deep_agent_test.analytics_deep_agent import build_analytics_deep_agent
from deep_agent_test.prompts import RUNNER_CONTINUE_INSTRUCTION_TEMPLATE
from deep_agent_test.settings import DeepAgentSettings, load_deep_agent_settings

EXIT_COMMANDS = {"exit", "quit", "q", "выход", "стоп"}
HITL_DECISION_COMMANDS = {"approve", "a", "ok", "да", "edit", "e", "reject", "r", "нет"}
TEST_DATA_DIR = Path(__file__).resolve().parent / "data"


def build_chat_agent(settings: DeepAgentSettings | None = None, data_tools: list[BaseTool] | None = None) -> Any:
    """Собирает аналитического DeepAgent для терминального чата.

    Args:
        settings: Настройки агента. Если ``None``, загружается JSON-конфиг по умолчанию.
        data_tools: Инструменты чтения данных. Если ``None``, используются production tools из конфига.

    Returns:
        Скомпилированный DeepAgent graph с моделью из корневого ``model.py``.
    """

    from model import embeddings as openrouter_embeddings
    from model import model as openrouter_model

    return build_analytics_deep_agent(
        openrouter_model,
        embeddings_model=openrouter_embeddings,
        settings=settings,
        data_tools=data_tools,
    )


def build_test_data_tools(data_dir: Path = TEST_DATA_DIR) -> list[BaseTool]:
    """Создает sync/async tools чтения тестовых CSV-данных агента.

    Args:
        data_dir: Папка с CSV-файлами ``hits``, ``cards_event`` и ``uko_event``.

    Returns:
        Список LangChain tools с одним инструментом ``read_table``, пригодным
        для синхронного ``agent.invoke`` и асинхронного ``agent.ainvoke``.
    """

    from examples.fake_spark_tools import build_fake_spark_tools

    raw_tool = build_fake_spark_tools(delay_seconds=0.0, data_dir=data_dir)[0]

    def read_table_sync(**kwargs: Any) -> Any:
        """Выполняет тестовый ``read_table`` из синхронного runner-а.

        Args:
            kwargs: Аргументы ``read_table``: table_name, select_columns, filters, max_rows и include_schema.

        Returns:
            DataFrame с результатом выборки или текстовая ошибка инструмента.
        """

        import asyncio

        return asyncio.run(raw_tool.ainvoke(kwargs))

    async def read_table_async(**kwargs: Any) -> Any:
        """Выполняет тестовый ``read_table`` из асинхронного runner-а.

        Args:
            kwargs: Аргументы ``read_table``: table_name, select_columns, filters, max_rows и include_schema.

        Returns:
            DataFrame с результатом выборки или текстовая ошибка инструмента.
        """

        return await raw_tool.ainvoke(kwargs)

    return [
        StructuredTool.from_function(
            func=read_table_sync,
            coroutine=read_table_async,
            name=raw_tool.name,
            description=raw_tool.description,
            args_schema=raw_tool.args_schema,
        )
    ]


def make_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Создает config LangGraph с идентификатором диалога.

    Args:
        thread_id: Идентификатор диалога для checkpointer и resume после interrupt.

    Returns:
        Словарь config для ``agent.invoke`` и ``agent.get_state``.
    """

    return {"configurable": {"thread_id": thread_id}}


def run_chat(settings: DeepAgentSettings | None = None, data_tools: list[BaseTool] | None = None) -> None:
    """Запускает интерактивный терминальный чат с агентом.

    Args:
        settings: Настройки агента. Если ``None``, загружается JSON-конфиг по умолчанию.
        data_tools: Инструменты чтения данных. Если ``None``, используются production tools из конфига.

    Returns:
        None.
    """

    settings = settings or load_deep_agent_settings()
    agent = build_chat_agent(settings=settings, data_tools=data_tools)
    config = make_config(settings.thread_id)
    loaded_skills_printed = False

    while True:
        user_message = input("Вы: ").strip()
        if user_message.lower() in EXIT_COMMANDS:
            return
        if not user_message:
            continue
        if user_message.lower() in HITL_DECISION_COMMANDS:
            continue

        result = invoke_user_message(agent, config, user_message)
        result = continue_until_agent_boundary(agent, config, result)
        loaded_skills_printed = print_loaded_skills_once(result, already_printed=loaded_skills_printed)
        print_turn_result(result)

        while True:
            interrupts = extract_interrupt_values(result)
            if not interrupts:
                break

            decisions = collect_human_decisions(interrupts)
            result = resume_with_decisions(agent, config, decisions)
            result = continue_until_agent_boundary(
                agent,
                config,
                result,
                require_progress=requires_progress_after_decisions(decisions),
            )
            loaded_skills_printed = print_loaded_skills_once(result, already_printed=loaded_skills_printed)
            print_turn_result(result)


def invoke_user_message(agent: Any, config: dict[str, Any], message: str) -> Any:
    """Отправляет новое пользовательское сообщение агенту.

    Args:
        agent: Скомпилированный DeepAgent graph.
        config: Config LangGraph с thread_id.
        message: Текст пользовательского сообщения.

    Returns:
        Результат ``agent.invoke``: состояние graph или interrupt payload.
    """

    return agent.invoke({"messages": [{"role": "user", "content": message}]}, config=config)


def resume_with_decisions(agent: Any, config: dict[str, Any], decisions: list[dict[str, Any]]) -> Any:
    """Продолжает выполнение после human-in-the-loop interrupt.

    Args:
        agent: Скомпилированный DeepAgent graph.
        config: Config LangGraph с тем же thread_id.
        decisions: Список решений пользователя для всех interrupted tool calls.

    Returns:
        Следующий результат ``agent.invoke``.
    """

    return agent.invoke(Command(resume={"decisions": decisions}), config=config)


def collect_human_decisions(interrupts: list[Any]) -> list[dict[str, Any]]:
    """Собирает решения пользователя по всем interrupt payload.

    Args:
        interrupts: Список значений ``Interrupt.value`` из результата LangGraph.

    Returns:
        Список решений в формате ``HumanInTheLoopMiddleware``.
    """

    decisions: list[dict[str, Any]] = []
    for interrupt_payload in interrupts:
        action_requests = interrupt_payload.get("action_requests", [])
        review_configs = interrupt_payload.get("review_configs", [])
        for index, action_request in enumerate(action_requests):
            review_config = review_configs[index]
            decisions.append(collect_single_decision(action_request, review_config))
    return decisions


def collect_single_decision(action_request: dict[str, Any], review_config: dict[str, Any]) -> dict[str, Any]:
    """Запрашивает у пользователя одно решение approve/edit/reject/respond.

    Args:
        action_request: Описание tool call, который требует решения пользователя.
        review_config: Разрешенные варианты решения для этого tool call.

    Returns:
        Решение пользователя в формате ``approve``, ``edit``, ``reject`` или ``respond``.
    """

    allowed = review_config.get("allowed_decisions", [])
    action_name = action_request.get("name", "<unknown>")
    args = action_request.get("args", {})

    if allowed == ["respond"]:
        question = args.get("question") or action_request.get("description") or "Уточните задачу."
        print()
        print("Агент:")
        print(question)
        answer = input("Вы: ").strip()
        return {"type": "respond", "message": answer}

    if action_name == "write_todos":
        print()
        print("Агент:")
        print("План анализа:")
        print(format_todos_for_user(args.get("todos", [])))
    else:
        print()
        print("Агент:")
        print(action_request.get("description") or f"Требуется решение для действия {action_name}.")

    aliases = {
        "a": "approve",
        "approve": "approve",
        "ok": "approve",
        "да": "approve",
        "e": "edit",
        "edit": "edit",
        "r": "reject",
        "reject": "reject",
        "нет": "reject",
    }

    while True:
        raw_decision = input(f"Решение ({', '.join(allowed)}): ").strip().lower()
        decision_type = aliases.get(raw_decision, raw_decision)
        if decision_type not in allowed:
            continue
        if decision_type == "approve":
            return {"type": "approve"}
        if decision_type == "reject":
            message = input("Причина отклонения: ").strip()
            return {"type": "reject", "message": message}
        if decision_type == "respond":
            message = input("Ответ агенту: ").strip()
            return {"type": "respond", "message": message}
        if decision_type == "edit":
            return collect_edit_feedback(action_name)


def collect_edit_feedback(action_name: str) -> dict[str, str]:
    """Читает текстовые правки пользователя на естественном языке.

    Args:
        action_name: Имя tool call, который пользователь хочет изменить.

    Returns:
        Решение HITL. Для плана возвращается ``edit`` с текстом правок, а для
        остальных tools возвращается ``reject`` с просьбой перестроить tool call.
    """

    feedback = input("Опишите правки обычным текстом: ").strip()
    if action_name == "write_todos":
        return {"type": "edit", "message": feedback}
    return {
        "type": "reject",
        "message": (
            f"Пользователь просит изменить вызов tool `{action_name}`. "
            f"Сформируй новый вызов инструмента с учетом правки: {feedback}"
        ),
    }


def continue_until_agent_boundary(
    agent: Any,
    config: dict[str, Any],
    result: Any,
    *,
    require_progress: bool = False,
) -> Any:
    """Продолжает выполнение до interrupt или state без незавершенных todo.

    Args:
        agent: Скомпилированный DeepAgent graph.
        config: Config LangGraph с thread_id.
        result: Последний результат ``agent.invoke`` или ``resume``.
        require_progress: Нужно ли требовать следующий tool/subagent шаг после HITL.

    Returns:
        Последний результат после автоматического продолжения или исходный результат.
    """

    current = result
    while True:
        if extract_interrupt_values(current):
            return current
        if not should_continue_agent_loop(current, require_progress=require_progress):
            return current
        continue_instruction = build_continue_instruction(current, require_progress=require_progress)
        current = invoke_user_message(agent, config, continue_instruction)
        if has_unfinished_todos(current) or has_completed_todos(current) or extract_interrupt_values(current):
            require_progress = False


def should_continue_agent_loop(result: Any, *, require_progress: bool) -> bool:
    """Определяет необходимость продолжения по state, без анализа текста ответа.

    Args:
        result: Результат ``agent.invoke`` или ``resume``.
        require_progress: Нужно ли требовать tool/subagent шаг после HITL.

    Returns:
        ``True``, если runner должен вернуть управление агенту.
    """

    if not isinstance(result, dict):
        return False
    if extract_interrupt_values(result):
        return False
    if last_message_has_tool_calls(result):
        return False
    return (
        has_unfinished_todos(result)
        or needs_final_response_after_completed_todos(result)
        or (require_progress and not has_completed_todos(result))
    )


def build_continue_instruction(result: Any, *, require_progress: bool) -> str:
    """Формирует служебную инструкцию продолжить выполнение без проверки фраз.

    Args:
        result: Текущий state graph.
        require_progress: Был ли предыдущий шаг HITL-ответом, требующим действия.

    Returns:
        Текст служебного сообщения для следующего ``agent.invoke``.
    """

    if has_only_final_response_todo_in_progress(result) or needs_final_response_after_completed_todos(result):
        return (
            "Служебная инструкция runner-а: сформируй финальный ответ пользователю сейчас. "
            "Не вызывай write_todos повторно, если todos уже отражают выполненную работу. "
            "Укажи вывод, проверенные таблицы, поля, фильтры, найденные значения и ограничения данных."
        )

    todos = result.get("todos") if isinstance(result, dict) else None
    todos_note = f" Текущий todo state: {todos}" if todos else ""
    progress_note = (
        " Предыдущий шаг был human-in-the-loop решением, после которого ожидается "
        "продолжение через tool/subagent."
        if require_progress
        else ""
    )
    return RUNNER_CONTINUE_INSTRUCTION_TEMPLATE.format(
        progress_note=progress_note,
        todos_note=todos_note,
    )


def requires_progress_after_decisions(decisions: list[dict[str, Any]]) -> bool:
    """Проверяет, должен ли agent loop продолжиться после HITL-решений.

    Args:
        decisions: Решения пользователя по interrupt.

    Returns:
        ``True`` для approve/edit/respond, потому что после них ожидается действие.
    """

    return any(decision.get("type") in {"approve", "edit", "respond"} for decision in decisions)


def print_loaded_skills_once(result: Any, *, already_printed: bool) -> bool:
    """Печатает список загруженных skills один раз после первого запуска агента.

    Args:
        result: Результат ``agent.invoke`` или ``resume`` с состоянием graph.
        already_printed: Признак, что список skills уже был выведен в этой сессии runner-а.

    Returns:
        ``True``, если список уже был напечатан или был напечатан текущим вызовом.
    """

    if already_printed or not isinstance(result, dict):
        return already_printed

    skill_paths = result.get("preloaded_skill_paths") or []
    if not skill_paths:
        return False

    print()
    print("Агент:")
    print("Загруженные skills:")
    for index, skill_path in enumerate(skill_paths, start=1):
        print(f"{index}. {skill_path}")
    print()
    return True


def print_turn_result(result: Any) -> None:
    """Печатает только содержательный ответ агента без промежуточной диагностики.

    Args:
        result: Результат ``agent.invoke``.

    Returns:
        None.
    """

    interrupts = extract_interrupt_values(result)
    if interrupts:
        return

    text = last_agent_response_text(result)
    if text:
        print()
        print("Агент:")
        print(text)
        print()


def extract_interrupt_values(result: Any) -> list[Any]:
    """Извлекает значения interrupt из результата LangGraph.

    Args:
        result: Результат ``agent.invoke``.

    Returns:
        Список ``Interrupt.value`` или пустой список, если interrupt нет.
    """

    if not isinstance(result, dict):
        return []

    raw_interrupts = result.get("__interrupt__") or ()
    values: list[Any] = []
    for item in raw_interrupts:
        value = getattr(item, "value", None)
        if value is None and isinstance(item, dict):
            value = item.get("value", item)
        values.append(value)
    return values


def last_agent_response_text(result: Any) -> str:
    """Возвращает новый содержательный ответ агента из последнего сообщения.

    Args:
        result: Результат ``agent.invoke`` после обычного шага или resume.

    Returns:
        Текст последнего сообщения, если это новый AI-ответ без tool calls.
        Пустая строка возвращается для ToolMessage и других промежуточных
        сообщений, чтобы не печатать старые ответы из истории повторно.
    """

    if not isinstance(result, dict):
        return ""

    messages = result.get("messages") or []
    if not messages:
        return ""
    message = messages[-1]
    if getattr(message, "tool_calls", None):
        return ""
    if type(message).__name__ != "AIMessage":
        return ""
    return message_to_text(message).strip()


def last_message_has_tool_calls(result: Any) -> bool:
    """Проверяет, содержит ли последнее сообщение tool calls.

    Args:
        result: Результат ``agent.invoke`` или ``resume``.

    Returns:
        ``True``, если последнее сообщение содержит tool calls.
    """

    if not isinstance(result, dict):
        return False
    messages = result.get("messages") or []
    if not messages:
        return False
    return bool(getattr(messages[-1], "tool_calls", None))


def has_unfinished_todos(result: Any) -> bool:
    """Проверяет наличие незавершенных todo в state graph.

    Args:
        result: Результат ``agent.invoke`` или ``resume``.

    Returns:
        ``True``, если есть todo со статусом не ``completed``.
    """

    todos = result.get("todos") if isinstance(result, dict) else None
    if not isinstance(todos, list):
        return False
    return any(isinstance(todo, dict) and todo.get("status") != "completed" for todo in todos)


def has_completed_todos(result: Any) -> bool:
    """Проверяет, что в state есть хотя бы один завершенный todo.

    Args:
        result: Результат ``agent.invoke`` или ``resume``.

    Returns:
        ``True``, если в state есть completed todo.
    """

    todos = result.get("todos") if isinstance(result, dict) else None
    if not isinstance(todos, list):
        return False
    return any(isinstance(todo, dict) and todo.get("status") == "completed" for todo in todos)


def has_only_final_response_todo_in_progress(result: Any) -> bool:
    """Проверяет, что единственный незавершенный todo относится к финальному ответу.

    Args:
        result: Результат ``agent.invoke`` или ``resume``.

    Returns:
        ``True``, если все рабочие пункты завершены, а незавершенным остался
        только пункт подготовки финального ответа.
    """

    todos = result.get("todos") if isinstance(result, dict) else None
    if not isinstance(todos, list):
        return False

    unfinished = [todo for todo in todos if isinstance(todo, dict) and todo.get("status") != "completed"]
    if len(unfinished) != 1:
        return False

    content = str(unfinished[0].get("content") or "").lower()
    final_markers = ("финаль", "итог", "ответ", "вывод")
    return any(marker in content for marker in final_markers)


def needs_final_response_after_completed_todos(result: Any) -> bool:
    """Проверяет, нужно ли продолжить graph ради финального AI-ответа.

    Args:
        result: Результат ``agent.invoke`` или ``resume``.

    Returns:
        ``True``, если все todo завершены, но последний message является
        результатом tool, а не содержательным AI-ответом.
    """

    if not isinstance(result, dict):
        return False
    todos = result.get("todos")
    if not isinstance(todos, list) or not todos:
        return False
    return not has_unfinished_todos(result) and last_message_is_tool_message(result)


def last_message_is_tool_message(result: Any) -> bool:
    """Проверяет, что последним сообщением в state является ToolMessage.

    Args:
        result: Результат ``agent.invoke`` или ``resume``.

    Returns:
        ``True``, если последний message создан инструментом.
    """

    if not isinstance(result, dict):
        return False
    messages = result.get("messages") or []
    if not messages:
        return False
    return type(messages[-1]).__name__ == "ToolMessage"


def format_todos_for_user(todos: list[dict[str, Any]]) -> str:
    """Форматирует список задач плана в короткий человекочитаемый текст.

    Args:
        todos: Список пунктов плана из аргументов ``write_todos``.

    Returns:
        Нумерованный список пунктов плана или сообщение об отсутствии плана.
    """

    if not todos:
        return "План не указан."

    status_labels = {
        "pending": "ожидает",
        "in_progress": "в работе",
        "completed": "готово",
    }
    lines = []
    for index, todo in enumerate(todos, start=1):
        content = str(todo.get("content") or "").strip() or "Без описания"
        status = status_labels.get(str(todo.get("status") or ""), str(todo.get("status") or ""))
        suffix = f" [{status}]" if status else ""
        lines.append(f"{index}. {content}{suffix}")
    return "\n".join(lines)


def message_to_text(message: Any) -> str:
    """Преобразует LangChain message или словарь сообщения в текст.

    Args:
        message: Сообщение LangChain, словарь или произвольный объект.

    Returns:
        Текстовое представление содержимого сообщения.
    """

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        return "\n".join(str(block) for block in content)
    if content is not None:
        return str(content)
    return str(message)


def main() -> int:
    """Запускает пример терминального чата на тестовых CSV-данных.

    Args:
        Отсутствуют.

    Returns:
        Код успешного завершения процесса.
    """

    settings = load_deep_agent_settings()  # Загружаем настройки агента из defaults.json или override-конфига.
    test_data_tools = build_test_data_tools(TEST_DATA_DIR)  # Создаем sync/async read_table для локальных CSV.
    run_chat(settings=settings, data_tools=test_data_tools)  # Запускаем чат с настройками и тестовыми tools.
    return 0  # Возвращаем код успешного завершения процесса для SystemExit.


if __name__ == "__main__":
    raise SystemExit(main())


# подтяни к сработке 3486d84b-4eba-4ba4-b044-94764fc9e7a4 информацию о городе в котором был пользователь по ip

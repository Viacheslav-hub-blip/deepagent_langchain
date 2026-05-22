"""End-to-end сценарий research-agent с run graph, веткой и диалоговым follow-up.

Содержит:
- ScenarioSandbox: минимальная песочница для dataframe-контекста.
- _load_example_dataframe: загрузка стартовых тестовых данных.
- build_agent: сборка ResearchAgent с моделью из ``model.py`` и тестовыми Spark tools.
- _format_message_content: приведение LangChain message content к строке.
- _last_ai_message_text: получение последнего AI-сообщения.
- _snapshot_existing_run_ids: фиксация существующих run_id перед запуском.
- _find_latest_run_dir: поиск нового run-каталога для мониторинга прогресса.
- _read_jsonl_rows: безопасное чтение JSONL-файлов lineage/artifacts.
- _print_progress_event: печать одного lineage event.
- _print_artifact_progress: печать одного artifact event.
- _monitor_run_progress: фоновый мониторинг lineage и artifacts.
- _run_with_progress: запуск awaitable с прогрессом и timeout.
- _print_run_result: печать результата через read API агента.
- _select_branch_source_node_id: выбор node, от которого стартует ветка.
- _collect_dataset_artifact_ids: сбор dataset artifacts для BranchRequest.
- _build_initial_query: запрос для базового исследования.
- _build_branch_request: запрос на ветвление с альтернативной гипотезой.
- _format_artifact_manifest: подготовка списка artifacts для dialog context.
- _build_dialog_messages: подготовка обычного chat-style follow-up.
- run_initial_research: первый end-to-end запуск.
- run_branch_research: запуск ветки от сохраненного lineage node.
- run_dialog_follow_up: чатовый follow-up поверх результатов base и branch.
- main: полный сценарий base run -> branch run -> dialog run.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.fake_spark_tools import build_fake_spark_tools  # noqa: E402
from planner_agent import ResearchAgent  # noqa: E402
from planner_agent.schemas.lineage import BranchRequest  # noqa: E402
from planner_agent.services.run_inspection_service import RunResult  # noqa: E402


RUN_TIMEOUT_SECONDS = 420
PROGRESS_POLL_SECONDS = 1.5
DIALOG_REPORT_CHARS = 12_000


class ScenarioSandbox:
    """Минимальная песочница для передачи dataframe-контекста в workspace tools.

    Args:
        dataframe: Стартовая таблица, которая будет доступна агенту как ``df_current``.

    Returns:
        Экземпляр песочницы с методами, которые ожидает ResearchAgent.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """Сохраняет dataframe в словарь переменных песочницы.

        Args:
            dataframe: Таблица pandas DataFrame для стартового контекста.

        Returns:
            None.
        """

        self.last_dataframe_variable = "df_current"
        self.globals: dict[str, Any] = {"df_current": dataframe}

    async def get_all_variable_previews(self) -> dict[str, str]:
        """Возвращает краткие описания всех переменных песочницы.

        Args:
            Отсутствуют.

        Returns:
            Словарь ``имя переменной -> текстовое описание``. Для DataFrame описание
            включает shape, список колонок, null counts и первые строки.
        """

        previews: dict[str, str] = {}
        for name, value in self.globals.items():
            if isinstance(value, pd.DataFrame):
                previews[name] = (
                    f"shape={value.shape}; "
                    f"columns={list(value.columns)}; "
                    f"null_counts={value.isna().sum().to_dict()}; "
                    f"head={value.head(3).to_dict(orient='records')}"
                )
            else:
                previews[name] = str(value)[:1_000]
        return previews

    async def add_variable(self, name: str, value: object) -> None:
        """Добавляет или обновляет переменную в песочнице.

        Args:
            name: Имя переменной.
            value: Значение переменной.

        Returns:
            None.
        """

        self.globals[name] = value
        if isinstance(value, pd.DataFrame):
            self.last_dataframe_variable = name

    async def get_variable(self, name: str) -> object:
        """Возвращает переменную из песочницы по имени.

        Args:
            name: Имя переменной.

        Returns:
            Значение переменной или ``None``, если переменная отсутствует.
        """

        return self.globals.get(name)


def _load_example_dataframe() -> pd.DataFrame:
    """Загружает стартовую таблицу для демонстрационного запуска.

    Args:
        Отсутствуют.

    Returns:
        DataFrame из ``examples/data/cspfs_repo_features3.hits_extra_info_129372427_view.csv`` или встроенную одну строку,
        если файл недоступен.
    """

    data_path = PROJECT_ROOT / "examples" / "data" / "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
    if data_path.exists():
        return pd.read_csv(data_path)

    return pd.DataFrame(
        [
            {
                "client_id": "client-42",
                "event_date": "2025-01-03",
                "event_type": "payment",
                "amount": 1500.0,
                "merchant_name": "AutoPay Mobile",
                "recipient": "self_account",
            }
        ]
    )


def build_agent() -> ResearchAgent:
    """Собирает ResearchAgent для ручного end-to-end сценария.

    Args:
        Отсутствуют.

    Returns:
        Экземпляр ResearchAgent с моделью из ``model.py``, тестовыми Spark tools,
        skills/memory/runs из каталога ``examples``.
    """

    from model import model as configured_model  # noqa: WPS433

    examples_root = PROJECT_ROOT / "examples"
    sandbox = ScenarioSandbox(_load_example_dataframe())
    spark_tools = build_fake_spark_tools(delay_seconds=0.7)

    return ResearchAgent(
        model=configured_model,
        sandbox=sandbox,
        tools=spark_tools,
        workspace_root=str(PROJECT_ROOT),
        sources_dir=str(examples_root / "data"),
        contexts_dir=str(examples_root / "skills"),
        skills_dir=str(examples_root / "skills"),
        memory_dir=str(examples_root / "memory"),
        runs_dir=str(examples_root / "runs"),
    )


def _format_message_content(content: object) -> str:
    """Преобразует содержимое LangChain message в строку для печати.

    Args:
        content: Содержимое сообщения. Обычно строка, иногда список блоков.

    Returns:
        Строковое представление содержимого.
    """

    if isinstance(content, str):
        return content
    return str(content)


def _last_ai_message_text(messages: list[BaseMessage]) -> str:
    """Возвращает текст последнего AI-сообщения.

    Args:
        messages: Список LangChain messages, который вернул ResearchAgent.

    Returns:
        Текст последнего AIMessage или последнего сообщения, если AIMessage не найден.
    """

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return _format_message_content(message.content)
    if messages:
        return _format_message_content(messages[-1].content)
    return ""


def _snapshot_existing_run_ids(runs_dir: Path) -> set[str]:
    """Фиксирует существующие run_id перед новым запуском.

    Args:
        runs_dir: Каталог, где ResearchAgent хранит запуски.

    Returns:
        Множество имен существующих run-каталогов.
    """

    if not runs_dir.exists():
        return set()
    return {path.name for path in runs_dir.iterdir() if path.is_dir()}


def _find_latest_run_dir(runs_dir: Path, ignored_run_ids: set[str]) -> Path | None:
    """Находит самый свежий run-каталог, которого не было до запуска.

    Args:
        runs_dir: Каталог с ResearchRun.
        ignored_run_ids: run_id, которые существовали до текущей фазы.

    Returns:
        Путь к новому run-каталогу или ``None``.
    """

    if not runs_dir.exists():
        return None
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and path.name not in ignored_run_ids
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Безопасно читает JSONL-файл.

    Args:
        path: Путь к JSONL-файлу.

    Returns:
        Список словарей из валидных строк файла.
    """

    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _print_progress_event(node: dict[str, Any]) -> None:
    """Печатает один lineage node в компактном формате.

    Args:
        node: JSON-представление StateNode.

    Returns:
        None.
    """

    node_type = node.get("node_type", "unknown")
    status = node.get("status", "unknown")
    title = node.get("title", "")
    summary = str(node.get("summary", "")).replace("\n", " ")[:220]
    created_at = node.get("created_at", "")
    print(
        f"[progress] {created_at} | {node_type} | {status} | {title} | {summary}",
        flush=True,
    )


def _print_artifact_progress(artifact: dict[str, Any]) -> None:
    """Печатает один artifact event в компактном формате.

    Args:
        artifact: JSON-представление Artifact.

    Returns:
        None.
    """

    artifact_id = artifact.get("artifact_id", "")
    kind = artifact.get("kind", "")
    uri = artifact.get("uri", "")
    print(f"[artifact] {artifact_id} | {kind} | {uri}", flush=True)


async def _monitor_run_progress(
        runs_dir: Path,
        ignored_run_ids: set[str],
        stop_event: asyncio.Event,
) -> None:
    """Печатает lineage nodes и artifacts во время выполнения одной фазы.

    Args:
        runs_dir: Каталог с ResearchRun.
        ignored_run_ids: run_id, существовавшие до старта фазы.
        stop_event: Событие остановки мониторинга.

    Returns:
        None.
    """

    printed_node_ids: set[str] = set()
    printed_artifact_ids: set[str] = set()
    active_run_dir: Path | None = None

    while not stop_event.is_set():
        active_run_dir = active_run_dir or _find_latest_run_dir(
            runs_dir=runs_dir,
            ignored_run_ids=ignored_run_ids,
        )
        if active_run_dir is not None:
            for node in _read_jsonl_rows(active_run_dir / "lineage.jsonl"):
                node_id = str(node.get("node_id") or "")
                if node_id and node_id not in printed_node_ids:
                    printed_node_ids.add(node_id)
                    _print_progress_event(node)

            for artifact in _read_jsonl_rows(active_run_dir / "artifacts.jsonl"):
                artifact_id = str(artifact.get("artifact_id") or "")
                if artifact_id and artifact_id not in printed_artifact_ids:
                    printed_artifact_ids.add(artifact_id)
                    _print_artifact_progress(artifact)

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=PROGRESS_POLL_SECONDS,
            )
        except TimeoutError:
            continue


async def _run_with_progress(
        *,
        label: str,
        runs_dir: Path,
        awaitable: Awaitable[list[BaseMessage]],
) -> list[BaseMessage]:
    """Запускает фазу агента с мониторингом прогресса и timeout.

    Args:
        label: Название фазы для печати в консоль.
        runs_dir: Каталог с ResearchRun.
        awaitable: Awaitable, который запускает агента и возвращает messages.

    Returns:
        Список LangChain messages из завершенного запуска.

    Raises:
        TimeoutError: Если фаза не завершилась за ``RUN_TIMEOUT_SECONDS``.
    """

    print(f"\n{label}\n{'=' * len(label)}", flush=True)
    ignored_run_ids = _snapshot_existing_run_ids(runs_dir)
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(
        _monitor_run_progress(
            runs_dir=runs_dir,
            ignored_run_ids=ignored_run_ids,
            stop_event=stop_event,
        )
    )
    try:
        return await asyncio.wait_for(awaitable, timeout=RUN_TIMEOUT_SECONDS)
    finally:
        stop_event.set()
        await monitor_task


def _print_run_result(agent: ResearchAgent, run_id: str, title: str) -> RunResult | None:
    """Печатает сохраненный результат ResearchRun через публичный read API.

    Args:
        agent: Экземпляр ResearchAgent.
        run_id: Идентификатор запуска.
        title: Заголовок блока печати.

    Returns:
        RunResult или ``None``, если запуск не найден.
    """

    result = agent.get_run_result(run_id)
    print(f"\n{title}\n{'=' * len(title)}")
    if result is None:
        print(f"Run result is unavailable for run_id={run_id}")
        return None

    print(f"run_id: {result.run.run_id}")
    print(f"parent_run_id: {result.run.parent_run_id}")
    print(f"source_node_id: {result.run.source_node_id}")
    print(f"node_count: {result.summary.node_count}")
    print(f"artifact_count: {result.summary.artifact_count}")
    print(f"final_report_node_id: {result.summary.final_report_node_id}")
    print(f"final_report_artifact_id: {result.summary.final_report_artifact_id}")
    print(f"final_report_chars: {len(result.final_report or '')}")
    print("\nGraph nodes:")
    for node in result.nodes:
        print(f"- {node.node_type} | {node.status} | {node.title} | {node.node_id}")

    print("\nArtifacts:")
    for artifact in result.artifacts:
        role = artifact.metadata.get("artifact_role", "")
        tool_name = artifact.metadata.get("tool_name", "")
        metadata = " | ".join(part for part in (role, tool_name) if part)
        suffix = f" | {metadata}" if metadata else ""
        print(f"- {artifact.artifact_id}: {artifact.kind}{suffix} | {artifact.uri}")
    return result


def _select_branch_source_node_id(result: RunResult) -> str:
    """Выбирает node, от которого будет создана новая ветка.

    Args:
        result: Полный результат базового ResearchRun.

    Returns:
        node_id финального отчета, если он есть, иначе последний node запуска.

    Raises:
        ValueError: Если в результате нет lineage nodes.
    """

    if result.summary.final_report_node_id:
        return result.summary.final_report_node_id
    if not result.nodes:
        raise ValueError("Cannot create branch: base run has no lineage nodes")
    return result.nodes[-1].node_id


def _collect_dataset_artifact_ids(result: RunResult) -> list[str]:
    """Собирает dataset artifacts, которые полезно явно передать в BranchRequest.

    Args:
        result: Полный результат базового ResearchRun.

    Returns:
        Список artifact_id с ``kind == "dataset"``.
    """

    return [
        artifact.artifact_id
        for artifact in result.artifacts
        if artifact.kind == "dataset"
    ]


def _build_initial_query() -> str:
    """Формирует пользовательский запрос для базового исследования.

    Args:
        Отсутствуют.

    Returns:
        Текст запроса для первого ResearchRun.
    """

    return (
        "Проведи end-to-end анализ клиента client-42 по CSI-сработке на дату "
        "2025-01-03. Используй spark_lookup_trigger_cases для получения сработок, "
        "spark_get_uko_events и spark_get_cards_events "
        "для выгрузки событий клиента за 90 дней "
        "за день. Найди повторяющиеся паттерны поведения, укажи какие данные "
        "были сохранены как artifacts, и явно отмечай неполноту данных, если "
        "используются preview или chunks."
    )


def _build_branch_request(base_result: RunResult) -> BranchRequest:
    """Создает BranchRequest для проверки альтернативной гипотезы.

    Args:
        base_result: Результат базового запуска, от которого создается ветка.

    Returns:
        BranchRequest в режиме ``what_if``.
    """

    source_node_id = _select_branch_source_node_id(base_result)
    return BranchRequest(
        source_run_id=base_result.run.run_id,
        source_node_id=source_node_id,
        branch_mode="what_if",
        artifact_refs=_collect_dataset_artifact_ids(base_result),
        include_artifacts=True,
        include_memory_snapshot=True,
        include_completed_tasks=True,
        new_task=(
            "Создай ветку от текущего состояния и проверь альтернативную гипотезу: "
            "сработка могла быть связана не с автопополнением, а с изменением "
            "получателя или recipient velocity. Переиспользуй уже созданные "
            "artifacts вместо повторной выгрузки, если их достаточно. Не вызывай "
            "инструмент выгрузки повторно для тех же данных без явной причины. Если "
            "используешь только preview/chunk, явно напиши, что вывод основан "
            "на части данных. Сравни эту гипотезу с базовой."
        ),
    )


def _format_artifact_manifest(result: RunResult) -> str:
    """Формирует текстовый список artifacts запуска для передачи в dialog context.

    Args:
        result: Результат ResearchRun, artifacts которого нужно описать.

    Returns:
        Многострочный текст с ``artifact_id``, ``kind``, ``uri`` и ключевой metadata.
    """

    lines = [
        f"run_id: {result.run.run_id}",
        f"final_report_node_id: {result.summary.final_report_node_id}",
        f"final_report_artifact_id: {result.summary.final_report_artifact_id}",
        "artifacts:",
    ]
    for artifact in result.artifacts:
        role = artifact.metadata.get("artifact_role", "")
        tool_name = artifact.metadata.get("tool_name", "")
        task_id = artifact.metadata.get("task_id", "")
        metadata = ", ".join(
            f"{key}={value}"
            for key, value in (
                ("role", role),
                ("tool", tool_name),
                ("task_id", task_id),
            )
            if value
        )
        metadata_suffix = f"; metadata: {metadata}" if metadata else ""
        lines.append(
            "- "
            f"artifact_id: {artifact.artifact_id}; "
            f"kind: {artifact.kind}; "
            f"uri: {artifact.uri}; "
            f"summary: {artifact.summary}"
            f"{metadata_suffix}"
        )
    return "\n".join(lines)


def _build_dialog_messages(base_result: RunResult, branch_result: RunResult) -> list[BaseMessage]:
    """Создает обычную chat-style историю для follow-up без UI.

    Args:
        base_result: Результат базового запуска.
        branch_result: Результат ветки.

    Returns:
        Список LangChain messages для ``ResearchAgent.ainvoke``.
    """

    base_report = (base_result.final_report or "")[:DIALOG_REPORT_CHARS]
    branch_report = (branch_result.final_report or "")[:DIALOG_REPORT_CHARS]
    base_artifacts = _format_artifact_manifest(base_result)
    branch_artifacts = _format_artifact_manifest(branch_result)
    return [
        HumanMessage(
            content=(
                "Ниже результаты двух исследовательских запусков. Я хочу "
                "получить короткое сравнение и рекомендацию, какую гипотезу "
                "проверять дальше. Используй не только текст отчетов, но и "
                "списки artifacts с их artifact_id и uri."
            )
        ),
        AIMessage(
            content=(
                f"Base run_id={base_result.run.run_id}\n\n"
                f"<base_artifacts>\n{base_artifacts}\n</base_artifacts>\n\n"
                f"<base_report>\n{base_report}\n</base_report>"
            )
        ),
        AIMessage(
            content=(
                f"Branch run_id={branch_result.run.run_id}\n"
                f"source_node_id={branch_result.run.source_node_id}\n\n"
                f"<branch_artifacts>\n{branch_artifacts}\n</branch_artifacts>\n\n"
                f"<branch_report>\n{branch_report}\n</branch_report>"
            )
        ),
        HumanMessage(
            content=(
                "Сравни базовую гипотезу и ветку. Что сильнее подтверждается "
                "данными, какие artifacts нужно открыть для проверки, и какой "
                "следующий шаг ты бы сделал? В ответе называй artifacts именно "
                "по artifact_id и uri из предоставленного списка."
            )
        ),
    ]


async def run_initial_research(agent: ResearchAgent, runs_dir: Path) -> RunResult:
    """Запускает базовое исследование и возвращает сохраненный RunResult.

    Args:
        agent: Экземпляр ResearchAgent.
        runs_dir: Каталог, где агент хранит ResearchRun.

    Returns:
        RunResult базового запуска.

    Raises:
        RuntimeError: Если агент не сохранил результат запуска.
    """

    messages = await _run_with_progress(
        label="Phase 1: base research run",
        runs_dir=runs_dir,
        awaitable=agent.ainvoke(
            {
                "user_query": _build_initial_query(),
                "session_id": "e2e-branch-dialog-session",
                "user_id": "manual-tester",
            },
            config={"recursion_limit": 40},
        ),
    )
    print("\nBase final message\n==================")
    print(_last_ai_message_text(messages))

    result = _print_run_result(agent, agent.last_run_id, "Base run result API")
    if result is None:
        raise RuntimeError("Base run finished but RunResult is unavailable")
    return result


async def run_branch_research(
        agent: ResearchAgent,
        runs_dir: Path,
        base_result: RunResult,
) -> RunResult:
    """Создает ветку от базового run и запускает уточненную гипотезу.

    Args:
        agent: Экземпляр ResearchAgent.
        runs_dir: Каталог, где агент хранит ResearchRun.
        base_result: Результат базового запуска.

    Returns:
        RunResult ветки.

    Raises:
        RuntimeError: Если агент не сохранил результат ветки.
    """

    branch_request = _build_branch_request(base_result)
    print("\nBranch request\n==============")
    print(f"source_run_id: {branch_request.source_run_id}")
    print(f"source_node_id: {branch_request.source_node_id}")
    print(f"branch_mode: {branch_request.branch_mode}")
    print(f"artifact_refs: {branch_request.artifact_refs}")

    messages = await _run_with_progress(
        label="Phase 2: branch run with alternative hypothesis",
        runs_dir=runs_dir,
        awaitable=agent.ainvoke_branch(
            branch_request,
            config={"recursion_limit": 40},
        ),
    )
    print("\nBranch final message\n====================")
    print(_last_ai_message_text(messages))

    result = _print_run_result(agent, agent.last_run_id, "Branch run result API")
    if result is None:
        raise RuntimeError("Branch run finished but RunResult is unavailable")
    return result


async def run_dialog_follow_up(
        agent: ResearchAgent,
        runs_dir: Path,
        base_result: RunResult,
        branch_result: RunResult,
) -> RunResult:
    """Запускает обычный chat-style follow-up поверх двух сохраненных запусков.

    Args:
        agent: Экземпляр ResearchAgent.
        runs_dir: Каталог, где агент хранит ResearchRun.
        base_result: Результат базового запуска.
        branch_result: Результат ветки.

    Returns:
        RunResult диалогового follow-up запуска.

    Raises:
        RuntimeError: Если агент не сохранил результат follow-up запуска.
    """

    messages = await _run_with_progress(
        label="Phase 3: chat-style dialog follow-up",
        runs_dir=runs_dir,
        awaitable=agent.ainvoke(
            {
                "messages": _build_dialog_messages(base_result, branch_result),
                "session_id": "e2e-branch-dialog-session",
                "user_id": "manual-tester",
            },
            config={"recursion_limit": 30},
        ),
    )
    print("\nDialog final message\n====================")
    print(_last_ai_message_text(messages))

    result = _print_run_result(agent, agent.last_run_id, "Dialog run result API")
    if result is None:
        raise RuntimeError("Dialog run finished but RunResult is unavailable")
    return result


async def main() -> None:
    """Выполняет полный сценарий dream vision без UI.

    Args:
        Отсутствуют.

    Returns:
        None. Результаты печатаются в консоль, а runs/artifacts сохраняются в
        ``examples/runs``.
    """

    runs_dir = PROJECT_ROOT / "examples" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent()

    print("Starting e2e branch/dialog scenario.")
    print(f"Timeout per phase: {RUN_TIMEOUT_SECONDS} seconds.")
    try:
        base_result = await run_initial_research(agent, runs_dir)
        branch_result = await run_branch_research(agent, runs_dir, base_result)
        dialog_result = await run_dialog_follow_up(
            agent,
            runs_dir,
            base_result,
            branch_result,
        )
    except TimeoutError:
        print("\nScenario failed: timeout.")
        print("Check the last [progress] or [llm] line above.")
        return
    except Exception as exc:
        print("\nScenario failed with error\n==========================")
        print(str(exc))
        return

    print("\nScenario completed\n==================")
    print(f"base_run_id: {base_result.run.run_id}")
    print(f"branch_run_id: {branch_result.run.run_id}")
    print(f"dialog_run_id: {dialog_result.run.run_id}")
    print(f"runs_dir: {runs_dir}")


if __name__ == "__main__":
    asyncio.run(main())

"""Запуск нескольких демонстрационных исследований с подробной инспекцией.

Содержит:
- ExampleQuery: описание одного демонстрационного запроса.
- parse_args: чтение CLI-параметров.
- build_agent: сборка ResearchAgent с локальными CSV-инструментами.
- run_examples: последовательный запуск демонстрационных запросов.
- run_single_example: запуск одного запроса и печать отчета.
- log_progress: печать progress-сообщения с немедленным flush.
- render_run_report: сбор человекочитаемого отчета по ResearchRun.
- render_run_header: форматирование шапки запуска.
- render_plan_timeline: вывод изменений плана по lineage snapshots.
- render_node_timeline: вывод заключений каждого lineage node.
- render_tool_calls: вывод вызовов инструментов, аргументов и preview результатов.
- render_artifact_summary: вывод краткого списка artifacts запуска.
- render_final_report: вывод финального ответа агента.
- load_snapshot_safe: безопасная загрузка snapshot узла.
- extract_plan: извлечение плана из snapshot.
- plan_signature: стабильная сигнатура плана для поиска изменений.
- render_plan_tasks: форматирование задач плана.
- render_node_outcome: форматирование результата конкретного node.
- render_tool_trace: форматирование одного tool trace artifact.
- read_artifact_preview: чтение preview текстового artifact.
- find_run_directory: поиск директории ResearchRun по путям artifacts.
- clip: ограничение длинного текста.
- json_pretty: безопасная JSON-сериализация.
- append_section: добавление секции в markdown-отчет.
- main: точка входа.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from examples.fake_spark_tools import build_fake_spark_tools
from planner_agent import ResearchAgent
from planner_agent.schemas.lineage import StateNode
from planner_agent.services.run_inspection_service import RunResult
from sandbox import ClientPythonSandbox


PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
DATA_DIR = EXAMPLES_DIR / "data"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "observability_examples"
DEFAULT_OUTPUT_PATH = DEFAULT_RUNS_DIR / "latest_observability_report.md"
DEFAULT_RECURSION_LIMIT = 90
PREVIEW_CHARS = 2_000


@dataclass(frozen=True)
class ExampleQuery:
    """Описывает один демонстрационный запуск агента.

    Args:
        title: Короткое название сценария для консольного отчета.
        query: Пользовательский запрос, который будет передан ResearchAgent.

    Returns:
        Не возвращает значение; dataclass используется как контейнер настроек.
    """

    title: str
    query: str


EXAMPLE_QUERIES: tuple[ExampleQuery, ...] = (
    # В тексты намеренно добавлены подсказки по ожидаемому уровню доказательности,
    # чтобы модель чаще использовала read_table и execute_python_code.
    ExampleQuery(
        title="Образовательные сработки",
        query=(
            "Сколько сработок связано с образовательными услугами? "
            "Используй доступные таблицы, покажи точные фильтры и верни число, "
            "долю от всех сработок и 3-5 примеров."
        ),
    ),
    ExampleQuery(
        title="Типичность транзакции клиента",
        query=(
            "Насколько типична транзакция `ae107b8e-4788-4073-9bb4-4f209a6e02aa` "
            "для клиента `epk_id = 2099007770421986000001`? "
            "Сравни сумму, назначение, получателя, канал и устройство с историей клиента."
        ),
    ),
    ExampleQuery(
        title="Краткая история клиента",
        query=(
            "Построй краткую историю клиента `epk_id = 2099007770421986000001`: "
            "ключевые операции, сработки, образовательные платежи, повторяющиеся "
            "получатели и краткий вывод по поведению."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    """Читает CLI-параметры демонстрационного запуска.

    Args:
        Отсутствуют. Параметры читаются из командной строки.

    Returns:
        Namespace с настройками модели, директорий и вывода.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Запускает три демонстрационных ResearchAgent-сценария и печатает "
            "подробную инспекцию плана, узлов, tools и artifacts."
        )
    )
    parser.add_argument(
        "--runs-dir",
        default=str(DEFAULT_RUNS_DIR),
        help="Директория для ResearchRun artifacts и lineage.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Markdown-файл, куда будет продублирован полный отчет.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=DEFAULT_RECURSION_LIMIT,
        help="LangGraph recursion_limit для каждого запуска.",
    )
    parser.add_argument(
        "--no-stream-console",
        action="store_true",
        help="Отключить потоковый вывод LangGraph-узлов во время запуска.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=len(EXAMPLE_QUERIES),
        help="Сколько первых примеров запустить.",
    )
    return parser.parse_args()


def build_agent(args: argparse.Namespace) -> ResearchAgent:
    """Собирает ResearchAgent для демонстрационных запросов.

    Args:
        args: CLI-настройки с моделью, директориями и режимом вывода.

    Returns:
        ResearchAgent с локальным `read_table`, sandbox и директориями проекта.
    """

    log_progress("Импортирую LLM из model.py: from model import model as llm")
    from model import model as llm

    log_progress("Собираю sandbox и локальный read_table tool")
    sandbox = ClientPythonSandbox(
        allowed_libraries={},
        working_directory=str(PROJECT_ROOT),
    )
    tools = build_fake_spark_tools(delay_seconds=0.0, data_dir=DATA_DIR)
    return ResearchAgent(
        model=llm,
        sandbox=sandbox,
        tools=tools,
        workspace_root=str(PROJECT_ROOT),
        sources_dir=str(DATA_DIR),
        contexts_dir=str(PROJECT_ROOT / "skills"),
        skills_dir=str(PROJECT_ROOT / "skills"),
        memory_dir=str(PROJECT_ROOT / "memory"),
        runs_dir=str(Path(args.runs_dir)),
        stream_console=not bool(args.no_stream_console),
    )


async def run_examples(agent: ResearchAgent, args: argparse.Namespace) -> str:
    """Запускает несколько примеров и собирает общий markdown-отчет.

    Args:
        agent: ResearchAgent для последовательного выполнения запросов.
        args: CLI-настройки запуска.

    Returns:
        Полный markdown-отчет по всем выполненным примерам.
    """

    reports: list[str] = []
    selected_queries = EXAMPLE_QUERIES[: max(0, args.max_examples)]
    for index, example in enumerate(selected_queries, start=1):
        report = await run_single_example(
            agent=agent,
            example=example,
            index=index,
            recursion_limit=args.recursion_limit,
        )
        reports.append(report)
    return "\n\n".join(reports)


async def run_single_example(
        *,
        agent: ResearchAgent,
        example: ExampleQuery,
        index: int,
        recursion_limit: int,
) -> str:
    """Запускает один запрос и печатает отчет по сохраненному run.

    Args:
        agent: ResearchAgent для выполнения запроса.
        example: Описание демонстрационного запроса.
        index: Порядковый номер примера в общем запуске.
        recursion_limit: Лимит шагов LangGraph.

    Returns:
        Markdown-отчет по одному ResearchRun.
    """

    print(f"\n\n{'=' * 100}", flush=True)
    print(f"ПРИМЕР {index}: {example.title}", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(example.query, flush=True)
    log_progress("Передаю запрос в ResearchAgent. Сейчас должны начаться вызовы LLM.")
    await agent.ainvoke(
        {
            "user_query": example.query,
            "session_id": "observability-examples",
            "user_id": "local-demo",
        },
        config={"recursion_limit": recursion_limit},
    )
    log_progress("ResearchAgent завершил run. Читаю lineage, snapshots и artifacts.")
    result = agent.get_run_result(include_nodes=True, include_artifacts=True)
    if result is None:
        raise RuntimeError("ResearchAgent не вернул RunResult после запуска.")

    report = render_run_report(result)
    print(report, flush=True)
    return report


def log_progress(message: str) -> None:
    """Печатает progress-сообщение с timestamp и немедленным flush.

    Args:
        message: Текст события, который нужно показать пользователю.

    Returns:
        None.
    """

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def render_run_report(result: RunResult) -> str:
    """Собирает полный человекочитаемый отчет по одному ResearchRun.

    Args:
        result: Полный read-only результат запуска.

    Returns:
        Markdown-текст с планом, узлами, инструментами, artifacts и финальным ответом.
    """

    lines: list[str] = []
    append_section(lines, "Run", render_run_header(result))
    append_section(lines, "Как обновлялся план", render_plan_timeline(result))
    append_section(lines, "Узлы и заключения", render_node_timeline(result))
    append_section(lines, "Вызовы инструментов", render_tool_calls(result))
    append_section(lines, "Artifacts", render_artifact_summary(result))
    append_section(lines, "Финальный ответ", render_final_report(result))
    return "\n".join(lines)


def render_run_header(result: RunResult) -> str:
    """Форматирует шапку запуска.

    Args:
        result: RunResult текущего запуска.

    Returns:
        Строка с идентификаторами, датами и количеством объектов.
    """

    run = result.run
    return "\n".join(
        [
            f"- run_id: `{run.run_id}`",
            f"- status: `{run.status}`",
            f"- query: {run.initial_user_query}",
            f"- nodes: {len(result.nodes)}",
            f"- artifacts: {len(result.artifacts)}",
        ]
    )


def render_plan_timeline(result: RunResult) -> str:
    """Показывает изменения плана по snapshot lineage nodes.

    Args:
        result: RunResult с nodes и final_state.

    Returns:
        Markdown-блок с версиями плана и статусами задач.
    """

    chunks: list[str] = []
    last_signature = ""
    for node in result.nodes:
        snapshot = load_snapshot_safe(result, node)
        plan = extract_plan(snapshot)
        if not plan:
            continue
        signature = plan_signature(plan)
        if signature == last_signature:
            continue
        last_signature = signature
        chunks.append(
            "\n".join(
                [
                    f"### {node.node_type} / {node.title}",
                    f"- node_id: `{node.node_id}`",
                    render_plan_tasks(plan),
                ]
            )
        )
    return "\n\n".join(chunks) if chunks else "План в snapshots не найден."


def render_node_timeline(result: RunResult) -> str:
    """Показывает summary и результат каждого lineage node.

    Args:
        result: RunResult с nodes и snapshots.

    Returns:
        Markdown-блок с последовательностью узлов.
    """

    blocks: list[str] = []
    for index, node in enumerate(result.nodes, start=1):
        snapshot = load_snapshot_safe(result, node)
        blocks.append(
            "\n".join(
                [
                    f"{index}. `{node.node_type}` - {node.title}",
                    f"   - status: `{node.status}`",
                    f"   - node_id: `{node.node_id}`",
                    f"   - summary: {clip(node.summary, 500)}",
                    render_node_outcome(snapshot),
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "Lineage nodes не найдены."


def render_tool_calls(result: RunResult) -> str:
    """Показывает вызовы tools по tool trace artifacts.

    Args:
        result: RunResult с artifacts запуска.

    Returns:
        Markdown-блок с именем tool, аргументами и preview результата.
    """

    traces = [
        artifact
        for artifact in result.artifacts
        if artifact.kind == "tool_trace"
        or artifact.metadata.get("artifact_role") in {
            "tool_call_trace",
            "tool_calls_trace",
        }
    ]
    if not traces:
        return "Tool trace artifacts не найдены."

    blocks = [render_tool_trace(index, artifact) for index, artifact in enumerate(traces, start=1)]
    return "\n\n".join(blocks)


def render_artifact_summary(result: RunResult) -> str:
    """Форматирует компактный список artifacts запуска.

    Args:
        result: RunResult с artifacts.

    Returns:
        Markdown-список artifacts.
    """

    if not result.artifacts:
        return "Artifacts не найдены."
    rows: list[str] = []
    for artifact in result.artifacts:
        role = artifact.metadata.get("artifact_role") or artifact.metadata.get("node_type") or "-"
        rows.append(
            "- "
            f"`{artifact.artifact_id}` | kind=`{artifact.kind}` | role=`{role}` | "
            f"summary={clip(artifact.summary, 220)}"
        )
    return "\n".join(rows)


def render_final_report(result: RunResult) -> str:
    """Возвращает финальный markdown-ответ агента.

    Args:
        result: RunResult текущего запуска.

    Returns:
        Финальный отчет или диагностическое сообщение.
    """

    return result.final_report or "Финальный отчет не найден."


def load_snapshot_safe(result: RunResult, node: StateNode) -> dict[str, Any] | None:
    """Берет snapshot node из final_state или с диска через state_ref.

    Args:
        result: RunResult, которому принадлежит node.
        node: Lineage node.

    Returns:
        Snapshot как словарь или ``None``, если файл недоступен.
    """

    if result.final_state and node.node_id == result.summary.final_report_node_id:
        return result.final_state
    if not node.state_ref:
        return None
    run_dir = find_run_directory(result)
    if run_dir is None:
        return None
    snapshot_path = run_dir / node.state_ref
    if not snapshot_path.exists():
        return None
    try:
        return json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def find_run_directory(result: RunResult) -> Path | None:
    """Находит директорию `runs/{run_id}` по URI artifacts.

    Args:
        result: RunResult с идентификатором запуска и списком artifacts.

    Returns:
        Путь к директории конкретного ResearchRun или ``None``, если artifacts
        отсутствуют либо путь не содержит run_id.
    """

    run_id = result.run.run_id
    for artifact in result.artifacts:
        path = Path(artifact.uri).resolve()
        for candidate in [path, *path.parents]:
            if candidate.name == run_id:
                return candidate
    return None


def extract_plan(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Извлекает план задач из snapshot.

    Args:
        snapshot: Snapshot lineage node или ``None``.

    Returns:
        Словарь задач плана, если он найден.
    """

    if not isinstance(snapshot, dict):
        return {}
    plan = snapshot.get("plan")
    return plan if isinstance(plan, dict) else {}


def plan_signature(plan: dict[str, Any]) -> str:
    """Создает стабильную сигнатуру плана для сравнения версий.

    Args:
        plan: Словарь задач плана.

    Returns:
        JSON-строка с ключевыми полями задач.
    """

    compact: dict[str, Any] = {}
    for task_id, task in sorted(plan.items(), key=lambda item: str(item[0])):
        if not isinstance(task, dict):
            continue
        compact[str(task_id)] = {
            "description": task.get("description"),
            "dependencies": task.get("dependencies"),
            "status": task.get("status"),
            "retry_count": task.get("retry_count"),
            "validation_passed": task.get("validation_passed"),
            "validation_score": task.get("validation_score"),
        }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True, default=str)


def render_plan_tasks(plan: dict[str, Any]) -> str:
    """Форматирует задачи плана.

    Args:
        plan: Словарь задач из snapshot.

    Returns:
        Markdown-список задач со статусами и зависимостями.
    """

    rows: list[str] = []
    for task_id, task in sorted(plan.items(), key=lambda item: str(item[0])):
        if not isinstance(task, dict):
            continue
        deps = task.get("dependencies") or []
        tools = task.get("suggested_tools") or []
        validation = task.get("validation_passed")
        rows.append(
            "- "
            f"task `{task_id}` [{task.get('status', 'unknown')}] "
            f"deps={deps} tools={tools} validation={validation}\n"
            f"  {clip(str(task.get('description') or ''), 300)}"
        )
    return "\n".join(rows)


def render_node_outcome(snapshot: dict[str, Any] | None) -> str:
    """Форматирует ключевой результат node по snapshot.

    Args:
        snapshot: Snapshot lineage node или ``None``.

    Returns:
        Строка с outcome для отображения под node.
    """

    if not isinstance(snapshot, dict):
        return "   - outcome: snapshot недоступен"

    task = snapshot.get("task")
    if isinstance(task, dict):
        result = (
            task.get("full_result")
            or task.get("result_preview")
            or task.get("validation_reason")
            or task.get("error_log")
            or ""
        )
        return (
            f"   - task_status: `{task.get('status')}`\n"
            f"   - task_result: {clip(str(result), 900)}"
        )

    if snapshot.get("final_report"):
        return f"   - final_report: {clip(str(snapshot.get('final_report')), 900)}"

    feedback = snapshot.get("feedback_context")
    if feedback:
        return f"   - feedback: {clip(json_pretty(feedback), 900)}"

    plan = extract_plan(snapshot)
    if plan:
        return f"   - plan_tasks: {len(plan)}"

    return "   - outcome: нет отдельного результата в snapshot"


def render_tool_trace(index: int, artifact: Any) -> str:
    """Форматирует один artifact с trace вызова tool.

    Args:
        index: Порядковый номер trace в отчете.
        artifact: Artifact из RunResult.

    Returns:
        Markdown-блок с параметрами и preview результата.
    """

    metadata = artifact.metadata or {}
    tool_name = metadata.get("tool_name") or metadata.get("stage") or "unknown"
    args_preview = metadata.get("args_preview")
    captured_refs = metadata.get("captured_artifact_refs") or []
    content_preview = read_artifact_preview(artifact.uri, max_chars=PREVIEW_CHARS)
    return "\n".join(
        [
            f"{index}. `{tool_name}`",
            f"   - artifact_id: `{artifact.artifact_id}`",
            f"   - role: `{metadata.get('artifact_role', '-')}`",
            f"   - captured: `{metadata.get('captured', False)}`",
            f"   - captured_artifact_refs: `{captured_refs}`",
            f"   - args: `{clip(str(args_preview or '-'), 900)}`",
            f"   - returned: {clip(content_preview or artifact.summary or '-', 1_500)}",
        ]
    )


def read_artifact_preview(uri: str, *, max_chars: int) -> str:
    """Читает preview текстового artifact.

    Args:
        uri: Локальный путь artifact.
        max_chars: Максимальный размер возвращаемого текста.

    Returns:
        Текстовое preview или пустую строку, если файл недоступен.
    """

    path = Path(uri)
    if not path.exists() or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:max_chars]
    except UnicodeDecodeError:
        return ""


def clip(value: str, max_chars: int) -> str:
    """Обрезает длинный текст для консольного отчета.

    Args:
        value: Исходный текст.
        max_chars: Максимальная длина результата.

    Returns:
        Обрезанный текст с маркером, если исходное значение было длиннее лимита.
    """

    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated]"


def json_pretty(value: Any) -> str:
    """Безопасно сериализует значение в читаемый JSON.

    Args:
        value: Произвольное JSON-подобное значение.

    Returns:
        Отформатированная JSON-строка или `str(value)` при ошибке.
    """

    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return str(value)


def append_section(lines: list[str], title: str, body: str) -> None:
    """Добавляет markdown-секцию в общий список строк.

    Args:
        lines: Список строк отчета, который будет изменен на месте.
        title: Заголовок секции.
        body: Тело секции.

    Returns:
        None.
    """

    lines.append(f"## {title}")
    lines.append("")
    lines.append(body)
    lines.append("")


def main() -> None:
    """Запускает демонстрационные сценарии и сохраняет общий markdown-отчет.

    Args:
        Отсутствуют. Настройки берутся из CLI и переменных окружения.

    Returns:
        None.
    """

    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_progress("Старт main_observability_examples.py")
    log_progress(f"Runs dir: {Path(args.runs_dir).resolve()}")
    log_progress(f"Output: {output_path.resolve()}")
    log_progress(f"Stream console: {not bool(args.no_stream_console)}")
    agent = build_agent(args)
    log_progress("Агент собран. Запускаю примеры.")
    report = asyncio.run(run_examples(agent, args))
    output_path.write_text(report, encoding="utf-8")
    log_progress(f"Отчет сохранен: {output_path.resolve()}")


if __name__ == "__main__":
    main()

"""Запуск research-agent на DeepSeek Flash для одного antifraud-кейса.

Содержит:
- _load_csv_table: загрузка CSV-таблицы тестового antifraud-датасета.
- _load_integrity_report: чтение контрольного отчета целостности датасета.
- _build_initial_globals: подготовка стартовых переменных песочницы.
- _build_sandbox: создание ClientPythonSandbox с pandas и таблицами кейса.
- _build_user_query: сборка полного запроса для агента из TASK_FOR_AGENT.md.
- _format_message_content: преобразование содержимого LangChain message в текст.
- _read_jsonl_rows: безопасное чтение JSONL-файлов lineage и artifacts.
- _find_latest_run_dir: поиск последнего каталога запуска агента.
- _print_progress_event: печать одного события lineage.
- _print_artifact_progress: печать одного artifact.
- _monitor_run_progress: фоновый мониторинг прогресса запуска.
- _validate_dataset_files: проверка наличия файлов датасета.
- _validate_dataset_expectations: проверка контрольных ожиданий кейса.
- _validate_spark_tool_reads_dataset: проверка чтения нужного data_dir через spark_query_table.
- run_offline_checks: запуск всех проверок без обращения к LLM.
- build_agent: сборка ResearchAgent с DeepSeek Flash и локальными tools.
- _invoke_agent_and_stop_monitor: запуск агента и остановка мониторинга.
- _invoke_agent_with_timeout: bounded-запуск агента с timeout.
- _classify_runtime_error: классификация сетевых/model/runtime ошибок.
- _print_run_result_api: печать результата через публичный API ResearchAgent.
- _print_post_run_analysis: краткий разбор шагов, artifacts и финального статуса.
- main: точка входа для офлайн-проверок и одного реального запуска.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.fake_spark_tools import build_fake_spark_tools  # noqa: E402
from planner_agent import ResearchAgent  # noqa: E402
from sandbox import ClientPythonSandbox  # noqa: E402


DATASET_DIR = PROJECT_ROOT / "test_one_dataset" / "one_client_antifraud_dataset"
RUNS_DIR = PROJECT_ROOT / "runs" / "deepseek_event_test"
TASK_FILE = DATASET_DIR / "TASK_FOR_AGENT.md"
INTEGRITY_REPORT_FILE = DATASET_DIR / "integrity_report.json"

HITS_FILE = "cspfs_repo_features3.hits_extra_info_129372427_view.csv"
CARDS_FILE = "csp_afpc_sss_inc.cards_event.csv"
UKO_FILE = "csp_afpc_sss_inc.uko_event.csv"
HISTORY_FILE = "csp_repo_features.history_automarking_big_148078_155487.csv"
TIMELINE_FILE = "demo_client_timeline.csv"
REQUIRED_DATA_FILES = [HITS_FILE, CARDS_FILE, UKO_FILE, HISTORY_FILE, TIMELINE_FILE]

KEY_EVENT_ID = "f9246b19-3bf5-4883-8076-d1d4356a6cf8"
CLIENT_USER_ID = "7770421986"
CLIENT_EPK_ID = "2099007770421986000001"
DAY_N = "2026-03-09"
DAY_N_COMPACT = "20260309"
REAL_RUN_TIMEOUT_SECONDS = 600
GRAPH_RECURSION_LIMIT = 60


def _load_csv_table(file_name: str) -> pd.DataFrame:
    """Загружает одну CSV-таблицу antifraud-датасета.

    Args:
        file_name: Имя CSV-файла внутри ``DATASET_DIR``.

    Returns:
        DataFrame с содержимым указанного CSV-файла.
    """

    return pd.read_csv(DATASET_DIR / file_name, low_memory=False)


def _load_integrity_report() -> dict[str, Any]:
    """Читает JSON-отчет целостности тестового датасета.

    Args:
        Отсутствуют.

    Returns:
        Словарь с контрольными метриками датасета.
    """

    if not INTEGRITY_REPORT_FILE.exists():
        return {}
    return json.loads(INTEGRITY_REPORT_FILE.read_text(encoding="utf-8"))


def _build_initial_globals() -> dict[str, Any]:
    """Готовит стартовые переменные песочницы для агента.

    Args:
        Отсутствуют.

    Returns:
        Словарь глобальных переменных: таблицы кейса, идентификаторы клиента и
        контрольный отчет. ``df_current`` указывает на таблицу сработок.
    """

    df_hits = _load_csv_table(HITS_FILE)
    return {
        "df_current": df_hits,
        "df_hits": df_hits,
        "df_cards": _load_csv_table(CARDS_FILE),
        "df_uko": _load_csv_table(UKO_FILE),
        "df_history_automarking": _load_csv_table(HISTORY_FILE),
        "df_timeline": _load_csv_table(TIMELINE_FILE),
        "integrity_report": _load_integrity_report(),
        "key_event_id": KEY_EVENT_ID,
        "client_user_id": CLIENT_USER_ID,
        "client_epk_id": CLIENT_EPK_ID,
        "day_n": DAY_N,
    }


def _build_sandbox() -> ClientPythonSandbox:
    """Создает песочницу для Python-анализа внутри worker-узлов.

    Args:
        Отсутствуют.

    Returns:
        ClientPythonSandbox с pandas и предзагруженными таблицами кейса.
    """

    sandbox = ClientPythonSandbox(
        allowed_libraries={"pd": pd},
        initial_globals=_build_initial_globals(),
    )
    sandbox.last_dataframe_variable = "df_current"
    return sandbox


def _build_user_query() -> str:
    """Собирает полный пользовательский запрос для запуска агента.

    Args:
        Отсутствуют.

    Returns:
        Текст запроса из ``TASK_FOR_AGENT.md`` с дополнительными правилами
        устойчивости к нарушенной консистентности данных.
    """

    base_query = TASK_FILE.read_text(encoding="utf-8")
    additions = """

Дополнительные правила для этого запуска:

- Нарушения консистентности, пустые поля и несовпадения форматов в данных не
  являются ошибкой запуска. Если встретишь такие случаи, зафиксируй их как
  ограничение анализа и продолжай.
- Обязательно проверь ключевую сработку, другие сработки клиента, транзакции в
  дни сработок, похожие прошлые транзакции по merchant/типу операции/примерной
  сумме и историю авторазметки.
- Если текущее поведение похоже на обычный паттерн клиента или часть риска
  объясняется его историей, предложи варианты улучшения антифрод-системы,
  изменения правил, исключений или дополнительных признаков.
- Не считай отсутствие части данных критической ошибкой. В итоговом отчете
  явно раздели подтвержденные факты, гипотезы и ограничения.
"""
    return f"{base_query.strip()}\n{additions.strip()}"


def _format_message_content(content: object) -> str:
    """Преобразует содержимое LangChain message в строку для печати.

    Args:
        content: Содержимое сообщения LangChain.

    Returns:
        Строковое представление ответа.
    """

    if isinstance(content, str):
        return content
    return str(content)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Читает JSONL-файл с пропуском неполных строк.

    Args:
        path: Путь к JSONL-файлу.

    Returns:
        Список валидных JSON-объектов из файла.
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


def _find_latest_run_dir(runs_dir: Path, ignored_run_ids: set[str]) -> Path | None:
    """Находит самый свежий каталог запуска, созданный после старта скрипта.

    Args:
        runs_dir: Каталог ResearchRun.
        ignored_run_ids: Идентификаторы запусков, которые уже существовали.

    Returns:
        Путь к новому каталогу запуска или ``None``.
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


def _print_progress_event(node: dict[str, Any]) -> None:
    """Печатает краткую строку по одному lineage-событию.

    Args:
        node: JSON-представление StateNode.

    Returns:
        None.
    """

    node_type = node.get("node_type", "unknown")
    status = node.get("status", "unknown")
    title = str(node.get("title", ""))
    summary = str(node.get("summary", "")).replace("\n", " ")[:240]
    created_at = node.get("created_at", "")
    print(
        f"[progress] {created_at} | {node_type} | {status} | {title} | {summary}",
        flush=True,
    )


def _print_artifact_progress(artifact: dict[str, Any]) -> None:
    """Печатает краткую строку по одному artifact.

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
    poll_interval_seconds: float = 2.0,
) -> None:
    """Фоном печатает новые lineage nodes и artifacts текущего запуска.

    Args:
        runs_dir: Каталог, куда агент пишет результаты запуска.
        ignored_run_ids: Запуски, существовавшие до текущего старта.
        stop_event: Событие остановки мониторинга.
        poll_interval_seconds: Пауза между проверками файлов.

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
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval_seconds)
        except TimeoutError:
            continue


def _validate_dataset_files() -> list[str]:
    """Проверяет наличие обязательных файлов тестового датасета.

    Args:
        Отсутствуют.

    Returns:
        Список диагностических сообщений.

    Raises:
        FileNotFoundError: Если отсутствует каталог или обязательный файл.
    """

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    if not TASK_FILE.exists():
        raise FileNotFoundError(f"Task file not found: {TASK_FILE}")
    missing_files = [
        file_name
        for file_name in REQUIRED_DATA_FILES
        if not (DATASET_DIR / file_name).exists()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing dataset files: {missing_files}")
    return [f"Dataset files OK: {DATASET_DIR}"]


def _validate_dataset_expectations() -> list[str]:
    """Проверяет контрольные ожидания antifraud-кейса без LLM.

    Args:
        Отсутствуют.

    Returns:
        Список сообщений об успешных проверках.

    Raises:
        AssertionError: Если датасет не соответствует контрольным ожиданиям.
    """

    hits = _load_csv_table(HITS_FILE)
    cards = _load_csv_table(CARDS_FILE)
    uko = _load_csv_table(UKO_FILE)
    history = _load_csv_table(HISTORY_FILE)

    messages: list[str] = []
    if len(hits) != 7:
        raise AssertionError(f"Expected 7 hits, got {len(hits)}")
    messages.append("hits_total OK: 7")

    key_hit = hits[hits["event_id"].astype(str) == KEY_EVENT_ID]
    if len(key_hit) != 1:
        raise AssertionError(f"Expected one key hit {KEY_EVENT_ID}, got {len(key_hit)}")
    messages.append(f"key_event_id OK: {KEY_EVENT_ID}")

    day_hits = hits[hits["event_time"].astype(str).str.startswith(DAY_N)]
    if len(day_hits) != 4:
        raise AssertionError(f"Expected 4 hits at {DAY_N}, got {len(day_hits)}")
    messages.append(f"day_n_hits OK: {len(day_hits)}")

    cards_ids = set(cards["event_id"].astype(str))
    uko_ids = set(uko["event_id"].astype(str))
    if KEY_EVENT_ID not in cards_ids:
        raise AssertionError("Key event is not present in cards_event")
    if KEY_EVENT_ID in uko_ids:
        raise AssertionError("Key event must not be duplicated in uko_event")
    messages.append("key_event_route OK: cards_event only")

    hit_ids = set(hits["event_id"].astype(str))
    missing_target = sorted(hit_ids - (cards_ids | uko_ids))
    duplicated_target = sorted(hit_ids & cards_ids & uko_ids)
    exactly_one_count = sum(
        (event_id in cards_ids) ^ (event_id in uko_ids)
        for event_id in hit_ids
    )
    if missing_target or duplicated_target or exactly_one_count != len(hit_ids):
        raise AssertionError(
            "Each hit must exist in exactly one target table. "
            f"missing={missing_target}; duplicated={duplicated_target}; exactly_one={exactly_one_count}"
        )
    messages.append(
        "hit_target_integrity OK: every hit exists in exactly one operational table"
    )

    table_user_ids = {
        "hits": sorted(hits["user_id"].astype(str).unique().tolist()),
        "cards": sorted(cards["user_id"].astype(str).unique().tolist()),
        "uko": sorted(uko["user_id"].astype(str).unique().tolist()),
        "history": sorted(history["user_id"].astype(str).unique().tolist()),
    }
    unexpected_users = {
        table_name: user_ids
        for table_name, user_ids in table_user_ids.items()
        if user_ids != [CLIENT_USER_ID]
    }
    if unexpected_users:
        raise AssertionError(f"Dataset must contain one client only: {unexpected_users}")
    messages.append(f"one_client_check OK: {CLIENT_USER_ID}")

    historical_hits = hits[hits["event_dt"].astype(str) < DAY_N_COMPACT]
    if historical_hits.empty:
        raise AssertionError("Expected historical hits before day N")
    messages.append(f"historical_hits OK: {len(historical_hits)}")
    return messages


async def _validate_spark_tool_reads_dataset() -> list[str]:
    """Проверяет, что spark_query_table читает именно тестовый каталог.

    Args:
        Отсутствуют.

    Returns:
        Список сообщений об успешной проверке.

    Raises:
        AssertionError: Если tool не вернул ключевую сработку.
    """

    tool = build_fake_spark_tools(delay_seconds=0.0, data_dir=DATASET_DIR)[0]
    result = await tool.ainvoke(
        {
            "table_name": "hits",
            "select_columns": ["event_id", "event_time", "user_id", "epk_id"],
            "filters": [{"column": "event_id", "operator": "eq", "value": KEY_EVENT_ID}],
            "max_rows": 5,
        }
    )
    if not isinstance(result, pd.DataFrame):
        raise AssertionError(f"spark_query_table returned non-DataFrame: {result}")
    if len(result) != 1:
        raise AssertionError(f"spark_query_table expected one key row, got {len(result)}")
    row = result.iloc[0]
    if str(row["user_id"]) != CLIENT_USER_ID or str(row["epk_id"]) != CLIENT_EPK_ID:
        raise AssertionError(f"spark_query_table returned wrong client row: {row.to_dict()}")
    source_file = result.attrs.get("spark_source_file", "")
    if source_file != HITS_FILE:
        raise AssertionError(f"spark_query_table returned wrong source file: {source_file}")
    return ["spark_query_table OK: reads test_one_dataset/one_client_antifraud_dataset"]


async def run_offline_checks() -> None:
    """Запускает все проверки, которые не обращаются к реальной LLM.

    Args:
        Отсутствуют.

    Returns:
        None. Успешные проверки печатаются в stdout.
    """

    print("Offline checks\n==============")
    check_messages = [
        *_validate_dataset_files(),
        *_validate_dataset_expectations(),
        *(await _validate_spark_tool_reads_dataset()),
    ]
    for message in check_messages:
        print(f"[offline-ok] {message}")


def build_agent() -> ResearchAgent:
    """Собирает ResearchAgent для реального DeepSeek/OpenRouter запуска.

    Args:
        Отсутствуют.

    Returns:
        Экземпляр ResearchAgent с локальным sandbox, fake Spark tools и
        каталогом запусков ``runs/deepseek_event_test``.
    """

    from model import model as deepseek_model

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    sandbox = _build_sandbox()
    spark_tools = build_fake_spark_tools(
        delay_seconds=0.2,
        data_dir=DATASET_DIR,
    )

    return ResearchAgent(
        model=deepseek_model,
        sandbox=sandbox,
        tools=spark_tools,
        code_generator_tool_names=set(),
        enable_workspace_tools=True,
        workspace_root=str(PROJECT_ROOT),
        sources_dir=str(DATASET_DIR),
        contexts_dir=str(PROJECT_ROOT / "skills"),
        skills_dir=str(PROJECT_ROOT / "skills"),
        memory_dir=str(PROJECT_ROOT / "memory"),
        runs_dir=str(RUNS_DIR),
        stream_console=False,
    )


async def _invoke_agent_and_stop_monitor(
    *,
    agent: ResearchAgent,
    user_query: str,
    stop_event: asyncio.Event,
    monitor_task: asyncio.Task[None],
) -> list[Any]:
    """Запускает агента и гарантированно останавливает монитор прогресса.

    Args:
        agent: Экземпляр ResearchAgent.
        user_query: Полный запрос для анализа.
        stop_event: Событие остановки мониторинга.
        monitor_task: Фоновая задача мониторинга.

    Returns:
        Список LangChain messages из финального состояния агента.
    """

    try:
        return await agent.ainvoke(
            {
                "user_query": user_query,
                "session_id": "deepseek-antifraud-event-session",
                "user_id": "manual-tester",
            },
            config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        )
    finally:
        stop_event.set()
        await monitor_task


async def _invoke_agent_with_timeout(agent: ResearchAgent, user_query: str) -> list[Any]:
    """Запускает агента с ограничением времени и мониторингом progress.

    Args:
        agent: Экземпляр ResearchAgent.
        user_query: Полный запрос для анализа.

    Returns:
        Список LangChain messages из финального состояния агента.

    Raises:
        TimeoutError: Если запуск не завершился за ``REAL_RUN_TIMEOUT_SECONDS``.
    """

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ignored_run_ids = {path.name for path in RUNS_DIR.iterdir() if path.is_dir()}
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(
        _monitor_run_progress(
            runs_dir=RUNS_DIR,
            ignored_run_ids=ignored_run_ids,
            stop_event=stop_event,
        )
    )

    return await asyncio.wait_for(
        _invoke_agent_and_stop_monitor(
            agent=agent,
            user_query=user_query,
            stop_event=stop_event,
            monitor_task=monitor_task,
        ),
        timeout=REAL_RUN_TIMEOUT_SECONDS,
    )


def _classify_runtime_error(exc: Exception) -> str:
    """Классифицирует ошибку запуска в человекочитаемый диагностический статус.

    Args:
        exc: Исключение, полученное при запуске агента.

    Returns:
        Короткое описание вероятного класса ошибки.
    """

    text = f"{exc.__class__.__name__}: {exc}".lower()
    if "timeout" in text:
        return "timeout"
    if any(marker in text for marker in ("credit", "quota", "insufficient", "402")):
        return "credits_or_quota"
    if any(marker in text for marker in ("rate limit", "429", "too many requests")):
        return "rate_limit"
    if any(marker in text for marker in ("401", "403", "api key", "auth", "unauthorized")):
        return "auth_or_key"
    if any(marker in text for marker in ("openrouter", "connection", "dns", "ssl", "network")):
        return "provider_or_network"
    if any(marker in text for marker in ("json", "structured output", "parse")):
        return "model_json_parse"
    if "tool" in text:
        return "tool_runtime"
    return "unknown_runtime_error"


def _print_run_result_api(agent: ResearchAgent) -> None:
    """Печатает сводку результата через публичный API агента.

    Args:
        agent: Экземпляр ResearchAgent после запуска.

    Returns:
        None.
    """

    result = agent.get_run_result()

    print("\nRun result API\n==============")
    if result is None:
        print("Run result is unavailable.")
        return

    print(f"run_id: {result.run.run_id}")
    print(f"node_count: {result.summary.node_count}")
    print(f"artifact_count: {result.summary.artifact_count}")
    print(f"final_report_node_id: {result.summary.final_report_node_id}")
    print(f"final_report_artifact_id: {result.summary.final_report_artifact_id}")
    print(f"final_report_chars: {len(result.final_report or '')}")
    print(f"messages_from_final_state: {len(result.messages)}")

    print("\nArtifacts")
    for artifact in agent.list_artifacts():
        role = artifact.metadata.get("artifact_role") or artifact.metadata.get("node_type") or ""
        tool_name = artifact.metadata.get("tool_name") or ""
        metadata_parts = [str(part) for part in (role, tool_name) if part]
        metadata_text = " | ".join(metadata_parts)
        if metadata_text:
            print(f"- {artifact.artifact_id}: {artifact.kind} | {metadata_text} | {artifact.uri}")
        else:
            print(f"- {artifact.artifact_id}: {artifact.kind} | {artifact.uri}")


def _print_post_run_analysis(agent: ResearchAgent) -> None:
    """Печатает краткий разбор шагов текущего запуска.

    Args:
        agent: Экземпляр ResearchAgent после запуска или ошибки.

    Returns:
        None.
    """

    print("\nPost-run step analysis\n======================")
    graph = agent.get_run_graph()
    if graph is None:
        latest_run_dir = _find_latest_run_dir(RUNS_DIR, ignored_run_ids=set())
        if latest_run_dir is None:
            print("No run graph or lineage directory found.")
            return
        nodes = _read_jsonl_rows(latest_run_dir / "lineage.jsonl")
        if not nodes:
            print(f"No lineage nodes found in {latest_run_dir}.")
            return
        for index, node in enumerate(nodes, start=1):
            node_type = node.get("node_type", "unknown")
            status = node.get("status", "unknown")
            title = str(node.get("title", "")).replace("\n", " ")[:120]
            summary = str(node.get("summary", "")).replace("\n", " ")[:180]
            print(f"{index}. {node_type} | {status} | {title} | {summary}")
        return

    for index, node in enumerate(graph.nodes, start=1):
        title = str(node.title or "").replace("\n", " ")[:120]
        summary = str(node.summary or "").replace("\n", " ")[:180]
        print(f"{index}. {node.node_type} | {node.status} | {title} | {summary}")


async def main() -> None:
    """Выполняет офлайн-проверки и один bounded-запуск DeepSeek Flash.

    Args:
        Отсутствуют. Если в ``sys.argv`` есть ``--offline-only``, реальный
        запуск модели пропускается.

    Returns:
        None. Все результаты печатаются в консоль, artifacts сохраняются в
        ``runs/deepseek_event_test``.
    """

    try:
        await run_offline_checks()
    except Exception as exc:
        print("\nOffline checks failed\n=====================")
        print(f"{exc.__class__.__name__}: {exc}")
        return

    if "--offline-only" in sys.argv:
        print("\nOffline-only mode: real model run skipped.")
        return

    agent = build_agent()
    user_query = _build_user_query()
    print("\nStarting DeepSeek Flash antifraud event run.")
    print("Model: deepseek/deepseek-v4-flash")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Runs dir: {RUNS_DIR}")
    print(f"Timeout: {REAL_RUN_TIMEOUT_SECONDS} seconds")

    try:
        messages = await _invoke_agent_with_timeout(agent, user_query)
    except TimeoutError as exc:
        print("\nRun failed: timeout\n===================")
        print(f"Diagnostic status: {_classify_runtime_error(exc)}")
        print(f"Timeout after {REAL_RUN_TIMEOUT_SECONDS} seconds.")
        _print_post_run_analysis(agent)
        return
    except Exception as exc:
        print("\nRun failed with controlled diagnostic\n=====================================")
        print(f"Diagnostic status: {_classify_runtime_error(exc)}")
        print(f"{exc.__class__.__name__}: {exc}")
        _print_post_run_analysis(agent)
        return

    final_message = messages[-1] if messages else None
    print("\nFinal report\n============")
    print(_format_message_content(final_message.content) if final_message else "")

    _print_run_result_api(agent)
    _print_post_run_analysis(agent)


if __name__ == "__main__":
    asyncio.run(main())

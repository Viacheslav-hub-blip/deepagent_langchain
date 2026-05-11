/**
 * React hook управления активным аналитическим run.
 *
 * Содержит:
 * - buildPayload: формирует payload запуска агента.
 * - reportTextFromResult: извлекает markdown-отчет из RunResult.
 * - initialQueryFromRun: извлекает исходный запрос из ResearchRun.
 * - useAnalyticsRun: управляет live запуском, историческим run, графом, отчетом и artifacts.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchRunArtifacts,
  fetchRunGraph,
  fetchRunResult,
  invokeRun,
  startLiveRun,
} from "../api.js";
import { buildUserGraph } from "../lib/userGraph.js";
import { isTerminalRunStatus, LIVE_POLL_INTERVAL_MS, stableMergeNodes } from "../lib/nodes.js";

/**
 * Формирует payload запуска агента из пользовательского запроса.
 *
 * @param {string} query Текст задачи пользователя.
 * @returns {object} Payload для `/runs/live` или `/runs/invoke`.
 */
function buildPayload(query) {
  return {
    user_query: query.trim(),
    session_id: "",
    user_id: null,
    filesystem_context: {},
    context_runs: [],
  };
}

/**
 * Возвращает markdown-отчет из результата run.
 *
 * @param {object | null} result Ответ `/runs/{run_id}/result`.
 * @returns {string} Текст итогового отчета.
 */
function reportTextFromResult(result) {
  return String(result?.final_report || "").trim();
}

/**
 * Возвращает исходный пользовательский запрос из ResearchRun.
 *
 * @param {object | null} run Запись ResearchRun.
 * @returns {string} Текст исходного запроса или заголовок run.
 */
function initialQueryFromRun(run) {
  return String(run?.initial_user_query || run?.title || "").trim();
}

/**
 * Управляет состоянием текущего запуска агента и восстановлением сохраненных runs.
 *
 * @returns {object} Состояние active run и методы запуска/загрузки.
 */
export function useAnalyticsRun() {
  const timerRef = useRef(null);
  const activeRunIdRef = useRef("");

  const [phase, setPhase] = useState("idle");
  const [statusText, setStatusText] = useState("");
  const [error, setError] = useState("");
  const [runId, setRunId] = useState("");
  const [run, setRun] = useState(null);
  const [rawNodes, setRawNodes] = useState([]);
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [messages, setMessages] = useState([]);
  const [reportText, setReportText] = useState("");
  const [reportLoading, setReportLoading] = useState(false);
  const [artifacts, setArtifacts] = useState([]);

  const nodes = useMemo(() => buildUserGraph(rawNodes).nodes, [rawNodes]);

  const stopPolling = useCallback(() => {
    if (timerRef.current) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const refreshArtifacts = useCallback(async (nextRunId) => {
    if (!nextRunId) {
      setArtifacts([]);
      return;
    }
    try {
      const items = await fetchRunArtifacts(nextRunId);
      setArtifacts(Array.isArray(items) ? items : []);
    } catch {
      setArtifacts([]);
    }
  }, []);

  const loadReport = useCallback(async (nextRunId) => {
    if (!nextRunId) return "";

    setReportLoading(true);
    try {
      const result = await fetchRunResult(nextRunId);
      const text = reportTextFromResult(result);
      setReportText(text);
      return text;
    } finally {
      setReportLoading(false);
    }
  }, []);

  const pollGraph = useCallback(async (nextRunId) => {
    if (!nextRunId || activeRunIdRef.current !== nextRunId) return;

    try {
      const graph = await fetchRunGraph(nextRunId);
      const incomingRawNodes = graph.nodes || [];
      const nextUserGraph = buildUserGraph(incomingRawNodes);
      const latestUserNodeId = nextUserGraph.nodes.at(-1)?.node_id || "";

      setRun(graph.run || null);
      setRawNodes((current) => stableMergeNodes(current, incomingRawNodes));
      setSelectedNodeId((current) => latestUserNodeId || current);
      refreshArtifacts(nextRunId);

      const nextStatus = graph.run?.status || "running";
      if (isTerminalRunStatus(nextStatus)) {
        stopPolling();
        const ok = nextStatus === "succeeded" || nextStatus === "completed";
        setPhase(ok ? "done" : "error");
        setStatusText(ok ? "Агент завершил работу." : `Run завершился со статусом: ${nextStatus}`);
        try {
          await loadReport(nextRunId);
          await refreshArtifacts(nextRunId);
        } catch (reportError) {
          setError(`Не удалось загрузить итоговый отчет: ${reportError.message}`);
        }
        return;
      }

      setPhase("running");
      setStatusText(`Агент работает. Шагов на графе: ${nextUserGraph.nodes.length}`);
    } catch (pollError) {
      setStatusText(`Жду граф запуска: ${pollError.message}`);
    }

    timerRef.current = window.setTimeout(() => pollGraph(nextRunId), LIVE_POLL_INTERVAL_MS);
  }, [loadReport, refreshArtifacts, stopPolling]);

  const start = useCallback(async (query) => {
    const cleanQuery = String(query || "").trim();
    if (!cleanQuery) {
      setError("Введите задачу для агента.");
      return false;
    }

    stopPolling();
    activeRunIdRef.current = "";
    setPhase("starting");
    setStatusText("Создаю live run...");
    setError("");
    setRunId("");
    setRun(null);
    setRawNodes([]);
    setSelectedNodeId("");
    setReportText("");
    setReportLoading(false);
    setArtifacts([]);
    setMessages([{ role: "user", content: cleanQuery }]);

    const payload = buildPayload(cleanQuery);

    try {
      const response = await startLiveRun(payload);
      activeRunIdRef.current = response.run_id;
      setRunId(response.run_id);
      setRun(response.run || null);
      setPhase("running");
      setStatusText("Run создан. Жду первые узлы графа...");
      pollGraph(response.run_id);
      return true;
    } catch (liveError) {
      try {
        setStatusText("Live endpoint недоступен. Запускаю совместимый режим...");
        const response = await invokeRun(payload);
        const nextRunId = response.run_id;
        const resultNodes = response.result?.nodes || [];

        activeRunIdRef.current = nextRunId;
        setRunId(nextRunId);
        setRun(response.result?.run || null);
        setRawNodes(resultNodes);
        setSelectedNodeId("");
        setPhase("done");
        setStatusText("Run завершен.");
        setReportText(reportTextFromResult(response.result));
        await refreshArtifacts(nextRunId);
        return true;
      } catch (fallbackError) {
        setPhase("error");
        setError(`Не удалось запустить анализ: ${fallbackError.message || liveError.message}`);
        setStatusText("");
        return false;
      }
    }
  }, [pollGraph, refreshArtifacts, stopPolling]);

  const loadRun = useCallback(async (nextRunId) => {
    if (!nextRunId) return false;

    stopPolling();
    activeRunIdRef.current = nextRunId;
    setPhase("loading");
    setStatusText("Загружаю сохраненный run...");
    setError("");
    setRunId(nextRunId);
    setRawNodes([]);
    setSelectedNodeId("");
    setReportText("");
    setArtifacts([]);

    try {
      const graph = await fetchRunGraph(nextRunId);
      setRun(graph.run || null);
      setRawNodes(graph.nodes || []);
      setMessages([
        {
          role: "user",
          content: initialQueryFromRun(graph.run) || "Сохраненный аналитический run",
        },
      ]);
      await Promise.all([loadReport(nextRunId), refreshArtifacts(nextRunId)]);
      const terminal = isTerminalRunStatus(graph.run?.status);
      setPhase(terminal ? "done" : "running");
      setStatusText(terminal ? "Run загружен." : "Run еще не завершен. Обновляю граф...");
      if (!terminal) {
        pollGraph(nextRunId);
      }
      return true;
    } catch (loadError) {
      setPhase("error");
      setError(`Не удалось загрузить run: ${loadError.message}`);
      setStatusText("");
      return false;
    }
  }, [loadReport, pollGraph, refreshArtifacts, stopPolling]);

  const reset = useCallback(() => {
    stopPolling();
    activeRunIdRef.current = "";
    setPhase("idle");
    setStatusText("");
    setError("");
    setRunId("");
    setRun(null);
    setRawNodes([]);
    setSelectedNodeId("");
    setMessages([]);
    setReportText("");
    setReportLoading(false);
    setArtifacts([]);
  }, [stopPolling]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  return {
    phase,
    statusText,
    error,
    runId,
    run,
    nodes,
    rawNodes,
    selectedNodeId,
    messages,
    reportText,
    reportLoading,
    artifacts,
    start,
    loadRun,
    reset,
    selectNode: setSelectedNodeId,
  };
}

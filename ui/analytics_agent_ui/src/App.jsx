/**
 * Корневой компонент нового UI агента аналитики.
 *
 * Содержит:
 * - normalizeSkills: нормализует ответ `/skills`.
 * - App: собирает layout истории, чата, skills и artifacts.
 */
import { useCallback, useEffect, useState } from "react";
import { BrainCircuit, PanelLeftOpen, PanelRightOpen } from "lucide-react";
import { fetchRuns, fetchSkills } from "./api.js";
import { ChatHistoryBar } from "./components/ChatHistoryBar.jsx";
import { ChatWorkspace } from "./components/ChatWorkspace.jsx";
import { ContextSidebar } from "./components/ContextSidebar.jsx";
import { useAnalyticsRun } from "./hooks/useAnalyticsRun.js";

/**
 * Приводит ответ skills endpoint к массиву SkillRecord.
 *
 * @param {object | null} payload Ответ `/skills`.
 * @returns {Array} Список skills.
 */
function normalizeSkills(payload) {
  return Array.isArray(payload?.skills) ? payload.skills : [];
}

/**
 * Отображает основное приложение агента аналитики.
 *
 * @returns {JSX.Element} Новый трехколоночный UI.
 */
export default function App() {
  const runState = useAnalyticsRun();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState("");
  const [skills, setSkills] = useState([]);
  const [skillsLoading, setSkillsLoading] = useState(false);
  const [skillsError, setSkillsError] = useState("");

  const refreshRuns = useCallback(async () => {
    setRunsLoading(true);
    setRunsError("");
    try {
      const payload = await fetchRuns();
      setRuns(Array.isArray(payload) ? payload : []);
    } catch (error) {
      setRunsError(error.message || "Не удалось загрузить историю.");
    } finally {
      setRunsLoading(false);
    }
  }, []);

  const refreshSkills = useCallback(async () => {
    setSkillsLoading(true);
    setSkillsError("");
    try {
      const payload = await fetchSkills();
      setSkills(normalizeSkills(payload));
    } catch (error) {
      setSkillsError(error.message || "Не удалось загрузить skills.");
    } finally {
      setSkillsLoading(false);
    }
  }, []);

  const submitQuery = useCallback(async (query) => {
    const started = await runState.start(query);
    if (started) {
      refreshRuns();
    }
  }, [refreshRuns, runState]);

  const selectRun = useCallback(async (runId) => {
    const loaded = await runState.loadRun(runId);
    if (loaded) {
      refreshRuns();
    }
  }, [refreshRuns, runState]);

  useEffect(() => {
    refreshRuns();
    refreshSkills();
  }, [refreshRuns, refreshSkills]);

  return (
    <div className="analytics-shell">
      <header className="app-header">
        <div className="app-brand">
          <span className="app-brand-mark">
            <BrainCircuit size={21} />
          </span>
          <div>
            <strong>Analytics Agent</strong>
            <small>{runState.runId ? `run ${runState.runId.slice(0, 8)}` : "рабочая область"}</small>
          </div>
        </div>

        <div className="app-header-actions">
          <button
            type="button"
            className={`header-toggle ${historyOpen ? "header-toggle--active" : ""}`}
            onClick={() => setHistoryOpen((value) => !value)}
            title={historyOpen ? "Скрыть историю" : "Показать историю"}
            aria-label={historyOpen ? "Скрыть историю" : "Показать историю"}
          >
            <PanelLeftOpen size={18} />
            История
          </button>
          <button
            type="button"
            className={`header-toggle ${contextOpen ? "header-toggle--active" : ""}`}
            onClick={() => setContextOpen((value) => !value)}
            title={contextOpen ? "Скрыть skills и артефакты" : "Показать skills и артефакты"}
            aria-label={contextOpen ? "Скрыть skills и артефакты" : "Показать skills и артефакты"}
          >
            <PanelRightOpen size={18} />
            Контекст
          </button>
        </div>
      </header>

      <div
        className={[
          "app-layout",
          historyOpen ? "app-layout--history-open" : "",
          contextOpen ? "app-layout--context-open" : "",
        ].filter(Boolean).join(" ")}
      >
        {historyOpen ? (
          <ChatHistoryBar
            runs={runs}
            activeRunId={runState.runId}
            loading={runsLoading}
            error={runsError}
            onSelectRun={selectRun}
            onRefresh={refreshRuns}
            onNewChat={runState.reset}
          />
        ) : null}
        <ChatWorkspace
          runState={runState}
          onSubmit={submitQuery}
          onReset={runState.reset}
        />
        {contextOpen ? (
          <ContextSidebar
            skills={skills}
            artifacts={runState.artifacts}
            skillsLoading={skillsLoading}
            skillsError={skillsError}
            runId={runState.runId}
          />
        ) : null}
      </div>
    </div>
  );
}

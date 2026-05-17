/**
 * Центральная область чата и графа выполнения.
 *
 * Содержит:
 * - isBusyPhase: определяет активную фазу run.
 * - resizeComposerTextarea: подгоняет высоту поля ввода под многострочный текст.
 * - primaryRawNodeIdForUserNode: выбирает raw node для inspector выбранного user-node.
 * - lastUserMessageContent: возвращает последний пользовательский запрос из сообщений.
 * - ChatWorkspace: отображает prompt, сообщения, граф и markdown-отчет.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Loader2, RotateCcw } from "lucide-react";
import { fetchNodeInspector } from "../api.js";
import { AgentCanvas } from "./AgentCanvas.jsx";
import { GraphNodeDetails } from "./GraphNodeDetails.jsx";
import { MarkdownBlock } from "./MarkdownBlock.jsx";

/**
 * Проверяет, выполняется ли сейчас запуск или загрузка.
 *
 * @param {string} phase Фаза hook состояния.
 * @returns {boolean} True, если ввод нужно временно заблокировать.
 */
function isBusyPhase(phase) {
  return ["starting", "running", "loading"].includes(String(phase || ""));
}

/**
 * Подгоняет высоту textarea под текущий многострочный текст.
 *
 * @param {HTMLTextAreaElement | null} element Поле ввода пользователя.
 * @returns {void}
 */
function resizeComposerTextarea(element) {
  if (!element) {
    return;
  }

  element.style.height = "auto";
  const maxHeight = Number.parseInt(window.getComputedStyle(element).maxHeight, 10) || 160;
  const nextHeight = Math.min(element.scrollHeight, maxHeight);
  element.style.height = `${nextHeight}px`;
  element.style.overflowY = element.scrollHeight > maxHeight ? "auto" : "hidden";
}

/**
 * Находит основной raw node id, связанный с пользовательским узлом графа.
 *
 * @param {object | null} userNode Выбранный user-facing node.
 * @returns {string} Идентификатор raw lineage node или пустая строка.
 */
function primaryRawNodeIdForUserNode(userNode) {
  const rawEvents = (userNode?.raw_events || []).filter(
    (event) => !String(event?.node_type || "").startsWith("synthetic")
  );
  const completed = [...rawEvents].reverse().find((event) => {
    const type = String(event?.node_type || "").toLowerCase();
    return type.includes("task_completed") || type.includes("task_failed") || type.includes("final");
  });
  return completed?.node_id || rawEvents.at(-1)?.node_id || "";
}

/**
 * Возвращает последний пользовательский запрос из истории сообщений.
 *
 * @param {Array} messages Сообщения текущего чата.
 * @returns {string} Последний user message.
 */
function lastUserMessageContent(messages) {
  const message = [...(messages || [])].reverse().find((item) => item?.role === "user");
  return String(message?.content || "").trim();
}

/**
 * Отображает центральную рабочую область аналитического агента.
 *
 * @param {object} props Свойства компонента.
 * @param {object} props.runState Состояние из useAnalyticsRun.
 * @param {Function} props.onSubmit Обработчик отправки нового запроса.
 * @param {Function} props.onReset Обработчик сброса текущего чата.
 * @returns {JSX.Element} Центральная область приложения.
 */
export function ChatWorkspace({ runState, onSubmit, onReset }) {
  const textareaRef = useRef(null);
  const hydratedRunIdRef = useRef("");
  const [query, setQuery] = useState("");
  const [nodeInspector, setNodeInspector] = useState(null);
  const [nodeInspectorLoading, setNodeInspectorLoading] = useState(false);
  const [nodeInspectorError, setNodeInspectorError] = useState("");
  const hasConversation = runState.messages.length > 0 || runState.runId;
  const busy = isBusyPhase(runState.phase);
  const selectedNode = useMemo(
    () => runState.nodes.find((node) => node.node_id === runState.selectedNodeId) || null,
    [runState.nodes, runState.selectedNodeId]
  );
  const lastUserQuery = useMemo(
    () => lastUserMessageContent(runState.messages),
    [runState.messages]
  );

  function submit(event) {
    event.preventDefault();
    if (busy) {
      return;
    }
    const value = query.trim();
    if (!value) {
      onSubmit("");
      return;
    }
    onSubmit(value);
  }

  /**
   * Сбрасывает активный чат и очищает composer только при явном старте нового чата.
   *
   * @returns {void}
   */
  function resetChat() {
    hydratedRunIdRef.current = "";
    setQuery("");
    onReset();
  }

  useEffect(() => {
    if (!runState.runId || hydratedRunIdRef.current === runState.runId) {
      return;
    }

    hydratedRunIdRef.current = runState.runId;
    if (!query.trim() && lastUserQuery) {
      setQuery(lastUserQuery);
    }
  }, [runState.runId, lastUserQuery, query]);

  useEffect(() => {
    resizeComposerTextarea(textareaRef.current);
  }, [query, busy, hasConversation]);

  useEffect(() => {
    let cancelled = false;
    setNodeInspector(null);
    setNodeInspectorError("");

    const rawNodeId = primaryRawNodeIdForUserNode(selectedNode);
    if (!runState.runId || !rawNodeId) {
      setNodeInspectorLoading(false);
      return () => {
        cancelled = true;
      };
    }

    setNodeInspectorLoading(true);
    fetchNodeInspector(runState.runId, rawNodeId)
      .then((payload) => {
        if (!cancelled) {
          setNodeInspector(payload);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setNodeInspectorError(error.message || "Не удалось загрузить детали узла.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setNodeInspectorLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [runState.runId, selectedNode]);

  return (
    <main className={`chat-workspace ${hasConversation ? "chat-workspace--active" : ""}`}>
      <div className="chat-scroll">
        {!hasConversation ? (
          <section className="empty-chat">
            <h1>что нужно сделать?</h1>
          </section>
        ) : (
          <section className="conversation">
            {runState.messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`chat-message chat-message--${message.role}`}>
                <span>{message.role === "user" ? "Вы" : "Агент"}</span>
                <p>{message.content}</p>
              </article>
            ))}

            <section className="graph-message">
              <div className="graph-message-head">
                <div>
                  <span className="panel-kicker">Task graph</span>
                  <h2>Граф работы агента</h2>
                </div>
                <div className={`run-status run-status--${runState.phase}`}>
                  {busy ? <Loader2 className="spin" size={15} /> : null}
                  {runState.statusText || "Ожидаю действия"}
                </div>
              </div>
              {runState.error ? <div className="panel-error">{runState.error}</div> : null}
              <AgentCanvas
                nodes={runState.nodes}
                selectedNodeId={runState.selectedNodeId}
                onSelectNode={runState.selectNode}
                phase={runState.phase}
              />
              <GraphNodeDetails
                node={selectedNode}
                inspector={nodeInspector}
                loading={nodeInspectorLoading}
                error={nodeInspectorError}
                onClose={() => runState.selectNode("")}
              />
            </section>

            {(runState.reportLoading || runState.reportText) ? (
              <article className="chat-message chat-message--agent report-message">
                <span>Финальный ответ</span>
                {runState.reportLoading ? (
                  <div className="loading-row">
                    <Loader2 className="spin" size={17} />
                    Загружаю итоговый отчет...
                  </div>
                ) : (
                  <MarkdownBlock>{runState.reportText}</MarkdownBlock>
                )}
              </article>
            ) : null}
          </section>
        )}
      </div>

      <form className={`composer ${hasConversation ? "composer--active" : "composer--initial"}`} onSubmit={submit}>
        {hasConversation ? (
          <button type="button" className="composer-reset" onClick={resetChat} title="Новый чат">
            <RotateCcw size={17} />
          </button>
        ) : null}
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Опишите задачу для агента"
          rows={1}
        />
        <button type="submit" className="composer-send" disabled={busy}>
          {busy ? <Loader2 className="spin" size={18} /> : <ArrowUp size={18} />}
        </button>
      </form>
    </main>
  );
}

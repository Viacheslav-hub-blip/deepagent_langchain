/**
 * Центральная область чата и графа выполнения.
 *
 * Содержит:
 * - isBusyPhase: определяет активную фазу run.
 * - ChatWorkspace: отображает prompt, сообщения, граф и markdown-отчет.
 */
import { useState } from "react";
import { ArrowUp, Loader2, RotateCcw } from "lucide-react";
import { AgentCanvas } from "./AgentCanvas.jsx";
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
 * Отображает центральную рабочую область аналитического агента.
 *
 * @param {object} props Свойства компонента.
 * @param {object} props.runState Состояние из useAnalyticsRun.
 * @param {Function} props.onSubmit Обработчик отправки нового запроса.
 * @param {Function} props.onReset Обработчик сброса текущего чата.
 * @returns {JSX.Element} Центральная область приложения.
 */
export function ChatWorkspace({ runState, onSubmit, onReset }) {
  const [query, setQuery] = useState("");
  const hasConversation = runState.messages.length > 0 || runState.runId;
  const busy = isBusyPhase(runState.phase);

  function submit(event) {
    event.preventDefault();
    const value = query.trim();
    if (!value) {
      onSubmit("");
      return;
    }
    setQuery("");
    onSubmit(value);
  }

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
            </section>

            {(runState.reportLoading || runState.reportText) ? (
              <article className="chat-message chat-message--agent report-message">
                <span>Отчет</span>
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
          <button type="button" className="composer-reset" onClick={onReset} title="Новый чат">
            <RotateCcw size={17} />
          </button>
        ) : null}
        <textarea
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Опишите задачу для агента"
          rows={1}
          disabled={busy}
        />
        <button type="submit" className="composer-send" disabled={busy}>
          {busy ? <Loader2 className="spin" size={18} /> : <ArrowUp size={18} />}
        </button>
      </form>
    </main>
  );
}

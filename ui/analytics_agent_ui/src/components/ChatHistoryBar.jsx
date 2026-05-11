/**
 * Левый bar истории чатов.
 *
 * Содержит:
 * - formatRunTime: форматирует дату запуска.
 * - runTitle: выбирает заголовок run.
 * - sortRuns: сортирует историю по времени.
 * - ChatHistoryBar: отображает список сохраненных runs.
 */
import { History, Loader2, MessageSquare, Plus, RefreshCw } from "lucide-react";

/**
 * Форматирует дату запуска для компактного отображения.
 *
 * @param {string} value ISO-строка даты.
 * @returns {string} Локализованная дата или пустая строка.
 */
function formatRunTime(value) {
  const date = new Date(value || "");
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

/**
 * Возвращает человекочитаемый заголовок run.
 *
 * @param {object} summary Краткая сводка run.
 * @returns {string} Заголовок для истории.
 */
function runTitle(summary) {
  const run = summary?.run || {};
  return String(run.initial_user_query || run.title || run.run_id || "Без названия").trim();
}

/**
 * Сортирует run summaries от новых к старым.
 *
 * @param {Array} runs Список RunSummary.
 * @returns {Array} Отсортированный список.
 */
function sortRuns(runs) {
  return [...(runs || [])].sort((a, b) => {
    const left = Date.parse(a?.run?.updated_at || a?.run?.created_at || "");
    const right = Date.parse(b?.run?.updated_at || b?.run?.created_at || "");
    return (Number.isNaN(right) ? 0 : right) - (Number.isNaN(left) ? 0 : left);
  });
}

/**
 * Отображает историю сохраненных аналитических запусков.
 *
 * @param {object} props Свойства компонента.
 * @param {Array} props.runs Список RunSummary.
 * @param {string} props.activeRunId Активный run_id.
 * @param {boolean} props.loading Признак загрузки истории.
 * @param {string} props.error Текст ошибки истории.
 * @param {Function} props.onSelectRun Обработчик выбора run.
 * @param {Function} props.onRefresh Обработчик обновления истории.
 * @param {Function} props.onNewChat Обработчик нового чата.
 * @returns {JSX.Element} Левый sidebar истории.
 */
export function ChatHistoryBar({
  runs,
  activeRunId,
  loading,
  error,
  onSelectRun,
  onRefresh,
  onNewChat,
}) {
  const orderedRuns = sortRuns(runs);

  return (
    <aside className="side-panel history-panel">
      <div className="panel-head">
        <div>
          <span className="panel-kicker">History</span>
          <h2>Чаты</h2>
        </div>
        <button type="button" className="icon-button" onClick={onRefresh} title="Обновить историю">
          {loading ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
        </button>
      </div>

      <button type="button" className="new-chat-button" onClick={onNewChat}>
        <Plus size={17} />
        Новый чат
      </button>

      {error ? <div className="panel-error">{error}</div> : null}

      <div className="history-list">
        {!orderedRuns.length && !loading ? (
          <div className="empty-panel-state">
            <History size={22} />
            <span>История пока пустая</span>
          </div>
        ) : null}

        {orderedRuns.map((summary) => {
          const run = summary.run || {};
          const active = run.run_id && run.run_id === activeRunId;
          return (
            <button
              key={run.run_id}
              type="button"
              className={`history-item ${active ? "history-item--active" : ""}`}
              onClick={() => onSelectRun(run.run_id)}
            >
              <MessageSquare size={16} />
              <span className="history-item-text">
                <strong>{runTitle(summary)}</strong>
                <small>
                  {formatRunTime(run.updated_at || run.created_at)}
                  {summary.artifact_count ? ` · ${summary.artifact_count} artifacts` : ""}
                </small>
              </span>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

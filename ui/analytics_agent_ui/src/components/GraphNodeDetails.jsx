/**
 * Панель просмотра выбранного узла графа.
 *
 * Содержит:
 * - artifactLabel: выбирает читаемое имя artifact.
 * - rawEventLabel: выбирает заголовок raw event.
 * - GraphNodeDetails: показывает содержимое узла после клика на карточку графа.
 */
import { Boxes, FileText, Loader2, X } from "lucide-react";
import { compactId } from "../lib/nodes.js";
import { MarkdownBlock } from "./MarkdownBlock.jsx";

/**
 * Возвращает читаемое имя artifact для панели деталей.
 *
 * @param {object} entry ArtifactDetails или Artifact.
 * @returns {string} Название artifact.
 */
function artifactLabel(entry) {
  const artifact = entry?.artifact || entry || {};
  const meta = artifact.metadata || {};
  return String(meta.original_filename || meta.filename || meta.export_filename || artifact.artifact_id || "artifact");
}

/**
 * Возвращает краткий заголовок raw event.
 *
 * @param {object} event Raw lineage event.
 * @param {number} index Индекс события.
 * @returns {string} Заголовок события.
 */
function rawEventLabel(event, index) {
  return String(event?.title || event?.node_type || `Raw event ${index + 1}`);
}

/**
 * Отображает подробности user-facing узла и inspector raw node.
 *
 * @param {object} props Свойства компонента.
 * @param {object | null} props.node Выбранный user-facing node.
 * @param {object | null} props.inspector NodeInspectorView для связанного raw node.
 * @param {boolean} props.loading Признак загрузки inspector.
 * @param {string} props.error Ошибка загрузки inspector.
 * @param {Function} props.onClose Обработчик закрытия панели.
 * @returns {JSX.Element | null} Панель деталей выбранного узла.
 */
export function GraphNodeDetails({ node, inspector, loading, error, onClose }) {
  if (!node) {
    return null;
  }

  const artifacts = inspector?.artifacts || [];
  const toolTraces = inspector?.tool_traces || [];
  const rawEvents = node.raw_events || [];

  return (
    <aside className="node-details-panel">
      <div className="node-details-header">
        <div>
          <span className="panel-kicker">Node details</span>
          <h3>{node.title || node.node_type || "Узел графа"}</h3>
        </div>
        <button type="button" className="icon-button" onClick={onClose} title="Закрыть детали узла">
          <X size={17} />
        </button>
      </div>

      <div className="node-details-grid">
        <div className="node-detail-kv">
          <span>Status</span>
          <strong>{node.status || "unknown"}</strong>
        </div>
        <div className="node-detail-kv">
          <span>Type</span>
          <strong>{node.group_role || node.node_type || "node"}</strong>
        </div>
        <div className="node-detail-kv">
          <span>Raw events</span>
          <strong>{rawEvents.length || "—"}</strong>
        </div>
        <div className="node-detail-kv">
          <span>ID</span>
          <code>{compactId(node.node_id, 10, 6)}</code>
        </div>
      </div>

      {node.summary ? (
        <section className="node-details-section">
          <div className="node-details-title">
            <FileText size={15} />
            <span>Содержимое узла</span>
          </div>
          <MarkdownBlock className="node-details-markdown">{node.summary}</MarkdownBlock>
        </section>
      ) : null}

      {node.task_description ? (
        <section className="node-details-section">
          <div className="node-details-title">
            <FileText size={15} />
            <span>Постановка задачи</span>
          </div>
          <MarkdownBlock className="node-details-markdown">{node.task_description}</MarkdownBlock>
        </section>
      ) : null}

      {node.task_result ? (
        <section className="node-details-section">
          <div className="node-details-title">
            <FileText size={15} />
            <span>Результат задачи</span>
          </div>
          <MarkdownBlock className="node-details-markdown">{node.task_result}</MarkdownBlock>
        </section>
      ) : null}

      {loading ? (
        <div className="node-details-state">
          <Loader2 className="spin" size={18} />
          Загружаю inspector raw node...
        </div>
      ) : null}
      {error ? <div className="panel-error node-details-error">{error}</div> : null}

      {artifacts.length || toolTraces.length ? (
        <section className="node-details-section">
          <div className="node-details-title">
            <Boxes size={15} />
            <span>Artifacts и tool traces</span>
          </div>
          <div className="node-artifact-detail-list">
            {[...artifacts, ...toolTraces].map((entry) => {
              const artifactId = entry?.artifact?.artifact_id || entry?.artifact_id || artifactLabel(entry);
              return (
                <article key={artifactId} className="node-artifact-detail">
                  <strong>{artifactLabel(entry)}</strong>
                  <small>{entry?.artifact?.kind || entry?.kind || "artifact"}</small>
                  {entry?.preview?.preview ? <pre>{entry.preview.preview}</pre> : null}
                </article>
              );
            })}
          </div>
        </section>
      ) : null}

      {rawEvents.length ? (
        <section className="node-details-section">
          <div className="node-details-title">
            <FileText size={15} />
            <span>Raw events</span>
          </div>
          <div className="raw-event-summary-list">
            {rawEvents.map((event, index) => (
              <div key={event.node_id || index} className="raw-event-summary">
                <strong>{rawEventLabel(event, index)}</strong>
                <span>{event.status || "unknown"} · {compactId(event.node_id, 8, 5)}</span>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </aside>
  );
}

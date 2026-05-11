/**
 * Карточка узла графа агента.
 *
 * Содержит:
 * - isTaskNode: проверяет, является ли узел задачей.
 * - taskTitle: формирует заголовок task-узла.
 * - taskDescription: выбирает описание task-узла.
 * - NodeCardShell: общая кликабельная оболочка карточки.
 * - NodeCard: отображает карточку узла на графе.
 */
import { AlertTriangle, Check, Clock3, Cpu, FileText, GitBranch, Sparkles } from "lucide-react";
import { compactId, getStatusTone, summarizeNode } from "../lib/nodes.js";
import { getUserNodeStage } from "../lib/userGraph.js";

const ICONS = {
  start: Sparkles,
  context: FileText,
  plan: Cpu,
  work: Cpu,
  review: AlertTriangle,
  replan: Clock3,
  answer: Check,
};

/**
 * Проверяет, является ли узел пользовательской задачей.
 *
 * @param {object} node Узел графа.
 * @returns {boolean} True для task-узла.
 */
function isTaskNode(node) {
  return node?.group_role === "task";
}

/**
 * Формирует короткий заголовок task-узла.
 *
 * @param {object} node Узел графа.
 * @returns {string} Заголовок карточки.
 */
function taskTitle(node) {
  const taskId = node?.task_id || "";
  return taskId ? `Task ${taskId}` : "Task";
}

/**
 * Выбирает описание task-узла.
 *
 * @param {object} node Узел графа.
 * @returns {string} Описание задачи.
 */
function taskDescription(node) {
  return node?.summary || node?.task_description || node?.title || "Описание задачи";
}

/**
 * Оборачивает карточку узла в кликабельную область с клавиатурной навигацией.
 *
 * @param {object} props Свойства оболочки.
 * @param {string} props.className CSS-классы карточки.
 * @param {Function} props.onClick Обработчик выбора.
 * @param {React.ReactNode} props.children Содержимое карточки.
 * @returns {JSX.Element} Кликабельная оболочка.
 */
function NodeCardShell({ className, onClick, children }) {
  function onKeyDown(event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onClick?.();
    }
  }

  return (
    <div role="button" tabIndex={0} className={className} onClick={onClick} onKeyDown={onKeyDown}>
      {children}
    </div>
  );
}

/**
 * Отображает карточку одного узла графа выполнения.
 *
 * @param {object} props Свойства компонента.
 * @param {object} props.node Узел графа.
 * @param {number} props.index Порядковый номер узла.
 * @param {boolean} props.active Выбран ли узел.
 * @param {boolean} props.current Является ли узел текущим.
 * @param {boolean} props.inBranch Входит ли узел в активную ветку.
 * @param {Function} props.onClick Обработчик выбора узла.
 * @param {Function | undefined} props.onBranchClick Необязательный обработчик branch.
 * @returns {JSX.Element} Карточка узла.
 */
export function NodeCard({ node, index, active, current, inBranch, onClick, onBranchClick }) {
  const stage = getUserNodeStage(node);
  const tone = getStatusTone(node.status);
  const Icon = ICONS[stage.id] || Cpu;
  const taskOnly = isTaskNode(node);

  const className = [
    "node-card",
    taskOnly ? "node-card--task-clean" : "",
    `node-card--${stage.id}`,
    `node-card--${tone}`,
    active ? "node-card--active" : "",
    current ? "node-card--current" : "",
    inBranch ? "node-card--branch" : "",
  ].filter(Boolean).join(" ");

  function branchClick(event) {
    event.stopPropagation();
    onBranchClick?.(node.node_id);
  }

  const branchButton = onBranchClick ? (
    <button
      type="button"
      className="node-branch-button"
      onClick={branchClick}
      title="Создать branch от этого узла"
      aria-label="Создать branch от этого узла"
    >
      <GitBranch size={14} />
    </button>
  ) : null;

  if (taskOnly) {
    const tools = Array.isArray(node.task_tools) ? node.task_tools : [];

    return (
      <NodeCardShell className={className} onClick={onClick}>
        {branchButton}
        <div className="task-clean-title">{taskTitle(node)}</div>
        <div className="task-clean-meta">
          <div className={`status-pill status-pill--${tone}`}>{node.status || "unknown"}</div>
        </div>
        {tools.length ? (
          <div className="task-clean-tools" aria-label="Инструменты задачи">
            {tools.map((name) => (
              <span key={name} className="tool-chip" title={name}>
                {name}
              </span>
            ))}
          </div>
        ) : null}
        <p className="task-clean-description">{taskDescription(node)}</p>
      </NodeCardShell>
    );
  }

  return (
    <NodeCardShell className={className} onClick={onClick}>
      {branchButton}
      <div className="node-card-top">
        <div className="node-icon">
          <Icon size={17} />
        </div>
        <div className={`status-pill status-pill--${tone}`}>{node.status || "unknown"}</div>
      </div>

      <div className="node-stage">
        <span>{stage.label} · #{String(index + 1).padStart(2, "0")}</span>
        {current ? <em>Active now</em> : inBranch ? <em>Branch path</em> : null}
      </div>

      <h3>{node.title || node.node_type || "Node"}</h3>
      <p>{summarizeNode(node)}</p>

      <div className="node-card-footer">
        <span>{node.raw_event_count ? `${node.raw_event_count} raw events` : node.node_type || "node"}</span>
        <code>{compactId(node.node_id)}</code>
      </div>
    </NodeCardShell>
  );
}

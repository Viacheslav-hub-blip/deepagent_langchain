/**
 * Карточка узла графа агента.
 *
 * Содержит:
 * - isTaskNode: проверяет, является ли узел задачей.
 * - taskTitle: формирует заголовок task-узла.
 * - nodeTitle: выбирает заголовок любого узла.
 * - NodeCardShell: общая кликабельная оболочка карточки.
 * - NodeCard: отображает карточку узла на графе.
 */
import { GitBranch } from "lucide-react";
import { getStatusTone } from "../lib/nodes.js";
import { getUserNodeStage } from "../lib/userGraph.js";

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
 * Выбирает заголовок узла для компактной карточки графа.
 *
 * @param {object} node Узел графа.
 * @returns {string} Заголовок карточки.
 */
function nodeTitle(node) {
  if (isTaskNode(node)) {
    return taskTitle(node);
  }
  return node?.title || node?.node_type || "Node";
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

  return (
    <NodeCardShell className={className} onClick={onClick}>
      {branchButton}
      <div className="node-card-compact-top">
        <div className={`status-pill status-pill--${tone}`}>{node.status || "unknown"}</div>
      </div>

      <h3>{nodeTitle(node)}</h3>
    </NodeCardShell>
  );
}

/**
 * Утилиты нормализации lineage nodes для UI.
 *
 * Содержит:
 * - normalizeNodeType: нормализует тип узла.
 * - getNodeStage: определяет стадию узла.
 * - getStatusTone: выбирает визуальный тон статуса.
 * - isTerminalRunStatus: проверяет терминальный статус run.
 * - compactId: сокращает длинные идентификаторы.
 * - stableMergeNodes: стабильно объединяет списки nodes.
 * - summarizeNode: выбирает текст summary узла.
 */

export const LIVE_POLL_INTERVAL_MS = 900;

export const STAGES = [
  { id: "start", label: "Start", match: ["run_started", "branch_started"] },
  { id: "context", label: "Context", match: ["context", "snapshot", "skill", "memory"] },
  { id: "plan", label: "Plan", match: ["plan", "planner", "task_scheduled", "scheduler"] },
  { id: "work", label: "Work", match: ["worker", "tool", "artifact", "task_completed", "task_failed"] },
  { id: "review", label: "Review", match: ["critic", "validator", "validation", "review"] },
  { id: "replan", label: "Replan", match: ["replan"] },
  { id: "answer", label: "Answer", match: ["final", "report", "responder", "answer"] },
];

/**
 * Приводит произвольный тип узла к нижнему регистру и безопасным символам.
 *
 * @param {string} value Исходное значение.
 * @returns {string} Нормализованный тип.
 */
export function normalizeNodeType(value) {
  return String(value || "node").toLowerCase().replace(/[^a-z0-9_\-]+/g, "_");
}

/**
 * Определяет стадию графа по типу, заголовку и summary узла.
 *
 * @param {object} node Lineage node.
 * @returns {object} Объект стадии.
 */
export function getNodeStage(node) {
  const combined = [
    normalizeNodeType(node?.node_type),
    normalizeNodeType(node?.title),
    normalizeNodeType(node?.summary),
  ].join(" ");
  return STAGES.find((stage) => stage.match.some((token) => combined.includes(token))) || STAGES[3];
}

/**
 * Возвращает CSS-тон для статуса.
 *
 * @param {string} status Статус run или node.
 * @returns {string} Имя визуального тона.
 */
export function getStatusTone(status) {
  const value = String(status || "").toLowerCase();
  if (["succeeded", "success", "completed", "done"].includes(value)) return "success";
  if (["failed", "error", "cancelled"].includes(value)) return "danger";
  if (["running", "started", "pending", "in_progress"].includes(value)) return "active";
  return "neutral";
}

/**
 * Проверяет, завершился ли run.
 *
 * @param {string} status Статус run.
 * @returns {boolean} True для терминального статуса.
 */
export function isTerminalRunStatus(status) {
  const value = String(status || "").toLowerCase();
  return ["succeeded", "failed", "cancelled", "completed"].includes(value);
}

/**
 * Сокращает длинный идентификатор с сохранением начала и конца.
 *
 * @param {string} value Исходный идентификатор.
 * @param {number} left Количество символов слева.
 * @param {number} right Количество символов справа.
 * @returns {string} Сокращенный идентификатор.
 */
export function compactId(value, left = 8, right = 5) {
  const text = String(value || "");
  if (text.length <= left + right + 3) return text;
  return `${text.slice(0, left)}...${text.slice(-right)}`;
}

/**
 * Объединяет предыдущий и новый список nodes без дергания существующего порядка.
 *
 * @param {Array} previousNodes Ранее показанные nodes.
 * @param {Array} incomingNodes Новые nodes из API.
 * @returns {Array} Объединенный список nodes.
 */
export function stableMergeNodes(previousNodes, incomingNodes) {
  const byId = new Map();
  for (const node of previousNodes || []) {
    if (node?.node_id) byId.set(node.node_id, node);
  }
  for (const node of incomingNodes || []) {
    if (!node?.node_id) continue;
    byId.set(node.node_id, { ...(byId.get(node.node_id) || {}), ...node });
  }
  return Array.from(byId.values());
}

/**
 * Выбирает краткое описание узла.
 *
 * @param {object} node Lineage node.
 * @returns {string} Текст summary.
 */
export function summarizeNode(node) {
  return node?.summary || node?.title || node?.node_type || "Node без описания";
}

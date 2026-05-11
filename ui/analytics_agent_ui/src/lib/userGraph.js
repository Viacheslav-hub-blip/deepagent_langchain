/**
 * Построение пользовательского графа поверх raw lineage events.
 *
 * Содержит:
 * - normalize: нормализует текст для поиска.
 * - eventText: собирает текст узла для классификации.
 * - sortEvents: сортирует raw events по времени.
 * - firstNonEmpty: выбирает первое непустое значение.
 * - getTaskId: извлекает идентификатор задачи.
 * - isStartEvent: определяет стартовый event.
 * - isContextEvent: определяет context event.
 * - isSchedulerEvent: определяет scheduler event.
 * - isPlanEvent: определяет planner event.
 * - isFinalEvent: определяет финальный event.
 * - isTaskEvent: определяет task/worker event.
 * - statusForEvents: вычисляет статус группы.
 * - statusForTaskEvents: вычисляет статус task-группы.
 * - summaryForEvents: выбирает summary группы.
 * - getTaskDescription: извлекает постановку задачи.
 * - getTaskResult: извлекает результат задачи.
 * - collectInvokedToolNames: собирает tool names.
 * - createGroupNode: создает user-facing node.
 * - buildUserGraph: строит компактный граф для UI.
 * - getRawEventsForUserNode: возвращает raw events user node.
 * - getUserNodeStage: определяет стадию user node.
 */
import { getNodeStage } from "./nodes.js";

/**
 * Нормализует строку для устойчивых сравнений.
 *
 * @param {string} value Исходное значение.
 * @returns {string} Нормализованный текст.
 */
function normalize(value) {
  return String(value || "").toLowerCase();
}

/**
 * Собирает основной текст event для классификации.
 *
 * @param {object} node Raw lineage node.
 * @returns {string} Текстовая сигнатура node.
 */
function eventText(node) {
  return `${normalize(node?.node_type)} ${normalize(node?.title)} ${normalize(node?.summary)}`;
}

/**
 * Сортирует events по created_at без потери исходного порядка при плохих датах.
 *
 * @param {Array} events Raw lineage events.
 * @returns {Array} Отсортированные events.
 */
function sortEvents(events) {
  return [...(events || [])].sort((a, b) => {
    const left = Date.parse(a?.created_at || "");
    const right = Date.parse(b?.created_at || "");
    if (Number.isNaN(left) || Number.isNaN(right)) return 0;
    return left - right;
  });
}

/**
 * Возвращает первое непустое значение из списка.
 *
 * @param {Array} values Возможные значения.
 * @returns {string} Первое непустое значение.
 */
function firstNonEmpty(values) {
  for (const value of values) {
    if (value !== undefined && value !== null && String(value).trim()) {
      return String(value).trim();
    }
  }
  return "";
}

/**
 * Извлекает task_id из metadata, title или summary.
 *
 * @param {object} node Raw lineage node.
 * @returns {string} Идентификатор задачи.
 */
function getTaskId(node) {
  const direct = firstNonEmpty([
    node?.metadata?.task_id,
    node?.metadata?.task_index,
    node?.metadata?.task_number,
    node?.metadata?.task?.task_id,
    node?.metadata?.task?.id,
    node?.task_id,
  ]);
  if (direct) return direct;

  const text = `${node?.title || ""} ${node?.summary || ""}`;
  const match = text.match(/\btask\s*(?:started|finished|completed|failed)?\s*[:#-]?\s*(\d+)\b/i)
    || text.match(/\bworker\s+started\s*:\s*task\s+(\d+)\b/i)
    || text.match(/\btask\s+(\d+)\b/i);
  if (match?.[1]) return match[1];

  const typeMatch = String(node?.node_type || "").match(/\btask[_-](\d+)\b/i);
  return typeMatch?.[1] || "";
}

/** @param {object} node Raw node. @returns {boolean} True для стартового event. */
function isStartEvent(node) {
  const text = eventText(node);
  return text.includes("run_started") || text.includes("branch_started") || text.includes("run started");
}

/** @param {object} node Raw node. @returns {boolean} True для context event. */
function isContextEvent(node) {
  const text = eventText(node);
  return text.includes("context") || text.includes("snapshot") || text.includes("skill") || text.includes("memory");
}

/** @param {object} node Raw node. @returns {boolean} True для scheduler event. */
function isSchedulerEvent(node) {
  const text = eventText(node);
  return text.includes("scheduler") || text.includes("task_scheduled") || text.includes("task scheduled");
}

/** @param {object} node Raw node. @returns {boolean} True для planner event. */
function isPlanEvent(node) {
  const text = eventText(node);
  return !isSchedulerEvent(node) && (
    text.includes("plan") || text.includes("planner") || text.includes("replan") || text.includes("replanner")
  );
}

/** @param {object} node Raw node. @returns {boolean} True для final/report event. */
function isFinalEvent(node) {
  const text = eventText(node);
  return text.includes("final") || text.includes("report") || text.includes("answer") || text.includes("responder");
}

/** @param {object} node Raw node. @returns {boolean} True для worker/task event. */
function isTaskEvent(node) {
  const text = eventText(node);
  return !isSchedulerEvent(node) && (
    text.includes("worker") ||
    text.includes("task_") ||
    text.includes("task ") ||
    text.includes("task:") ||
    text.includes("validation") ||
    text.includes("validator") ||
    text.includes("critic")
  );
}

/**
 * Вычисляет статус группы raw events.
 *
 * @param {Array} events Raw events.
 * @returns {string} Статус группы.
 */
function statusForEvents(events) {
  const values = (events || []).map((event) => normalize(event?.status));
  if (values.some((value) => ["failed", "error", "cancelled"].includes(value))) return "failed";
  if (values.some((value) => ["running", "started", "pending", "in_progress"].includes(value))) return "running";
  if (values.some((value) => ["succeeded", "success", "completed", "done"].includes(value))) return "succeeded";
  return events?.at(-1)?.status || "unknown";
}

/**
 * Вычисляет статус task-группы.
 *
 * @param {Array} events Raw task events.
 * @returns {string} Статус задачи.
 */
function statusForTaskEvents(events) {
  const text = eventText({ summary: (events || []).map(eventText).join(" ") });
  if (text.includes("task_failed") || text.includes("failed")) return "failed";
  if (text.includes("task_completed") || text.includes("validation_completed")) return "succeeded";
  return statusForEvents(events);
}

/**
 * Выбирает summary группы raw events.
 *
 * @param {Array} events Raw events.
 * @returns {string} Summary группы.
 */
function summaryForEvents(events) {
  const sorted = sortEvents(events);
  const preferred = sorted.find((event) => event.summary && event.summary.length > 80) || sorted.at(-1) || sorted[0];
  return preferred?.summary || preferred?.title || "События выполнения.";
}

/**
 * Извлекает постановку задачи из metadata или первого worker event.
 *
 * @param {Array} events Raw task events.
 * @param {string} taskId Идентификатор задачи.
 * @returns {string} Описание задачи.
 */
function getTaskDescription(events, taskId) {
  const sorted = sortEvents(events);
  const metadataValue = sorted
    .map((event) => event?.metadata?.task?.description || event?.metadata?.task_description || event?.metadata?.description)
    .find(Boolean);
  const raw = metadataValue || sorted[0]?.summary || sorted[0]?.title || `Задача ${taskId}`;
  return String(raw)
    .replace(/^\s*worker\s+started\s*:\s*task\s+\w+\s*[:—-]?\s*/i, "")
    .replace(/^\s*task\s*#?\s*[\w-]+\s*[:—-]?\s*/i, "")
    .trim() || `Задача ${taskId}`;
}

/**
 * Извлекает результат задачи из завершающих raw events.
 *
 * @param {Array} events Raw task events.
 * @returns {string} Текст результата задачи.
 */
function getTaskResult(events) {
  const sorted = sortEvents(events);
  const result = [...sorted].reverse().find((event) => {
    const text = eventText(event);
    return event.summary && (
      text.includes("task_completed") ||
      text.includes("task_finished") ||
      text.includes("validation_completed") ||
      text.includes("worker_result")
    );
  });
  return result?.summary || "";
}

/**
 * Собирает имена вызванных инструментов из metadata.
 *
 * @param {Array} events Raw task events.
 * @returns {Array<string>} Уникальные имена tools.
 */
function collectInvokedToolNames(events) {
  const names = [];
  const seen = new Set();
  for (const event of events || []) {
    for (const name of event?.metadata?.invoked_tool_names || []) {
      const clean = String(name || "").trim();
      if (clean && !seen.has(clean)) {
        seen.add(clean);
        names.push(clean);
      }
    }
    const single = String(event?.metadata?.tool_name || "").trim();
    if (single && !seen.has(single)) {
      seen.add(single);
      names.push(single);
    }
  }
  return names;
}

/**
 * Создает user-facing node из группы raw events.
 *
 * @param {object} options Настройки группового узла.
 * @returns {object} Узел пользовательского графа.
 */
function createGroupNode({
  id,
  type,
  title,
  summary,
  status,
  events,
  layoutRow,
  layoutPhase,
  layoutLabel,
  iteration = null,
  taskId = null,
  groupRole = null,
  taskDescription = "",
  taskResult = "",
  taskTools = [],
}) {
  const rawEvents = sortEvents(events);
  return {
    node_id: id,
    node_type: type,
    title,
    summary,
    status,
    created_at: rawEvents[0]?.created_at,
    updated_at: rawEvents.at(-1)?.created_at,
    raw_events: rawEvents,
    raw_event_count: rawEvents.length,
    parent_ids: [],
    is_user_group: true,
    layout_row: layoutRow,
    layout_phase: layoutPhase,
    layout_label: layoutLabel,
    iteration,
    task_id: taskId,
    group_role: groupRole,
    task_description: taskDescription,
    task_result: taskResult,
    task_tools: taskTools,
  };
}

/**
 * Строит компактный пользовательский граф из raw lineage nodes.
 *
 * @param {Array} rawNodes Raw lineage nodes из API.
 * @returns {object} Узлы графа и map соответствий raw/user nodes.
 */
export function buildUserGraph(rawNodes) {
  const ordered = sortEvents(Array.isArray(rawNodes) ? rawNodes : []);
  if (!ordered.length) {
    return { nodes: [], rawToGroup: new Map(), groupToRaw: new Map() };
  }

  const startEvents = [];
  const contextEvents = [];
  const planEvents = [];
  const finalEvents = [];
  const taskBuckets = new Map();
  const otherEvents = [];

  for (const node of ordered) {
    if (isStartEvent(node)) {
      startEvents.push(node);
      continue;
    }
    if (isFinalEvent(node)) {
      finalEvents.push(node);
      continue;
    }
    if (isContextEvent(node) && !planEvents.length && !taskBuckets.size) {
      contextEvents.push(node);
      continue;
    }
    if (isPlanEvent(node) || isSchedulerEvent(node)) {
      planEvents.push(node);
      continue;
    }

    const taskId = getTaskId(node);
    if (taskId && isTaskEvent(node)) {
      if (!taskBuckets.has(taskId)) taskBuckets.set(taskId, []);
      taskBuckets.get(taskId).push(node);
      continue;
    }

    otherEvents.push(node);
  }

  const groupedNodes = [];
  let row = 0;

  if (startEvents.length) {
    groupedNodes.push(createGroupNode({
      id: "user-start",
      type: "user_start",
      title: "Запуск исследования",
      summary: summaryForEvents(startEvents),
      status: statusForEvents(startEvents),
      events: startEvents,
      layoutRow: row,
      layoutPhase: "start",
      layoutLabel: "Запуск исследования",
    }));
    row += 1;
  }

  if (contextEvents.length) {
    groupedNodes.push(createGroupNode({
      id: "user-context",
      type: "user_context",
      title: "Сбор контекста",
      summary: summaryForEvents(contextEvents),
      status: statusForEvents(contextEvents),
      events: contextEvents,
      layoutRow: row,
      layoutPhase: "context",
      layoutLabel: "Сбор контекста",
    }));
    row += 1;
  }

  if (planEvents.length || taskBuckets.size) {
    groupedNodes.push(createGroupNode({
      id: "user-plan-1",
      type: "user_plan",
      title: "Планирование",
      summary: planEvents.length ? summaryForEvents(planEvents) : `План сформировал ${taskBuckets.size || "несколько"} задач.`,
      status: planEvents.length ? statusForEvents(planEvents) : "succeeded",
      events: planEvents.length ? planEvents : [{
        node_id: "synthetic-plan-1",
        node_type: "synthetic_plan",
        title: "Планирование",
        status: "succeeded",
        summary: "Синтетический блок планирования.",
        parent_ids: [],
      }],
      layoutRow: row,
      layoutPhase: "plan",
      layoutLabel: "Планирование",
      iteration: 1,
      groupRole: "plan",
    }));
    row += 1;
  }

  for (const taskId of Array.from(taskBuckets.keys()).sort((a, b) => String(a).localeCompare(String(b), "ru", { numeric: true }))) {
    const events = taskBuckets.get(taskId);
    const taskResult = getTaskResult(events);
    const taskDescription = getTaskDescription(events, taskId);
    const status = statusForTaskEvents(events);
    groupedNodes.push(createGroupNode({
      id: `user-task-${taskId}`,
      type: "user_task",
      title: `Task ${taskId}`,
      summary: taskResult && status !== "running" ? taskResult : taskDescription,
      status,
      events,
      layoutRow: row,
      layoutPhase: "tasks",
      layoutLabel: "Задачи плана",
      iteration: 1,
      taskId,
      groupRole: "task",
      taskDescription,
      taskResult,
      taskTools: collectInvokedToolNames(events),
    }));
  }

  if (taskBuckets.size) {
    row += 1;
  }

  if (otherEvents.length) {
    groupedNodes.push(createGroupNode({
      id: "user-other",
      type: "user_step",
      title: "Дополнительные события",
      summary: summaryForEvents(otherEvents),
      status: statusForEvents(otherEvents),
      events: otherEvents,
      layoutRow: row,
      layoutPhase: "tasks",
      layoutLabel: "Дополнительные события",
      groupRole: "other",
    }));
    row += 1;
  }

  if (finalEvents.length) {
    groupedNodes.push(createGroupNode({
      id: "user-final",
      type: "user_final",
      title: "Финальный отчет",
      summary: summaryForEvents(finalEvents),
      status: statusForEvents(finalEvents),
      events: finalEvents,
      layoutRow: row,
      layoutPhase: "final",
      layoutLabel: "Финальный отчет",
      groupRole: "final",
    }));
  }

  for (const node of groupedNodes) {
    if (node.group_role === "task" && groupedNodes.some((candidate) => candidate.node_id === "user-plan-1")) {
      node.parent_ids = ["user-plan-1"];
      continue;
    }
    const previous = groupedNodes
      .filter((candidate) => candidate.layout_row < node.layout_row)
      .sort((a, b) => b.layout_row - a.layout_row)[0];
    node.parent_ids = previous ? [previous.node_id] : [];
  }
  if (groupedNodes[0]) groupedNodes[0].parent_ids = [];

  const rawToGroup = new Map();
  const groupToRaw = new Map();
  for (const group of groupedNodes) {
    groupToRaw.set(group.node_id, group.raw_events.map((event) => event.node_id));
    for (const event of group.raw_events) {
      rawToGroup.set(event.node_id, group.node_id);
    }
  }

  return { nodes: groupedNodes, rawToGroup, groupToRaw };
}

/**
 * Возвращает raw events, из которых собран user-facing node.
 *
 * @param {object} userNode Пользовательский узел графа.
 * @returns {Array} Raw events.
 */
export function getRawEventsForUserNode(userNode) {
  return userNode?.raw_events || [];
}

/**
 * Определяет стадию user-facing node.
 *
 * @param {object} node Узел графа.
 * @returns {object} Стадия узла.
 */
export function getUserNodeStage(node) {
  if (!node?.is_user_group) return getNodeStage(node);
  const phase = normalize(node.layout_phase);
  const type = normalize(node.node_type);
  if (phase === "start" || type.includes("start")) return { id: "start", label: "Start" };
  if (phase === "context" || type.includes("context")) return { id: "context", label: "Context" };
  if (phase === "plan" || type.includes("plan")) return { id: "plan", label: "Plan" };
  if (phase === "tasks" || type.includes("task")) return { id: "work", label: "Task" };
  if (phase === "final" || type.includes("final")) return { id: "answer", label: "Answer" };
  return getNodeStage(node);
}

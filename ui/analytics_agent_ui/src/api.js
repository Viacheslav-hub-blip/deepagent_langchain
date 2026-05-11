/**
 * API-клиент для нового UI агента аналитики.
 *
 * Содержит:
 * - getApiBase: возвращает базовый путь API.
 * - setApiBase: сохраняет базовый путь API.
 * - apiFetch: выполняет JSON-запрос к API.
 * - startLiveRun: создает live run агента.
 * - invokeRun: запускает совместимый синхронный run.
 * - fetchRuns: загружает историю запусков.
 * - fetchRunGraph: загружает граф выбранного run.
 * - fetchRunResult: загружает итоговый отчет run.
 * - fetchRunArtifacts: загружает artifacts run.
 * - fetchSkills: загружает доступные skills.
 * - artifactFileUrl: формирует ссылку скачивания artifact.
 */

const API_BASE_STORAGE_KEY = "analyticsAgentApiBase";

/**
 * Возвращает базовый URL API из localStorage или значение по умолчанию.
 *
 * @returns {string} Базовый путь API.
 */
export function getApiBase() {
  return localStorage.getItem(API_BASE_STORAGE_KEY) || "/api/v1";
}

/**
 * Сохраняет нормализованный базовый URL API.
 *
 * @param {string} value Новый базовый путь API.
 * @returns {string} Сохраненный базовый путь API.
 */
export function setApiBase(value) {
  const normalized = String(value || "/api/v1").trim() || "/api/v1";
  localStorage.setItem(API_BASE_STORAGE_KEY, normalized);
  return normalized;
}

/**
 * Выполняет запрос к backend API и разбирает JSON или текстовый ответ.
 *
 * @param {string} path Относительный путь endpoint.
 * @param {RequestInit} options Настройки fetch.
 * @returns {Promise<unknown>} Тело успешного ответа.
 * @throws {Error} Ошибка HTTP или текст ошибки backend.
 */
export async function apiFetch(path, options = {}) {
  const base = getApiBase().replace(/\/$/, "");
  const response = await fetch(`${base}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    const detail = typeof payload === "object"
      ? payload.detail || JSON.stringify(payload)
      : payload;
    throw new Error(detail || `HTTP ${response.status}`);
  }

  return payload;
}

/**
 * Создает live run агента для последующего polling графа.
 *
 * @param {object} payload Параметры запуска агента.
 * @returns {Promise<object>} Ответ с run_id и ResearchRun.
 */
export function startLiveRun(payload) {
  return apiFetch("/runs/live", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Запускает fallback endpoint агента, если live endpoint недоступен.
 *
 * @param {object} payload Параметры запуска агента.
 * @returns {Promise<object>} Ответ синхронного запуска.
 */
export function invokeRun(payload) {
  return apiFetch("/runs/invoke", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Загружает историю сохраненных запусков.
 *
 * @returns {Promise<Array>} Список RunSummary.
 */
export function fetchRuns() {
  return apiFetch("/runs");
}

/**
 * Загружает lineage graph выбранного run.
 *
 * @param {string} runId Идентификатор run.
 * @returns {Promise<object>} RunGraph payload.
 */
export function fetchRunGraph(runId) {
  return apiFetch(`/runs/${encodeURIComponent(runId)}/graph`);
}

/**
 * Загружает итоговый результат выбранного run.
 *
 * @param {string} runId Идентификатор run.
 * @returns {Promise<object>} RunResult payload.
 */
export function fetchRunResult(runId) {
  return apiFetch(`/runs/${encodeURIComponent(runId)}/result`);
}

/**
 * Загружает список artifacts выбранного run.
 *
 * @param {string} runId Идентификатор run.
 * @returns {Promise<Array>} Список Artifact.
 */
export function fetchRunArtifacts(runId) {
  return apiFetch(`/runs/${encodeURIComponent(runId)}/artifacts`);
}

/**
 * Загружает список skills, доступных агенту.
 *
 * @returns {Promise<object>} Ответ SkillListView.
 */
export function fetchSkills() {
  return apiFetch("/skills");
}

/**
 * Формирует прямую ссылку скачивания artifact.
 *
 * @param {string} runId Идентификатор run.
 * @param {string} artifactId Идентификатор artifact.
 * @returns {string} URL скачивания файла artifact.
 */
export function artifactFileUrl(runId, artifactId) {
  const base = getApiBase().replace(/\/$/, "");
  return `${base}/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(artifactId)}/file`;
}

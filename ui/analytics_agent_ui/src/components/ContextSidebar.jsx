/**
 * Правый bar контекста агента.
 *
 * Содержит:
 * - artifactName: выбирает имя artifact.
 * - artifactMeta: форматирует метаданные artifact.
 * - artifactViewerKind: выбирает режим просмотра artifact.
 * - isTextArtifact: проверяет, можно ли читать artifact как текст.
 * - ContextSidebar: отображает skills и artifacts активного run.
 */
import { useEffect, useState } from "react";
import { BookOpen, Boxes, Download, Eye, FileText, Layers, Loader2 } from "lucide-react";
import {
  artifactFileUrl,
  fetchArtifactBlob,
  fetchArtifactText,
  fetchSkill,
} from "../api.js";
import { ContentViewerModal } from "./ContentViewerModal.jsx";

const EMPTY_VIEWER = {
  open: false,
  title: "",
  subtitle: "",
  kind: "text",
  mimeType: "",
  content: "",
  blobUrl: "",
  downloadUrl: "",
  loading: false,
  error: "",
};

/**
 * Возвращает короткое имя artifact для UI.
 *
 * @param {object} artifact Artifact payload.
 * @returns {string} Имя или идентификатор artifact.
 */
function artifactName(artifact) {
  const meta = artifact?.metadata || {};
  return String(meta.original_filename || meta.filename || meta.export_filename || artifact?.artifact_id || "artifact");
}

/**
 * Форматирует основные технические метаданные artifact.
 *
 * @param {object} artifact Artifact payload.
 * @returns {string} Строка метаданных.
 */
function artifactMeta(artifact) {
  return [artifact?.kind, artifact?.mime_type].filter(Boolean).join(" · ") || "artifact";
}

/**
 * Проверяет, относится ли artifact к текстовым типам для полного просмотра.
 *
 * @param {object} artifact Artifact payload.
 * @returns {boolean} True для text, markdown, json, csv и jsonl.
 */
function isTextArtifact(artifact) {
  const mime = String(artifact?.mime_type || "").toLowerCase();
  const name = artifactName(artifact).toLowerCase();
  return (
    mime.startsWith("text/") ||
    ["application/json", "application/jsonl", "application/x-jsonlines"].includes(mime) ||
    /\.(md|markdown|txt|json|jsonl|csv|tsv)$/i.test(name)
  );
}

/**
 * Выбирает режим viewer для artifact по MIME-типу и имени файла.
 *
 * @param {object} artifact Artifact payload.
 * @returns {string} Тип viewer: image, pdf, markdown, text или fallback.
 */
function artifactViewerKind(artifact) {
  const mime = String(artifact?.mime_type || "").toLowerCase();
  const name = artifactName(artifact).toLowerCase();
  if (mime.startsWith("image/")) return "image";
  if (mime === "application/pdf" || name.endsWith(".pdf")) return "pdf";
  if (mime.includes("markdown") || /\.(md|markdown)$/i.test(name)) return "markdown";
  if (isTextArtifact(artifact)) return "text";
  return "fallback";
}

/**
 * Отображает skills агента и artifacts выбранного запуска.
 *
 * @param {object} props Свойства компонента.
 * @param {Array} props.skills Список SkillRecord.
 * @param {Array} props.artifacts Список Artifact.
 * @param {boolean} props.skillsLoading Признак загрузки skills.
 * @param {string} props.skillsError Ошибка загрузки skills.
 * @param {string} props.runId Активный run_id.
 * @returns {JSX.Element} Правый sidebar контекста.
 */
export function ContextSidebar({ skills, artifacts, skillsLoading, skillsError, runId }) {
  const [viewer, setViewer] = useState(EMPTY_VIEWER);

  useEffect(() => {
    return () => {
      if (viewer.blobUrl) {
        URL.revokeObjectURL(viewer.blobUrl);
      }
    };
  }, [viewer.blobUrl]);

  /**
   * Закрывает viewer и очищает временные данные preview.
   *
   * @returns {void}
   */
  function closeViewer() {
    setViewer(EMPTY_VIEWER);
  }

  /**
   * Открывает модальное окно с полным содержимым skill.
   *
   * @param {object} skill SkillRecord из списка skills.
   * @returns {Promise<void>} Завершается после загрузки содержимого.
   */
  async function openSkill(skill) {
    setViewer({
      ...EMPTY_VIEWER,
      open: true,
      title: skill.name,
      subtitle: "Skill.md",
      kind: "skill",
      loading: true,
    });

    try {
      const payload = await fetchSkill(skill.name);
      setViewer((current) => ({
        ...current,
        content: String(payload?.content || ""),
        loading: false,
      }));
    } catch (error) {
      setViewer((current) => ({
        ...current,
        error: error.message || "Не удалось загрузить skill.",
        loading: false,
      }));
    }
  }

  /**
   * Открывает artifact в подходящем режиме просмотра.
   *
   * @param {object} artifact Artifact payload.
   * @returns {Promise<void>} Завершается после загрузки preview.
   */
  async function openArtifact(artifact) {
    if (!runId) {
      return;
    }

    const title = artifactName(artifact);
    const mimeType = artifact?.mime_type || "";
    const kind = artifactViewerKind(artifact);
    const downloadUrl = artifactFileUrl(runId, artifact.artifact_id);

    setViewer({
      ...EMPTY_VIEWER,
      open: true,
      title,
      subtitle: artifactMeta(artifact),
      kind,
      mimeType,
      downloadUrl,
      loading: kind !== "fallback",
      content: kind === "fallback" ? "Preview для этого типа artifact недоступен прямо в UI." : "",
    });

    if (kind === "fallback") {
      return;
    }

    try {
      if (kind === "image" || kind === "pdf") {
        const payload = await fetchArtifactBlob(runId, artifact.artifact_id);
        setViewer((current) => ({
          ...current,
          blobUrl: payload.objectUrl,
          mimeType: payload.mimeType || mimeType,
          loading: false,
        }));
        return;
      }

      const payload = await fetchArtifactText(runId, artifact.artifact_id);
      setViewer((current) => ({
        ...current,
        content: payload?.content ?? "Artifact недоступен как UTF-8 текст.",
        loading: false,
      }));
    } catch (error) {
      setViewer((current) => ({
        ...current,
        error: error.message || "Не удалось загрузить artifact.",
        loading: false,
      }));
    }
  }

  return (
    <aside className="side-panel context-panel">
      <div className="panel-head">
        <div>
          <span className="panel-kicker">Context</span>
          <h2>Skills и артефакты</h2>
        </div>
        <Layers size={20} />
      </div>

      <section className="context-section">
        <div className="context-section-title">
          <BookOpen size={16} />
          <span>Загруженные skills</span>
          {skillsLoading ? <Loader2 className="spin" size={15} /> : null}
        </div>
        {skillsError ? <div className="panel-error">{skillsError}</div> : null}
        <div className="skill-list">
          {!skills.length && !skillsLoading ? (
            <div className="empty-panel-state">
              <BookOpen size={20} />
              <span>Skills не найдены</span>
            </div>
          ) : null}
          {skills.map((skill) => (
            <article key={skill.name} className="skill-card">
              <strong>{skill.name}</strong>
              <p>{skill.description || "Описание отсутствует."}</p>
              {skill.category ? <small>{skill.category}</small> : null}
              <button type="button" className="context-card-action" onClick={() => openSkill(skill)}>
                <Eye size={14} />
                Просмотр
              </button>
            </article>
          ))}
        </div>
      </section>

      <section className="context-section context-section--artifacts">
        <div className="context-section-title">
          <Boxes size={16} />
          <span>Артефакты run</span>
        </div>
        <div className="artifact-list">
          {!runId ? (
            <div className="empty-panel-state">
              <FileText size={20} />
              <span>Выберите или запустите чат</span>
            </div>
          ) : null}
          {runId && !artifacts.length ? (
            <div className="empty-panel-state">
              <FileText size={20} />
              <span>Артефакты пока не созданы</span>
            </div>
          ) : null}
          {artifacts.map((artifact) => (
            <article key={artifact.artifact_id} className="artifact-card">
              <FileText size={16} />
              <span>
                <strong>{artifactName(artifact)}</strong>
                <small>{artifactMeta(artifact)}</small>
              </span>
              <div className="artifact-card-actions">
                <button
                  type="button"
                  className="context-icon-action"
                  onClick={() => openArtifact(artifact)}
                  title="Открыть artifact в UI"
                  aria-label="Открыть artifact в UI"
                >
                  <Eye size={15} />
                </button>
                <a
                  className="context-icon-action"
                  href={artifactFileUrl(runId, artifact.artifact_id)}
                  download
                  title="Скачать artifact"
                  aria-label="Скачать artifact"
                >
                  <Download size={15} />
                </a>
              </div>
            </article>
          ))}
        </div>
      </section>
      <ContentViewerModal
        open={viewer.open}
        title={viewer.title}
        subtitle={viewer.subtitle}
        kind={viewer.kind}
        mimeType={viewer.mimeType}
        content={viewer.content}
        blobUrl={viewer.blobUrl}
        downloadUrl={viewer.downloadUrl}
        loading={viewer.loading}
        error={viewer.error}
        onClose={closeViewer}
      />
    </aside>
  );
}

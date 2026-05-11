/**
 * Правый bar контекста агента.
 *
 * Содержит:
 * - artifactName: выбирает имя artifact.
 * - artifactMeta: форматирует метаданные artifact.
 * - ContextSidebar: отображает skills и artifacts активного run.
 */
import { BookOpen, Boxes, Download, FileText, Layers, Loader2 } from "lucide-react";
import { artifactFileUrl } from "../api.js";

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
            <a
              key={artifact.artifact_id}
              className="artifact-card"
              href={artifactFileUrl(runId, artifact.artifact_id)}
              download
            >
              <FileText size={16} />
              <span>
                <strong>{artifactName(artifact)}</strong>
                <small>{artifactMeta(artifact)}</small>
              </span>
              <Download size={15} />
            </a>
          ))}
        </div>
      </section>
    </aside>
  );
}

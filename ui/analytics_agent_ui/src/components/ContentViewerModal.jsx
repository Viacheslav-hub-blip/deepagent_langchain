/**
 * Модальное окно просмотра skills и artifacts.
 *
 * Содержит:
 * - isMarkdownContent: определяет markdown-содержимое.
 * - ContentViewerModal: показывает текст, markdown, изображения, PDF и fallback для скачивания.
 */
import { Download, FileText, Loader2, X } from "lucide-react";
import { MarkdownBlock } from "./MarkdownBlock.jsx";

/**
 * Проверяет, нужно ли отображать содержимое через markdown renderer.
 *
 * @param {string} kind Тип содержимого viewer.
 * @param {string} mimeType MIME-тип artifact или пустая строка.
 * @returns {boolean} True, если содержимое лучше показывать как markdown.
 */
function isMarkdownContent(kind, mimeType) {
  const normalizedKind = String(kind || "").toLowerCase();
  const normalizedMime = String(mimeType || "").toLowerCase();
  return (
    normalizedKind === "skill" ||
    normalizedKind === "markdown" ||
    normalizedMime.includes("markdown")
  );
}

/**
 * Показывает полное содержимое skill или artifact без выхода из UI.
 *
 * @param {object} props Свойства компонента.
 * @param {boolean} props.open Открыто ли окно.
 * @param {string} props.title Заголовок окна.
 * @param {string} props.subtitle Дополнительное описание содержимого.
 * @param {string} props.kind Тип содержимого: skill, markdown, text, image, pdf или fallback.
 * @param {string} props.mimeType MIME-тип artifact.
 * @param {string} props.content Текстовое содержимое.
 * @param {string} props.blobUrl Object URL для image/PDF preview.
 * @param {string} props.downloadUrl Ссылка скачивания artifact.
 * @param {boolean} props.loading Признак загрузки.
 * @param {string} props.error Текст ошибки загрузки.
 * @param {Function} props.onClose Обработчик закрытия.
 * @returns {JSX.Element | null} Модальное окно просмотра или null.
 */
export function ContentViewerModal({
  open,
  title,
  subtitle,
  kind,
  mimeType,
  content,
  blobUrl,
  downloadUrl,
  loading,
  error,
  onClose,
}) {
  if (!open) {
    return null;
  }

  const normalizedKind = String(kind || "text").toLowerCase();
  const showMarkdown = isMarkdownContent(normalizedKind, mimeType);

  return (
    <div className="content-viewer-overlay" role="presentation" onMouseDown={onClose}>
      <section
        className="content-viewer-modal"
        role="dialog"
        aria-modal="true"
        aria-label={title || "Просмотр содержимого"}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="content-viewer-header">
          <div className="content-viewer-title">
            <FileText size={18} />
            <div>
              <h2>{title || "Просмотр"}</h2>
              {subtitle ? <p>{subtitle}</p> : null}
            </div>
          </div>
          <div className="content-viewer-actions">
            {downloadUrl ? (
              <a className="content-viewer-icon-button" href={downloadUrl} download title="Скачать файл">
                <Download size={17} />
              </a>
            ) : null}
            <button
              type="button"
              className="content-viewer-icon-button"
              onClick={onClose}
              title="Закрыть"
              aria-label="Закрыть"
            >
              <X size={18} />
            </button>
          </div>
        </header>

        <div className="content-viewer-body">
          {loading ? (
            <div className="content-viewer-state">
              <Loader2 className="spin" size={22} />
              <span>Загружаю содержимое...</span>
            </div>
          ) : error ? (
            <div className="panel-error content-viewer-error">{error}</div>
          ) : normalizedKind === "image" && blobUrl ? (
            <img className="content-viewer-image" src={blobUrl} alt={title || "Artifact preview"} />
          ) : normalizedKind === "pdf" && blobUrl ? (
            <iframe className="content-viewer-frame" src={blobUrl} title={title || "Artifact PDF"} />
          ) : showMarkdown ? (
            <MarkdownBlock>{content}</MarkdownBlock>
          ) : content ? (
            <pre className="content-viewer-pre">{content}</pre>
          ) : (
            <div className="content-viewer-state">
              <FileText size={22} />
              <span>Preview для этого типа файла недоступен. Используйте скачивание.</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

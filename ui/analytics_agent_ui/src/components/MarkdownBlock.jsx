import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Рендер markdown (ответы модели, отчёты). Стили: .report-markdown.
 */
export function MarkdownBlock({ children, className = "" }) {
  const text = typeof children === "string" ? children : "";
  if (!text.trim()) {
    return null;
  }
  return (
    <div className={`report-markdown markdown-block ${className}`.trim()}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}

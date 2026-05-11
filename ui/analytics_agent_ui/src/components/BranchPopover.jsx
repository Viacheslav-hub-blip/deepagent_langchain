import { useEffect, useState } from "react";
import { GitBranch, Loader2, X } from "lucide-react";

export function BranchPopover({
  node,
  position,
  busy,
  error,
  onCreateBranch,
  onClose,
  zoom = 1,
}) {
  const [newTask, setNewTask] = useState("");
  const [branchMode, setBranchMode] = useState("what_if");

  useEffect(() => {
    setNewTask("");
    setBranchMode("what_if");
  }, [node?.node_id]);

  if (!node || !position) {
    return null;
  }

  async function submit(event) {
    event.preventDefault();

    const ok = await onCreateBranch({
      userNodeId: node.node_id,
      newTask,
      branchMode,
    });

    if (ok) {
      setNewTask("");
      onClose();
    }
  }

  return (
    <form
      className="branch-popover"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        "--branch-popover-inverse-scale": String(1 / Math.max(0.1, zoom)),
      }}
      onSubmit={submit}
      onMouseDown={(event) => event.stopPropagation()}
      onClick={(event) => event.stopPropagation()}
    >
      <div className="branch-popover-header">
        <div className="branch-popover-title">
          <GitBranch size={15} />
          <span>Новый branch от узла</span>
        </div>
        <button type="button" className="branch-popover-close" onClick={onClose}>
          <X size={15} />
        </button>
      </div>

      <div className="branch-source">
        <strong>{node.title || node.node_type}</strong>
        <span>{node.raw_event_count ? `${node.raw_event_count} raw events` : node.node_type}</span>
      </div>

      <label className="branch-field">
        <span>Что проверить в новой ветке?</span>
        <textarea
          value={newTask}
          onChange={(event) => setNewTask(event.target.value)}
          placeholder="Например: проверь альтернативную гипотезу по устройству клиента"
          rows={4}
          autoFocus
          disabled={busy}
        />
      </label>

      <label className="branch-field">
        <span>Branch mode</span>
        <select value={branchMode} onChange={(event) => setBranchMode(event.target.value)} disabled={busy}>
          <option value="what_if">what_if</option>
        </select>
      </label>

      {error ? <div className="branch-error">{error}</div> : null}

      <button type="submit" className="branch-submit" disabled={busy || !newTask.trim()}>
        {busy ? <Loader2 className="spin" size={16} /> : <GitBranch size={16} />}
        {busy ? "Запускаю branch…" : "Запустить branch"}
      </button>
    </form>
  );
}

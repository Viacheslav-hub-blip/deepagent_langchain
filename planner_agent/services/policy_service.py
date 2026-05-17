"""Deny-by-default policy engine for corporate agent execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from planner_agent.schemas.policy import PolicyDecision

from ._json import append_jsonl


EXECUTABLE_DENYLIST = {
    ".exe",
    ".dll",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    ".com",
    ".scr",
    ".msi",
    ".vbs",
    ".js",
    ".jar",
    ".pyz",
    ".app",
    ".dmg",
}

DEFAULT_DENIED_TOOLS = {
    "web_search",
    "web_extract",
    "browser",
    "browser_open",
    "browser_click",
    "browser_type",
    "terminal",
    "raw_shell",
    "shell",
    "write_file",
    "patch",
    "execute_code",
}

DEFAULT_ALLOWED_TOOLS = {
    "execute_python_code",
    "read_table",
    "list_skills",
    "load_skill",
}

WRITE_PATH_KEYS = ("path", "file_path", "output_path", "filename")


class PolicyEngine:
    def __init__(
        self,
        *,
        allowed_tools: set[str] | None = None,
        denied_tools: set[str] | None = None,
        runs_dir: str | Path = "runs",
    ) -> None:
        self.allowed_tools = allowed_tools or set(DEFAULT_ALLOWED_TOOLS)
        self.denied_tools = denied_tools or set(DEFAULT_DENIED_TOOLS)
        self.runs_dir = Path(runs_dir)

    def evaluate_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
        user_id: str | None = None,
    ) -> PolicyDecision:
        args = args or {}

        if tool_name in self.denied_tools or tool_name.startswith("browser_"):
            return self._decision(tool_name, "deny", "Tool is denied by corporate policy.", args, run_id, node_id, user_id)

        executable_path = self._find_executable_write(args)
        if executable_path:
            return self._decision(
                tool_name,
                "deny",
                f"Executable persistent writes are denied: {executable_path}",
                args,
                run_id,
                node_id,
                user_id,
            )

        if tool_name not in self.allowed_tools:
            return self._decision(tool_name, "review", "Tool is not in the approved MVP allowlist.", args, run_id, node_id, user_id)

        return self._decision(tool_name, "allow", "Tool is approved.", args, run_id, node_id, user_id)

    def audit(self, decision: PolicyDecision) -> PolicyDecision:
        append_jsonl(self.runs_dir / "audit.jsonl", decision)
        if decision.run_id:
            append_jsonl(self.runs_dir / decision.run_id / "audit.jsonl", decision)
        return decision

    def evaluate_and_audit(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
        user_id: str | None = None,
    ) -> PolicyDecision:
        return self.audit(
            self.evaluate_tool_call(
                tool_name,
                args,
                run_id=run_id,
                node_id=node_id,
                user_id=user_id,
            )
        )

    @staticmethod
    def _find_executable_write(args: dict[str, Any]) -> str | None:
        for key in WRITE_PATH_KEYS:
            value = args.get(key)
            if not value:
                continue
            suffix = Path(str(value)).suffix.lower()
            if suffix in EXECUTABLE_DENYLIST:
                return str(value)
        return None

    @staticmethod
    def _decision(
        tool_name: str,
        decision: str,
        reason: str,
        args: dict[str, Any],
        run_id: str | None,
        node_id: str | None,
        user_id: str | None,
    ) -> PolicyDecision:
        return PolicyDecision(
            run_id=run_id,
            node_id=node_id,
            user_id=user_id,
            tool_name=tool_name,
            decision=decision,  # type: ignore[arg-type]
            reason=reason,
            metadata={"args_preview": args},
        )


__all__ = [
    "DEFAULT_ALLOWED_TOOLS",
    "DEFAULT_DENIED_TOOLS",
    "EXECUTABLE_DENYLIST",
    "PolicyEngine",
]

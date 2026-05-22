"""Host-side Python sandbox for research-agent workers.

Exports:
- ClientPythonSandbox: executes generated Python in a restricted environment.
"""

from .sandbox import ClientPythonSandbox

__all__ = ["ClientPythonSandbox"]

"""Sandbox adapter boundary.

The current graph accepts a ``ClientPythonSandbox`` instance from the host
application. This module documents the boundary for v2 without importing the
external sandbox package at module import time.
"""

from __future__ import annotations

from typing import Any, Protocol


class PythonSandboxProtocol(Protocol):
    last_dataframe_variable: str | None
    globals: dict[str, Any]
    working_directory: Any | None

    async def get_all_variable_previews(self) -> dict[str, str]:
        ...

    async def add_variable(self, name: str, value: Any) -> None:
        ...

    async def get_variable(self, name: str) -> Any:
        ...

    @staticmethod
    def get_installed_packages() -> dict[str, str]:
        """Возвращает словарь {имя_пакета: версия} установленных pip-пакетов.

        Используется в системном промпте worker-а, чтобы модель знала,
        какие библиотеки доступны для импорта при генерации кода.
        """
        ...


__all__ = ["PythonSandboxProtocol"]

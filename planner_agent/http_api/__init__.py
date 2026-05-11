"""Опциональный HTTP-слой (FastAPI) поверх ResearchAgent и RunInspectionService.

Требует зависимостей ``[api]`` из pyproject (fastapi, uvicorn). Модуль отдает
только SDK/API endpoints и не монтирует frontend-статику.
"""

from __future__ import annotations

from .app import create_app
from .config import ApiServices, ApiSettings, build_api_services

__all__ = ["ApiServices", "ApiSettings", "build_api_services", "create_app"]

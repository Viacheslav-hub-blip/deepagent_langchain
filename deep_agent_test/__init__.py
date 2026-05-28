"""Пакет production-ready аналитического native DeepAgents агента.

Содержит:
- build_analytics_deep_agent: сборка аналитического DeepAgent.
- load_deep_agent_settings: загрузка настроек агента из JSON-конфига.
"""

from deep_agent_test.analytics_deep_agent import build_analytics_deep_agent
from deep_agent_test.settings import load_deep_agent_settings

__all__ = ["build_analytics_deep_agent", "load_deep_agent_settings"]

"""Middleware вывода prompt-ов перед первичным планированием агента.

Содержит:
- PromptDebugConsoleMiddleware: middleware печати system/user prompt перед первым вызовом supervisor-модели.
- PromptDebugConsoleMiddleware.wrap_model_call: вывод prompt-ов и передача запроса дальше без изменений.
- _last_user_prompt: извлечение последнего пользовательского prompt из истории сообщений.
- _message_content_to_text: преобразование содержимого LangChain message в текст.
- _system_prompt_to_text: преобразование system prompt из ModelRequest в текст.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

from deep_agent_test.plan_approval_middleware import INTERNAL_RUNNER_MESSAGE_PREFIX


@dataclass
class PromptDebugConsoleMiddleware(AgentMiddleware):
    """Печатает system prompt и user prompt перед первым вызовом модели supervisor-а.

    Args:
        enabled: Нужно ли печатать prompt-ы в консоль.

    Returns:
        Middleware, которое не меняет запрос модели и используется только для диагностики.
    """

    enabled: bool = False
    _printed: bool = False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Выводит prompt-ы перед первичным планированием и вызывает модель.

        Args:
            request: Запрос модели с итоговым system prompt, сообщениями и state.
            handler: Функция реального вызова модели.

        Returns:
            Ответ модели без изменений.
        """

        if self.enabled and not self._printed and not request.state.get("approved_plan_user_key"):
            self._printed = True
            print()
            print("Агент:")
            print("System prompt перед генерацией плана:")
            print(_system_prompt_to_text(request.system_message))
            print()
            print("User prompt перед генерацией плана:")
            print(_last_user_prompt(request.messages))
            print()
        return handler(request)


def _last_user_prompt(messages: list[Any]) -> str:
    """Извлекает последний пользовательский prompt из истории сообщений.

    Args:
        messages: Сообщения из ``ModelRequest.messages``.

    Returns:
        Текст последнего настоящего пользовательского сообщения или пустую строку.
    """

    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = _message_content_to_text(message.content).strip()
        if not content or content.startswith(INTERNAL_RUNNER_MESSAGE_PREFIX):
            continue
        return content
    return ""


def _message_content_to_text(content: Any) -> str:
    """Преобразует содержимое сообщения LangChain в текст.

    Args:
        content: Строка, список content-блоков или произвольное значение.

    Returns:
        Текстовое представление содержимого сообщения.
    """

    if isinstance(content, list):
        return "\n".join(_message_content_to_text(item) for item in content)
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        return str(content)
    if content is None:
        return ""
    return str(content)


def _system_prompt_to_text(system_message: Any) -> str:
    """Преобразует system prompt из ``ModelRequest`` в текст.

    Args:
        system_message: System prompt в формате строки, сообщения LangChain или другого объекта.

    Returns:
        Текст system prompt, который передается модели.
    """

    content = getattr(system_message, "content", system_message)
    return _message_content_to_text(content)


__all__ = ["PromptDebugConsoleMiddleware"]

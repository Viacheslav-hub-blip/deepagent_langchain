"""Переопределение описаний инструментов перед вызовом модели.

Содержит:
- PromptToolDescriptionsMiddleware: middleware для замены tool descriptions и
  добавления уточняющего prompt-блока.
- _rewrite_tool_descriptions: копирование tool metadata с новыми описаниями.
- _append_system_prompt: добавление русского prompt-блока к system message.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class PromptToolDescriptionsMiddleware(AgentMiddleware):
    """Заменяет описания tools и добавляет уточняющий prompt-блок.

    Args:
        tool_descriptions: Новые описания tools, где ключ — имя инструмента.
        system_prompt_append: Дополнительные инструкции, которые нужно добавить в
            system prompt перед вызовом модели.

    Returns:
        Middleware, который не меняет реализацию tools, а меняет только prompt-level
        metadata, видимую модели.
    """

    tool_descriptions: Mapping[str, str]
    system_prompt_append: str = ""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Синхронно подменяет описания инструментов в запросе к модели.

        Args:
            request: Исходный запрос к модели с tools и system message.
            handler: Следующий обработчик model call.

        Returns:
            Ответ модели после вызова следующего обработчика.
        """

        return handler(self._override_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Асинхронно подменяет описания инструментов в запросе к модели.

        Args:
            request: Исходный запрос к модели с tools и system message.
            handler: Следующий асинхронный обработчик model call.

        Returns:
            Ответ модели после вызова следующего обработчика.
        """

        return await handler(self._override_request(request))

    def _override_request(self, request: ModelRequest) -> ModelRequest:
        """Создаёт копию запроса с обновлёнными descriptions и prompt-блоком.

        Args:
            request: Исходный запрос к модели.

        Returns:
            Новый ``ModelRequest`` с prompt-only изменениями.
        """

        tools = _rewrite_tool_descriptions(request.tools, self.tool_descriptions)
        system_message = _append_system_prompt(request.system_message, self.system_prompt_append)
        return request.override(tools=tools, system_message=system_message)


def _rewrite_tool_descriptions(
    tools: list[BaseTool | dict[str, Any]],
    descriptions: Mapping[str, str],
) -> list[BaseTool | dict[str, Any]]:
    """Возвращает копию списка tools с переопределёнными descriptions.

    Args:
        tools: Инструменты, которые будут переданы модели.
        descriptions: Новые описания по имени инструмента.

    Returns:
        Новый список tools; сами функции инструментов не изменяются.
    """

    rewritten: list[BaseTool | dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        description = descriptions.get(str(name)) if name else None
        if description is None:
            rewritten.append(tool)
            continue
        if isinstance(tool, dict):
            copied = dict(tool)
            copied["description"] = description
            rewritten.append(copied)
        else:
            rewritten.append(tool.model_copy(update={"description": description}))
    return rewritten


def _append_system_prompt(system_message: SystemMessage | None, text: str) -> SystemMessage | None:
    """Добавляет текстовый блок к system message.

    Args:
        system_message: Текущее системное сообщение или ``None``.
        text: Текст, который нужно добавить в конец system prompt.

    Returns:
        Обновлённое системное сообщение или исходное значение, если текст пустой.
    """

    if not text.strip():
        return system_message
    if system_message is None:
        return SystemMessage(content=text)
    return SystemMessage(
        content=[
            *system_message.content_blocks,
            {"type": "text", "text": f"\n\n{text}"},
        ]
    )


__all__ = ["PromptToolDescriptionsMiddleware"]

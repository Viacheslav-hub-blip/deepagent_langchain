"""Утилиты для получения структурированного JSON-ответа от chat model.

Содержит:
- _content_to_text: преобразование content блока LangChain message в текст.
- _response_to_text: извлечение текста из ответа модели.
- _strip_code_fences: удаление markdown code fences.
- _extract_balanced_json: поиск сбалансированного JSON внутри текста.
- _json_candidates: построение кандидатов для JSON-парсинга.
- parse_structured_response: валидация ответа модели по Pydantic-схеме.
- invoke_structured_output: вызов модели и получение Pydantic-объекта.
"""

import json
from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel


StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

# OpenRouter-совместимые модели могут зависать на OpenAI parse endpoint,
# поэтому fallback на with_structured_output по умолчанию выключен.
ENABLE_STRUCTURED_OUTPUT_FALLBACK: bool = False
STRUCTURED_OUTPUT_PARSE_RETRIES: int = 1


def _content_to_text(content: object) -> str:
    """Преобразует содержимое LangChain message в строку.

    Args:
        content: Строка, список content blocks или произвольный объект.

    Returns:
        Текстовое представление содержимого.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                nested_content = item.get("content")
                if isinstance(nested_content, str):
                    parts.append(nested_content)
                    continue
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def _response_to_text(response: object) -> str:
    """Извлекает текст из ответа chat model.

    Args:
        response: AIMessage, dict или произвольный объект ответа модели.

    Returns:
        Текст, который можно передать в JSON-парсер.
    """

    if isinstance(response, AIMessage):
        return _content_to_text(response.content)

    content = getattr(response, "content", None)
    if content is not None:
        return _content_to_text(content)

    if isinstance(response, dict):
        for key in ("content", "text", "output", "response", "result"):
            if key in response:
                return _content_to_text(response[key])

    return str(response)


def _strip_code_fences(text: str) -> str:
    """Удаляет markdown-обертку ```json ... ``` вокруг ответа.

    Args:
        text: Сырой текст ответа модели.

    Returns:
        Текст без внешних markdown code fences.
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("json\n"):
        cleaned = cleaned[5:].strip()
    elif cleaned.lower() == "json":
        cleaned = ""
    return cleaned


def _extract_balanced_json(text: str) -> str | None:
    """Находит первый сбалансированный JSON-объект или массив в тексте.

    Args:
        text: Текст, который может содержать JSON внутри пояснений модели.

    Returns:
        JSON-фрагмент или ``None``, если подходящий фрагмент не найден.
    """

    openers = {"{": "}", "[": "]"}
    starts = [index for index, char in enumerate(text) if char in openers]

    for start in starts:
        stack: list[str] = []
        in_string = False
        escaped = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue

            if char in openers:
                stack.append(openers[char])
                continue

            if stack and char == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start:index + 1].strip()
                continue

    return None


def _json_candidates(text: str) -> list[str]:
    """Строит список возможных JSON-кандидатов для парсинга.

    Args:
        text: Сырой текст ответа модели.

    Returns:
        Список строк-кандидатов: исходный текст, текст без code fences и
        найденный сбалансированный JSON.
    """

    candidates: list[str] = []

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    without_fences = _strip_code_fences(stripped)
    if without_fences and without_fences not in candidates:
        candidates.append(without_fences)

    balanced = _extract_balanced_json(without_fences or stripped)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    return candidates


def parse_structured_response(schema: type[StructuredModelT], response: object) -> StructuredModelT:
    """Парсит ответ модели в Pydantic-схему.

    Args:
        schema: Pydantic-модель ожидаемого структурированного ответа.
        response: Ответ модели или текст ответа.

    Returns:
        Экземпляр ``schema``.

    Raises:
        ValueError: Если ни один JSON-кандидат не прошел валидацию.
    """

    raw_text = _response_to_text(response)
    errors: list[str] = []

    for candidate in _json_candidates(raw_text):
        try:
            return schema.model_validate_json(candidate)
        except Exception as exc:
            errors.append(str(exc))
            try:
                return schema.model_validate(json.loads(candidate))
            except Exception as nested_exc:
                errors.append(str(nested_exc))

    raise ValueError(
        "Could not parse structured output. "
        f"Raw response: {raw_text[:1000]!r}. "
        f"Errors: {' | '.join(errors[:4]) or 'no JSON candidates found'}"
    )


async def invoke_structured_output(
    llm: BaseChatModel,
    schema: type[StructuredModelT],
    messages: list[BaseMessage],
) -> StructuredModelT:
    """Вызывает модель и возвращает структурированный Pydantic-ответ.

    Используется обычный ``ainvoke`` и локальный JSON-парсинг. Это лучше
    совместимо с OpenRouter и моделями, у которых OpenAI parse endpoint может
    отвечать медленно или нестабильно. Fallback на ``with_structured_output``
    оставлен как флаг для будущей диагностики, но по умолчанию отключен.

    Args:
        llm: Chat model, совместимая с LangChain.
        schema: Pydantic-модель ожидаемого ответа.
        messages: Сообщения, передаваемые в модель.

    Returns:
        Экземпляр ``schema``.

    Raises:
        ValueError: Если не удалось получить или распарсить структурированный
            ответ ни обычным вызовом, ни fallback-вызовом.
    """

    raw_exc: Exception | None = None
    current_messages = list(messages)
    for attempt in range(STRUCTURED_OUTPUT_PARSE_RETRIES + 1):
        try:
            raw_response = await llm.ainvoke(current_messages)
        except Exception:
            raise

        try:
            parsed = parse_structured_response(schema, raw_response)
            return parsed
        except Exception as parse_exc:
            raw_exc = parse_exc
            if attempt >= STRUCTURED_OUTPUT_PARSE_RETRIES:
                break
            current_messages = [
                *messages,
                HumanMessage(
                    content=(
                        "Предыдущий ответ не удалось распарсить как JSON для заданной "
                        "Pydantic-схемы. Повтори ответ заново: верни только один "
                        "валидный JSON-объект без markdown fences, комментариев, "
                        "префиксов, суффиксов и обрезанных строк. Все строковые "
                        "значения должны быть закрыты кавычками."
                    )
                ),
            ]

    assert raw_exc is not None
    if not ENABLE_STRUCTURED_OUTPUT_FALLBACK:
        raise ValueError(
            "Structured output raw parsing failed. "
            "with_structured_output fallback is disabled to avoid OpenRouter hangs. "
            f"Raw parser error: {raw_exc}"
        ) from raw_exc

    try:
        parsed = await llm.with_structured_output(schema).ainvoke(messages)
        return parsed
    except Exception as structured_exc:
        raise ValueError(
            "Structured output failed. "
            f"Raw parser error: {raw_exc}. "
            f"Structured parser error: {structured_exc}"
        ) from structured_exc

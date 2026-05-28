"""Middleware подстановки few-shot примеров в prompt аналитического DeepAgent.

Содержит:
- SelectedFewShotExamples: структурированный результат LLM-выбора примеров.
- FewShotExamplesMiddleware: middleware поиска, выбора и добавления few-shot примеров.
- FewShotExamplesMiddleware.before_agent: поиск и кеширование few-shot примеров.
- FewShotExamplesMiddleware.wrap_model_call: добавление few-shot примеров в system prompt.
- select_few_shot_examples_with_llm: выбор подходящих примеров из top-10 кандидатов.
- load_full_example_markdown: загрузка полного markdown выбранных примеров.
- build_few_shot_prompt_block: сборка prompt-блока с выбранными примерами.
- extract_last_user_query: извлечение последнего пользовательского запроса.
- build_user_query_key: построение ключа пользовательского запроса для кеша.
- _format_candidates_for_rerank: подготовка кандидатов для LLM-реранжирования.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from deepagents.middleware._utils import append_to_system_message

from deep_agent_test.agent_logging import DeepAgentEventLogger
from deep_agent_test.few_shot_examples_index import FewShotExamplesStore, FewShotSearchResult
from deep_agent_test.plan_approval_middleware import AnalyticsPlanState, INTERNAL_RUNNER_MESSAGE_PREFIX
from deep_agent_test.prompts import FEW_SHOT_PROMPT_BLOCK_TEMPLATE, FEW_SHOT_RERANK_SYSTEM_PROMPT


class SelectedFewShotExamples(BaseModel):
    """Результат выбора few-shot примеров через LLM.

    Args:
        names: Названия выбранных примеров. Если подходящих примеров нет, список должен быть пустым.

    Returns:
        Список названий примеров, полные markdown-файлы которых нужно добавить в prompt.
    """

    names: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class FewShotExamplesMiddleware(AgentMiddleware[AnalyticsPlanState]):
    """Находит few-shot примеры и добавляет их в system prompt перед планированием.

    Args:
        store: Хранилище few-shot индекса с векторным поиском.
        model: Chat model для выбора подходящих примеров из top-10 кандидатов.
        top_k: Количество кандидатов после векторного поиска.
        max_examples: Максимальное количество примеров для подстановки в prompt.
        event_logger: Файловый логгер для записи выбранных few-shot примеров.

    Returns:
        Middleware, которое сохраняет выбранные примеры в state и добавляет их в prompt.
    """

    store: FewShotExamplesStore
    model: Any
    top_k: int = 10
    max_examples: int = 3
    event_logger: DeepAgentEventLogger | None = None

    state_schema = AnalyticsPlanState

    def before_agent(
        self,
        state: AnalyticsPlanState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        """Ищет и кеширует few-shot примеры для последнего пользовательского запроса.

        Args:
            state: Текущее состояние агента.
            runtime: Runtime текущего запуска graph.

        Returns:
            Обновление state с выбранными примерами или ``None``, если кеш актуален.
        """

        user_query = extract_last_user_query(state.get("messages", []))
        if not user_query:
            return None

        user_key = build_user_query_key(user_query)
        if state.get("few_shot_examples_user_key") == user_key:
            return None

        candidates = self.store.search(query=user_query, top_k=self.top_k)
        selected = select_few_shot_examples_with_llm(
            model=self.model,
            user_query=user_query,
            candidates=candidates,
            max_examples=self.max_examples,
        )
        examples_block = load_full_example_markdown(selected)
        if self.event_logger is not None:
            self.event_logger.log_few_shot_selection(
                {
                    "user_query": user_query,
                    "candidate_count": len(candidates),
                    "candidates": _format_candidates_for_rerank(candidates),
                    "selected": [candidate.name for candidate in selected],
                }
            )

        return {
            "few_shot_examples_user_key": user_key,
            "few_shot_example_names": [candidate.name for candidate in selected],
            "few_shot_examples": examples_block,
        }

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Добавляет выбранные few-shot примеры в system prompt текущего вызова модели.

        Args:
            request: Запрос модели с текущими сообщениями, tools и state.
            handler: Функция реального вызова модели.

        Returns:
            Ответ модели после вызова handler.
        """

        examples = request.state.get("few_shot_examples")
        if not examples:
            return handler(request)

        system_message = append_to_system_message(
            request.system_message,
            build_few_shot_prompt_block(examples),
        )
        return handler(request.override(system_message=system_message))


def select_few_shot_examples_with_llm(
    model: Any,
    user_query: str,
    candidates: list[FewShotSearchResult],
    max_examples: int,
) -> list[FewShotSearchResult]:
    """Выбирает подходящие few-shot примеры из top-k кандидатов.

    Args:
        model: Chat model, поддерживающая ``with_structured_output``.
        user_query: Последний пользовательский запрос.
        candidates: Кандидаты после векторного поиска.
        max_examples: Максимальное количество выбранных примеров.

    Returns:
        Список выбранных кандидатов в порядке исходного ранжирования.
    """

    if not candidates or max_examples <= 0:
        return []

    structured_model = model.with_structured_output(SelectedFewShotExamples)
    result = structured_model.invoke(
        [
            SystemMessage(content=FEW_SHOT_RERANK_SYSTEM_PROMPT),
            HumanMessage(
                content=json.dumps(
                    {
                        "user_query": user_query,
                        "max_examples": max_examples,
                        "candidates": _format_candidates_for_rerank(candidates),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ]
    )

    selected_names = set(result.names[:max_examples])
    if not selected_names:
        return []

    selected: list[FewShotSearchResult] = []
    for candidate in candidates:
        if candidate.name in selected_names:
            selected.append(candidate)
        if len(selected) >= max_examples:
            break
    return selected


def load_full_example_markdown(selected: list[FewShotSearchResult]) -> str:
    """Загружает полное содержимое markdown-файлов выбранных примеров.

    Args:
        selected: Список выбранных кандидатов с абсолютными путями к файлам.

    Returns:
        Markdown-блок с полным содержимым выбранных примеров.
    """

    blocks: list[str] = []
    for candidate in selected:
        text = Path(candidate.absolute_path).read_text(encoding="utf-8").strip()
        blocks.append(f"## Few-shot example: {candidate.name}\n\n{text}")
    return "\n\n".join(blocks)


def build_few_shot_prompt_block(examples_markdown: str) -> str:
    """Формирует блок few-shot примеров для system prompt.

    Args:
        examples_markdown: Полный markdown выбранных few-shot примеров.

    Returns:
        Готовый текстовый блок для добавления в system prompt.
    """

    return FEW_SHOT_PROMPT_BLOCK_TEMPLATE.format(examples_markdown=examples_markdown)


def extract_last_user_query(messages: list[Any]) -> str:
    """Извлекает последний настоящий пользовательский запрос из истории сообщений.

    Args:
        messages: История сообщений LangChain agent state.

    Returns:
        Текст последнего HumanMessage, который не является служебной инструкцией runner-а.
    """

    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        content = str(message.content).strip()
        if not content or content.startswith(INTERNAL_RUNNER_MESSAGE_PREFIX):
            continue
        return content
    return ""


def build_user_query_key(user_query: str) -> str:
    """Строит стабильный ключ пользовательского запроса для кеширования few-shot примеров.

    Args:
        user_query: Текст пользовательского запроса.

    Returns:
        sha256-ключ нормализованного текста запроса.
    """

    normalized_query = " ".join(user_query.split())
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()


def _format_candidates_for_rerank(candidates: list[FewShotSearchResult]) -> list[dict[str, Any]]:
    """Подготавливает кандидатов для LLM-реранжирования.

    Args:
        candidates: Список кандидатов после векторного поиска.

    Returns:
        Список словарей с названием, описанием и vector_score.
    """

    return [
        {
            "name": candidate.name,
            "description": candidate.description,
            "vector_score": round(candidate.vector_score, 6),
        }
        for candidate in candidates
    ]


__all__ = [
    "FewShotExamplesMiddleware",
    "SelectedFewShotExamples",
    "build_few_shot_prompt_block",
    "build_user_query_key",
    "extract_last_user_query",
    "load_full_example_markdown",
    "select_few_shot_examples_with_llm",
]

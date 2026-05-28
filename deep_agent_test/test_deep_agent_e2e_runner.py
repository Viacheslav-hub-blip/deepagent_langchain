"""E2E-тесты native DeepAgent с авто-approve плана и подробным трейсом.

Как использовать:
1. Укажи свои запросы в ``E2E_REQUESTS`` ниже.
2. Запусти тест с ``RUN_DEEP_AGENT_E2E=1``.
3. В консоли будут видны: сгенерированный план, вызванные инструменты и финальный ответ.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any

from deep_agent_test.run_native_analytics_chat import (
    build_chat_agent,
    build_test_data_tools,
    continue_until_agent_boundary,
    extract_interrupt_values,
    format_todos_for_user,
    invoke_user_message,
    iter_tool_calls,
    iter_tool_results,
    last_agent_response_text,
    load_deep_agent_settings,
    make_config,
    resume_with_decisions,
)

# Редактируй список запросов под свои e2e-сценарии.
E2E_REQUESTS: list[str] = [
    "подтяни к сработке 3486d84b-4eba-4ba4-b044-94764fc9e7a4 информацию о городе в котором был пользователь по ip",
]

# Ответ по умолчанию, если агент запросит уточнение через decision type=respond.
DEFAULT_RESPOND_MESSAGE = (
    "Продолжай по имеющимся данным. Если данных недостаточно, явно опиши ограничения результата."
)
MAX_LOG_PREVIEW_CHARS = 1200


@unittest.skipUnless(
    os.environ.get("RUN_DEEP_AGENT_E2E") == "1",
    "E2E отключены. Установи RUN_DEEP_AGENT_E2E=1 для запуска.",
)
class DeepAgentE2ERunnerTests(unittest.TestCase):
    """Проверяет e2e-проход агента с авто-approve плана."""

    def test_e2e_requests_with_auto_approve(self) -> None:
        """Запускает e2e по каждому запросу из E2E_REQUESTS."""

        settings = load_deep_agent_settings()
        test_tools = build_test_data_tools()
        agent = build_chat_agent(settings=settings, data_tools=test_tools)

        for case_index, query in enumerate(E2E_REQUESTS, start=1):
            with self.subTest(case=case_index, query=query):
                config = make_config(f"{settings.thread_id}-e2e-{case_index}")
                print()
                print(f"=== E2E CASE {case_index} ===")
                print(f"Запрос: {query}")
                print("-" * 80)
                result = invoke_user_message(agent, config, query)
                result = continue_until_agent_boundary(agent, config, result)
                seen = {"tool_calls": 0, "tool_results": 0}
                self._print_new_progress(result, seen)

                safety_steps = 0
                while True:
                    interrupts = extract_interrupt_values(result)
                    if not interrupts:
                        break
                    self._print_interrupts(interrupts)
                    decisions = self._auto_decisions(interrupts)
                    result = resume_with_decisions(agent, config, decisions)
                    result = continue_until_agent_boundary(agent, config, result, require_progress=True)
                    self._print_new_progress(result, seen)
                    safety_steps += 1
                    self.assertLess(
                        safety_steps,
                        20,
                        "Слишком много HITL-циклов в e2e. Похоже на зацикливание.",
                    )

                final_message = last_agent_response_text(result)
                print("-" * 80)
                print("Финальное сообщение:")
                print(final_message or "<пусто>")
                print("=" * 80)
                self.assertTrue(final_message.strip(), "Агент не вернул финальное сообщение.")
                self._assert_structured_task_outputs_quality(result)

    def _auto_decisions(self, interrupts: list[Any]) -> list[dict[str, str]]:
        """Формирует решения HITL: план всегда approve."""

        decisions: list[dict[str, str]] = []
        for payload in interrupts:
            action_requests = payload.get("action_requests", [])
            review_configs = payload.get("review_configs", [])
            for index, action_request in enumerate(action_requests):
                allowed = review_configs[index].get("allowed_decisions", [])
                action_name = str(action_request.get("name") or "")
                if action_name == "write_todos" and "approve" in allowed:
                    decisions.append({"type": "approve"})
                    continue
                if "respond" in allowed:
                    decisions.append({"type": "respond", "message": DEFAULT_RESPOND_MESSAGE})
                    continue
                if "approve" in allowed:
                    decisions.append({"type": "approve"})
                    continue
                decisions.append({"type": "reject", "message": "Автотест: неподдерживаемый тип решения."})
        return decisions

    def _print_interrupts(self, interrupts: list[Any]) -> None:
        """Печатает план и типы ожидаемых решений."""

        for payload in interrupts:
            action_requests = payload.get("action_requests", [])
            for action_request in action_requests:
                action_name = str(action_request.get("name") or "")
                if action_name == "write_todos":
                    print()
                    print("План анализа (auto-approve):")
                    print(format_todos_for_user(action_request.get("args", {}).get("todos", [])))
                    print("-" * 80)
                else:
                    print(f"HITL действие: {action_name}")

    def _print_new_progress(self, result: Any, seen: dict[str, int]) -> None:
        """Печатает новые tool calls и новые tool results."""

        if not isinstance(result, dict):
            return
        messages = result.get("messages") or []

        all_calls = list(iter_tool_calls(messages))
        new_calls = all_calls[seen["tool_calls"] :]
        if new_calls:
            print()
            print("Новые вызовы инструментов:")
        for index, (tool_name, args) in enumerate(new_calls, start=1):
            print(f"[call #{seen['tool_calls'] + index}] `{tool_name}`")
            print(self._format_for_console(args))
            print("-" * 80)
        seen["tool_calls"] = len(all_calls)

        all_results = list(iter_tool_results(messages))
        new_results = all_results[seen["tool_results"] :]
        if new_results:
            print()
            print("Новые результаты инструментов:")
        for index, (tool_name, content) in enumerate(new_results, start=1):
            print(f"[result #{seen['tool_results'] + index}] `{tool_name}`")
            print(self._format_for_console(content))
            print("-" * 80)
        seen["tool_results"] = len(all_results)

    def _format_for_console(self, value: Any) -> str:
        """Форматирует данные для читаемого консольного вывода."""

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return "<empty>"
            if len(text) > MAX_LOG_PREVIEW_CHARS:
                return f"{text[:MAX_LOG_PREVIEW_CHARS]}... <truncated {len(text) - MAX_LOG_PREVIEW_CHARS} chars>"
            return text

        try:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except TypeError:
            text = str(value)
        if len(text) > MAX_LOG_PREVIEW_CHARS:
            return f"{text[:MAX_LOG_PREVIEW_CHARS]}... <truncated {len(text) - MAX_LOG_PREVIEW_CHARS} chars>"
        return text

    def _assert_structured_task_outputs_quality(self, result: Any) -> None:
        """Проверяет базовое качество structured output от subagent-ов."""

        if not isinstance(result, dict):
            return
        messages = result.get("messages") or []
        for tool_name, content in iter_tool_results(messages):
            if tool_name != "task" or not isinstance(content, str):
                continue
            parsed = self._try_parse_json(content)
            if not isinstance(parsed, dict):
                continue
            status = str(parsed.get("status") or "")
            rows_count = parsed.get("rows_count")
            if status != "success" or not isinstance(rows_count, int) or rows_count <= 0:
                continue
            preview_rows = parsed.get("preview_rows")
            key_values = parsed.get("key_values_for_next_step")
            has_preview_payload = (
                isinstance(preview_rows, list)
                and bool(preview_rows)
                and isinstance(preview_rows[0], dict)
                and bool(preview_rows[0])
            )
            has_key_values = isinstance(key_values, dict) and bool(key_values)
            self.assertTrue(
                has_preview_payload or has_key_values,
                (
                    "task(status=success, rows_count>0) должен возвращать либо непустой preview_rows, "
                    "либо непустой key_values_for_next_step."
                ),
            )

    def _try_parse_json(self, value: str) -> dict[str, Any] | None:
        """Пытается распарсить JSON-строку."""

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


if __name__ == "__main__":
    unittest.main()

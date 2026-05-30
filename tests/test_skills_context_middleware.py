"""Тесты кэширования и ролей PreloadedSkillsContextMiddleware."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from deep_agent_test.skills_context_middleware import PreloadedSkillsContextMiddleware


class _FakeStructuredModel:
    """Заглушка structured-output модели: возвращает заранее заданные пути и считает вызовы."""

    def __init__(self, counter: dict[str, int], paths: list[str]) -> None:
        self._counter = counter
        self._paths = paths

    def invoke(self, _messages: object) -> object:
        self._counter["calls"] += 1

        class _Result:
            def __init__(self, paths: list[str]) -> None:
                self.paths = paths

        return _Result(self._paths)


class _FakeModel:
    """Заглушка chat-модели с with_structured_output."""

    def __init__(self, counter: dict[str, int], paths: list[str]) -> None:
        self._counter = counter
        self._paths = paths

    def with_structured_output(self, _schema: object) -> _FakeStructuredModel:
        return _FakeStructuredModel(self._counter, self._paths)


def _write_skill(root: Path, name: str, description: str) -> str:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"name: {name}\ndescription: {description}\n\n# {name}\nТело skill {name}.\n",
        encoding="utf-8",
    )
    return f"/skills/{name}/SKILL.md"


def _state(query: str) -> dict[str, object]:
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=query)]}


class SkillsContextMiddlewareTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.hit_path = _write_skill(self.root, "hit-table", "Сработки антифрода.")
        _write_skill(self.root, "cards-event-table", "Карточный канал.")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_supervisor_selects_and_caches(self) -> None:
        counter = {"calls": 0}
        shared: dict[str, object] = {}
        supervisor = PreloadedSkillsContextMiddleware(
            skills_root=self.root,
            model=_FakeModel(counter, [self.hit_path]),
            select_skills=True,
            shared_selection=shared,
        )

        update = supervisor.before_agent(_state("найди сработки"), runtime=None)

        self.assertIsNotNone(update)
        self.assertEqual(update["preloaded_skill_paths"], [self.hit_path])
        self.assertEqual(counter["calls"], 1)
        self.assertEqual(shared["entry"]["user_query"], "найди сработки")

    def test_supervisor_does_not_recompute_same_query(self) -> None:
        counter = {"calls": 0}
        shared: dict[str, object] = {}
        supervisor = PreloadedSkillsContextMiddleware(
            skills_root=self.root,
            model=_FakeModel(counter, [self.hit_path]),
            select_skills=True,
            shared_selection=shared,
        )

        supervisor.before_agent(_state("найди сработки"), runtime=None)
        supervisor.before_agent(_state("найди сработки"), runtime=None)

        self.assertEqual(counter["calls"], 1)

    def test_subagent_reuses_supervisor_selection_without_model_call(self) -> None:
        counter = {"calls": 0}
        shared: dict[str, object] = {}
        supervisor = PreloadedSkillsContextMiddleware(
            skills_root=self.root,
            model=_FakeModel(counter, [self.hit_path]),
            select_skills=True,
            shared_selection=shared,
        )
        subagent = PreloadedSkillsContextMiddleware(
            skills_root=self.root,
            model=_FakeModel(counter, [self.hit_path]),
            select_skills=False,
            shared_selection=shared,
        )

        supervisor.before_agent(_state("найди сработки"), runtime=None)
        update = subagent.before_agent(_state("Прочитай event_description"), runtime=None)

        self.assertIsNotNone(update)
        self.assertEqual(update["preloaded_skill_paths"], [self.hit_path])
        self.assertEqual(counter["calls"], 1)

    def test_subagent_without_cache_returns_none(self) -> None:
        counter = {"calls": 0}
        shared: dict[str, object] = {}
        subagent = PreloadedSkillsContextMiddleware(
            skills_root=self.root,
            model=_FakeModel(counter, [self.hit_path]),
            select_skills=False,
            shared_selection=shared,
        )

        update = subagent.before_agent(_state("Прочитай event_description"), runtime=None)

        self.assertIsNone(update)
        self.assertEqual(counter["calls"], 0)


if __name__ == "__main__":
    unittest.main()

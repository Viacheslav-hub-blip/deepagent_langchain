"""Тесты backend facade для запуска research-agent без UI.

Содержит:
- FakeGraph: тестовый graph с методом ainvoke.
- ResearchAgentFacadeTests: проверки запуска и инспекции через ResearchAgent.
"""

from __future__ import annotations

import tempfile
import unittest
from asyncio import run

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from planner_agent.models import AgentState
from planner_agent.research_agent import ResearchAgent, ResearchAgentInput
from planner_agent.schemas.lineage import BranchRequest, StateNode
from planner_agent.services.artifact_service import ArtifactService
from planner_agent.services.lineage_service import LineageService


class FakeGraph:
    """Минимальный graph, который имитирует успешный запуск агента в тестах."""

    def __init__(
            self,
            lineage_service: LineageService,
            artifact_service: ArtifactService,
    ) -> None:
        """Сохраняет сервисы, куда будет записан тестовый run.

        Args:
            lineage_service: Сервис записи lineage nodes.
            artifact_service: Сервис записи artifacts.

        Returns:
            None.
        """

        self.lineage_service = lineage_service
        self.artifact_service = artifact_service

    async def ainvoke(
            self,
            state: AgentState,
            config: dict | None = None,
    ) -> dict:
        """Имитирует LangGraph ainvoke и возвращает финальное состояние.

        Args:
            state: Входное состояние агента.
            config: Неиспользуемый LangGraph config.

        Returns:
            Словарь с финальным состоянием.
        """

        run_record = (
            self.lineage_service.get_run(state.run_id)
            if state.run_id
            else None
        )
        if run_record is None:
            run_record = self.lineage_service.create_run(
                initial_user_query=state.initial_user_query,
                session_id=state.session_id,
                user_id=state.user_id,
            )
        final_node = StateNode(
            run_id=run_record.run_id,
            node_type="final_report",
            title="Final report",
            status="succeeded",
        )
        artifact = self.artifact_service.write_artifact(
            run_id=run_record.run_id,
            node_id=final_node.node_id,
            kind="report",
            filename="final_report.md",
            content="# Report\n\nFacade works",
            mime_type="text/markdown",
            summary="Facade works",
        )
        final_node.artifact_refs = [artifact.artifact_id]
        self.lineage_service.append_node(
            final_node,
            state={
                "run_id": run_record.run_id,
                "final_report": "# Report\n\nFacade works",
            },
        )
        return {
            **state.model_dump(),
            "run_id": run_record.run_id,
            "final_report": "# Report\n\nFacade works",
            "messages": state.messages + [AIMessage(content="# Report\n\nFacade works")],
        }


class ResearchAgentFacadeTests(unittest.TestCase):
    """Проверяет удобный backend API вокруг существующего graph."""

    def test_ainvoke_returns_langchain_messages_and_keeps_state_for_inspection(self) -> None:
        """Проверяет, что facade возвращает run_id, nodes, artifacts и inspector."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            agent = ResearchAgent(
                graph=FakeGraph(lineage, artifacts),
                lineage_service=lineage,
                artifact_service=artifacts,
                runs_dir=tmp,
            )

            messages = run(
                agent.ainvoke(
                    "Analyze behavior",
                    session_id="session-1",
                    user_id="user-1",
                )
            )

            self.assertTrue(messages)
            self.assertTrue(all(isinstance(message, BaseMessage) for message in messages))
            self.assertEqual(messages[-1].content, "# Report\n\nFacade works")
            self.assertIsNotNone(agent.last_state)
            assert agent.last_state is not None
            self.assertTrue(agent.last_run_id)
            self.assertEqual(agent.last_state.final_report, "# Report\n\nFacade works")

            nodes = agent.inspector().list_nodes(agent.last_run_id)
            artifacts = agent.inspector().list_artifacts(agent.last_run_id)
            self.assertEqual(len(nodes), 1)
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(
                agent.inspector().get_final_report(agent.last_run_id),
                "# Report\n\nFacade works",
            )
            self.assertEqual(agent.get_final_report(), "# Report\n\nFacade works")
            self.assertEqual(len(agent.list_artifacts()), 1)
            artifact_id = agent.list_artifacts()[0].artifact_id
            artifact_details = agent.get_artifact_details(artifact_id, preview_chars=9)
            self.assertIsNotNone(artifact_details)
            assert artifact_details is not None
            self.assertEqual(artifact_details.preview.preview, "# Report\n")
            self.assertEqual(artifact_details.artifact.metadata["content_kind"], "report")
            artifact_preview = agent.preview_artifact(artifact_id, preview_chars=1)
            self.assertIsNotNone(artifact_preview)
            assert artifact_preview is not None
            self.assertEqual(artifact_preview.preview, "#")

            graph = agent.get_run_graph()
            self.assertIsNotNone(graph)
            assert graph is not None
            self.assertEqual(len(graph.nodes), 1)

            details = agent.get_node_details(graph.nodes[0].node_id)
            self.assertIsNotNone(details)
            assert details is not None
            self.assertEqual(details.node.node_type, "final_report")
            self.assertEqual(len(details.artifacts), 1)

            result = agent.get_run_result()
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.final_report, "# Report\n\nFacade works")
            self.assertEqual(result.summary.node_count, 1)
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(agent.list_run_summaries()[0].run.run_id, agent.last_run_id)

    def test_agent_supports_langchain_runnable_methods(self) -> None:
        """Проверяет invoke/ainvoke/batch/abatch в стиле LangChain Runnable."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            agent = ResearchAgent(
                graph=FakeGraph(lineage, artifacts),
                lineage_service=lineage,
                artifact_service=artifacts,
                runs_dir=tmp,
            )

            self.assertIsInstance(agent, Runnable)

            sync_messages = agent.invoke(
                {"user_query": "Analyze sync", "session_id": "session-sync"}
            )
            sync_state = agent.last_state
            async_messages = run(
                agent.ainvoke(
                    ResearchAgentInput(
                        user_query="Analyze async",
                        session_id="session-async",
                    )
                )
            )
            async_state = agent.last_state
            batch_results = agent.batch(["Batch one", "Batch two"])
            abatch_results = run(agent.abatch(["Async batch one", "Async batch two"]))

            self.assertEqual(sync_messages[-1].content, "# Report\n\nFacade works")
            self.assertEqual(async_messages[-1].content, "# Report\n\nFacade works")
            self.assertIsNotNone(sync_state)
            self.assertIsNotNone(async_state)
            assert sync_state is not None
            assert async_state is not None
            self.assertEqual(sync_state.session_id, "session-sync")
            self.assertEqual(async_state.session_id, "session-async")
            self.assertEqual(len(batch_results), 2)
            self.assertEqual(len(abatch_results), 2)
            self.assertTrue(all(isinstance(item, list) for item in batch_results))
            self.assertTrue(all(isinstance(item, list) for item in abatch_results))
            self.assertTrue(all(item[-1].content == "# Report\n\nFacade works" for item in batch_results))
            self.assertTrue(all(item[-1].content == "# Report\n\nFacade works" for item in abatch_results))

    def test_agent_accepts_optional_context_runs_without_breaking_langchain_input(self) -> None:
        """Проверяет необязательный dialog context поверх существующих runs."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            agent = ResearchAgent(
                graph=FakeGraph(lineage, artifacts),
                lineage_service=lineage,
                artifact_service=artifacts,
                runs_dir=tmp,
            )

            context_run = lineage.create_run(initial_user_query="Base analysis")
            context_node = StateNode(
                run_id=context_run.run_id,
                node_type="final_report",
                title="Final report",
                status="succeeded",
            )
            context_artifact = artifacts.write_artifact(
                run_id=context_run.run_id,
                node_id=context_node.node_id,
                kind="report",
                filename="final_report.md",
                content="# Base report\n\nAlternative hypothesis was stronger.",
                mime_type="text/markdown",
                summary="Base report",
            )
            context_node.artifact_refs = [context_artifact.artifact_id]
            lineage.append_node(
                context_node,
                state={
                    "run_id": context_run.run_id,
                    "final_report": "# Base report\n\nAlternative hypothesis was stronger.",
                },
            )

            messages = agent.invoke(
                {
                    "user_query": "Compare prior run with current hypothesis",
                    "context_runs": [
                        {
                            "run_id": context_run.run_id,
                            "role": "base",
                            "artifact_refs": [context_artifact.artifact_id],
                        }
                    ],
                }
            )

            self.assertEqual(messages[-1].content, "# Report\n\nFacade works")
            self.assertIsNotNone(agent.last_state)
            assert agent.last_state is not None
            dialog_context = agent.last_state.ephemeral_recalls.get("dialog_context", "")
            self.assertIn(context_run.run_id, dialog_context)
            self.assertIn("Base report", dialog_context)
            self.assertIn(context_artifact.artifact_id, dialog_context)
            self.assertEqual(
                agent.last_state.filesystem_context.get("dialog_context"),
                dialog_context,
            )

    def test_agent_can_create_and_invoke_branch_from_node(self) -> None:
        """Проверяет backend-only сценарий branch_from node -> restore state -> ainvoke."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            agent = ResearchAgent(
                graph=FakeGraph(lineage, artifacts),
                lineage_service=lineage,
                artifact_service=artifacts,
                runs_dir=tmp,
            )

            source_messages = run(agent.ainvoke("Analyze source"))
            source_run_id = agent.last_run_id
            source_graph = agent.get_run_graph()
            self.assertTrue(source_messages)
            self.assertIsNotNone(source_graph)
            assert source_graph is not None
            source_node_id = source_graph.nodes[-1].node_id

            branch_messages = run(
                agent.ainvoke_branch(
                    BranchRequest(
                        source_run_id=source_run_id,
                        source_node_id=source_node_id,
                        new_task="Continue as branch",
                        branch_mode="continue",
                    )
                )
            )

            self.assertTrue(branch_messages)
            self.assertEqual(branch_messages[-1].content, "# Report\n\nFacade works")
            self.assertIsNotNone(agent.last_state)
            assert agent.last_state is not None
            branch_run = lineage.get_run(agent.last_run_id)
            self.assertIsNotNone(branch_run)
            assert branch_run is not None
            self.assertEqual(branch_run.parent_run_id, source_run_id)
            self.assertEqual(branch_run.source_node_id, source_node_id)
            self.assertEqual(agent.last_state.initial_user_query, "Continue as branch")

            branch_nodes = lineage.get_nodes(branch_run.run_id)
            self.assertEqual(branch_nodes[0].node_type, "branch_started")
            self.assertEqual(branch_nodes[-1].node_type, "final_report")

    def test_stream_methods_yield_single_final_result(self) -> None:
        """Проверяет stream/astream как совместимый single-result stream."""

        async def _collect_async_stream(agent: ResearchAgent) -> list[list[BaseMessage]]:
            """Собирает messages из async stream.

            Args:
                agent: ResearchAgent для проверки astream.

            Returns:
                Список ответов из async iterator.
            """

            outputs: list[list[BaseMessage]] = []
            async for item in agent.astream("Async streamed analysis"):
                outputs.append(item)
            return outputs

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            agent = ResearchAgent(
                graph=FakeGraph(lineage, artifacts),
                lineage_service=lineage,
                artifact_service=artifacts,
                runs_dir=tmp,
            )

            stream_results = list(agent.stream("Streamed analysis"))
            async_outputs = run(_collect_async_stream(agent))

            self.assertEqual(len(stream_results), 1)
            self.assertEqual(stream_results[0][-1].content, "# Report\n\nFacade works")
            self.assertEqual(len(async_outputs), 1)
            self.assertEqual(async_outputs[0][-1].content, "# Report\n\nFacade works")


if __name__ == "__main__":
    unittest.main()

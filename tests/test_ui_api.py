"""Тесты API слоя для research-agent UI.

Содержит:
- UiApiTests: проверки HTTP endpoints поверх RunInspectionService.
"""

from __future__ import annotations

import tempfile
import unittest

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from planner_agent import ResearchAgent
from planner_agent.models import AgentState
from planner_agent.schemas.lineage import StateNode
from planner_agent.services.artifact_service import ArtifactService
from planner_agent.services.lineage_service import LineageService
from planner_agent.services.run_inspection_service import RunInspectionService
from planner_agent.http_api import ApiSettings, create_app
from planner_agent.http_api.config import ApiServices


class ApiFakeGraph:
    """Тестовый graph для проверки invoke endpoints без внешнего LLM.

    Args:
        lineage_service: Сервис записи lineage.
        artifact_service: Сервис записи artifacts.

    Returns:
        Объект с методом ``ainvoke`` для ResearchAgent.
    """

    def __init__(
            self,
            lineage_service: LineageService,
            artifact_service: ArtifactService,
    ) -> None:
        """Сохраняет сервисы для тестового запуска.

        Args:
            lineage_service: Сервис lineage.
            artifact_service: Сервис artifacts.

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
        """Имитирует успешный запуск агента.

        Args:
            state: Входное состояние агента.
            config: Неиспользуемый config.

        Returns:
            Словарь финального состояния.
        """

        run_record = self.lineage_service.get_run(state.run_id) if state.run_id else None
        if run_record is None:
            run_record = self.lineage_service.create_run(
                initial_user_query=state.initial_user_query,
                session_id=state.session_id,
                user_id=state.user_id,
            )
        parent_ids = state.parent_node_ids or (
            [state.current_node_id] if state.current_node_id else []
        )
        final_node = StateNode(
            run_id=run_record.run_id,
            node_type="final_report",
            title="Final report",
            parent_ids=parent_ids,
            status="succeeded",
        )
        report = self.artifact_service.write_artifact(
            run_id=run_record.run_id,
            node_id=final_node.node_id,
            kind="report",
            filename="final_report.md",
            content="# API report\n\nInvoke works",
            mime_type="text/markdown",
            summary="Invoke works",
        )
        final_node.artifact_refs = [report.artifact_id]
        self.lineage_service.append_node(
            final_node,
            state={
                "run_id": run_record.run_id,
                "final_report": "# API report\n\nInvoke works",
                "messages": [
                    message.model_dump(mode="json")
                    for message in [
                        *state.messages,
                        AIMessage(content="# API report\n\nInvoke works"),
                    ]
                ],
            },
        )
        return {
            **state.model_dump(),
            "run_id": run_record.run_id,
            "final_report": "# API report\n\nInvoke works",
            "messages": state.messages + [AIMessage(content="# API report\n\nInvoke works")],
        }


class UiApiTests(unittest.TestCase):
    """Проверяет минимальные HTTP endpoints для будущего UI."""

    def test_reads_run_graph_node_inspector_and_artifacts(self) -> None:
        """Проверяет чтение runs, graph, Node Inspector и artifact preview через API."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            services = ApiServices(
                lineage_service=lineage,
                artifact_service=artifacts,
                inspection_service=RunInspectionService(lineage, artifacts),
            )
            client = TestClient(
                create_app(
                    settings=ApiSettings(api_prefix="/api/v1"),
                    services=services,
                )
            )

            run_record = lineage.create_run(initial_user_query="Inspect API")
            root_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="context_snapshot",
                title="Context",
                status="succeeded",
                state={"run_id": run_record.run_id, "memory_snapshot": "Frozen"},
            )
            final_node = StateNode(
                run_id=run_record.run_id,
                node_type="final_report",
                title="Final report",
                parent_ids=[root_node.node_id],
                status="succeeded",
            )
            report_artifact = artifacts.write_artifact(
                run_id=run_record.run_id,
                node_id=final_node.node_id,
                kind="report",
                filename="final_report.md",
                content="# Report\n\nDone",
                mime_type="text/markdown",
                summary="Done",
            )
            final_node.artifact_refs = [report_artifact.artifact_id]
            lineage.append_node(
                final_node,
                state={
                    "run_id": run_record.run_id,
                    "final_report": "# Report\n\nDone",
                    "messages": [{"type": "ai", "content": "# Report\n\nDone"}],
                },
            )

            health = client.get("/api/v1/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")

            runs = client.get("/api/v1/runs")
            self.assertEqual(runs.status_code, 200)
            self.assertEqual(runs.json()[0]["run"]["run_id"], run_record.run_id)

            graph = client.get(f"/api/v1/runs/{run_record.run_id}/graph")
            self.assertEqual(graph.status_code, 200)
            self.assertEqual(len(graph.json()["nodes"]), 2)

            inspector = client.get(
                f"/api/v1/runs/{run_record.run_id}/nodes/{final_node.node_id}/inspector",
                params={"preview_chars": 8, "snapshot_preview_chars": 12},
            )
            self.assertEqual(inspector.status_code, 200)
            inspector_payload = inspector.json()
            self.assertEqual(inspector_payload["node"]["node_id"], final_node.node_id)
            self.assertEqual(
                inspector_payload["parent_nodes"][0]["node_id"],
                root_node.node_id,
            )
            self.assertEqual(
                inspector_payload["artifacts"][0]["artifact"]["artifact_id"],
                report_artifact.artifact_id,
            )

            preview = client.get(
                f"/api/v1/runs/{run_record.run_id}/artifacts/{report_artifact.artifact_id}/preview",
                params={"preview_chars": 8},
            )
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.json()["preview"], "# Report")

            text = client.get(
                f"/api/v1/runs/{run_record.run_id}/artifacts/{report_artifact.artifact_id}/text",
                params={"max_chars": 3},
            )
            self.assertEqual(text.status_code, 200)
            self.assertEqual(text.json()["content"], "# R")

            file_response = client.get(
                f"/api/v1/runs/{run_record.run_id}/artifacts/{report_artifact.artifact_id}/file",
            )
            self.assertEqual(file_response.status_code, 200)
            self.assertIn("# Report", file_response.text)
            disposition = file_response.headers.get("content-disposition", "")
            self.assertIn("attachment", disposition.lower())

            dialog_context = client.post(
                "/api/v1/dialog-context",
                json={
                    "user_query": "Compare this run with another hypothesis",
                    "context_runs": [
                        {
                            "run_id": run_record.run_id,
                            "role": "base",
                            "artifact_refs": [report_artifact.artifact_id],
                            "max_report_chars": 100,
                            "max_artifact_preview_chars": 20,
                        }
                    ],
                },
            )
            self.assertEqual(dialog_context.status_code, 200)
            dialog_payload = dialog_context.json()
            self.assertEqual(
                dialog_payload["user_query"],
                "Compare this run with another hypothesis",
            )
            self.assertIn(
                run_record.run_id,
                dialog_payload["context"]["rendered_context"],
            )
            self.assertIn(
                report_artifact.artifact_id,
                dialog_payload["context"]["rendered_context"],
            )
            self.assertEqual(
                dialog_payload["context"]["context_runs"][0]["ref"]["role"],
                "base",
            )

    def test_creates_branch_metadata_without_running_agent(self) -> None:
        """Проверяет создание branch run через API без запуска LLM."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            services = ApiServices(
                lineage_service=lineage,
                artifact_service=artifacts,
                inspection_service=RunInspectionService(lineage, artifacts),
            )
            client = TestClient(create_app(services=services))

            run_record = lineage.create_run(initial_user_query="Base")
            source_node = lineage.create_state_node(
                run_id=run_record.run_id,
                node_type="final_report",
                title="Final report",
                status="succeeded",
                state={"run_id": run_record.run_id, "final_report": "Done"},
            )

            response = client.post(
                "/api/v1/branches",
                json={
                    "source_run_id": run_record.run_id,
                    "source_node_id": source_node.node_id,
                    "new_task": "Check alternative hypothesis",
                    "branch_mode": "what_if",
                    "include_artifacts": True,
                    "include_memory_snapshot": True,
                    "include_completed_tasks": True,
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["run"]["parent_run_id"], run_record.run_id)
            self.assertEqual(payload["run"]["source_node_id"], source_node.node_id)
            self.assertIsNotNone(payload["branch_started_node_id"])

    def test_invokes_agent_when_agent_is_configured(self) -> None:
        """Проверяет endpoint запуска агента через API."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            inspection = RunInspectionService(lineage, artifacts)
            agent = ResearchAgent(
                graph=ApiFakeGraph(lineage, artifacts),
                lineage_service=lineage,
                artifact_service=artifacts,
                runs_dir=tmp,
            )
            services = ApiServices(
                lineage_service=lineage,
                artifact_service=artifacts,
                inspection_service=inspection,
                agent=agent,
            )
            client = TestClient(create_app(services=services))

            response = client.post(
                "/api/v1/runs/invoke",
                json={
                    "user_query": "Run API analysis",
                    "session_id": "api-session",
                    "user_id": "api-user",
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["run_id"])
            self.assertEqual(payload["messages"][-1]["content"], "# API report\n\nInvoke works")
            self.assertEqual(payload["result"]["final_report"], "# API report\n\nInvoke works")
            self.assertEqual(payload["result"]["run"]["session_id"], "api-session")

    def test_invoke_returns_503_without_configured_agent(self) -> None:
        """Проверяет понятную ошибку, если API создан без агента."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            services = ApiServices(
                lineage_service=lineage,
                artifact_service=artifacts,
                inspection_service=RunInspectionService(lineage, artifacts),
            )
            client = TestClient(create_app(services=services))

            response = client.post(
                "/api/v1/runs/invoke",
                json={"user_query": "Run without agent"},
            )

            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"], "agent_not_configured")

    def test_invokes_branch_when_agent_is_configured(self) -> None:
        """Проверяет endpoint создания и запуска ветки через API."""

        with tempfile.TemporaryDirectory() as tmp:
            lineage = LineageService(tmp)
            artifacts = ArtifactService(tmp)
            inspection = RunInspectionService(lineage, artifacts)
            agent = ResearchAgent(
                graph=ApiFakeGraph(lineage, artifacts),
                lineage_service=lineage,
                artifact_service=artifacts,
                runs_dir=tmp,
            )
            services = ApiServices(
                lineage_service=lineage,
                artifact_service=artifacts,
                inspection_service=inspection,
                agent=agent,
            )
            client = TestClient(create_app(services=services))

            source_run = lineage.create_run(initial_user_query="Base")
            source_node = lineage.create_state_node(
                run_id=source_run.run_id,
                node_type="final_report",
                title="Final report",
                status="succeeded",
                state={"run_id": source_run.run_id, "final_report": "Done"},
            )

            response = client.post(
                "/api/v1/branches/invoke",
                json={
                    "source_run_id": source_run.run_id,
                    "source_node_id": source_node.node_id,
                    "new_task": "Run branch via API",
                    "branch_mode": "what_if",
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["run_id"])
            self.assertEqual(payload["result"]["run"]["parent_run_id"], source_run.run_id)
            self.assertEqual(payload["messages"][-1]["content"], "# API report\n\nInvoke works")


if __name__ == "__main__":
    unittest.main()

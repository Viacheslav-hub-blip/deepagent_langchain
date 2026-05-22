"""Проверка создания branch от любого шага lineage без UI.

Запуск из корня репозитория::

    python examples/branch_from_graph_step_demo.py

Скрипт создаёт временный каталог runs, строит короткий lineage из нескольких
узлов и для каждого из них вызывает ``LineageService.branch_from``, проверяя
что новый ResearchRun содержит узел ``branch_started`` и корректные ссылки на
родительский run и исходный узел.

При ``ModuleNotFoundError`` задайте ``PYTHONPATH`` на корень репозитория или
выполните ``pip install -e .``, если проект оформлен как пакет.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from planner_agent.schemas.lineage import BranchRequest
from planner_agent.services.lineage_service import LineageService


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lineage = LineageService(tmp)

        run = lineage.create_run(
            initial_user_query="Демонстрационный анализ",
            session_id="branch-demo-session",
            user_id="branch-demo-user",
        )

        n_context = lineage.create_state_node(
            run_id=run.run_id,
            node_type="context_snapshot",
            title="Context",
            parent_ids=[],
            summary="Снимок контекста",
            state={"run_id": run.run_id, "step": "context"},
        )
        lineage.create_state_node(
            run_id=run.run_id,
            node_type="plan_created",
            title="Plan",
            parent_ids=[n_context.node_id],
            summary="План",
            state={"run_id": run.run_id, "step": "plan"},
        )

        lineage_nodes = lineage.get_nodes(run.run_id)
        assert len(lineage_nodes) == 2, f"ожидалось 2 узла, получено {len(lineage_nodes)}"

        for source in lineage_nodes:
            branch = lineage.branch_from(
                BranchRequest(
                    source_run_id=run.run_id,
                    source_node_id=source.node_id,
                    new_task=f"Ветка после узла {source.node_type}",
                    branch_mode="what_if",
                )
            )
            assert branch.parent_run_id == run.run_id
            assert branch.source_node_id == source.node_id

            br_nodes = lineage.get_nodes(branch.run_id)
            branch_started = [n for n in br_nodes if n.node_type == "branch_started"]
            assert len(branch_started) == 1, br_nodes
            meta = branch_started[0].metadata or {}
            assert meta.get("source_node_id") == source.node_id

        print("OK: branch_from успешно для каждого шага графа:", [n.node_id for n in lineage_nodes])


if __name__ == "__main__":
    main()

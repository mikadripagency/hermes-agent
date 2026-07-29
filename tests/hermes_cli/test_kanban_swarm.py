import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_swarm import (
    SwarmWorkerSpec,
    create_swarm,
    latest_blackboard,
    post_blackboard_update,
)

pytestmark = pytest.mark.usefixtures("claimed_completion_for_kanban_fixtures")


def test_create_swarm_builds_parallel_workers_verifier_and_synthesizer(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Map the target market and produce a decision memo.",
            workers=[
                SwarmWorkerSpec(profile="researcher-a", title="Market scan", body="Find competitors"),
                SwarmWorkerSpec(profile="researcher-b", title="Customer scan", body="Find customer pains"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
            tenant="intel",
            created_by="orchestrator",
        )

        root = kb.get_task(conn, created.root_id)
        workers = [kb.get_task(conn, tid) for tid in created.worker_ids]
        verifier = kb.get_task(conn, created.verifier_id)
        synthesizer = kb.get_task(conn, created.synthesizer_id)

        assert root.status == "done"
        assert root.assignee == "orchestrator"
        assert [task.status for task in workers] == ["ready", "ready"]
        assert [task.assignee for task in workers] == ["researcher-a", "researcher-b"]
        assert verifier.status == "todo"
        assert synthesizer.status == "todo"
        assert set(kb.parent_ids(conn, created.verifier_id)) == set(created.worker_ids)
        assert kb.parent_ids(conn, created.synthesizer_id) == [created.verifier_id]
        assert all(created.root_id in (task.body or "") for task in workers)
    finally:
        conn.close()


def test_create_swarm_idempotent_retry_recovers_crash_after_root_claim(
    tmp_path, monkeypatch,
):
    conn = kb.connect(tmp_path / "kanban.db")
    original_complete_task = kb.complete_task
    crashed = False

    def crash_once_after_claim(*args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("crash after claim")
        return original_complete_task(*args, **kwargs)

    monkeypatch.setattr(kb, "complete_task", crash_once_after_claim)
    swarm_kwargs = {
        "goal": "Recover one durable swarm.",
        "workers": [
            SwarmWorkerSpec(profile="researcher", title="Research", body="Find proof")
        ],
        "verifier_assignee": "reviewer",
        "synthesizer_assignee": "writer",
        "created_by": "orchestrator",
        "idempotency_key": "swarm-crash-retry",
    }

    try:
        with pytest.raises(RuntimeError, match="crash after claim"):
            create_swarm(conn, **swarm_kwargs)

        created = create_swarm(conn, **swarm_kwargs)

        root = kb.get_task(conn, created.root_id)
        assert root is not None and root.status == "done"
        assert latest_blackboard(conn, created.root_id)["topology"] == (
            created.as_dict() | {"goal": swarm_kwargs["goal"]}
        )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 4
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'claimed'",
            (created.root_id,),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_create_swarm_idempotent_retry_reuses_partial_graph(tmp_path, monkeypatch):
    conn = kb.connect(tmp_path / "kanban.db")
    original_create_task = kb.create_task
    crashed = False

    def crash_once_after_first_worker(*args, **kwargs):
        nonlocal crashed
        task_id = original_create_task(*args, **kwargs)
        child_key = str(kwargs.get("idempotency_key", ""))
        if not crashed and child_key.endswith(":worker:0"):
            crashed = True
            raise RuntimeError("crash after first worker")
        return task_id

    monkeypatch.setattr(kb, "create_task", crash_once_after_first_worker)
    swarm_kwargs = {
        "goal": "Recover one partial swarm.",
        "workers": [
            SwarmWorkerSpec(profile="a", title="A", body="A"),
            SwarmWorkerSpec(profile="b", title="B", body="B"),
        ],
        "verifier_assignee": "reviewer",
        "synthesizer_assignee": "writer",
        "idempotency_key": "swarm-partial-retry",
    }

    try:
        with pytest.raises(RuntimeError, match="crash after first worker"):
            create_swarm(conn, **swarm_kwargs)

        created = create_swarm(conn, **swarm_kwargs)

        root = kb.get_task(conn, created.root_id)
        assert root is not None and root.status == "done"
        assert len(created.worker_ids) == len(set(created.worker_ids)) == 2
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 5
    finally:
        conn.close()


def test_swarm_blackboard_merges_structured_updates(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Collect evidence.",
            workers=[SwarmWorkerSpec(profile="researcher", title="Evidence", body="Find proof")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        post_blackboard_update(
            conn,
            created.root_id,
            author="researcher",
            key="sources",
            value=["https://example.com/a"],
        )
        post_blackboard_update(
            conn,
            created.root_id,
            author="reviewer",
            key="risks",
            value={"missing_primary_source": True},
        )

        board = latest_blackboard(conn, created.root_id)
        assert board["sources"] == ["https://example.com/a"]
        assert board["risks"] == {"missing_primary_source": True}
        assert board["_authors"]["sources"] == "researcher"
    finally:
        conn.close()


def test_swarm_verifier_and_synthesis_are_dependency_gated(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        created = create_swarm(
            conn,
            goal="Research two branches then verify and synthesize.",
            workers=[
                SwarmWorkerSpec(profile="a", title="Branch A", body="A"),
                SwarmWorkerSpec(profile="b", title="Branch B", body="B"),
            ],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )

        kb.complete_task(
            conn,
            created.worker_ids[0],
            summary="A done",
            metadata={"confidence": 0.8},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "todo"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(conn, created.worker_ids[1], summary="B done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.verifier_id).status == "ready"
        assert kb.get_task(conn, created.synthesizer_id).status == "todo"

        kb.complete_task(
            conn,
            created.verifier_id,
            summary="Verified both branches",
            metadata={"gate": "pass"},
        )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, created.synthesizer_id).status == "ready"
    finally:
        conn.close()

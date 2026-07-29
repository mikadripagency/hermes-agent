from concurrent.futures import ThreadPoolExecutor
import shutil
import threading

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


def test_create_swarm_rejects_legacy_partial_graph_without_writes(tmp_path):
    source_path = tmp_path / "legacy.db"
    conn = kb.connect(source_path)
    goal = "Do not duplicate a copied legacy child."
    root_key = "legacy-partial-swarm"
    try:
        root = kb.create_task(
            conn,
            title=f"Swarm: {goal}",
            body=(
                "Kanban Swarm v1 planning/root card. This card is completed "
                "immediately so parallel workers can start while it remains the "
                "shared blackboard and audit anchor.\n\n"
                f"Goal:\n{goal}"
            ),
            assignee="swarm-orchestrator",
            created_by="swarm-orchestrator",
            idempotency_key=root_key,
            evidence_contract_na_reason="swarm topology anchor; no terminal delivery gate",
        )
        claimed = kb.claim_task(conn, root, claimer="legacy-swarm")
        assert claimed is not None and claimed.current_run_id is not None
        assert kb.complete_task(
            conn,
            root,
            summary=f"{root}/run {claimed.current_run_id} · legacy root",
            expected_run_id=claimed.current_run_id,
        )
        legacy_child = kb.create_task(
            conn,
            title="Legacy worker",
            body="Legacy worker body",
            assignee="researcher",
            created_by="swarm-orchestrator",
            parents=[root],
            evidence_contract_na_reason=(
                "specialist swarm output is gated by the downstream verifier"
            ),
        )
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        copied_path = tmp_path / "copied.db"
        shutil.copy2(source_path, copied_path)
        conn = kb.connect(copied_path)
        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "tasks",
                "task_runs",
                "task_events",
                "completion_deliveries",
                "kanban_notify_subs",
            )
        }

        with pytest.raises(RuntimeError, match="legacy partial swarm graph"):
            create_swarm(
                conn,
                goal=goal,
                workers=[
                    SwarmWorkerSpec(
                        profile="researcher",
                        title="Legacy worker",
                        body="Legacy worker body",
                    )
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key=root_key,
            )

        after = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
        assert after == before
        assert kb.get_task(conn, legacy_child) is not None
    finally:
        conn.close()


def test_create_swarm_retry_does_not_complete_foreign_run(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    original_claim_task = kb.claim_task
    swarm_kwargs = {
        "goal": "Leave a foreign dispatcher run alone.",
        "workers": [
            SwarmWorkerSpec(profile="researcher", title="Research", body="Find proof")
        ],
        "verifier_assignee": "reviewer",
        "synthesizer_assignee": "writer",
        "created_by": "orchestrator",
        "idempotency_key": "foreign-run-swarm",
    }

    def crash_before_helper_claim(*args, **kwargs):
        raise RuntimeError("crash before helper claim")

    try:
        monkeypatch.setattr(kb, "claim_task", crash_before_helper_claim)
        with pytest.raises(RuntimeError, match="crash before helper claim"):
            create_swarm(conn, **swarm_kwargs)
        monkeypatch.setattr(kb, "claim_task", original_claim_task)

        root = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ?",
            (swarm_kwargs["idempotency_key"],),
        ).fetchone()[0]
        foreign_conn = kb.connect(db_path)
        try:
            foreign = original_claim_task(
                foreign_conn,
                root,
                claimer="dispatcher:foreign-worker",
            )
            assert foreign is not None and foreign.current_run_id is not None
            foreign_run_id = foreign.current_run_id
        finally:
            foreign_conn.close()

        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "tasks",
                "task_runs",
                "task_events",
                "completion_deliveries",
                "kanban_notify_subs",
            )
        }
        with pytest.raises(RuntimeError, match="owned by another claimer"):
            create_swarm(conn, **swarm_kwargs)
        after = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }

        assert after == before
        root_task = kb.get_task(conn, root)
        assert root_task is not None and root_task.status == "running"
        assert root_task.current_run_id == foreign_run_id
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'completed'",
            (root,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_create_swarm_rejects_changed_idempotent_manifest_without_writes(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    kwargs = {
        "goal": "Keep one stable manifest.",
        "workers": [SwarmWorkerSpec(profile="a", title="A", body="A")],
        "verifier_assignee": "reviewer",
        "synthesizer_assignee": "writer",
        "idempotency_key": "stable-manifest-swarm",
    }
    try:
        create_swarm(conn, **kwargs)
        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "tasks",
                "task_runs",
                "task_events",
                "completion_deliveries",
                "kanban_notify_subs",
            )
        }

        changed = dict(kwargs)
        changed["workers"] = [
            SwarmWorkerSpec(profile="b", title="B", body="changed")
        ]
        with pytest.raises(ValueError, match="different swarm manifest"):
            create_swarm(conn, **changed)

        after = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
        assert after == before
    finally:
        conn.close()


def test_create_swarm_concurrent_changed_manifests_cannot_form_hybrid_graph(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.connect(db_path).close()
    barrier = threading.Barrier(2)

    def construct(worker_count):
        conn = kb.connect(db_path)
        try:
            barrier.wait()
            return create_swarm(
                conn,
                goal="Choose exactly one concurrent manifest.",
                workers=[
                    SwarmWorkerSpec(
                        profile=f"worker-{index}",
                        title=f"Worker {index}",
                        body=f"Body {index}",
                    )
                    for index in range(worker_count)
                ],
                verifier_assignee="reviewer",
                synthesizer_assignee="writer",
                idempotency_key="concurrent-manifest-swarm",
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(construct, count) for count in (1, 2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except ValueError as exc:
                outcomes.append(exc)

    created = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    rejected = [outcome for outcome in outcomes if isinstance(outcome, ValueError)]
    assert len(created) == len(rejected) == 1
    assert "different swarm manifest" in str(rejected[0])

    conn = kb.connect(db_path)
    try:
        graph = created[0]
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == (
            len(graph.worker_ids) + 3
        )
        assert set(kb.parent_ids(conn, graph.verifier_id)) == set(graph.worker_ids)
        assert kb.parent_ids(conn, graph.synthesizer_id) == [graph.verifier_id]
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

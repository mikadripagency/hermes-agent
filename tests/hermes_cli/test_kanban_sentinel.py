"""Regression tests for the profile-local Kanban sentinel."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("explicit_delivery_contract_for_kanban_fixtures")


def _load():
    from scripts import kanban_sentinel as sentinel

    return sentinel


def _running_copy(tmp_path: Path, *, now: int, pid: int = 4242, task_id: str = "t_watch"):
    from hermes_cli import kanban_db as kb

    source = tmp_path / "source.db"
    conn = kb.connect(source)
    tid = kb.create_task(
        conn,
        title="sentinel fixture",
        assignee="developer",
        max_runtime_seconds=1800,
        evidence_contract_na_reason="sentinel fixture",
    )
    assert tid == task_id or tid.startswith("t_")
    claimed = kb.claim_task(
        conn,
        tid,
        ttl_seconds=300,
        claimer="fixture-host:1",
        execution_profile="developer",
    )
    assert claimed is not None and claimed.current_run_id is not None
    run_id = int(claimed.current_run_id)
    kb._set_worker_pid(conn, tid, pid)
    assert kb.heartbeat_worker(conn, tid, expected_run_id=run_id)
    conn.execute(
        "UPDATE tasks SET started_at=?, last_heartbeat_at=?, claim_expires=? WHERE id=?",
        (now - 40, now - 10, now + 290, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at=?, last_heartbeat_at=?, claim_expires=? WHERE id=?",
        (now - 40, now - 10, now + 290, run_id),
    )
    conn.commit()
    conn.close()

    copied = tmp_path / "copy.db"
    src = sqlite3.connect(source)
    dst = sqlite3.connect(copied)
    src.backup(dst)
    src.close()
    dst.close()
    return copied, tid, run_id


def _task_row(conn, tid):
    return conn.execute(
        "SELECT id, assignee, status, started_at, last_heartbeat_at, "
        "current_run_id, worker_pid, max_runtime_seconds, claim_expires "
        "FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()


def test_fresh_direct_spawn_with_no_paseo_agent_is_healthy(tmp_path, monkeypatch):
    sentinel = _load()
    now = 2_000_000
    db, tid, _ = _running_copy(tmp_path, now=now)
    monkeypatch.setattr(sentinel, "_pid_alive", lambda pid: pid == 4242)

    conn = sentinel._connect_ro(db)
    try:
        verdict = sentinel._classify(conn, _task_row(conn, tid), now, [])
    finally:
        conn.close()

    assert verdict["verdict"] == "healthy"
    assert verdict["worker_mode"] == "direct"


def test_fresh_paseo_reattach_grace_is_healthy(tmp_path, monkeypatch):
    sentinel = _load()
    now = 2_000_000
    db, tid, run_id = _running_copy(tmp_path, now=now)
    monkeypatch.setattr(sentinel, "_pid_alive", lambda _pid: False)
    agents = [{
        "id": "agent-current",
        "shortId": "agent-c",
        "status": "running",
        "labels": {
            "kanban_task": tid,
            "kanban_run": str(run_id),
            "kanban_profile": "developer",
        },
    }]

    conn = sentinel._connect_ro(db)
    try:
        verdict = sentinel._classify(conn, _task_row(conn, tid), now, agents)
    finally:
        conn.close()

    assert verdict["verdict"] == "healthy"
    assert verdict["worker_mode"] == "paseo"
    assert verdict["agent_short_id"] == "agent-c"


def test_startup_grace_treats_missing_worker_telemetry_as_unknown_not_dead(
    tmp_path, monkeypatch
):
    sentinel = _load()
    now = 2_000_000
    db, tid, _ = _running_copy(tmp_path, now=now)
    monkeypatch.setattr(sentinel, "_pid_alive", lambda _pid: False)

    conn = sentinel._connect_ro(db)
    try:
        verdict = sentinel._classify(conn, _task_row(conn, tid), now, [])
    finally:
        conn.close()

    assert verdict["verdict"] == "healthy"


def test_stale_current_run_is_a_real_orphan_after_grace(tmp_path, monkeypatch):
    sentinel = _load()
    now = 2_000_000
    db, tid, run_id = _running_copy(tmp_path, now=now)
    monkeypatch.setattr(sentinel, "_pid_alive", lambda _pid: False)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE tasks SET current_run_id=?, started_at=?, claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (run_id + 999, now - 900, now - 300, now - 900, tid),
    )
    conn.commit()
    conn.close()

    conn = sentinel._connect_ro(db)
    try:
        verdict = sentinel._classify(conn, _task_row(conn, tid), now, [])
    finally:
        conn.close()

    assert verdict["verdict"] == "agent_dead_task_running"
    assert verdict["orphan_reason"] == "current_run_missing_or_not_running"


def test_dead_direct_worker_with_expired_claim_is_exactly_once_actionable(
    tmp_path, monkeypatch, capsys
):
    sentinel = _load()
    from hermes_cli import kanban_db as kb

    now = 2_000_000
    db, tid, run_id = _running_copy(tmp_path, now=now)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE tasks SET started_at=?, last_heartbeat_at=?, claim_expires=? WHERE id=?",
        (now - 4000, now - 4000, now - 300, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at=?, last_heartbeat_at=?, claim_expires=? WHERE id=?",
        (now - 4000, now - 4000, now - 300, run_id),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(sentinel, "DB_PATH", db)
    monkeypatch.setattr(sentinel, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(sentinel, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(sentinel, "_load_paseo_agents", lambda: [])
    monkeypatch.setattr(sentinel, "_load_task_agents", lambda *_args: [])
    monkeypatch.setattr(sentinel.time, "time", lambda: now)

    allowed = []
    for _ in range(2):
        assert sentinel.main() == 0
        data = json.loads(capsys.readouterr().out)
        allowed.extend(
            item for item in data["anomalies"]
            if item["task_id"] == tid and item["action_allowed"]
        )

    assert len(allowed) == 1
    assert allowed[0]["verdict"] == "agent_dead_task_running"
    assert allowed[0]["orphan_reason"] == "worker_gone_claim_expired"

    conn = kb.connect(db)
    assert kb.release_stale_claims(conn) == 1
    task = conn.execute(
        "SELECT status, claim_lock, claim_expires FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    reclaimed = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='reclaimed'",
        (tid,),
    ).fetchone()[0]
    conn.close()
    assert tuple(task) == ("running", None, None)
    assert reclaimed == 1


def test_linked_paseo_agent_archived_after_grace_is_a_real_orphan(
    tmp_path, monkeypatch
):
    sentinel = _load()
    from hermes_cli import kanban_db as kb

    now = 2_000_000
    db, tid, run_id = _running_copy(tmp_path, now=now)
    conn = kb.connect(db)
    kb.add_comment(conn, tid, "developer", "paseo_agent=agent-archived")
    conn.execute(
        "UPDATE tasks SET started_at=?, last_heartbeat_at=?, claim_expires=? WHERE id=?",
        (now - 900, now - 900, now - 300, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at=?, last_heartbeat_at=?, claim_expires=? WHERE id=?",
        (now - 900, now - 900, now - 300, run_id),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(sentinel, "_pid_alive", lambda _pid: False)
    archived = [{
        "id": "agent-archived",
        "status": "archived",
        "labels": {
            "kanban_task": tid,
            "kanban_run": str(run_id),
            "kanban_profile": "developer",
        },
    }]

    conn = sentinel._connect_ro(db)
    try:
        verdict = sentinel._classify(conn, _task_row(conn, tid), now, archived)
    finally:
        conn.close()

    assert verdict["verdict"] == "agent_dead_task_running"
    assert verdict["orphan_reason"] == "linked_agent_unreachable"


def test_missing_paseo_telemetry_alone_never_becomes_reclaim_evidence(
    tmp_path, monkeypatch
):
    sentinel = _load()
    now = 2_000_000
    db, tid, _ = _running_copy(tmp_path, now=now)
    monkeypatch.setattr(sentinel, "_pid_alive", lambda _pid: True)

    conn = sentinel._connect_ro(db)
    try:
        verdict = sentinel._classify(conn, _task_row(conn, tid), now, None)
    finally:
        conn.close()

    assert verdict["verdict"] == "healthy"
    assert verdict["paseo_listing_available"] is False


def test_copy_db_reads_stay_consistent_during_live_heartbeats(tmp_path, monkeypatch):
    sentinel = _load()
    from hermes_cli import kanban_db as kb

    now = 2_000_000
    db, tid, run_id = _running_copy(tmp_path, now=now)
    monkeypatch.setattr(sentinel, "_pid_alive", lambda _pid: True)
    stop = threading.Event()
    errors = []

    def writer():
        conn = kb.connect(db)
        try:
            for _ in range(30):
                if not kb.heartbeat_worker(conn, tid, expected_run_id=run_id):
                    errors.append("heartbeat fenced")
                    return
        finally:
            conn.close()
            stop.set()

    thread = threading.Thread(target=writer)
    thread.start()
    verdicts = []
    while not stop.is_set():
        conn = sentinel._connect_ro(db)
        try:
            verdicts.append(sentinel._classify(conn, _task_row(conn, tid), now, []))
        finally:
            conn.close()
    thread.join(timeout=10)

    assert not errors
    assert verdicts
    assert {item["verdict"] for item in verdicts} == {"healthy"}

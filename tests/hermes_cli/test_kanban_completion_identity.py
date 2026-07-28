from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claimed_delivery(conn):
    task_id = kb.create_task(
        conn,
        title="ship with attributable completion",
        assignee="developer",
        evidence_contract_na_reason="identity contract test",
    )
    task = kb.claim_task(conn, task_id)
    assert task is not None and task.current_run_id is not None
    return task_id, task.current_run_id


def _cardinalities(conn):
    return {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "tasks",
            "task_runs",
            "task_events",
            "completion_deliveries",
            "kanban_notify_subs",
        )
    }


@pytest.mark.parametrize(
    "summary",
    [
        "run {run_id} · missing task identity",
        "{task_id} · missing run identity",
        "t_wrong/run {run_id} · wrong task identity",
        "{task_id}/run 999999 · wrong run identity",
    ],
    ids=("missing-task", "missing-run", "wrong-task", "wrong-run"),
)
def test_claimed_delivery_rejects_noncanonical_completion_identity_without_side_effects(
    kanban_home, summary,
):
    with kb.connect_closing() as conn:
        task_id, run_id = _claimed_delivery(conn)
        before = _cardinalities(conn)

        with pytest.raises(ValueError, match="authoritative completion identity"):
            kb.complete_task(
                conn,
                task_id,
                summary=summary.format(task_id=task_id, run_id=run_id),
            )

        after = _cardinalities(conn)
        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)

    assert after == before
    assert task is not None and task.status == "running"
    assert task.current_run_id == run_id
    assert run is not None and run.status == "running" and run.ended_at is None


def test_claimed_delivery_accepts_authoritative_canonical_identity(kanban_home):
    with kb.connect_closing() as conn:
        task_id, run_id = _claimed_delivery(conn)
        summary = f"{task_id}/run {run_id} · tests green · production verified"

        assert kb.complete_task(conn, task_id, summary=summary)

        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)
        completed = [event for event in kb.list_events(conn, task_id) if event.kind == "completed"]

    assert task is not None and task.status == "done"
    assert run is not None and run.summary == summary
    assert len(completed) == 1
    assert completed[0].payload["summary"] == summary


def test_dashboard_style_completion_revalidates_identity_under_terminal_lock(
    kanban_home, monkeypatch,
):
    with kb.connect_closing() as conn:
        task_id, old_run_id = _claimed_delivery(conn)
        old_summary = f"{task_id}/run {old_run_id} · stale dashboard submission"

        def replace_run_between_preflight_and_terminal_lock(*_args):
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
                    "ended_at=2 WHERE id=?",
                    (old_run_id,),
                )
                cursor = conn.execute(
                    "INSERT INTO task_runs "
                    "(task_id, profile, status, started_at) "
                    "VALUES (?, 'developer', 'running', 3)",
                    (task_id,),
                )
                conn.execute(
                    "UPDATE tasks SET current_run_id=? WHERE id=?",
                    (cursor.lastrowid, task_id),
                )
            return [], []

        monkeypatch.setattr(
            kb, "_verify_created_cards", replace_run_between_preflight_and_terminal_lock
        )

        with pytest.raises(ValueError, match=f"{task_id}/run {old_run_id + 1}"):
            kb.complete_task(
                conn,
                task_id,
                summary=old_summary,
                created_cards=["trigger-between-locks"],
            )

        task = kb.get_task(conn, task_id)
        completed = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "completed"
        ]

    assert task is not None and task.status == "running"
    assert task.current_run_id == old_run_id + 1
    assert completed == []


def test_unclaimed_and_system_inbox_completions_remain_compatible(kanban_home):
    with kb.connect_closing() as conn:
        legacy_id = kb.create_task(
            conn,
            title="legacy manual completion",
            evidence_contract_na_reason="legacy compatibility fixture",
        )
        inbox_id = kb.create_system_inbox_task(
            conn,
            title="persistent process inbox",
            initial_status="blocked",
        )

        assert kb.complete_task(conn, legacy_id, summary="legacy summary without identity")
        assert kb.complete_task(conn, inbox_id, summary="inbox maintenance summary")

    with kb.connect_closing() as conn:
        assert kb.latest_summary(conn, legacy_id) == "legacy summary without identity"
        assert kb.latest_summary(conn, inbox_id) == "inbox maintenance summary"


def test_cli_surfaces_claimed_delivery_identity_rejection(kanban_home):
    with kb.connect_closing() as conn:
        task_id, _ = _claimed_delivery(conn)

    output = kc.run_slash(
        f"complete {task_id} --summary 'summary without task and run'"
    )

    assert "authoritative completion identity" in output
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "running"


def test_dashboard_api_returns_conflict_for_claimed_delivery_identity_rejection(
    kanban_home,
):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException
    from plugins.kanban.dashboard import plugin_api

    with kb.connect_closing() as conn:
        task_id, _ = _claimed_delivery(conn)

    with pytest.raises(HTTPException) as exc_info:
        plugin_api.update_task(
            task_id,
            plugin_api.UpdateTaskBody(
                status="done", summary="summary without task and run"
            ),
            board="default",
        )

    assert exc_info.value.status_code == 409
    assert "authoritative completion identity" in str(exc_info.value.detail)
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "running"

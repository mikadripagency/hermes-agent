"""Done-time completion delivery hardening (t_18088232).

``complete_task`` must always (a) resolve the shared orchestration channel
subscription so the terminal Done ping reaches #orchestration and not only
the origin thread, and (b) write a durable ``completion_deliveries`` ledger
row — for EVERY task, no task-kind gating — that the gateway notifier
upgrades to ``acknowledged`` with the real platform receipt.
``notify-reconcile`` stays the manual backstop, not the primary path.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


ORCH_CHAT_ID = "C0ORCHTEST"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_channel_directory(home: Path, *, name="orchestration", chat_id=ORCH_CHAT_ID):
    (home / "channel_directory.json").write_text(
        json.dumps(
            {
                "updated_at": None,
                "platforms": {
                    "slack": [
                        {"id": chat_id, "name": name, "type": "private"},
                        {"id": "D0OTHER", "name": "someone", "type": "dm"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def _completed_event_id(conn, task_id):
    return next(
        event.id for event in kb.list_events(conn, task_id)
        if event.kind == "completed"
    )


# --- done-time ledger row ----------------------------------------------------

def test_complete_task_writes_pending_ledger_row_for_every_task(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="plain legacy-style task")
        assert kb.complete_task(conn, tid, summary="done")
        event_id = _completed_event_id(conn, tid)
        row = conn.execute(
            "SELECT * FROM completion_deliveries WHERE event_id = ?", (event_id,)
        ).fetchone()
        assert row is not None
        assert row["task_id"] == tid
        assert row["state"] == "pending"
        assert row["handoff_version"] == 1
        assert row["receipt_id"] is None


def test_init_migrates_legacy_single_event_delivery_key(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy-completion-ledger.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE completion_deliveries ("
            "event_id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, "
            "handoff_version INTEGER NOT NULL, platform TEXT, chat_id TEXT, "
            "thread_id TEXT, notifier_profile TEXT, "
            "state TEXT NOT NULL DEFAULT 'pending', receipt_id TEXT, "
            "created_at INTEGER NOT NULL, acknowledged_at INTEGER)"
        )
        conn.execute(
            "INSERT INTO completion_deliveries VALUES "
            "(42, 't_legacy', 1, 'slack', 'CDM', '', 'developer', "
            "'acknowledged', 'legacy-receipt', 1, 2)"
        )
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        pk = [
            row["name"] for row in conn.execute(
                "PRAGMA table_info(completion_deliveries)"
            ) if row["pk"]
        ]
        preserved = conn.execute(
            "SELECT chat_id, receipt_id FROM completion_deliveries WHERE event_id = 42"
        ).fetchone()
        conn.execute(
            "INSERT INTO completion_deliveries "
            "(event_id, task_id, handoff_version, platform, chat_id, thread_id, "
            "notifier_profile, state, receipt_id, created_at, acknowledged_at) "
            "VALUES (42, 't_legacy', 1, 'slack', 'CORCH', '', 'developer', "
            "'acknowledged', 'sibling-receipt', 1, 2)"
        )
        count = conn.execute(
            "SELECT COUNT(*) FROM completion_deliveries WHERE event_id = 42"
        ).fetchone()[0]
    assert pk == ["event_id", "platform", "chat_id", "thread_id", "notifier_profile"]
    assert dict(preserved) == {"chat_id": "CDM", "receipt_id": "legacy-receipt"}
    assert count == 2


# --- orchestration route resolution -------------------------------------------

def test_complete_task_installs_orchestration_subscription(kanban_home):
    _write_channel_directory(kanban_home)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="route me")
        assert kb.complete_task(conn, tid, summary="done")
        event_id = _completed_event_id(conn, tid)

        subs = kb.list_notify_subs(conn, tid)
        orch = [s for s in subs if s["chat_id"] == ORCH_CHAT_ID]
        assert len(orch) == 1
        assert orch[0]["platform"] == "slack"
        # Cursor parked just before the completed event: exactly the Done
        # ping is delivered, none of the task's earlier terminal history.
        assert orch[0]["last_event_id"] == event_id - 1
        _, _, events = kb.claim_unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="slack",
            chat_id=ORCH_CHAT_ID,
            kinds=("completed", "blocked", "gave_up", "crashed", "timed_out",
                   "status", "archived", "unblocked"),
        )
        assert [e.id for e in events] == [event_id]
        assert events[0].kind == "completed"


def test_orchestration_subscription_is_idempotent_and_keeps_cursor(kanban_home):
    _write_channel_directory(kanban_home)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="already subscribed")
        kb.add_notify_sub(conn, task_id=tid, platform="slack", chat_id=ORCH_CHAT_ID)
        assert kb.complete_task(conn, tid, summary="done")
        orch = [
            s for s in kb.list_notify_subs(conn, tid)
            if s["chat_id"] == ORCH_CHAT_ID
        ]
        assert len(orch) == 1
        # Pre-existing subscription keeps its own cursor (0 here).
        assert orch[0]["last_event_id"] == 0


def test_orchestration_route_noop_without_channel_directory(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="no directory")
        assert kb.complete_task(conn, tid, summary="done")
        assert kb.list_notify_subs(conn, tid) == []


def test_orchestration_route_disabled_by_empty_config(kanban_home):
    _write_channel_directory(kanban_home)
    (kanban_home / "config.yaml").write_text(
        "kanban:\n  orchestration_channel: ''\n", encoding="utf-8"
    )
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="opted out")
        assert kb.complete_task(conn, tid, summary="done")
        assert kb.list_notify_subs(conn, tid) == []


def test_orchestration_route_accepts_raw_chat_id_config(kanban_home):
    _write_channel_directory(kanban_home)
    (kanban_home / "config.yaml").write_text(
        f"kanban:\n  orchestration_channel: '{ORCH_CHAT_ID}'\n", encoding="utf-8"
    )
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="raw id route")
        assert kb.complete_task(conn, tid, summary="done")
        orch = [
            s for s in kb.list_notify_subs(conn, tid)
            if s["chat_id"] == ORCH_CHAT_ID
        ]
        assert len(orch) == 1


# --- primary acknowledgement path ---------------------------------------------

def _complete_with_pending_row(conn):
    tid = kb.create_task(conn, title="ack me")
    assert kb.complete_task(conn, tid, summary="done")
    return tid, _completed_event_id(conn, tid)


def test_acknowledge_upgrades_pending_row(kanban_home):
    with kb.connect_closing() as conn:
        tid, event_id = _complete_with_pending_row(conn)
        assert kb.acknowledge_completion_delivery(
            conn,
            task_id=tid,
            event_id=event_id,
            platform="slack",
            chat_id=ORCH_CHAT_ID,
            thread_id="",
            notifier_profile="developer",
            receipt_id="1784300000.000001",
        )
        row = conn.execute(
            "SELECT * FROM completion_deliveries WHERE event_id = ?", (event_id,)
        ).fetchone()
        assert row["state"] == "acknowledged"
        assert row["receipt_id"] == "1784300000.000001"
        assert row["chat_id"] == ORCH_CHAT_ID
        assert row["acknowledged_at"] is not None
        acked = [
            e for e in kb.list_events(conn, tid)
            if e.kind == "completion_delivery_acknowledged"
        ]
        assert len(acked) == 1
        assert acked[0].payload["completed_event_id"] == event_id


def test_acknowledge_inserts_row_for_pre_ledger_completions(kanban_home):
    with kb.connect_closing() as conn:
        tid, event_id = _complete_with_pending_row(conn)
        # Simulate a completion recorded before the done-time ledger shipped.
        conn.execute("DELETE FROM completion_deliveries WHERE event_id = ?", (event_id,))
        conn.commit()
        assert kb.acknowledge_completion_delivery(
            conn,
            task_id=tid,
            event_id=event_id,
            platform="slack",
            chat_id=ORCH_CHAT_ID,
            receipt_id="1784300000.000002",
        )
        row = conn.execute(
            "SELECT state, receipt_id FROM completion_deliveries WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        assert dict(row) == {
            "state": "acknowledged",
            "receipt_id": "1784300000.000002",
        }


def test_acknowledge_is_idempotent_per_route_and_keeps_siblings_independent(kanban_home):
    with kb.connect_closing() as conn:
        tid, event_id = _complete_with_pending_row(conn)
        kwargs = dict(
            task_id=tid,
            event_id=event_id,
            platform="slack",
            chat_id=ORCH_CHAT_ID,
            thread_id="",
            notifier_profile="developer",
        )
        assert kb.acknowledge_completion_delivery(
            conn, receipt_id="first-receipt", **kwargs
        )
        # Same route + receipt: idempotent True, no second audit event.
        assert kb.acknowledge_completion_delivery(
            conn, receipt_id="first-receipt", **kwargs
        )
        # A different receipt cannot overwrite this route.
        assert not kb.acknowledge_completion_delivery(
            conn, receipt_id="second-receipt", **kwargs
        )
        # A sibling destination has its own receipt and audit event.
        assert kb.acknowledge_completion_delivery(
            conn,
            task_id=tid,
            event_id=event_id,
            platform="slack",
            chat_id="COTHER",
            receipt_id="other-route",
        )
        rows = conn.execute(
            "SELECT chat_id, receipt_id FROM completion_deliveries "
            "WHERE event_id = ? ORDER BY chat_id",
            (event_id,),
        ).fetchall()
        assert [dict(row) for row in rows] == [
            {"chat_id": ORCH_CHAT_ID, "receipt_id": "first-receipt"},
            {"chat_id": "COTHER", "receipt_id": "other-route"},
        ]
        acked = [
            e for e in kb.list_events(conn, tid)
            if e.kind == "completion_delivery_acknowledged"
        ]
        assert len(acked) == 2


def test_acknowledge_rejects_non_completed_event_or_wrong_task(kanban_home):
    with kb.connect_closing() as conn:
        tid, event_id = _complete_with_pending_row(conn)
        other = kb.create_task(conn, title="other task")
        assert not kb.acknowledge_completion_delivery(
            conn,
            task_id=other,
            event_id=event_id,
            platform="slack",
            chat_id=ORCH_CHAT_ID,
            receipt_id="r",
        )
        created_event = next(
            e.id for e in kb.list_events(conn, tid) if e.kind == "created"
        )
        assert not kb.acknowledge_completion_delivery(
            conn,
            task_id=tid,
            event_id=created_event,
            platform="slack",
            chat_id=ORCH_CHAT_ID,
            receipt_id="r",
        )


# --- reconcile stays the backstop ---------------------------------------------

def test_reconcile_backstop_still_upgrades_done_time_pending_row(kanban_home):
    """A completion whose notifier ack was missed (pending ledger row +
    durable subscription receipt) is still repairable via the manual
    reconcile path — the done-time pending row must not block it."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="backstop", assignee="developer")
        kb.add_notify_sub(
            conn, task_id=tid, platform="slack", chat_id=ORCH_CHAT_ID,
            notifier_profile="developer",
        )
        assert kb.complete_task(conn, tid, summary="done")
        event_id = _completed_event_id(conn, tid)
        _, cursor, events = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="slack", chat_id=ORCH_CHAT_ID,
            kinds=("completed",),
        )
        assert [e.id for e in events] == [event_id]
        kb.advance_notify_cursor(
            conn, task_id=tid, platform="slack", chat_id=ORCH_CHAT_ID,
            new_cursor=cursor, message_id="1784300000.000003",
            message_event_id=event_id,
        )
        assert kb.reconcile_completion_delivery(
            conn,
            task_id=tid,
            event_id=event_id,
            platform="slack",
            chat_id=ORCH_CHAT_ID,
            thread_id="",
            notifier_profile="developer",
        ) == "1784300000.000003"
        row = conn.execute(
            "SELECT state, receipt_id FROM completion_deliveries WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        assert dict(row) == {
            "state": "acknowledged",
            "receipt_id": "1784300000.000003",
        }

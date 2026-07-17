from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_completed_receipt(conn):
    task_id = kb.create_task(conn, title="historical completion", assignee="developer")
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="slack",
        chat_id="CORCH",
        notifier_profile="developer",
    )
    assert kb.complete_task(conn, task_id, summary="done")
    event = next(event for event in kb.list_events(conn, task_id) if event.kind == "completed")
    old_cursor, cursor, events = kb.claim_unseen_events_for_sub(
        conn,
        task_id=task_id,
        platform="slack",
        chat_id="CORCH",
        kinds=("completed",),
    )
    assert old_cursor < cursor
    assert [item.id for item in events] == [event.id]
    kb.advance_notify_cursor(
        conn,
        task_id=task_id,
        platform="slack",
        chat_id="CORCH",
        new_cursor=cursor,
        message_id="1784203491.451939",
        message_event_id=event.id,
    )
    return task_id, event.id


def _reconcile(conn, task_id, completed_event_id, **overrides):
    route = {
        "task_id": task_id,
        "event_id": completed_event_id,
        "platform": "slack",
        "chat_id": "CORCH",
        "thread_id": "",
        "notifier_profile": "developer",
    }
    route.update(overrides)
    return kb.reconcile_completion_delivery(
        conn,
        task_id=str(route["task_id"]),
        event_id=int(route["event_id"]),
        platform=str(route["platform"]),
        chat_id=str(route["chat_id"]),
        thread_id=str(route["thread_id"]),
        notifier_profile=str(route["notifier_profile"]),
    )


def test_reconcile_completion_delivery_from_authoritative_receipt_is_idempotent(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)

        assert _reconcile(conn, task_id, event_id) == "1784203491.451939"
        assert _reconcile(conn, task_id, event_id) == "1784203491.451939"

        row = conn.execute(
            "SELECT * FROM completion_deliveries WHERE event_id = ?", (event_id,)
        ).fetchone()
        assert dict(row) | {"created_at": 0, "acknowledged_at": 0} == {
            "event_id": event_id,
            "task_id": task_id,
            "handoff_version": 1,
            "platform": "slack",
            "chat_id": "CORCH",
            "thread_id": "",
            "notifier_profile": "developer",
            "state": "acknowledged",
            "receipt_id": "1784203491.451939",
            "created_at": 0,
            "acknowledged_at": 0,
        }
        reconciled = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "completion_delivery_reconciled"
        ]
        assert len(reconciled) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"event_id": 999999},
        {"chat_id": "COTHER"},
        {"platform": "telegram"},
        {"notifier_profile": "default"},
    ],
)
def test_reconcile_completion_delivery_rejects_event_route_and_profile_mismatch(
    kanban_home, overrides
):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        with pytest.raises(ValueError):
            _reconcile(conn, task_id, event_id, **overrides)
        assert conn.execute(
            "SELECT COUNT(*) FROM completion_deliveries"
        ).fetchone()[0] == 0


def test_reconcile_completion_delivery_rejects_conflicting_existing_delivery(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        conn.execute(
            "INSERT INTO completion_deliveries "
            "(event_id, task_id, handoff_version, platform, chat_id, thread_id, "
            "notifier_profile, state, receipt_id, created_at, acknowledged_at) "
            "VALUES (?, ?, 1, 'slack', 'COTHER', '', 'developer', "
            "'acknowledged', 'different-receipt', 1, 1)",
            (event_id, task_id),
        )

        with pytest.raises(ValueError, match="conflicting"):
            _reconcile(conn, task_id, event_id)


def test_reconcile_completion_delivery_rejects_task_mismatch_and_blank_receipt(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        other_task_id = kb.create_task(conn, title="other", assignee="developer")

        with pytest.raises(ValueError, match="does not match"):
            _reconcile(conn, other_task_id, event_id)

        conn.execute(
            "UPDATE kanban_notify_subs SET last_message_id = '   ' WHERE task_id = ?",
            (task_id,),
        )
        with pytest.raises(ValueError, match="unambiguous"):
            _reconcile(conn, task_id, event_id)


def test_reconcile_completion_delivery_rejects_ambiguous_event_receipts(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="slack",
            chat_id="COTHER",
            notifier_profile="developer",
        )
        _, cursor, events = kb.claim_unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform="slack",
            chat_id="COTHER",
            kinds=("completed",),
        )
        assert [event.id for event in events] == [event_id]
        kb.advance_notify_cursor(
            conn,
            task_id=task_id,
            platform="slack",
            chat_id="COTHER",
            new_cursor=cursor,
            message_id="1784203492.000001",
            message_event_id=event_id,
        )

        with pytest.raises(ValueError, match="unambiguous"):
            _reconcile(conn, task_id, event_id)
        assert conn.execute(
            "SELECT COUNT(*) FROM completion_deliveries"
        ).fetchone()[0] == 0


def test_reconcile_completion_delivery_does_not_requeue_or_duplicate_send(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        before = kb.list_notify_subs(conn, task_id)[0]

        _reconcile(conn, task_id, event_id)

        after = kb.list_notify_subs(conn, task_id)[0]
        assert after["last_event_id"] == before["last_event_id"] == event_id
        assert after["pending_event_id"] is None
        _, _, events = kb.claim_unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform="slack",
            chat_id="CORCH",
            kinds=("completed",),
        )
        assert events == []

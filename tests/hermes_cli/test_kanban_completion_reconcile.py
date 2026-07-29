from pathlib import Path

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb

pytestmark = pytest.mark.usefixtures(
    "explicit_delivery_contract_for_kanban_fixtures",
    "claimed_completion_for_kanban_fixtures",
)


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


def test_verify_legacy_slack_sender_requires_receipt_author_match(monkeypatch):
    monkeypatch.setattr(cli, "_slack_tokens_for_active_profile", lambda: ["secret"])

    def fake_api(_token, method, **_params):
        if method == "auth.test":
            return {"ok": True, "user_id": "UDEFAULT"}
        return {
            "ok": True,
            "messages": [{"ts": "1784203491.451939", "user": "UDEFAULT"}],
        }

    monkeypatch.setattr(cli, "_slack_api", fake_api)
    assert cli._verify_legacy_slack_sender(
        "CORCH", "1784203491.451939"
    ) == "UDEFAULT"


def test_verify_legacy_slack_sender_rejects_other_actor(monkeypatch):
    monkeypatch.setattr(cli, "_slack_tokens_for_active_profile", lambda: ["secret"])
    monkeypatch.setattr(
        cli,
        "_slack_api",
        lambda _token, method, **_params: (
            {"ok": True, "user_id": "UDEFAULT"}
            if method == "auth.test"
            else {"ok": True, "messages": [{"ts": "1.2", "user": "UOTHER"}]}
        ),
    )
    with pytest.raises(RuntimeError, match="not authored"):
        cli._verify_legacy_slack_sender("CORCH", "1.2")


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


def test_reconcile_verified_legacy_sender_keeps_unstamped_route_truthful(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        conn.execute(
            "UPDATE kanban_notify_subs SET notifier_profile = NULL WHERE task_id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE completion_deliveries SET notifier_profile = '' WHERE event_id = ?",
            (event_id,),
        )
        conn.commit()

        assert kb.reconcile_completion_delivery(
            conn,
            task_id=task_id,
            event_id=event_id,
            platform="slack",
            chat_id="CORCH",
            notifier_profile="default",
            verified_legacy_sender_id="UDEFAULT",
            verified_legacy_receipt_id="1784203491.451939",
        ) == "1784203491.451939"

        row = conn.execute(
            "SELECT notifier_profile, state, receipt_id FROM completion_deliveries "
            "WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        assert dict(row) == {
            "notifier_profile": "",
            "state": "acknowledged",
            "receipt_id": "1784203491.451939",
        }
        event = next(
            item for item in kb.list_events(conn, task_id)
            if item.kind == "completion_delivery_reconciled"
        )
        assert event.payload["notifier_profile"] == ""
        assert event.payload["verified_legacy_sender_profile"] == "default"
        assert event.payload["verified_legacy_sender_id"] == "UDEFAULT"


def test_reconcile_legacy_receipt_change_is_rejected(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        conn.execute(
            "UPDATE kanban_notify_subs SET notifier_profile = NULL WHERE task_id = ?",
            (task_id,),
        )
        conn.execute(
            "UPDATE completion_deliveries SET notifier_profile = '' WHERE event_id = ?",
            (event_id,),
        )
        conn.commit()

        with pytest.raises(ValueError, match="changed before reconciliation"):
            kb.reconcile_completion_delivery(
                conn,
                task_id=task_id,
                event_id=event_id,
                platform="slack",
                chat_id="CORCH",
                notifier_profile="default",
                verified_legacy_sender_id="UDEFAULT",
                verified_legacy_receipt_id="different-receipt",
            )
        assert conn.execute(
            "SELECT state FROM completion_deliveries WHERE event_id = ?",
            (event_id,),
        ).fetchone()["state"] == "pending"


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
        # complete_task writes a pending done-time ledger row; a rejected
        # reconcile must never have acknowledged it.
        assert conn.execute(
            "SELECT COUNT(*) FROM completion_deliveries WHERE state = 'acknowledged'"
        ).fetchone()[0] == 0


def test_reconcile_completion_delivery_preserves_sibling_route(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        # Turn the done-time pending row into an acknowledged sibling route.
        conn.execute(
            "UPDATE completion_deliveries SET platform = 'slack', "
            "chat_id = 'COTHER', thread_id = '', notifier_profile = 'developer', "
            "state = 'acknowledged', receipt_id = 'different-receipt', "
            "acknowledged_at = 1 WHERE event_id = ?",
            (event_id,),
        )

        assert _reconcile(conn, task_id, event_id) == "1784203491.451939"
        rows = conn.execute(
            "SELECT chat_id, receipt_id FROM completion_deliveries "
            "WHERE event_id = ? ORDER BY chat_id",
            (event_id,),
        ).fetchall()
        assert [dict(row) for row in rows] == [
            {"chat_id": "CORCH", "receipt_id": "1784203491.451939"},
            {"chat_id": "COTHER", "receipt_id": "different-receipt"},
        ]


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


def test_reconcile_completion_delivery_rejects_in_flight_receipt(kanban_home):
    with kb.connect_closing() as conn:
        task_id, event_id = _seed_completed_receipt(conn)
        conn.execute(
            "UPDATE kanban_notify_subs SET pending_event_id = ? WHERE task_id = ?",
            (event_id, task_id),
        )

        with pytest.raises(ValueError, match="not durably acknowledged"):
            _reconcile(conn, task_id, event_id)
        assert conn.execute(
            "SELECT COUNT(*) FROM completion_deliveries WHERE state = 'acknowledged'"
        ).fetchone()[0] == 0


def test_reconcile_completion_delivery_uses_exact_route_with_unrelated_receipts(kanban_home):
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

        assert _reconcile(conn, task_id, event_id) == "1784203491.451939"
        row = conn.execute(
            "SELECT chat_id, receipt_id FROM completion_deliveries WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        assert dict(row) == {
            "chat_id": "CORCH",
            "receipt_id": "1784203491.451939",
        }


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

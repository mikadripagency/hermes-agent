"""Block-time delivery hardening — the "silently-parked blocked task" fix.

Before this, a task that transitioned to ``blocked`` emitted a ``blocked``
event but had **no** delivery ledger row and often **no** durable
``#orchestration`` subscription, so the block question never reached the owner
and the task sat parked and unwatched (tasks stuck 8-39h). ``block_task`` now
gives a human-facing ``blocked`` event the same durable delivery guarantees
``complete_task`` gives a ``completed`` event:

* a durable ``#orchestration`` Slack subscription (cursor just before the
  blocked event) so exactly the block ping is delivered, and
* a pending ``completion_deliveries`` ledger row (the table is reused; rows are
  keyed by the blocked event's own distinct ``event_id``, so they never collide
  with completion rows), upgraded to ``acknowledged`` by
  :func:`acknowledge_blocked_delivery` once the real platform receipt lands.

The gateway silent-until-Done filter is also adjusted so a ``blocked`` event
passes through even for ``developer``-assigned tasks (that filter is what muted
these), while all other non-completed kinds stay suppressed for developer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb

pytestmark = pytest.mark.usefixtures(
    "explicit_delivery_contract_for_kanban_fixtures",
    "claimed_completion_for_kanban_fixtures",
)


ORCH_CHAT_ID = "C0ORCHTEST"

_ALL_KINDS = (
    "completed", "blocked", "gave_up", "crashed", "timed_out",
    "status", "archived", "unblocked",
)


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


def _running_task(conn, *, title="t", assignee="worker"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer=assignee)
    assert claimed is not None
    return tid


def _blocked_event_id(conn, task_id):
    return next(
        event.id for event in kb.list_events(conn, task_id)
        if event.kind == "blocked"
    )


# --- durable block delivery ledger + subscription ----------------------------

def test_block_writes_pending_ledger_row_for_every_task(kanban_home):
    """Even with no orchestration route, a route-less pending row is written."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="need a key", kind="capability")
        event_id = _blocked_event_id(conn, tid)
        row = conn.execute(
            "SELECT * FROM completion_deliveries WHERE event_id = ?", (event_id,)
        ).fetchone()
        assert row is not None
        assert row["task_id"] == tid
        assert row["state"] == "pending"
        assert row["receipt_id"] is None


def test_block_installs_orchestration_subscription(kanban_home):
    _write_channel_directory(kanban_home)
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="which env?", kind="needs_input")
        event_id = _blocked_event_id(conn, tid)

        orch = [s for s in kb.list_notify_subs(conn, tid) if s["chat_id"] == ORCH_CHAT_ID]
        assert len(orch) == 1
        assert orch[0]["platform"] == "slack"
        # Cursor parked just before the blocked event: exactly the block ping
        # is delivered, none of the task's earlier history.
        assert orch[0]["last_event_id"] == event_id - 1
        _, _, events = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="slack", chat_id=ORCH_CHAT_ID, kinds=_ALL_KINDS,
        )
        assert [e.id for e in events] == [event_id]
        assert events[0].kind == "blocked"

        # A per-route pending ledger row is materialized for the orchestration sub.
        delivery = conn.execute(
            "SELECT notifier_profile, state FROM completion_deliveries "
            "WHERE event_id = ? AND chat_id = ?",
            (event_id, ORCH_CHAT_ID),
        ).fetchone()
        assert delivery["state"] == "pending"


def test_block_stamps_orchestration_route_with_worker_profile(kanban_home):
    _write_channel_directory(kanban_home)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="dev block", assignee="developer")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        claimed = kb.claim_task(conn, tid, claimer="developer-test")
        assert claimed is not None
        assert kb.block_task(
            conn, tid, reason="need input", kind="needs_input",
            expected_run_id=claimed.current_run_id,
        )
        orch = [s for s in kb.list_notify_subs(conn, tid) if s["chat_id"] == ORCH_CHAT_ID]
        assert len(orch) == 1
        assert orch[0]["notifier_profile"] == "developer"


# --- ack path + cross-kind guards --------------------------------------------

def test_acknowledge_blocked_upgrades_pending_row(kanban_home):
    _write_channel_directory(kanban_home)
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="developer")
        assert kb.block_task(conn, tid, reason="need input", kind="needs_input")
        event_id = _blocked_event_id(conn, tid)
        assert kb.acknowledge_blocked_delivery(
            conn, task_id=tid, event_id=event_id, platform="slack",
            chat_id=ORCH_CHAT_ID, notifier_profile="developer", receipt_id="R1",
        )
        row = conn.execute(
            "SELECT state, receipt_id FROM completion_deliveries "
            "WHERE event_id = ? AND chat_id = ?",
            (event_id, ORCH_CHAT_ID),
        ).fetchone()
        assert row["state"] == "acknowledged"
        assert row["receipt_id"] == "R1"


def test_acknowledge_blocked_rejects_completed_event(kanban_home):
    """The blocked ack must never acknowledge a completion row."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="done not blocked", assignee="developer")
        assert kb.complete_task(conn, tid, summary="done")
        completed_id = next(
            e.id for e in kb.list_events(conn, tid) if e.kind == "completed"
        )
        assert not kb.acknowledge_blocked_delivery(
            conn, task_id=tid, event_id=completed_id, platform="slack",
            chat_id="C1", notifier_profile="developer", receipt_id="R1",
        )


def test_acknowledge_completion_rejects_blocked_event(kanban_home):
    """The completion ack must never acknowledge a blocked row (cross-guard)."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn, assignee="developer")
        assert kb.block_task(conn, tid, reason="need input", kind="needs_input")
        event_id = _blocked_event_id(conn, tid)
        assert not kb.acknowledge_completion_delivery(
            conn, task_id=tid, event_id=event_id, platform="slack",
            chat_id="C1", notifier_profile="developer", receipt_id="R1",
        )


# --- non-human-facing blocks are NOT delivered -------------------------------

def test_dependency_block_installs_no_delivery(kanban_home):
    """A dependency wait routes to todo and never gets a delivery/sub."""
    _write_channel_directory(kanban_home)
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="waiting on parent", kind="dependency")
        assert kb.get_task(conn, tid).status == "todo"
        # No 'blocked' event, no orchestration sub, no delivery ledger rows.
        assert not any(e.kind == "blocked" for e in kb.list_events(conn, tid))
        assert not kb.list_notify_subs(conn, tid)
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM completion_deliveries WHERE task_id = ?", (tid,)
        ).fetchone()
        assert rows["n"] == 0


def test_loop_breaker_triage_installs_no_blocked_delivery(kanban_home):
    """When the loop breaker routes to triage it emits block_loop_detected,
    not 'blocked', so no block delivery is installed for that transition."""
    _write_channel_directory(kanban_home)
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        # Drive same-cause re-blocks up to the recurrence limit.
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT):
            kb.block_task(conn, tid, reason="x", kind="capability")
            if kb.get_task(conn, tid).status == "blocked":
                kb.unblock_task(conn, tid)
                with kb.write_txn(conn):
                    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
                kb.claim_task(conn, tid, claimer="worker")
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        assert any(e.kind == "block_loop_detected" for e in kb.list_events(conn, tid))
        # The final (triage) transition did not add a 'blocked' event.
        loop_ev = next(
            e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"
        )
        assert not any(
            e.kind == "blocked" and e.id > loop_ev.id
            for e in kb.list_events(conn, tid)
        )


# --- gateway silent-until-Done filter ----------------------------------------

async def _run_notifier_once(runner, watcher_coro):
    _orig_sleep = asyncio.sleep

    async def _fast_sleep(_):
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep):
        await asyncio.wait_for(watcher_coro, timeout=10.0)


@pytest.mark.asyncio
async def test_developer_blocked_passes_silent_filter_and_is_acked(kanban_home):
    """A developer-assigned blocked event is delivered (not muted) and the
    delivery ledger row is acknowledged with the real receipt."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dev blocked", assignee="developer")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            notifier_profile="developer",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        claimed = kb.claim_task(conn, tid, claimer="developer")
        assert kb.block_task(
            conn, tid, reason="which region?", kind="needs_input",
            expected_run_id=claimed.current_run_id,
        )
        event_id = _blocked_event_id(conn, tid)
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_notifier_profile = "developer"

    fake_adapter = MagicMock()

    async def _send_and_stop(chat_id, msg, metadata=None):
        runner._running = False
        return MagicMock(success=True, message_id="RBLOCK1")

    fake_adapter.send = AsyncMock(side_effect=_send_and_stop)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    await _run_notifier_once(runner, runner._kanban_notifier_watcher(interval=1))

    fake_adapter.send.assert_called_once()
    assert "blocked" in fake_adapter.send.call_args[0][1]

    conn = kb.connect()
    try:
        row = conn.execute(
            "SELECT state, receipt_id FROM completion_deliveries WHERE event_id = ?",
            (event_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["state"] == "acknowledged"
    assert row["receipt_id"] == "RBLOCK1"


@pytest.mark.asyncio
async def test_developer_crashed_still_suppressed(kanban_home):
    """Non-completed, non-blocked kinds stay silent for developer tasks."""
    from gateway.run import GatewayRunner
    from gateway.config import Platform

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dev crashed", assignee="developer")
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat1",
            notifier_profile="developer",
        )
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._kanban_sub_fail_counts = {}
    runner._kanban_notifier_profile = "developer"

    fake_adapter = MagicMock()
    ticks = {"n": 0}

    async def _send(chat_id, msg, metadata=None):
        return MagicMock(success=True, message_id="X")

    fake_adapter.send = AsyncMock(side_effect=_send)
    runner.adapters = {Platform.TELEGRAM: fake_adapter}

    _orig_sleep = asyncio.sleep

    async def _fast_sleep_and_stop(_):
        # sleep#1 is the notifier's initial start-up delay (before the while
        # loop); leave running=True through it so one full tick's _collect
        # runs, then stop on the next sleep.
        ticks["n"] += 1
        if ticks["n"] >= 2:
            runner._running = False
        await _orig_sleep(0)

    with patch("gateway.run.asyncio.sleep", side_effect=_fast_sleep_and_stop):
        await asyncio.wait_for(
            runner._kanban_notifier_watcher(interval=1), timeout=10.0
        )

    # crashed is suppressed for developer → no send.
    fake_adapter.send.assert_not_called()

import asyncio
from pathlib import Path
from types import SimpleNamespace


from gateway.config import Platform
from gateway.kanban_watchers import _format_completed_message
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return SimpleNamespace(success=True, message_id="1783796000.123456")


def test_completed_message_is_concise_and_omits_technical_ids():
    message = _format_completed_message(
        "Agent Board bereinigen",
        "PR #158 ist gemerged. Vercel Production läuft auf ff04ae4 und wurde geprüft.",
    )

    assert message == "Agent Board bereinigen fertig. PR #158 gemerged, Production geprüft."
    assert "t_" not in message and "ff04ae4" not in message


def test_completed_message_collapses_lines_and_strips_technical_ids():
    message = _format_completed_message("Task t_94f38ab5\ncommit deadbeef rollout", "")
    assert message == "Task commit rollout fertig."
    assert len(message.splitlines()) == 1


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter, platform=Platform.TELEGRAM):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {platform: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_dedupes_board_slugs_pointing_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["text"] == "notify once fertig."


def test_kanban_notifier_claim_prevents_second_watcher_send(tmp_path, monkeypatch):
    db_path = tmp_path / "single-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter1)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter2)))

    assert len(adapter1.sent) == 1
    assert adapter2.sent == []


def test_late_subscription_delivers_completion_and_records_receipt(tmp_path, monkeypatch):
    db_path = tmp_path / "late-subscription.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="late", assignee="worker")
        kb.complete_task(conn, tid, summary="done")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    conn = kb.connect()
    try:
        sub = kb.list_notify_subs(conn, tid)[0]
    finally:
        conn.close()
    assert len(adapter.sent) == 1
    assert sub["last_message_id"] == "1783796000.123456"
    assert sub["last_message_event_id"] == sub["last_event_id"]


def test_slack_completion_without_receipt_stays_retryable(tmp_path, monkeypatch):
    db_path = tmp_path / "missing-receipt.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="missing receipt", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="slack", chat_id="C123")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    adapter.send = lambda *args, **kwargs: asyncio.sleep(0, result=SimpleNamespace(success=True, message_id=None))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter, Platform.SLACK)))

    conn = kb.connect()
    try:
        sub = kb.list_notify_subs(conn, tid)[0]
    finally:
        conn.close()
    assert sub["last_event_id"] == 0
    assert sub["last_message_id"] is None
    assert [ev.kind for ev in _unseen_events_at(tid, "slack", "C123")] == ["completed"]


def _unseen_events_at(tid, platform, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn, task_id=tid, platform=platform, chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_notification_idempotency_key_is_thread_scoped(tmp_path, monkeypatch):
    db_path = tmp_path / "thread-scoped.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="threaded", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1", thread_id="A")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1", thread_id="B")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 2
    assert {item["metadata"]["thread_id"] for item in adapter.sent} == {"A", "B"}
    assert len({item["metadata"]["client_msg_id"] for item in adapter.sent}) == 2


def test_kanban_notifier_rewinds_claim_if_adapter_disconnects(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_db_path_is_test_isolated_from_real_home():
    hermes_home = Path(kb.kanban_home())
    production_db = Path.home() / ".hermes" / "kanban.db"
    assert kb.kanban_db_path().resolve() != production_db.resolve()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
    finally:
        conn.close()

    assert kb.kanban_db_path().resolve().is_relative_to(hermes_home.resolve())
    assert kb.kanban_db_path().resolve() != production_db.resolve()


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


def test_kanban_notifier_rewinds_claim_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter rewinds the claim so the next tick can retry.

    This is the second rewind path (distinct from the adapter-disconnect path
    in test_kanban_notifier_rewinds_claim_if_adapter_disconnects). Here the
    adapter is connected and the send call actually fires; the claim must
    still rewind so the event isn't lost when send() raises mid-tick.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Send was attempted (so we exercised the failure path, not just the
    # disconnect path) and the claim was rewound — the unseen-events query
    # still returns the event for retry on the next tick.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_notifier_keeps_subscription_after_repeated_send_failures(tmp_path, monkeypatch):
    db_path = tmp_path / "repeated-send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)
    for _ in range(3):
        runner._running = True
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    conn = kb.connect()
    try:
        assert len(kb.list_notify_subs(conn, tid)) == 1
    finally:
        conn.close()
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "crashed" in adapter.sent[0]["text"].lower()

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "crashed" in adapter.sent[1]["text"].lower()


def test_notifier_owning_profile_adapter_no_default_fallback(tmp_path, monkeypatch):
    """A subscription owned by a secondary profile whose profile-adapter
    registry entry EXISTS but lacks this platform must NOT fall back to the
    default profile's same-platform adapter — the notifier must route through
    the shared ``_authorization_adapter`` chokepoint, which forbids that
    fallback (gateway/authz_mixin.py). Delivering via the default profile's bot
    is the exact cross-profile mis-delivery this whole change exists to fix
    (`[230002] Bot can NOT be out of the chat`).

    Mutation check: reverting kanban_watchers.py's adapter selection to the old
    inline ``if adapter is None: adapter = self.adapters.get(plat)`` fallback
    makes this test FAIL (the default adapter receives the delivery).
    """
    db_path = tmp_path / "profile-no-fallback.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="owned by beta", assignee="worker")
        # Subscription is owned by profile "beta".
        kb.add_notify_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-beta",
            notifier_profile="beta",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    default_adapter = RecordingAdapter()
    other_adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    # Default profile has a telegram adapter …
    runner.adapters = {Platform.TELEGRAM: default_adapter}
    # … and profile "beta" HAS a non-empty registry entry (so it passes the
    # notifier's upstream skip-filter, which only skips owning profiles with NO
    # adapter at all), but that entry does NOT contain a telegram adapter — beta
    # connected a different platform (discord). The telegram sub owned by beta
    # must therefore resolve to NO adapter, not silently borrow the default
    # profile's telegram bot.
    runner._profile_adapters = {"beta": {Platform.DISCORD: other_adapter}}
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The default profile's adapter must never receive beta's notification.
    assert default_adapter.sent == [], (
        "Owning-profile subscription must not fall back to the default "
        f"profile's adapter; got {default_adapter.sent!r}"
    )
    assert other_adapter.sent == [], (
        f"beta's discord adapter must not receive a telegram sub; got {other_adapter.sent!r}"
    )
    # The claim is rewound (adapter resolved to None → treated as disconnected),
    # so the event is still unseen and will deliver once beta's adapter connects.
    assert [ev.kind for ev in _unseen_terminal_events_for(tid, "chat-beta")] == ["completed"]


def test_notifier_uses_active_profile_adapter_for_matching_owner(tmp_path, monkeypatch):
    """A stamped subscription owned by the active profile uses self.adapters."""
    db_path = tmp_path / "active-profile.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="active profile", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-active",
            notifier_profile="developer",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._profile_adapters = {}
    runner._kanban_notifier_profile = "developer"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [item["text"] for item in adapter.sent] == ["active profile fertig."]


def _unseen_terminal_events_for(tid, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Permanent-failure handling + backoff circuit breaker
# ---------------------------------------------------------------------------


class PermanentFailingAdapter:
    """Adapter whose send() raises a permanent Slack error (channel gone)."""

    def __init__(self, error="channel_not_found"):
        self.error = error
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError(self.error)


def _seed_sub_failure_state(tid, *, platform="slack", chat_id="C1",
                            consecutive_failures=0, next_retry_at=None):
    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE kanban_notify_subs SET consecutive_failures = ?, next_retry_at = ? "
                "WHERE task_id = ? AND platform = ? AND chat_id = ?",
                (consecutive_failures, next_retry_at, tid, platform, chat_id),
            )
    finally:
        conn.close()


def _read_sub(tid, *, platform="slack", chat_id="C1"):
    conn = kb.connect()
    try:
        for s in kb.list_notify_subs(conn, tid):
            if s["platform"] == platform and s["chat_id"] == chat_id:
                return s
        return None
    finally:
        conn.close()


def test_notify_retry_delay_backoff_is_capped():
    # Monotonic non-decreasing and capped at the 10-minute ceiling.
    delays = [kb.notify_retry_delay(n) for n in range(1, 20)]
    assert delays == sorted(delays)
    assert max(delays) == kb.NOTIFY_RETRY_BACKOFF_CAP
    assert delays[0] == kb.NOTIFY_RETRY_BACKOFF_BASE


def test_record_notify_failure_increments_and_sets_backoff(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "rec-fail.db"))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="slack", chat_id="C1")
        kb.complete_task(conn, tid, summary="done")
        old, new, events = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="slack", chat_id="C1", kinds=["completed"],
        )
        assert events
        n1 = kb.record_notify_failure(
            conn, task_id=tid, platform="slack", chat_id="C1",
            claimed_cursor=new, old_cursor=old, now=1000,
        )
        assert n1 == 1
        sub = kb.list_notify_subs(conn, tid)[0]
        assert sub["consecutive_failures"] == 1
        assert sub["next_retry_at"] == 1000 + kb.notify_retry_delay(1)
        # cursor rewound so the event is re-claimable
        assert sub["last_event_id"] == old
    finally:
        conn.close()


def test_advance_notify_cursor_resets_failure_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "reset.db"))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="x", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="slack", chat_id="C1")
        kb.complete_task(conn, tid, summary="done")
        old, new, _ = kb.claim_unseen_events_for_sub(
            conn, task_id=tid, platform="slack", chat_id="C1", kinds=["completed"],
        )
        # Simulate a prior failure state, then a successful advance.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE kanban_notify_subs SET consecutive_failures = 4, next_retry_at = 999999999",
            )
        kb.advance_notify_cursor(
            conn, task_id=tid, platform="slack", chat_id="C1", new_cursor=new,
        )
        sub = kb.list_notify_subs(conn, tid)[0]
        assert sub["consecutive_failures"] == 0
        assert sub["next_retry_at"] is None
    finally:
        conn.close()


def test_notifier_drops_sub_after_consecutive_permanent_failures(tmp_path, monkeypatch):
    """A channel_not_found sub is dropped after the failure threshold instead
    of being retried forever (regression: 85k retries over days)."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "perm-drop.db"))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="dead channel", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="slack", chat_id="C1")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    # One failure short of the drop threshold; next_retry in the past so the
    # gate does not skip this tick.
    _seed_sub_failure_state(tid, consecutive_failures=2, next_retry_at=None)

    adapter = PermanentFailingAdapter("channel_not_found")
    runner = _make_runner(adapter, Platform.SLACK)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.attempts == 1
    assert _read_sub(tid) is None, "permanent-failure sub should be dropped"


def test_notifier_backoff_gate_skips_subscription_not_yet_due(tmp_path, monkeypatch):
    """A sub whose next_retry_at is in the future is not even attempted."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "backoff-skip.db"))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="backing off", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="slack", chat_id="C1")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    import time as _t
    _seed_sub_failure_state(tid, consecutive_failures=5, next_retry_at=int(_t.time()) + 3600)

    adapter = PermanentFailingAdapter("channel_not_found")
    runner = _make_runner(adapter, Platform.SLACK)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert adapter.attempts == 0, "backoff gate must skip a not-yet-due sub"
    assert _read_sub(tid) is not None, "sub must survive while backing off"


def test_notifier_transient_failure_retained_with_backoff(tmp_path, monkeypatch):
    """A transient error keeps the sub but records a backoff so it can't spam."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "transient.db"))
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="rate limited", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="slack", chat_id="C1")
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = PermanentFailingAdapter("ratelimited")  # transient error string
    runner = _make_runner(adapter, Platform.SLACK)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    sub = _read_sub(tid)
    assert sub is not None, "transient failure must not drop the sub"
    assert sub["consecutive_failures"] == 1
    assert sub["next_retry_at"] is not None

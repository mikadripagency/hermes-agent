"""Tests for kb.specify_triage_task — the DB-layer atomic promotion
from the triage column to todo. LLM-free by design."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

pytestmark = pytest.mark.usefixtures("explicit_delivery_contract_for_kanban_fixtures")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _create_triage(conn, title="rough idea", body=None, assignee=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        triage=True,
    )


def test_specify_promotes_triage_to_todo(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="rough idea")
        assert kb.get_task(conn, tid).status == "triage"
    with kb.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="Refined: rough idea",
            body="**Goal**\nDo the thing.",
            author="specifier-bot",
        )
    assert ok is True
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    # No parents → recompute_ready should have flipped it past todo to ready.
    assert task.status == "ready"
    assert task.title == "Refined: rough idea"
    assert "**Goal**" in (task.body or "")


def test_specify_with_open_parent_lands_in_todo_not_ready(kanban_home):
    # Parent-gated specified tasks must not jump the dispatcher — they go
    # to todo and wait for parent completion like any other gated task.
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent work")
        child = _create_triage(conn, title="child idea")
        kb.link_tasks(conn, parent, child)
        # After linking with an open parent, triage status should still be
        # 'triage' (linking doesn't touch triage tasks).
        assert kb.get_task(conn, child).status == "triage"
    with kb.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            child,
            body="full spec",
            author="specifier",
        )
    assert ok is True
    with kb.connect() as conn:
        t = kb.get_task(conn, child)
    # Parent still open → specified child sits in 'todo', not 'ready'.
    assert t.status == "todo"


def test_specify_refuses_non_triage_task(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="normal task")
        assert kb.get_task(conn, tid).status == "ready"
    with kb.connect() as conn:
        ok = kb.specify_triage_task(conn, tid, body="won't apply")
    assert ok is False
    with kb.connect() as conn:
        # Status unchanged.
        assert kb.get_task(conn, tid).status == "ready"


def test_specify_returns_false_for_unknown_id(kanban_home):
    with kb.connect() as conn:
        ok = kb.specify_triage_task(conn, "t_does_not_exist", body="x")
    assert ok is False


def test_specify_rejects_blank_title(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="rough")
    with kb.connect() as conn, pytest.raises(ValueError):
        kb.specify_triage_task(conn, tid, title="   ", body="ok")


def test_specify_emits_event_with_verbatim_original(kanban_home):
    original_title = "rough ORIGINAL_TITLE_SENTINEL"
    original_body = "x" * 5000 + " ORIGINAL_BODY_SENTINEL"
    with kb.connect() as conn:
        tid = _create_triage(conn, title=original_title, body=original_body)
    with kb.connect() as conn:
        kb.specify_triage_task(
            conn, tid, title="new", body="b", author="ace"
        )
    with kb.connect() as conn:
        events = kb.list_events(conn, tid)
    kinds = [e.kind for e in events]
    assert "specified" in kinds
    # The specified event records which fields actually changed as a
    # JSON payload under task_events.payload.
    spec_ev = next(e for e in events if e.kind == "specified")
    assert spec_ev.payload is not None
    fields = spec_ev.payload.get("changed_fields") or []
    assert "title" in fields
    assert "body" in fields
    assert spec_ev.payload["old_title"] == original_title
    assert spec_ev.payload["old_body"] == original_body


def test_specify_records_original_snapshot_with_or_without_author(kanban_home):
    # With author → snapshot is attributed to that author.
    with kb.connect() as conn:
        tid1 = _create_triage(conn, title="original a", body="original body a")
        kb.specify_triage_task(
            conn, tid1, title="A-spec", body="b", author="ace"
        )
        comments1 = kb.list_comments(conn, tid1)
    assert len(comments1) == 1
    assert "original a" in comments1[0].body
    assert "original body a" in comments1[0].body
    assert comments1[0].author == "ace"

    # Without author → the original still must not be lost.
    with kb.connect() as conn:
        tid2 = _create_triage(conn, title="original b", body="original body b")
        kb.specify_triage_task(conn, tid2, title="B-spec", body="b")
        comments2 = kb.list_comments(conn, tid2)
    assert len(comments2) == 1
    assert comments2[0].author == "kanban-specifier"
    assert "original b" in comments2[0].body
    assert "original body b" in comments2[0].body


def test_specify_skips_comment_when_nothing_changed(kanban_home):
    # Create triage task with title and body already set; pass identical
    # values to specify. Should promote to todo but skip audit comment.
    with kb.connect() as conn:
        tid = _create_triage(conn, title="same", body="same body")
    with kb.connect() as conn:
        ok = kb.specify_triage_task(
            conn,
            tid,
            title="same",
            body="same body",
            author="ace",
        )
    assert ok is True
    with kb.connect() as conn:
        # Promoted.
        assert kb.get_task(conn, tid).status in {"todo", "ready"}
        # No audit comment because neither field changed.
        assert kb.list_comments(conn, tid) == []


def test_specify_with_only_body_preserves_title(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="keep this title")
    with kb.connect() as conn:
        kb.specify_triage_task(conn, tid, body="new body only")
    with kb.connect() as conn:
        t = kb.get_task(conn, tid)
    assert t.title == "keep this title"
    assert t.body == "new body only"


def test_specify_empty_body_snapshot_is_auditable(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="empty-body original", body=None)
        kb.specify_triage_task(conn, tid, body="new body")
        event = next(e for e in kb.list_events(conn, tid) if e.kind == "specified")
        comments = kb.list_comments(conn, tid)

    assert event.payload is not None
    assert event.payload["old_title"] == "empty-body original"
    assert event.payload["old_body"] is None
    assert len(comments) == 1
    assert "empty-body original" in comments[0].body
    assert "--- ORIGINAL BODY ---\n\n--- END ORIGINAL SPEC ---" in comments[0].body


def test_each_specify_keeps_its_own_previous_version(kanban_home):
    with kb.connect() as conn:
        tid = _create_triage(conn, title="version one", body="body one")
        assert kb.specify_triage_task(
            conn, tid, title="version two", body="body two"
        )
        # Exercise the supported path that can return a specified task to
        # triage: two same-cause blocks across an unblock trip the loop breaker.
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        assert kb.block_task(conn, tid, reason="need input", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        assert kb.block_task(conn, tid, reason="still need input", kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "triage"
        assert kb.specify_triage_task(
            conn, tid, title="version three", body="body three"
        )
        specified = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
        comments = kb.list_comments(conn, tid)

    payloads = [e.payload for e in specified]
    assert all(payload is not None for payload in payloads)
    assert [payload["old_title"] for payload in payloads if payload is not None] == [
        "version one",
        "version two",
    ]
    assert [payload["old_body"] for payload in payloads if payload is not None] == [
        "body one",
        "body two",
    ]
    assert len(comments) == 2
    assert "version one" in comments[0].body
    assert "version two" in comments[1].body


def test_specify_second_call_noop_false(kanban_home):
    # Promoting twice must not crash and the second call returns False
    # because the task is no longer in triage.
    with kb.connect() as conn:
        tid = _create_triage(conn, title="once")
    with kb.connect() as conn:
        assert kb.specify_triage_task(conn, tid, body="spec") is True
    with kb.connect() as conn:
        assert kb.specify_triage_task(conn, tid, body="spec again") is False

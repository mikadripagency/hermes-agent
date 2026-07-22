"""Regression tests for the per-project concurrency cap in the dispatcher.

By default, no single project (repo / experiment lane, grouped by
``tasks.project_id``) gets more than three active delivery lanes, even if the
global ``max_in_progress`` / per-profile caps would allow it. Tasks with no
project (NULL ``project_id``) are exempt. Mirrors
``test_kanban_per_profile_cap.py``.
"""
from __future__ import annotations

import concurrent.futures
import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Spin up a fresh HERMES_HOME with kanban DB + alpha/beta profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_per_project_cap_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _set_project(kb, conn, task_id, project_id):
    """Directly stamp a project_id on a task, bypassing the projects registry.

    ``create_task`` validates project_id against the projects.db registry and
    drops unknown ids; the dispatcher's grouping logic only reads the raw
    ``tasks.project_id`` column, so setting it directly is the right unit-level
    fixture here."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET project_id = ? WHERE id = ?", (project_id, task_id)
        )


def test_no_cap_all_tasks_dispatched(isolated_kanban_home_with_profiles):
    """Baseline: with no per-project cap, all ready tasks dispatch."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            tid = kb.create_task(conn, title=f"p{i}", assignee="alpha")
            _set_project(kb, conn, tid, "proj_a")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            max_in_progress_per_project=None,
        )
    assert len(res.spawned) == 5
    assert not res.skipped_per_project_capped


def test_cap_2_limits_single_project(isolated_kanban_home_with_profiles):
    """With cap=2: 2 dispatched from proj_a, remaining 3 deferred."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            tid = kb.create_task(conn, title=f"a{i}", assignee="alpha")
            _set_project(kb, conn, tid, "proj_a")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_project=2,
        )
    capped = [c[1] for c in res.skipped_per_project_capped]
    assert len(res.spawned) == 2
    assert capped == ["proj_a", "proj_a", "proj_a"]


def test_cap_is_independent_per_project(isolated_kanban_home_with_profiles):
    """Two projects each get up to the cap; they don't share the budget."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            tid = kb.create_task(conn, title=f"a{i}", assignee="alpha")
            _set_project(kb, conn, tid, "proj_a")
        for i in range(3):
            tid = kb.create_task(conn, title=f"b{i}", assignee="beta")
            _set_project(kb, conn, tid, "proj_b")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_project=2,
        )
    capped = [c[1] for c in res.skipped_per_project_capped]
    assert len(res.spawned) == 4  # 2 from each project
    assert capped.count("proj_a") == 1
    assert capped.count("proj_b") == 1


def test_null_project_tasks_are_exempt(isolated_kanban_home_with_profiles):
    """Tasks with no project_id are never counted or capped."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        # 5 project-less tasks — all should dispatch despite cap=1.
        for i in range(5):
            kb.create_task(conn, title=f"n{i}", assignee="alpha")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_project=1,
        )
    assert len(res.spawned) == 5
    assert not res.skipped_per_project_capped


def test_pre_existing_running_counts_against_cap(isolated_kanban_home_with_profiles):
    """A task already 'running' in a project counts toward the per-project cap."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        running = kb.create_task(conn, title="running", assignee="alpha")
        _set_project(kb, conn, running, "proj_a")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test:1' WHERE id = ?",
                (running,),
            )
        for i in range(2):
            tid = kb.create_task(conn, title=f"a{i}", assignee="alpha")
            _set_project(kb, conn, tid, "proj_a")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_project=1,
        )
    assert len(res.spawned) == 0  # proj_a already at cap
    assert len(res.skipped_per_project_capped) == 2


@pytest.mark.parametrize("cap", [0, -1, "abc", None])
def test_invalid_cap_treated_as_no_cap(isolated_kanban_home_with_profiles, cap):
    """Non-positive-int caps mean 'no cap' — fall through, never crash."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            tid = kb.create_task(conn, title=f"a{i}", assignee="alpha")
            _set_project(kb, conn, tid, "proj_a")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_project=cap,
        )
    assert not res.skipped_per_project_capped
    assert len(res.spawned) == 3


def test_capped_tasks_dispatched_on_subsequent_tick(isolated_kanban_home_with_profiles):
    """A task deferred because its project was at cap is eligible next tick."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            tid = kb.create_task(conn, title=f"a{i}", assignee="alpha")
            _set_project(kb, conn, tid, "proj_a")

    with kb.connect_closing() as conn:
        res1 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_project=1,
        )
    assert len(res1.spawned) == 1
    assert len(res1.skipped_per_project_capped) == 2

    spawned_id = res1.spawned[0][0]
    with kb.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (spawned_id,),
            )

    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_project=1,
        )
    assert len(res2.spawned) == 1
    assert len(res2.skipped_per_project_capped) == 1
    assert res2.spawned[0][0] != spawned_id


def test_dispatch_result_has_skipped_per_project_capped_field():
    """Schema-level invariant: DispatchResult exposes skipped_per_project_capped."""
    from hermes_cli.kanban_db import DispatchResult
    r = DispatchResult()
    assert hasattr(r, "skipped_per_project_capped")
    assert r.skipped_per_project_capped == []


def test_cap_3_claims_only_three_of_four_project_siblings(
    isolated_kanban_home_with_profiles,
):
    """A fourth sibling stays ready while an unrelated project still runs."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        siblings = []
        for i in range(4):
            task_id = kb.create_task(
                conn, title=f"repo-a-{i}", assignee="alpha", priority=10 - i,
            )
            _set_project(kb, conn, task_id, "project-a")
            siblings.append(task_id)
        unrelated = kb.create_task(
            conn, title="repo-b", assignee="beta", priority=1,
        )
        _set_project(kb, conn, unrelated, "project-b")

    with kb.connect_closing() as conn:
        result = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
        )
        sibling_statuses = [kb.get_task(conn, task_id).status for task_id in siblings]
        unrelated_status = kb.get_task(conn, unrelated).status

    assert sibling_statuses == ["running", "running", "running", "ready"]
    assert unrelated_status == "running"
    assert [item[0] for item in result.skipped_per_project_capped] == [siblings[3]]


def test_running_review_and_branch_blocked_tasks_share_project_slots(
    isolated_kanban_home_with_profiles,
):
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        running = kb.create_task(conn, title="running", assignee="alpha")
        review = kb.create_task(conn, title="review", assignee="alpha")
        blocked = kb.create_task(
            conn, title="blocked", assignee="alpha", initial_status="blocked",
        )
        waiting = kb.create_task(conn, title="waiting", assignee="alpha")
        other = kb.create_task(conn, title="other", assignee="beta")
        for task_id in (running, review, blocked, waiting):
            _set_project(kb, conn, task_id, "project-a")
        _set_project(kb, conn, other, "project-b")
        kb.claim_task(conn, running)
        kb.claim_task(conn, review)
        assert kb.block_task(conn, review, reason="review-required: ready")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET branch_name = 'project-a/blocked' WHERE id = ?",
                (blocked,),
            )

    with kb.connect_closing() as conn:
        result = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert kb.get_task(conn, waiting).status == "ready"
        assert kb.get_task(conn, other).status == "running"

    assert [item[0] for item in result.skipped_per_project_capped] == [waiting]


def test_system_inbox_does_not_consume_project_slot(
    isolated_kanban_home_with_profiles,
):
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(2):
            task_id = kb.create_task(conn, title=f"active-{i}", assignee="alpha")
            _set_project(kb, conn, task_id, "project-a")
            kb.claim_task(conn, task_id)
        inbox = kb.create_task(
            conn,
            title="inbox",
            assignee="alpha",
            initial_status="blocked",
            task_kind="system_inbox",
        )
        candidate = kb.create_task(conn, title="candidate", assignee="alpha")
        for task_id in (inbox, candidate):
            _set_project(kb, conn, task_id, "project-a")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET branch_name = 'project-a/inbox' WHERE id = ?",
                (inbox,),
            )

    with kb.connect_closing() as conn:
        result = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert kb.get_task(conn, candidate).status == "running"

    assert not result.skipped_per_project_capped


def test_concurrent_claims_cannot_take_the_same_last_project_slot(
    isolated_kanban_home_with_profiles,
):
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(2):
            active = kb.create_task(conn, title=f"active-{i}", assignee="alpha")
            _set_project(kb, conn, active, "project-a")
            assert kb.claim_task(conn, active) is not None
        candidates = []
        for i in range(2):
            candidate = kb.create_task(
                conn, title=f"candidate-{i}", assignee="alpha",
            )
            _set_project(kb, conn, candidate, "project-a")
            candidates.append(candidate)

    def claim(task_id):
        with kb.connect_closing() as conn:
            return kb.claim_task(conn, task_id, claimer=f"test:{task_id}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(claim, candidates))

    assert sum(claimed is not None for claimed in claims) == 1
    with kb.connect_closing() as conn:
        assert sorted(kb.get_task(conn, task_id).status for task_id in candidates) == [
            "ready",
            "running",
        ]


def test_unclaimed_running_recovery_reuses_its_existing_project_slot(
    isolated_kanban_home_with_profiles,
):
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        recovery = kb.create_task(
            conn, title="recovery", assignee="alpha", priority=10,
        )
        _set_project(kb, conn, recovery, "project-a")
        assert kb.claim_task(conn, recovery) is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_lock = NULL, claim_expires = NULL WHERE id = ?",
                (recovery,),
            )
        for i in range(2):
            active = kb.create_task(conn, title=f"active-{i}", assignee="alpha")
            _set_project(kb, conn, active, "project-a")
            assert kb.claim_task(conn, active) is not None
        waiting = kb.create_task(conn, title="waiting", assignee="alpha")
        _set_project(kb, conn, waiting, "project-a")

    with kb.connect_closing() as conn:
        result = kb.dispatch_once(conn, spawn_fn=_fake_spawn)
        assert kb.get_task(conn, recovery).claim_lock is not None
        assert kb.get_task(conn, waiting).status == "ready"

    assert result.spawned[0][0] == recovery
    assert [item[0] for item in result.skipped_per_project_capped] == [waiting]

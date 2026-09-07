"""Regression coverage for disposable reviewer Kanban isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from hermes_cli import kanban_db as kb


_REPO_ROOT = Path(__file__).resolve().parents[2]
_KANBAN_AUTHORITY_ENV = {
    "HERMES_KANBAN_TASK": "t_parent",
    "HERMES_KANBAN_RUN_ID": "710",
    "HERMES_KANBAN_CLAIM_LOCK": "parent-claim",
    "HERMES_KANBAN_DB": "set-by-test",
    "HERMES_KANBAN_BOARD": "default",
}


def _run_create(env: dict[str, str], title: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "create",
            title,
            "--evidence-na-reason",
            "disposable isolation fixture",
            "--json",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _board_counts(db_path: Path) -> dict[str, int]:
    with kb.connect(db_path=db_path) as conn:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "tasks",
                "task_events",
                "task_comments",
                "completion_deliveries",
            )
        }


def _webhook_child_env(*, profile_home: Path, live_db: Path) -> dict[str, str]:
    from tools.environments.local import _make_run_env

    inherited = {
        **_KANBAN_AUTHORITY_ENV,
        "HERMES_HOME": str(profile_home),
        "HERMES_KANBAN_DB": str(live_db),
        "HERMES_SESSION_SOURCE": "webhook",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(_REPO_ROOT),
    }
    with patch.dict(os.environ, inherited, clear=True):
        return _make_run_env({})


def test_webhook_reviewer_child_cannot_mutate_inherited_live_board(tmp_path):
    live_db = tmp_path / "live" / "kanban.db"
    kb.init_db(db_path=live_db)
    profile_home = tmp_path / "disposable-reviewer-home"
    profile_home.mkdir(parents=True)
    before = _board_counts(live_db)

    child_env = _webhook_child_env(profile_home=profile_home, live_db=live_db)
    result = _run_create(child_env, "must-not-reach-live-board")

    # hermes_cli.main historically does not propagate subcommand return codes;
    # rejection is proved by the diagnostic plus the zero DB delta below.
    assert "disposable Kanban access requires" in result.stderr
    assert _board_counts(live_db) == before
    for name in _KANBAN_AUTHORITY_ENV:
        assert name not in child_env
    assert child_env["_HERMES_DISPOSABLE_AGENT"] == "1"


def test_disposable_reviewer_can_opt_into_explicit_temporary_board(tmp_path):
    live_db = tmp_path / "live" / "kanban.db"
    kb.init_db(db_path=live_db)
    profile_home = tmp_path / "disposable-reviewer-home"
    profile_home.mkdir(parents=True)
    child_env = _webhook_child_env(profile_home=profile_home, live_db=live_db)
    before = _board_counts(live_db)

    isolated_home = tmp_path / "disposable"
    isolated_home.mkdir()
    isolated_db = isolated_home / "kanban.db"
    child_env.update(
        {
            "HERMES_HOME": str(isolated_home),
            "HERMES_KANBAN_DB": str(isolated_db),
            "HERMES_KANBAN_BOARD": "default",
        }
    )
    result = _run_create(child_env, "isolated-probe")

    assert result.returncode == 0, result.stderr
    created = json.loads(result.stdout)
    assert _board_counts(live_db) == before
    with kb.connect(db_path=isolated_db) as conn:
        task = kb.get_task(conn, created["id"])
        assert task is not None and task.title == "isolated-probe"
        events = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (created["id"],),
        ).fetchall()
        assert [row["kind"] for row in events] == ["created"]
        assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM completion_deliveries").fetchone()[0] == 0


def test_archive_probe_shapes_has_no_completion_or_delivery(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    with kb.connect(db_path=db_path) as conn:
        blocked_id = kb.create_task(
            conn,
            title="PR331 isolated gate probe",
            assignee="developer",
            required_evidence=["pr_merged"],
        )
        claimed = kb.claim_task(
            conn,
            blocked_id,
            claimer="fixture",
            execution_profile="developer",
        )
        assert claimed is not None and claimed.current_run_id is not None
        assert kb.block_task(
            conn,
            blocked_id,
            reason="synthetic fixture",
            kind="capability",
            expected_run_id=claimed.current_run_id,
        )
        ready_id = kb.create_task(
            conn,
            title="authority fixture",
            assignee="developer",
            required_evidence=["pr_merged"],
        )

        delivery_counts_before = {
            task_id: conn.execute(
                "SELECT COUNT(*) FROM completion_deliveries WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0]
            for task_id in (blocked_id, ready_id)
        }

        for task_id in (blocked_id, ready_id):
            assert kb.archive_task(conn, task_id) is True
            assert kb.archive_task(conn, task_id) is False
            archived = kb.get_task(conn, task_id)
            assert archived is not None and archived.status == "archived"
            event_kinds = [
                row["kind"]
                for row in conn.execute(
                    "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
                    (task_id,),
                )
            ]
            assert "archived" in event_kinds
            assert "completed" not in event_kinds
            assert conn.execute(
                "SELECT COUNT(*) FROM completion_deliveries WHERE task_id = ?",
                (task_id,),
            ).fetchone()[0] == delivery_counts_before[task_id]
            assert conn.execute(
                "SELECT COUNT(*) FROM completion_deliveries d "
                "JOIN task_events e ON e.id = d.event_id "
                "WHERE d.task_id = ? AND e.kind = 'completed'",
                (task_id,),
            ).fetchone()[0] == 0

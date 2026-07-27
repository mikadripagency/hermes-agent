"""Creation-time workspace validation (t_5ee8ac9c).

A worktree task with no resolvable anchor (no explicit workspace_path, no
project primary repo, no board default_workdir) is rejected at CREATION with
an actionable error instead of being persisted as a row that can only ever
spawn_fail -> gave_up. Scratch tasks that clearly reference a registered
project's repo get an advisory ``dispatch_warning`` event (never a reject).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import projects_db as pdb
from tools import kanban_tools

pytestmark = pytest.mark.usefixtures("explicit_delivery_contract_for_kanban_fixtures")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_project(name="Research Hub", repo=None, tmp_path=None):
    repo = repo or str(tmp_path / "Drip-Research-Hub")
    with pdb.connect_closing() as pc:
        pid = pdb.create_project(pc, name=name, folders=[repo])
        return pdb.get_project(pc, pid)


# --- reject path -----------------------------------------------------------

def test_worktree_without_anchor_rejected_at_creation(kanban_home):
    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match=r"--project .*worktree:"):
            kb.create_task(conn, title="anchorless", workspace_kind="worktree")
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_worktree_with_unresolvable_project_rejected_at_creation(kanban_home):
    """A project id that does not resolve is dropped (documented behavior);
    the task must then hit the same anchor guard instead of persisting a
    worktree row with a NULL path."""
    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match="worktree"):
            kb.create_task(
                conn,
                title="dangling project",
                workspace_kind="worktree",
                project_id="no-such-project",
            )


def test_kanban_create_tool_reports_anchor_error(kanban_home):
    result = kanban_tools._handle_create(
        {
            "title": "anchorless via tool",
            "assignee": "developer",
            "workspace_kind": "worktree",
        }
    )
    assert "Error" in result or "error" in result
    assert "worktree" in result
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


# --- accept paths ----------------------------------------------------------

def test_worktree_with_explicit_path_accepted(kanban_home, tmp_path):
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="anchored",
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "repo"),
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.workspace_path == str(tmp_path / "repo")


def test_worktree_with_project_anchor_accepted(kanban_home, tmp_path):
    proj = _make_project(tmp_path=tmp_path)
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="project anchored",
            workspace_kind="worktree",
            project_id=proj.slug,
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.workspace_path is not None
        assert task.project_id == proj.id


def test_worktree_with_board_default_workdir_accepted(kanban_home, tmp_path):
    repo = tmp_path / "boardrepo"
    repo.mkdir()
    kb.create_board("anchor-board", default_workdir=str(repo))
    with kb.connect(board="anchor-board") as conn:
        tid = kb.create_task(
            conn, title="board anchored", workspace_kind="worktree",
            board="anchor-board",
        )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.workspace_path == str(repo)


# --- scratch-task advisory warning ------------------------------------------

def _events_of_kind(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def test_scratch_task_referencing_project_repo_warns_but_creates(kanban_home, tmp_path):
    _make_project(tmp_path=tmp_path)
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="Fix the flaky loader",
            body="Please patch the retry logic in Drip-Research-Hub and add tests.",
        )
        task = kb.get_task(conn, tid)
        assert task is not None and task.workspace_kind == "scratch"
        warnings = _events_of_kind(conn, tid, "dispatch_warning")
        assert len(warnings) == 1
        payload = warnings[0].payload
        assert payload["kind"] == "scratch_task_references_project"
        assert payload["matched_by"] in {"repo_basename", "primary_path", "slug"}
        assert "worktree" in payload["hint"]


def test_scratch_task_without_project_reference_has_no_warning(kanban_home, tmp_path):
    _make_project(tmp_path=tmp_path)
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Summarize this PDF", body="No repo work.")
        assert _events_of_kind(conn, tid, "dispatch_warning") == []


def test_worktree_task_never_gets_scratch_warning(kanban_home, tmp_path):
    proj = _make_project(tmp_path=tmp_path)
    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="Work on Drip-Research-Hub",
            workspace_kind="worktree",
            project_id=proj.slug,
        )
        assert _events_of_kind(conn, tid, "dispatch_warning") == []

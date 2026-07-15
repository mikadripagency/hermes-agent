"""Tests: config-gated Paseo-agent kanban worker launcher.

Covers :mod:`hermes_cli.paseo_spawn` and its wiring into the dispatcher's
default spawn resolution (:func:`hermes_cli.kanban_db._resolve_default_spawn`).

The paseo CLI is mocked entirely at the ``subprocess`` boundary — no real
daemon, no real agents. Each test installs a fake ``subprocess.run`` that
dispatches on the paseo subcommand (``status`` / ``ls`` / ``run``) and a fake
``subprocess.Popen`` that stands in for the tracked watch-loop child. The
watch loop itself (``watch_worker``) is tested with scripted state sequences
plus one real-subprocess end-to-end run against a real kanban DB.
"""

from __future__ import annotations

import json
import subprocess


def _make_task(kb, *, assignee: str = "w", task_id: str = "t_paseo", workspace_kind: str = "dir"):
    return kb.Task(
        id=task_id,
        title="Implement the widget with a rather long title that exceeds sixty characters for truncation",
        body=None,
        assignee=assignee,
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind=workspace_kind,
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=7,
        max_runtime_seconds=1800,
    )


def _profile_home(tmp_path, monkeypatch, *, assignee: str = "w", paseo_cfg: str = ""):
    root = tmp_path / ".hermes"
    (root / "profiles" / assignee).mkdir(parents=True)
    (root / "profiles" / assignee / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\n" + paseo_cfg, encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


class _FakeProc:
    def __init__(self, pid=9999):
        self.pid = pid


class _CompletedRun:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _install_fake_paseo(
    monkeypatch,
    *,
    status_rc: int = 0,
    ls_agents=None,
    run_stdout=None,
    run_stderr="Created workspace ws_abc123 - repo (main)\n",
    run_rc: int = 0,
    archive_rc: int = 0,
):
    """Patch subprocess.run/Popen to emulate the paseo CLI. Returns a recorder."""
    if run_stdout is None:
        run_stdout = json.dumps(
            {"agentId": "ag_1234567", "status": "running", "provider": "hermes", "cwd": "/ws"}
        )
    calls: dict = {"run": [], "ls": [], "status": [], "archive": [], "popen": []}

    def fake_run(cmd, *args, **kwargs):
        sub = cmd[1] if len(cmd) > 1 else None
        if sub == "status":
            calls["status"].append(list(cmd))
            return _CompletedRun(returncode=status_rc)
        if sub == "ls":
            calls["ls"].append(list(cmd))
            return _CompletedRun(returncode=0, stdout=json.dumps(ls_agents or []))
        if sub == "archive":
            calls["archive"].append(list(cmd))
            return _CompletedRun(returncode=archive_rc, stderr="" if archive_rc == 0 else "archive failed")
        if sub == "run":
            calls["run"].append(list(cmd))
            return _CompletedRun(returncode=run_rc, stdout=run_stdout, stderr=run_stderr)
        raise AssertionError(f"unexpected paseo subcommand: {cmd!r}")

    def fake_popen(cmd, *args, **kwargs):
        calls["popen"].append(list(cmd))
        return _FakeProc(pid=9999)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


def test_enabled_healthy_launches_agent_with_contract(monkeypatch, tmp_path):
    """enabled + healthy daemon → `paseo run` with full env/labels/title + watcher pid."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    # Skip the DB-backed comment write; asserted separately.
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)

    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert pid == 9999
    assert len(calls["run"]) == 1
    run_cmd = calls["run"][0]
    assert run_cmd[:4] == ["paseo", "run", "-d", "--json"]
    assert "--provider" in run_cmd and run_cmd[run_cmd.index("--provider") + 1] == "hermes"
    assert "--cwd" in run_cmd and run_cmd[run_cmd.index("--cwd") + 1] == str(workspace)
    # Workers must never pause on permission prompts: default ACP mode is
    # dont_ask (valid hermes ACP mode ids: default / accept_edits / dont_ask).
    assert "--mode" in run_cmd and run_cmd[run_cmd.index("--mode") + 1] == "dont_ask"
    # Title carries id + truncated title (<=60 chars, ellipsised).
    title = run_cmd[run_cmd.index("--title") + 1]
    assert title.startswith("t_paseo · ")
    assert title.endswith("…")
    # Labels.
    joined = " ".join(run_cmd)
    assert "--label kanban_task=t_paseo" in joined
    assert "--label kanban_run=7" in joined
    assert "--label kanban_board=" in joined
    # Env contract flows via --env (task id -> kanban toolset auto-append).
    env_pairs = [run_cmd[i + 1] for i, a in enumerate(run_cmd) if a == "--env"]
    assert "HERMES_KANBAN_TASK=t_paseo" in env_pairs
    assert any(p.startswith("HERMES_KANBAN_DB=") for p in env_pairs)
    assert any(p.startswith("HERMES_PROFILE=") for p in env_pairs)
    assert "HERMES_KANBAN_RUN_ID=7" in env_pairs
    # Prompt is the final positional arg.
    assert run_cmd[-1] == "work kanban task t_paseo"
    # Watcher: the tracked child is the task-state watch loop, NOT `paseo wait`
    # (ACP agents flap to idle between turns; only the kanban DB decides).
    import sys as _sys
    import time as _time

    assert len(calls["popen"]) == 1
    watch_cmd = calls["popen"][0]
    assert watch_cmd[:4] == [_sys.executable, "-m", "hermes_cli.paseo_spawn", "--watch"]
    assert watch_cmd[watch_cmd.index("--task") + 1] == "t_paseo"
    assert watch_cmd[watch_cmd.index("--agent") + 1] == "ag_1234567"
    assert "--db" in watch_cmd
    assert watch_cmd[watch_cmd.index("--run") + 1] == "7"
    # Deadline ≈ now + max_runtime + slack.
    deadline = float(watch_cmd[watch_cmd.index("--deadline") + 1])
    assert abs(deadline - (_time.time() + 1800 + 300)) < 60


def test_daemon_down_falls_back_to_default_spawn(monkeypatch, tmp_path):
    """`paseo status` failure → automatic fallback to _default_spawn (no paseo run)."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    calls = _install_fake_paseo(monkeypatch, status_rc=1)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    # No agent was created; the watcher was never spawned.
    assert calls["run"] == []
    # _default_spawn ran instead: its Popen launches the `hermes ... chat` CLI.
    assert len(calls["popen"]) == 1
    fallback_cmd = calls["popen"][0]
    assert fallback_cmd[0] == "hermes"
    assert "chat" in fallback_cmd
    assert pid == 9999


def test_actively_working_agent_reattaches_without_second_run(monkeypatch, tmp_path):
    """An ACTIVELY working labeled agent → watcher only: no archive, no new run."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(
        monkeypatch,
        ls_agents=[{"id": "ag_existing1", "status": "running", "shortId": "ag_exis"}],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert calls["run"] == []  # no duplicate agent
    assert calls["archive"] == []  # a working agent is never archived
    assert len(calls["popen"]) == 1
    watch_cmd = calls["popen"][0]
    assert "--watch" in watch_cmd
    assert watch_cmd[watch_cmd.index("--agent") + 1] == "ag_existing1"
    assert pid == 9999


def test_idle_agent_archived_and_fresh_agent_spawned(monkeypatch, tmp_path):
    """An idle labeled agent is a stale remnant: archive it, launch fresh.

    A Paseo agent maps to a RUN. The idle agent's env carries the OLD run id /
    claim lock (kanban tools would reject it; no heartbeats flow) — regression
    guard for t_6811e170 runs 324/325 which re-attached to a finished agent and
    timed out.
    """
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(
        monkeypatch,
        ls_agents=[{"id": "ag_stale", "status": "idle"}],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    # Stale agent archived (no --force: it isn't running).
    assert len(calls["archive"]) == 1
    assert calls["archive"][0][:3] == ["paseo", "archive", "ag_stale"]
    assert "--force" not in calls["archive"][0]
    # Fresh agent created with the NEW run's env.
    assert len(calls["run"]) == 1
    run_cmd = calls["run"][0]
    env_pairs = [run_cmd[i + 1] for i, a in enumerate(run_cmd) if a == "--env"]
    assert "HERMES_KANBAN_RUN_ID=7" in env_pairs
    assert "--label kanban_run=7" in " ".join(run_cmd)
    # Watcher tracks the NEW agent.
    watch_cmd = calls["popen"][0]
    assert watch_cmd[watch_cmd.index("--agent") + 1] == "ag_1234567"
    assert pid == 9999


def test_errored_agent_archived_and_fresh_agent_spawned(monkeypatch, tmp_path):
    """An errored agent is stale too → archived, fresh agent launched."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(
        monkeypatch,
        ls_agents=[{"id": "ag_dead", "status": "error"}],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    assert len(calls["archive"]) == 1
    assert calls["archive"][0][2] == "ag_dead"
    assert len(calls["run"]) == 1  # fresh agent created


def test_archive_failure_still_spawns_fresh_agent(monkeypatch, tmp_path):
    """`paseo archive` failing must not block the fresh spawn."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(
        monkeypatch,
        ls_agents=[{"id": "ag_stuck", "status": "idle"}],
        archive_rc=1,
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert len(calls["archive"]) == 1  # attempted
    assert len(calls["run"]) == 1  # fresh agent still launched
    assert len(calls["popen"]) == 1  # watcher still spawned
    assert pid == 9999


def test_mode_omitted_when_configured_empty(monkeypatch, tmp_path):
    """kanban.paseo_spawn.mode: '' → no --mode flag (provider default)."""
    _profile_home(
        tmp_path,
        monkeypatch,
        paseo_cfg="kanban:\n  paseo_spawn:\n    mode: ''\n",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    assert "--mode" not in calls["run"][0]


def test_mode_override_from_config(monkeypatch, tmp_path):
    """A custom kanban.paseo_spawn.mode value is passed through."""
    _profile_home(
        tmp_path,
        monkeypatch,
        paseo_cfg="kanban:\n  paseo_spawn:\n    mode: accept_edits\n",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    run_cmd = calls["run"][0]
    assert run_cmd[run_cmd.index("--mode") + 1] == "accept_edits"


def test_mixed_agents_prefers_active_over_stale(monkeypatch, tmp_path):
    """With one idle and one running agent, re-attach to the running one."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(
        monkeypatch,
        ls_agents=[
            {"id": "ag_old_idle", "status": "idle"},
            {"id": "ag_working", "status": "running"},
        ],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    assert calls["run"] == []
    assert calls["archive"] == []  # active agent found → no archiving pass
    watch_cmd = calls["popen"][0]
    assert watch_cmd[watch_cmd.index("--agent") + 1] == "ag_working"


def test_comment_failure_does_not_break_spawn(monkeypatch, tmp_path):
    """A failing linkage-comment write must not fail the spawn."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    def _boom(*a, **k):
        raise RuntimeError("db unavailable")

    # connect_closing raising simulates any comment-write failure.
    monkeypatch.setattr(kb, "connect_closing", _boom)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert pid == 9999
    assert len(calls["run"]) == 1  # agent still launched
    assert len(calls["popen"]) == 1  # watcher still spawned


def test_disabled_uses_default_spawn(monkeypatch, tmp_path):
    """kanban.paseo_spawn.enabled false → dispatcher default is _default_spawn."""
    _profile_home(tmp_path, monkeypatch)  # no paseo_spawn config
    from hermes_cli import kanban_db as kb

    assert kb._resolve_default_spawn() is kb._default_spawn


def test_enabled_selects_paseo_spawn(monkeypatch, tmp_path):
    """kanban.paseo_spawn.enabled true → dispatcher default is spawn_via_paseo."""
    _profile_home(
        tmp_path,
        monkeypatch,
        paseo_cfg="kanban:\n  paseo_spawn:\n    enabled: true\n",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    assert kb._resolve_default_spawn() is paseo_spawn.spawn_via_paseo


def test_run_failure_raises(monkeypatch, tmp_path):
    """A nonzero `paseo run` exit is surfaced (recorded as a spawn failure)."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    _install_fake_paseo(monkeypatch, run_rc=1, run_stdout="", run_stderr="boom")
    workspace = tmp_path / "ws"
    workspace.mkdir()

    import pytest

    with pytest.raises(RuntimeError):
        paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)


# ---------------------------------------------------------------------------
# Project-linked workspace resolution (repo-worktree tasks)
# ---------------------------------------------------------------------------


def _write_workspace_registry(tmp_path, monkeypatch, entries):
    home = tmp_path / ".paseo"
    (home / "projects").mkdir(parents=True, exist_ok=True)
    (home / "projects" / "workspaces.json").write_text(
        json.dumps(entries), encoding="utf-8"
    )
    monkeypatch.setenv("PASEO_HOME", str(home))
    return home


def test_worktree_task_uses_registered_workspace(monkeypatch, tmp_path):
    """A worktree task with an existing registration → --workspace, no create."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    workspace = tmp_path / "wt"
    workspace.mkdir()
    _write_workspace_registry(
        tmp_path,
        monkeypatch,
        [
            {"workspaceId": "wks_archived", "cwd": str(workspace), "archivedAt": "2026-01-01"},
            {"workspaceId": "wks_live0001", "cwd": str(workspace), "archivedAt": None},
        ],
    )
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    created = []
    monkeypatch.setattr(
        paseo_spawn, "_create_directory_workspace",
        lambda *a, **k: created.append(a) or "wks_should_not_be_used",
    )
    calls = _install_fake_paseo(monkeypatch)

    paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="worktree"), str(workspace), board=None
    )

    run_cmd = calls["run"][0]
    assert run_cmd[run_cmd.index("--workspace") + 1] == "wks_live0001"
    assert "--cwd" in run_cmd  # cwd is still pinned
    assert created == []  # existing registration reused, nothing created


def test_worktree_task_creates_workspace_when_unregistered(monkeypatch, tmp_path):
    """No registration → daemon workspace created, its id passed as --workspace."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    workspace = tmp_path / "wt"
    workspace.mkdir()
    _write_workspace_registry(tmp_path, monkeypatch, [])
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    monkeypatch.setattr(
        paseo_spawn, "_create_directory_workspace", lambda bin_, path: "wks_fresh001"
    )
    calls = _install_fake_paseo(monkeypatch)

    paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="worktree"), str(workspace), board=None
    )

    run_cmd = calls["run"][0]
    assert run_cmd[run_cmd.index("--workspace") + 1] == "wks_fresh001"


def test_workspace_resolution_failure_falls_back_to_cwd(monkeypatch, tmp_path):
    """Workspace create failing → plain --cwd run, spawn proceeds."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    workspace = tmp_path / "wt"
    workspace.mkdir()
    _write_workspace_registry(tmp_path, monkeypatch, [])
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    monkeypatch.setattr(
        paseo_spawn, "_create_directory_workspace", lambda bin_, path: None
    )
    calls = _install_fake_paseo(monkeypatch)

    pid = paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="worktree"), str(workspace), board=None
    )

    run_cmd = calls["run"][0]
    assert "--workspace" not in run_cmd
    assert run_cmd[run_cmd.index("--cwd") + 1] == str(workspace)
    assert pid == 9999  # spawn unaffected


def test_scratch_task_keeps_plain_cwd(monkeypatch, tmp_path):
    """Non-worktree tasks never touch workspace resolution."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    workspace = tmp_path / "scratch"
    workspace.mkdir()
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    looked = []
    monkeypatch.setattr(
        paseo_spawn, "_find_registered_workspace",
        lambda path: looked.append(path) or None,
    )
    calls = _install_fake_paseo(monkeypatch)

    paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="scratch"), str(workspace), board=None
    )

    assert "--workspace" not in calls["run"][0]
    assert looked == []  # resolution short-circuits on workspace_kind


def test_create_directory_workspace_invokes_daemon_client(monkeypatch, tmp_path):
    """_create_directory_workspace shells node with the client.js URI and parses the id."""
    from hermes_cli import paseo_spawn

    client_js = tmp_path / "dist" / "utils" / "client.js"
    client_js.parent.mkdir(parents=True)
    client_js.write_text("// stub", encoding="utf-8")
    monkeypatch.setattr(paseo_spawn, "_paseo_client_js", lambda bin_: client_js)
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)

    captured = {}

    def fake_run(cmd, *a, **k):
        captured["cmd"] = list(cmd)
        return _CompletedRun(returncode=0, stdout='{"workspaceId": "wks_new42"}\n')

    monkeypatch.setattr(subprocess, "run", fake_run)

    ws = paseo_spawn._create_directory_workspace("paseo", "/some/worktree")

    assert ws == "wks_new42"
    cmd = captured["cmd"]
    assert cmd[0] == "/usr/bin/node"
    assert "--input-type=module" in cmd
    assert cmd[-2] == client_js.as_uri()
    assert cmd[-1] == "/some/worktree"


def test_create_directory_workspace_failure_returns_none(monkeypatch, tmp_path):
    """Daemon errors and missing prerequisites both yield None (no raise)."""
    from hermes_cli import paseo_spawn

    # Missing client.js → None without shelling out.
    monkeypatch.setattr(paseo_spawn, "_paseo_client_js", lambda bin_: None)
    assert paseo_spawn._create_directory_workspace("paseo", "/x") is None

    # Daemon error (nonzero rc) → None.
    client_js = tmp_path / "client.js"
    client_js.write_text("// stub", encoding="utf-8")
    monkeypatch.setattr(paseo_spawn, "_paseo_client_js", lambda bin_: client_js)
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda name: "/usr/bin/node")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _CompletedRun(returncode=1, stderr="daemon down"),
    )
    assert paseo_spawn._create_directory_workspace("paseo", "/x") is None


def test_find_registered_workspace_matches_realpath(tmp_path, monkeypatch):
    """Registry lookup resolves symlinks and skips archived entries."""
    from hermes_cli import paseo_spawn

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)
    _write_workspace_registry(
        tmp_path,
        monkeypatch,
        [{"workspaceId": "wks_sym", "cwd": str(link), "archivedAt": None}],
    )

    assert paseo_spawn._find_registered_workspace(str(real_dir)) == "wks_sym"
    assert paseo_spawn._find_registered_workspace(str(tmp_path / "other")) is None


# ---------------------------------------------------------------------------
# Watch loop (the tracked worker PID)
# ---------------------------------------------------------------------------


def _watch(paseo_spawn, monkeypatch, *, task_states, inspect_results, deadline=None, run_id=1,
           gone_threshold=3):
    """Run watch_worker with scripted task-state and inspect sequences.

    ``task_states`` / ``inspect_results`` are lists consumed one entry per
    poll; the final entry repeats forever. ``time.sleep`` is a no-op.
    """
    polls = {"n": 0}

    def fake_state(db_path, task_id):
        i = min(polls["n"], len(task_states) - 1)
        return task_states[i]

    def fake_inspect(paseo_bin, agent_id):
        i = min(polls["n"], len(inspect_results) - 1)
        polls["n"] += 1
        return inspect_results[i]

    monkeypatch.setattr(paseo_spawn, "_read_task_run_state", fake_state)
    monkeypatch.setattr(paseo_spawn, "_inspect_agent_ok", fake_inspect)

    return paseo_spawn.watch_worker(
        task_id="t_w",
        agent_id="ag_w",
        db_path="/nonexistent.db",
        run_id=run_id,
        deadline=deadline,
        poll_seconds=0,
        gone_threshold=gone_threshold,
    ), polls["n"]


def test_watch_exits_0_on_terminal_task_state(monkeypatch):
    """Task done in the DB → exit 0, even while the agent looks alive."""
    from hermes_cli import paseo_spawn

    rc, _ = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1), ("running", 1), ("done", 1)],
        inspect_results=[True],
    )
    assert rc == 0


def test_watch_ignores_agent_idle_while_task_running(monkeypatch):
    """Agent flapping to idle (inspect ok) must NOT terminate the watcher.

    The loop keeps polling through many agent-alive polls while the task is
    running, and only exits when the task's DB state settles. This is the
    regression guard for the t_6811e170 false protocol violation.
    """
    from hermes_cli import paseo_spawn

    # 20 polls of a healthy (idle-flapping) agent with the task running — the
    # watcher must survive all of them and exit 0 only on the DB transition.
    rc, polls = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)] * 20 + [("blocked", 1)],
        inspect_results=[True],
    )
    assert rc == 0
    assert polls >= 20  # it genuinely kept watching


def test_watch_exits_1_when_agent_gone_consecutively(monkeypatch):
    """Agent gone for N consecutive polls while task still running → exit 1."""
    from hermes_cli import paseo_spawn

    rc, polls = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)],
        inspect_results=[False],
        gone_threshold=3,
    )
    assert rc == 1
    assert polls == 3  # exactly the consecutive threshold


def test_watch_transient_inspect_failures_reset(monkeypatch):
    """Non-consecutive inspect failures (daemon blips) never reach the threshold."""
    from hermes_cli import paseo_spawn

    # Pattern: 2 bad, 1 good (reset), 2 bad, 1 good (reset), … then task done.
    inspect_seq = [False, False, True] * 4 + [True]
    task_seq = [("running", 1)] * (len(inspect_seq) - 1) + [("done", 1)]
    rc, _ = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=task_seq,
        inspect_results=inspect_seq,
        gone_threshold=3,
    )
    assert rc == 0  # blips never accumulated to the threshold


def test_watch_exits_1_on_deadline(monkeypatch):
    """Deadline in the past + task still running → exit 1."""
    import time

    from hermes_cli import paseo_spawn

    rc, _ = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)],
        inspect_results=[True],
        deadline=time.time() - 10,
    )
    assert rc == 1


def test_watch_exits_0_when_run_superseded(monkeypatch):
    """current_run_id moved past our run → our watch is stale → exit 0."""
    from hermes_cli import paseo_spawn

    rc, _ = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 2)],  # a newer run owns the task
        inspect_results=[True],
        run_id=1,
    )
    assert rc == 0


def test_watch_exits_0_when_task_deleted(monkeypatch):
    """Task row gone from the DB → nothing left to watch → exit 0."""
    from hermes_cli import paseo_spawn

    rc, _ = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=[("__missing__", None)],
        inspect_results=[True],
    )
    assert rc == 0


def test_watch_survives_transient_db_errors(monkeypatch):
    """None from the DB reader (locked/busy) keeps the loop alive."""
    from hermes_cli import paseo_spawn

    rc, _ = _watch(
        paseo_spawn,
        monkeypatch,
        task_states=[None, None, ("running", 1), ("done", 1)],
        inspect_results=[True],
    )
    assert rc == 0


def test_read_task_run_state_against_real_db(tmp_path, monkeypatch):
    """The read-only reader works against a real kanban schema, and never writes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    try:
        tid = kb.create_task(conn, title="watched")
        conn.execute(
            "UPDATE tasks SET status='running', current_run_id=42 WHERE id=?", (tid,)
        )
        conn.commit()
    finally:
        conn.close()

    assert paseo_spawn._read_task_run_state(str(db_path), tid) == ("running", 42)
    assert paseo_spawn._read_task_run_state(str(db_path), "t_nope") == ("__missing__", None)
    # Unreadable path → transient (None), not a crash.
    assert paseo_spawn._read_task_run_state(str(tmp_path / "missing.db"), tid) is None


def test_inspect_agent_ok_parsing(monkeypatch):
    """_inspect_agent_ok maps inspect --json states to live/gone correctly."""
    import json as _json

    from hermes_cli import paseo_spawn

    def make_run(rc, payload):
        def fake_run(cmd, *a, **k):
            return _CompletedRun(returncode=rc, stdout=_json.dumps(payload))

        return fake_run

    ok = {"Id": "ag_1", "Status": "idle", "Archived": False, "ArchivedAt": None}
    monkeypatch.setattr(subprocess, "run", make_run(0, ok))
    assert paseo_spawn._inspect_agent_ok("paseo", "ag_1") is True

    archived = dict(ok, Archived=True, ArchivedAt="2026-07-14T00:00:00Z")
    monkeypatch.setattr(subprocess, "run", make_run(0, archived))
    assert paseo_spawn._inspect_agent_ok("paseo", "ag_1") is False

    errored = dict(ok, Status="error")
    monkeypatch.setattr(subprocess, "run", make_run(0, errored))
    assert paseo_spawn._inspect_agent_ok("paseo", "ag_1") is False

    monkeypatch.setattr(subprocess, "run", make_run(1, {}))  # not found / daemon down
    assert paseo_spawn._inspect_agent_ok("paseo", "ag_1") is False


def test_watch_entrypoint_end_to_end(tmp_path, monkeypatch):
    """`python -m hermes_cli.paseo_spawn --watch` exits 0 once the task settles."""
    import subprocess as real_subprocess
    import sys
    import threading
    import time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db as kb

    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    tid = kb.create_task(conn, title="e2e")
    conn.execute("UPDATE tasks SET status='running', current_run_id=1 WHERE id=?", (tid,))
    conn.commit()

    # Stub paseo bin: inspect always reports a live agent.
    stub = tmp_path / "paseo"
    stub.write_text(
        "#!/bin/sh\n"
        "echo '{\"Id\": \"ag_1\", \"Status\": \"idle\", \"Archived\": false}'\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)

    def settle():
        # Own connection: sqlite3 connections are not shareable across threads.
        import sqlite3

        time.sleep(1.0)
        c2 = sqlite3.connect(str(db_path), timeout=10)
        try:
            c2.execute("UPDATE tasks SET status='done' WHERE id=?", (tid,))
            c2.commit()
        finally:
            c2.close()

    t = threading.Thread(target=settle)
    t.start()
    try:
        proc = real_subprocess.run(
            [
                sys.executable, "-m", "hermes_cli.paseo_spawn",
                "--watch", "--task", tid, "--agent", "ag_1",
                "--db", str(db_path), "--run", "1",
                "--paseo-bin", str(stub),
            ],
            env={**__import__("os").environ, "HERMES_PASEO_WATCH_POLL": "0.2"},
            timeout=30,
            capture_output=True,
            text=True,
        )
    finally:
        t.join()
        conn.close()
    assert proc.returncode == 0, proc.stderr

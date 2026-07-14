"""Tests: config-gated Paseo-agent kanban worker launcher.

Covers :mod:`hermes_cli.paseo_spawn` and its wiring into the dispatcher's
default spawn resolution (:func:`hermes_cli.kanban_db._resolve_default_spawn`).

The paseo CLI is mocked entirely at the ``subprocess`` boundary — no real
daemon, no real agents. Each test installs a fake ``subprocess.run`` that
dispatches on the paseo subcommand (``status`` / ``ls`` / ``run``) and a fake
``subprocess.Popen`` that stands in for the ``paseo wait`` watcher.
"""

from __future__ import annotations

import json
import subprocess


def _make_task(kb, *, assignee: str = "w", task_id: str = "t_paseo"):
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
        workspace_kind="dir",
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
):
    """Patch subprocess.run/Popen to emulate the paseo CLI. Returns a recorder."""
    if run_stdout is None:
        run_stdout = json.dumps(
            {"agentId": "ag_1234567", "status": "running", "provider": "hermes", "cwd": "/ws"}
        )
    calls: dict = {"run": [], "ls": [], "status": [], "popen": []}

    def fake_run(cmd, *args, **kwargs):
        sub = cmd[1] if len(cmd) > 1 else None
        if sub == "status":
            calls["status"].append(list(cmd))
            return _CompletedRun(returncode=status_rc)
        if sub == "ls":
            calls["ls"].append(list(cmd))
            return _CompletedRun(returncode=0, stdout=json.dumps(ls_agents or []))
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
    # Watcher: `paseo wait <id> --json --timeout <max_runtime + slack>`.
    assert len(calls["popen"]) == 1
    wait_cmd = calls["popen"][0]
    assert wait_cmd[:3] == ["paseo", "wait", "ag_1234567"]
    assert "--timeout" in wait_cmd
    assert wait_cmd[wait_cmd.index("--timeout") + 1] == str(1800 + 300)


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


def test_existing_agent_reattaches_without_second_run(monkeypatch, tmp_path):
    """A live labeled agent → watcher only, no second `paseo run`."""
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

    assert calls["run"] == []  # double-spawn guard fired
    assert len(calls["popen"]) == 1
    assert calls["popen"][0][:3] == ["paseo", "wait", "ag_existing1"]
    assert pid == 9999


def test_idle_agent_is_reattachable(monkeypatch, tmp_path):
    """An idle (non-archived) agent still counts as re-attachable."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(
        monkeypatch,
        ls_agents=[{"id": "ag_idle", "status": "idle"}],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    assert calls["run"] == []
    assert calls["popen"][0][2] == "ag_idle"


def test_errored_agent_not_reattached(monkeypatch, tmp_path):
    """An errored agent is NOT re-attachable → a fresh agent is launched."""
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
    assert len(calls["run"]) == 1  # errored agent ignored, new one created


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

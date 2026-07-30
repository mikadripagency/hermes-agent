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

import pytest
import yaml

pytestmark = pytest.mark.usefixtures("explicit_delivery_contract_for_kanban_fixtures")


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


def _profile_home(
    tmp_path,
    monkeypatch,
    *,
    assignee: str = "w",
    paseo_cfg: str = "",
    provider_profile: str | None = None,
):
    root = tmp_path / ".hermes"
    (root / "profiles" / assignee).mkdir(parents=True)
    (root / "profiles" / assignee / "config.yaml").write_text(
        "toolsets:\n  - kanban\n", encoding="utf-8"
    )
    config = yaml.safe_load("toolsets:\n  - kanban\n" + paseo_cfg) or {}
    config.setdefault("kanban", {}).setdefault("paseo_spawn", {})["profile"] = (
        provider_profile or assignee
    )
    root.joinpath("config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
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
    ls_rc: int = 0,
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
            return _CompletedRun(
                returncode=ls_rc,
                stdout=json.dumps(ls_agents or []),
                stderr="" if ls_rc == 0 else "ls failed",
            )
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
    assert "--label kanban_profile=w" in joined
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


def test_no_max_runtime_derives_fallback_deadline(monkeypatch, tmp_path):
    """A task with no max_runtime still gets a bounded --deadline (FIX 3).

    t_05292e4e's default-profile watcher was spawned deadline-less and hung
    unbounded. When max_runtime is absent we fall back to the canonical
    kb.DEFAULT_WORKER_MAX_RUNTIME_SECONDS (7200s), the same bound the sentinel
    uses, plus slack.
    """
    import dataclasses
    import time as _time

    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    task = dataclasses.replace(_make_task(kb), max_runtime_seconds=None)
    paseo_spawn.spawn_via_paseo(task, str(workspace), board=None)

    watch_cmd = calls["popen"][0]
    assert "--deadline" in watch_cmd, "deadline-less watcher regression"
    deadline = float(watch_cmd[watch_cmd.index("--deadline") + 1])
    expected = _time.time() + kb.DEFAULT_WORKER_MAX_RUNTIME_SECONDS + 300
    assert abs(deadline - expected) < 60


def test_no_max_runtime_fallback_deadline_is_config_tunable(monkeypatch, tmp_path):
    """kanban.default_max_runtime_seconds overrides the 7200s fallback bound
    for deadline-less tasks (two E2E tasks doing legitimate work timed out at
    exactly ~7200s under the hardcoded constant)."""
    import dataclasses
    import time as _time

    root = _profile_home(tmp_path, monkeypatch)
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\nkanban:\n  default_max_runtime_seconds: 10800\n"
        "  paseo_spawn:\n    profile: w\n",
        encoding="utf-8",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    task = dataclasses.replace(_make_task(kb), max_runtime_seconds=None)
    paseo_spawn.spawn_via_paseo(task, str(workspace), board=None)

    watch_cmd = calls["popen"][0]
    deadline = float(watch_cmd[watch_cmd.index("--deadline") + 1])
    expected = _time.time() + 10800 + 300
    assert abs(deadline - expected) < 60


def test_explicit_max_runtime_beats_configured_default(monkeypatch, tmp_path):
    """A per-task max_runtime_seconds always wins over the configured fallback."""
    import time as _time

    root = _profile_home(tmp_path, monkeypatch)
    root.joinpath("config.yaml").write_text(
        "toolsets:\n  - kanban\nkanban:\n  default_max_runtime_seconds: 10800\n"
        "  paseo_spawn:\n    profile: w\n",
        encoding="utf-8",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    watch_cmd = calls["popen"][0]
    deadline = float(watch_cmd[watch_cmd.index("--deadline") + 1])
    assert abs(deadline - (_time.time() + 1800 + 300)) < 60


def test_default_worker_max_runtime_seconds_fallback_and_validation(monkeypatch, tmp_path):
    """Helper unit: config value wins when positive; missing/invalid/zero
    values fall back to the canonical 7200s constant."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import config as hconfig
    from hermes_cli import kanban_db as kb

    monkeypatch.setattr(
        hconfig, "load_config",
        lambda: {"kanban": {"default_max_runtime_seconds": 4321}},
    )
    assert kb.default_worker_max_runtime_seconds() == 4321

    monkeypatch.setattr(hconfig, "load_config", lambda: {"kanban": {}})
    assert kb.default_worker_max_runtime_seconds() == kb.DEFAULT_WORKER_MAX_RUNTIME_SECONDS

    monkeypatch.setattr(
        hconfig, "load_config",
        lambda: {"kanban": {"default_max_runtime_seconds": 0}},
    )
    assert kb.default_worker_max_runtime_seconds() == kb.DEFAULT_WORKER_MAX_RUNTIME_SECONDS

    monkeypatch.setattr(
        hconfig, "load_config",
        lambda: {"kanban": {"default_max_runtime_seconds": "not-a-number"}},
    )
    assert kb.default_worker_max_runtime_seconds() == kb.DEFAULT_WORKER_MAX_RUNTIME_SECONDS


def test_watcher_receives_idle_stall_default(monkeypatch, tmp_path):
    """The watcher is spawned with the default idle-stall threshold (FIX 1)."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    watch_cmd = calls["popen"][0]
    assert "--idle-stall" in watch_cmd
    assert watch_cmd[watch_cmd.index("--idle-stall") + 1] == str(
        int(paseo_spawn.WATCH_IDLE_STALL_SECONDS)
    )


def test_watcher_idle_stall_disabled_omits_flag(monkeypatch, tmp_path):
    """kanban.paseo_spawn.idle_stall_seconds: 0 omits --idle-stall entirely."""
    _profile_home(
        tmp_path,
        monkeypatch,
        paseo_cfg=(
            "kanban:\n"
            "  paseo_spawn:\n"
            "    enabled: true\n"
            "    idle_stall_seconds: 0\n"
        ),
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    watch_cmd = calls["popen"][0]
    assert "--idle-stall" not in watch_cmd


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


def test_provider_profile_mismatch_falls_back_before_paseo_spawn(monkeypatch, tmp_path):
    """A default task cannot be claimed then launched by the developer provider."""
    _profile_home(
        tmp_path,
        monkeypatch,
        assignee="default",
        provider_profile="developer",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    fallback = []
    monkeypatch.setattr(
        kb,
        "_default_spawn",
        lambda task, workspace, *, board=None: fallback.append(
            (task.assignee, workspace, board)
        )
        or 4242,
    )
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(
        _make_task(kb, assignee="default"), str(workspace), board=None
    )

    assert pid == 4242
    assert fallback == [("default", str(workspace), None)]
    assert calls["status"] == [] and calls["run"] == []


def test_actively_working_agent_reattaches_without_second_run(monkeypatch, tmp_path):
    """An ACTIVELY working labeled agent → watcher only: no archive, no new run."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(
        monkeypatch,
        ls_agents=[{
            "id": "ag_existing1",
            "status": "running",
            "shortId": "ag_exis",
            "labels": ["kanban_run=7", "kanban_profile=w"],
        }],
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


def test_active_agent_from_old_run_is_force_archived_before_fresh_spawn(
    monkeypatch, tmp_path
):
    """A running agent is reusable only for the exact run and profile labels."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    old = {"id": "ag_old", "status": "running"}

    def list_agents(_bin, _task_id, *, run_id=None, profile=None):
        return [old] if run_id is None and profile is None else []

    monkeypatch.setattr(paseo_spawn, "_list_task_agents", list_agents)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    pid = paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert pid == 9999
    assert len(calls["archive"]) == 1
    assert calls["archive"][0][:3] == ["paseo", "archive", "ag_old"]
    assert "--force" in calls["archive"][0]
    assert len(calls["run"]) == 1


def test_active_stale_agent_archive_failure_prevents_second_worker(
    monkeypatch, tmp_path
):
    """A failed forced retirement cannot create two live workers for one task."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch, archive_rc=1)
    old = {"id": "ag_old", "status": "running"}
    monkeypatch.setattr(
        paseo_spawn,
        "_list_task_agents",
        lambda _bin, _task_id, *, run_id=None, profile=None: (
            [old] if run_id is None and profile is None else []
        ),
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with pytest.raises(RuntimeError, match="could not retire stale Paseo agent"):
        paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert len(calls["archive"]) == 1
    assert "--force" in calls["archive"][0]
    assert calls["run"] == [] and calls["popen"] == []


def test_superseded_run_retires_agent_before_watcher_reports_settled(monkeypatch):
    """A reclaimed Paseo agent cannot keep using native shell after run advance."""
    from hermes_cli import paseo_spawn

    retired = []
    monkeypatch.setattr(
        paseo_spawn,
        "_read_task_run_state",
        lambda *_args: ("running", 8),
    )
    monkeypatch.setattr(
        paseo_spawn,
        "_archive_agent",
        lambda paseo_bin, agent_id, task_id, *, force=False: retired.append(
            (paseo_bin, agent_id, task_id, force)
        ) or True,
    )

    result = paseo_spawn.watch_worker(
        task_id="t_stale",
        agent_id="ag_stale",
        db_path="unused.db",
        run_id=7,
        paseo_bin="paseo",
        poll_seconds=0,
    )

    assert result == paseo_spawn.WATCH_EXIT_TASK_SETTLED
    assert retired == [("paseo", "ag_stale", "t_stale", True)]


def test_watcher_sigterm_retires_daemon_agent_before_exit(monkeypatch):
    """Manual reclaim kills the ACP agent, not only its local watcher PID."""
    import signal

    from hermes_cli import paseo_spawn

    installed = []
    retired = []
    monkeypatch.setattr(
        signal,
        "signal",
        lambda signum, handler: installed.append((signum, handler)),
    )
    monkeypatch.setattr(
        paseo_spawn,
        "_archive_agent",
        lambda paseo_bin, agent_id, task_id, *, force=False: retired.append(
            (paseo_bin, agent_id, task_id, force)
        ) or True,
    )

    paseo_spawn._install_watch_termination_handler(
        paseo_bin="paseo",
        agent_id="ag_stale",
        task_id="t_stale",
    )

    assert installed[0][0] == signal.SIGTERM
    with pytest.raises(SystemExit, match=str(128 + int(signal.SIGTERM))):
        installed[0][1](signal.SIGTERM, None)
    assert retired == [("paseo", "ag_stale", "t_stale", True)]


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


def test_archive_failure_blocks_fresh_agent(monkeypatch, tmp_path):
    """A failed retirement cannot create a second native-shell owner."""
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

    with pytest.raises(RuntimeError, match="could not retire stale Paseo agent"):
        paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert len(calls["archive"]) == 1  # attempted
    assert calls["run"] == []
    assert calls["popen"] == []


def test_agent_discovery_failure_blocks_spawn(monkeypatch, tmp_path):
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch, ls_rc=1)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    with pytest.raises(RuntimeError, match="agent discovery failed"):
        paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)

    assert len(calls["ls"]) == 1
    assert calls["run"] == [] and calls["popen"] == []


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


def test_mode_yaml_false_omits_flag(monkeypatch, tmp_path):
    """kanban.paseo_spawn.mode: off → YAML bool False → no --mode flag.

    Regression: `str(False)` used to leak a bogus `--mode False`.
    """
    _profile_home(
        tmp_path,
        monkeypatch,
        paseo_cfg="kanban:\n  paseo_spawn:\n    mode: off\n",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    run_cmd = calls["run"][0]
    assert "--mode" not in run_cmd
    assert "False" not in run_cmd


def test_mode_yaml_true_falls_back_to_default(monkeypatch, tmp_path):
    """kanban.paseo_spawn.mode: on → YAML bool True → safe default dont_ask.

    Regression: `str(True)` used to leak a bogus `--mode True`.
    """
    _profile_home(
        tmp_path,
        monkeypatch,
        paseo_cfg="kanban:\n  paseo_spawn:\n    mode: on\n",
    )
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    calls = _install_fake_paseo(monkeypatch)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    run_cmd = calls["run"][0]
    assert run_cmd[run_cmd.index("--mode") + 1] == "dont_ask"
    assert "True" not in run_cmd


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
            {
                "id": "ag_working",
                "status": "running",
                "labels": ["kanban_run=7", "kanban_profile=w"],
            },
        ],
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    paseo_spawn.spawn_via_paseo(_make_task(kb), str(workspace), board=None)
    assert calls["run"] == []
    assert len(calls["archive"]) == 1
    assert calls["archive"][0][2] == "ag_old_idle"
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


def _write_project_registry(tmp_path, monkeypatch, entries):
    home = tmp_path / ".paseo"
    (home / "projects").mkdir(parents=True, exist_ok=True)
    (home / "projects" / "projects.json").write_text(
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


def test_scratch_task_not_under_project_keeps_plain_cwd(monkeypatch, tmp_path):
    """A non-worktree task outside any registered project → plain --cwd.

    Its cwd isn't inside a registered project root, so grouping short-circuits
    before any workspace registration lookup/create.
    """
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    # A registered project that does NOT contain the scratch cwd.
    (tmp_path / "elsewhere").mkdir()
    _write_project_registry(
        tmp_path,
        monkeypatch,
        [{"projectId": str(tmp_path / "elsewhere"), "rootPath": str(tmp_path / "elsewhere"),
          "kind": "non_git", "archivedAt": None}],
    )
    workspace = tmp_path / "scratch"
    workspace.mkdir()
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    looked = []
    monkeypatch.setattr(
        paseo_spawn, "_find_registered_workspace",
        lambda path: looked.append(path) or None,
    )
    created = []
    monkeypatch.setattr(
        paseo_spawn, "_create_directory_workspace",
        lambda *a, **k: created.append(a) or "wks_should_not_be_used",
    )
    calls = _install_fake_paseo(monkeypatch)

    paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="scratch"), str(workspace), board=None
    )

    run_cmd = calls["run"][0]
    assert "--workspace" not in run_cmd
    assert run_cmd[run_cmd.index("--cwd") + 1] == str(workspace)
    assert looked == []  # resolution short-circuits: not inside a project
    assert created == []


def test_dir_task_under_project_groups_via_root_workspace(monkeypatch, tmp_path):
    """A dir task inside a registered project → workspace anchored at the ROOT.

    The grouping workspace is created for the PROJECT ROOT (so the daemon
    attaches it to the existing project), while the agent still runs in the
    subdirectory via --cwd.
    """
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    project_root = tmp_path / "Apex-Experiments"
    subdir = project_root / "shinie-ipl-main-image"
    subdir.mkdir(parents=True)
    _write_project_registry(
        tmp_path,
        monkeypatch,
        [{"projectId": str(project_root), "rootPath": str(project_root),
          "kind": "non_git", "archivedAt": None}],
    )
    _write_workspace_registry(tmp_path, monkeypatch, [])  # nothing registered yet
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    created_paths = []

    def fake_create(bin_, path):
        created_paths.append(path)
        return "wks_root001"

    monkeypatch.setattr(paseo_spawn, "_create_directory_workspace", fake_create)
    calls = _install_fake_paseo(monkeypatch)

    paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="dir"), str(subdir), board=None
    )

    run_cmd = calls["run"][0]
    # Grouping workspace is created for the project ROOT, not the subdir.
    assert created_paths == [str(project_root)]
    assert run_cmd[run_cmd.index("--workspace") + 1] == "wks_root001"
    # ...but the agent still runs in the subdirectory.
    assert run_cmd[run_cmd.index("--cwd") + 1] == str(subdir)


def test_dir_task_reuses_existing_root_registration(monkeypatch, tmp_path):
    """A second dir task under the same project reuses the root workspace."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    project_root = tmp_path / "Apex-Experiments"
    subdir = project_root / "another-experiment"
    subdir.mkdir(parents=True)
    _write_project_registry(
        tmp_path,
        monkeypatch,
        [{"projectId": str(project_root), "rootPath": str(project_root),
          "kind": "non_git", "archivedAt": None}],
    )
    # An existing non-archived workspace registered AT THE ROOT.
    _write_workspace_registry(
        tmp_path,
        monkeypatch,
        [
            {"workspaceId": "wks_archived", "cwd": str(project_root), "archivedAt": "2026-01-01"},
            {"workspaceId": "wks_rootlive", "cwd": str(project_root), "archivedAt": None},
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
        _make_task(kb, workspace_kind="dir"), str(subdir), board=None
    )

    run_cmd = calls["run"][0]
    assert run_cmd[run_cmd.index("--workspace") + 1] == "wks_rootlive"
    assert run_cmd[run_cmd.index("--cwd") + 1] == str(subdir)
    assert created == []  # existing root registration reused


def test_dir_task_resolution_failure_falls_back_to_cwd(monkeypatch, tmp_path):
    """A dir task inside a project whose root workspace can't be created → --cwd."""
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    project_root = tmp_path / "Apex-Experiments"
    subdir = project_root / "exp"
    subdir.mkdir(parents=True)
    _write_project_registry(
        tmp_path,
        monkeypatch,
        [{"projectId": str(project_root), "rootPath": str(project_root),
          "kind": "non_git", "archivedAt": None}],
    )
    _write_workspace_registry(tmp_path, monkeypatch, [])
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    monkeypatch.setattr(
        paseo_spawn, "_create_directory_workspace", lambda bin_, path: None
    )
    calls = _install_fake_paseo(monkeypatch)

    pid = paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="dir"), str(subdir), board=None
    )

    run_cmd = calls["run"][0]
    assert "--workspace" not in run_cmd
    assert run_cmd[run_cmd.index("--cwd") + 1] == str(subdir)
    assert pid == 9999


def test_dir_task_at_project_root_is_not_reanchored(monkeypatch, tmp_path):
    """cwd == a registered project root → not a strict subdir → plain --cwd.

    The exact-match case already groups correctly on a bare run; re-anchoring
    would risk latching onto a stray per-subdir project. Strict-ancestor only.
    """
    _profile_home(tmp_path, monkeypatch)
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    project_root = tmp_path / "Apex-Experiments"
    project_root.mkdir()
    _write_project_registry(
        tmp_path,
        monkeypatch,
        [{"projectId": str(project_root), "rootPath": str(project_root),
          "kind": "non_git", "archivedAt": None}],
    )
    _write_workspace_registry(tmp_path, monkeypatch, [])
    monkeypatch.setattr(paseo_spawn, "_record_linkage_comment", lambda *a, **k: None)
    created = []
    monkeypatch.setattr(
        paseo_spawn, "_create_directory_workspace",
        lambda *a, **k: created.append(a) or "wks_x",
    )
    calls = _install_fake_paseo(monkeypatch)

    paseo_spawn.spawn_via_paseo(
        _make_task(kb, workspace_kind="dir"), str(project_root), board=None
    )

    assert "--workspace" not in calls["run"][0]
    assert created == []


def test_registered_project_root_deepest_match_wins(tmp_path, monkeypatch):
    """The deepest strict-ancestor project root is chosen; symlinks resolve."""
    from hermes_cli import paseo_spawn

    outer = tmp_path / "Projects"
    inner = outer / "Apex-Experiments"
    leaf = inner / "shinie" / "deep"
    leaf.mkdir(parents=True)
    _write_project_registry(
        tmp_path,
        monkeypatch,
        [
            {"projectId": str(outer), "rootPath": str(outer), "kind": "non_git", "archivedAt": None},
            {"projectId": str(inner), "rootPath": str(inner), "kind": "non_git", "archivedAt": None},
            {"projectId": str(tmp_path / "gone"), "rootPath": str(tmp_path / "gone"),
             "kind": "non_git", "archivedAt": "2026-01-01"},  # archived: ignored
        ],
    )

    # Deepest strict ancestor of the leaf is `inner`, not `outer`.
    assert paseo_spawn._registered_project_root_for_cwd(str(leaf)) == str(inner)
    # A path outside every project → None.
    assert paseo_spawn._registered_project_root_for_cwd(str(tmp_path / "unrelated")) is None
    # Exact match on a root → excluded (not a strict ancestor).
    assert paseo_spawn._registered_project_root_for_cwd(str(inner)) == str(outer)


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
    monkeypatch.setattr(paseo_spawn, "_archive_agent", lambda *args, **kwargs: True)

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


def test_watch_retires_agent_before_terminal_exit(monkeypatch):
    from hermes_cli import paseo_spawn

    retired = []
    monkeypatch.setattr(
        paseo_spawn, "_read_task_run_state", lambda *_args: ("blocked", 1)
    )
    monkeypatch.setattr(
        paseo_spawn,
        "_archive_agent",
        lambda paseo_bin, agent_id, task_id, *, force=False: retired.append(
            (paseo_bin, agent_id, task_id, force)
        ) or True,
    )

    assert paseo_spawn.watch_worker(
        task_id="t_w",
        agent_id="ag_w",
        db_path="unused.db",
        run_id=1,
        paseo_bin="paseo",
        poll_seconds=0,
    ) == paseo_spawn.WATCH_EXIT_TASK_SETTLED
    assert retired == [("paseo", "ag_w", "t_w", True)]


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

    monkeypatch.setattr(paseo_spawn, "_archive_agent", lambda *args, **kwargs: True)
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


# ---------------------------------------------------------------------------
# Soft-stall exit: idle agent + frozen heartbeat (FIX 1 / t_18088232 run 349)
# ---------------------------------------------------------------------------

_STALL_NOW = 1_000_000.0


def _watch_stall(
    paseo_spawn,
    monkeypatch,
    *,
    task_states,
    inspect_states,
    heartbeat_ages,
    idle_stall_seconds,
    run_id=1,
    gone_threshold=10_000,
):
    """Run watch_worker scripting (ok, status) inspects + heartbeat ages.

    ``inspect_states`` yields ``(ok, status)`` tuples (what
    ``_inspect_agent_state`` returns). ``heartbeat_ages`` yields the age of the
    task's last heartbeat in seconds-ago (or ``None`` for "no heartbeat"). The
    clock is frozen at ``_STALL_NOW`` and the poll index advances once per loop
    iteration via a stubbed ``time.sleep`` — so a "frozen" heartbeat is just a
    large age. The final entry of each list repeats forever.
    """
    import time as _time

    polls = {"n": 0}

    def fake_state(db_path, task_id):
        return task_states[min(polls["n"], len(task_states) - 1)]

    def fake_inspect_state(paseo_bin, agent_id):
        return inspect_states[min(polls["n"], len(inspect_states) - 1)]

    def fake_hb(db_path, task_id):
        age = heartbeat_ages[min(polls["n"], len(heartbeat_ages) - 1)]
        return None if age is None else _STALL_NOW - age

    def fake_sleep(_seconds):
        polls["n"] += 1

    monkeypatch.setattr(paseo_spawn, "_read_task_run_state", fake_state)
    monkeypatch.setattr(paseo_spawn, "_inspect_agent_state", fake_inspect_state)
    monkeypatch.setattr(paseo_spawn, "_read_task_heartbeat", fake_hb)
    monkeypatch.setattr(paseo_spawn, "_archive_agent", lambda *args, **kwargs: True)
    monkeypatch.setattr(_time, "time", lambda: _STALL_NOW)
    monkeypatch.setattr(_time, "sleep", fake_sleep)

    rc = paseo_spawn.watch_worker(
        task_id="t_w",
        agent_id="ag_w",
        db_path="/nonexistent.db",
        run_id=run_id,
        deadline=None,
        poll_seconds=0,
        gone_threshold=gone_threshold,
        idle_stall_seconds=idle_stall_seconds,
    )
    return rc, polls["n"]


def test_watch_idle_fresh_heartbeat_keeps_watching(monkeypatch):
    """Idle agent with a FRESH heartbeat (between turns) must keep watching."""
    from hermes_cli import paseo_spawn

    rc, polls = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)] * 6 + [("done", 1)],
        inspect_states=[(True, "idle")],
        heartbeat_ages=[10],  # 10s ago — well within the 900s threshold
        idle_stall_seconds=900,
    )
    assert rc == 0
    assert polls >= 6  # it genuinely kept watching through the idle polls


def test_watch_idle_frozen_heartbeat_exits_1(monkeypatch):
    """Idle agent whose heartbeat is frozen past the threshold → exit 1."""
    from hermes_cli import paseo_spawn

    rc, _ = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)],  # never settles
        inspect_states=[(True, "idle")],
        heartbeat_ages=[1000],  # 1000s > 900s threshold
        idle_stall_seconds=900,
    )
    assert rc == 1


def test_watch_busy_frozen_heartbeat_keeps_watching(monkeypatch):
    """A busy (running) agent is never stall-exited, even if its heartbeat lags.

    Auto-heartbeat fires per agent loop iteration, so a working agent
    heartbeats; only ``idle`` qualifies as a stall candidate. Here the agent
    reports ``running`` with a stale heartbeat and the watcher must keep going
    until the task settles.
    """
    from hermes_cli import paseo_spawn

    rc, polls = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)] * 5 + [("done", 1)],
        inspect_states=[(True, "running")],
        heartbeat_ages=[1000],  # frozen, but agent is busy → ignored
        idle_stall_seconds=900,
    )
    assert rc == 0
    assert polls >= 5


def test_watch_idle_stall_transient_inspect_failure_not_idle(monkeypatch):
    """A transient inspect failure (ok=False) must never count as idle.

    Even with a frozen heartbeat, an inspect that failed to read a status
    cannot trip the soft-stall exit. With a high gone-threshold the failures
    don't accumulate to "gone", so the watcher settles on the task state.
    """
    from hermes_cli import paseo_spawn

    rc, _ = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)] * 4 + [("done", 1)],
        inspect_states=[(False, None)],  # inspect failed → not idle
        heartbeat_ages=[1000],  # frozen, but must be ignored
        idle_stall_seconds=900,
        gone_threshold=10_000,
    )
    assert rc == 0


def test_watch_idle_no_heartbeat_keeps_watching(monkeypatch):
    """Idle agent but heartbeat unreadable (None) → cannot judge → keep watching."""
    from hermes_cli import paseo_spawn

    rc, _ = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)] * 3 + [("done", 1)],
        inspect_states=[(True, "idle")],
        heartbeat_ages=[None],  # locked/never-heartbeated → don't stall
        idle_stall_seconds=900,
    )
    assert rc == 0


def test_watch_terminal_state_wins_over_idle_stall(monkeypatch):
    """A terminal task exits 0 first, even when idle + heartbeat is frozen."""
    from hermes_cli import paseo_spawn

    rc, _ = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("done", 1)],
        inspect_states=[(True, "idle")],
        heartbeat_ages=[100_000],  # ancient, but the task is already terminal
        idle_stall_seconds=900,
    )
    assert rc == 0


def test_watch_idle_stall_threshold_configurable(monkeypatch):
    """The stall threshold is configurable: same frozen heartbeat, two verdicts."""
    from hermes_cli import paseo_spawn

    # Heartbeat 200s old: past a 100s threshold → stall (exit 1).
    rc_tight, _ = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)],
        inspect_states=[(True, "idle")],
        heartbeat_ages=[200],
        idle_stall_seconds=100,
    )
    assert rc_tight == 1

    # Same 200s-old heartbeat, but a 100000s threshold → still watching → 0.
    rc_loose, _ = _watch_stall(
        paseo_spawn,
        monkeypatch,
        task_states=[("running", 1)] * 3 + [("done", 1)],
        inspect_states=[(True, "idle")],
        heartbeat_ages=[200],
        idle_stall_seconds=100_000,
    )
    assert rc_loose == 0


def test_watch_idle_stall_disabled_ignores_frozen_heartbeat(monkeypatch):
    """idle_stall_seconds=0 disables the check entirely (uses liveness path)."""
    from hermes_cli import paseo_spawn

    # With the check disabled, watch_worker uses _inspect_agent_ok for
    # liveness; stub it to "alive". A frozen heartbeat must be ignored.
    polls = {"n": 0}
    task_states = [("running", 1)] * 4 + [("done", 1)]

    def fake_state(db_path, task_id):
        return task_states[min(polls["n"], len(task_states) - 1)]

    def fake_ok(paseo_bin, agent_id):
        polls["n"] += 1
        return True

    def _should_not_be_called(*a, **k):  # pragma: no cover - guard
        raise AssertionError("heartbeat read while stall disabled")

    monkeypatch.setattr(paseo_spawn, "_read_task_run_state", fake_state)
    monkeypatch.setattr(paseo_spawn, "_inspect_agent_ok", fake_ok)
    monkeypatch.setattr(paseo_spawn, "_read_task_heartbeat", _should_not_be_called)
    monkeypatch.setattr(paseo_spawn, "_archive_agent", lambda *args, **kwargs: True)

    rc = paseo_spawn.watch_worker(
        task_id="t_w",
        agent_id="ag_w",
        db_path="/nonexistent.db",
        run_id=1,
        poll_seconds=0,
        idle_stall_seconds=0,
    )
    assert rc == 0


def test_inspect_agent_state_extracts_status(monkeypatch):
    """_inspect_agent_state returns (ok, lowercased status); failures → (False, None)."""
    import json as _json

    from hermes_cli import paseo_spawn

    def make_run(rc, payload):
        def fake_run(cmd, *a, **k):
            return _CompletedRun(returncode=rc, stdout=_json.dumps(payload))

        return fake_run

    idle = {"Id": "ag_1", "Status": "Idle", "Archived": False, "ArchivedAt": None}
    monkeypatch.setattr(subprocess, "run", make_run(0, idle))
    assert paseo_spawn._inspect_agent_state("paseo", "ag_1") == (True, "idle")

    running = dict(idle, Status="running")
    monkeypatch.setattr(subprocess, "run", make_run(0, running))
    assert paseo_spawn._inspect_agent_state("paseo", "ag_1") == (True, "running")

    errored = dict(idle, Status="error")
    monkeypatch.setattr(subprocess, "run", make_run(0, errored))
    assert paseo_spawn._inspect_agent_state("paseo", "ag_1") == (False, "error")

    # Daemon down / not found → (False, None), never mistaken for idle.
    monkeypatch.setattr(subprocess, "run", make_run(1, {}))
    assert paseo_spawn._inspect_agent_state("paseo", "ag_1") == (False, None)


def test_read_task_heartbeat_against_real_db(tmp_path, monkeypatch):
    """_read_task_heartbeat reads last_heartbeat_at read-only; NULL/missing → None."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db as kb
    from hermes_cli import paseo_spawn

    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    try:
        tid = kb.create_task(conn, title="hb")
        conn.execute(
            "UPDATE tasks SET status='running', last_heartbeat_at=123456 WHERE id=?",
            (tid,),
        )
        conn.commit()
    finally:
        conn.close()

    assert paseo_spawn._read_task_heartbeat(str(db_path), tid) == 123456
    assert paseo_spawn._read_task_heartbeat(str(db_path), "t_nope") is None
    assert paseo_spawn._read_task_heartbeat(str(tmp_path / "missing.db"), tid) is None

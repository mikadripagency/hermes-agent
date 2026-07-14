"""Launch kanban workers as Paseo agents (config-gated).

The default kanban dispatcher launches each worker as a fire-and-forget
``hermes chat`` subprocess (:func:`hermes_cli.kanban_db._default_spawn`). This
module provides an *alternative* launcher that instead runs the worker as a
Paseo agent (provider ``hermes`` == ChingLing via ``hermes --profile developer
acp``). The point: every dev task then has exactly one linked Paseo chat that
Mika can open from mobile or desktop, while *all* existing worker-protocol
semantics are preserved:

* The env contract (:func:`hermes_cli.kanban_db._build_worker_env`) is passed to
  the agent's provider process via ``paseo run --env``, so the kanban toolset
  auto-appends (``HERMES_KANBAN_TASK``), heartbeats fire, goal-loop mode works,
  and board/DB/branch pins resolve exactly as for a normal worker.
* The dispatcher still receives a *PID* whose lifetime tracks the agent: a
  tracked ``paseo wait <agentId>`` watcher child. When the agent goes idle the
  watcher exits, so the dispatcher's PID-liveness / protocol-violation logic
  behaves identically to the ``hermes chat`` path (watcher exit rc 0 + task
  still running == protocol violation, detected downstream).

Config keys (under ``kanban.paseo_spawn`` in the profile ``config.yaml``):

* ``enabled`` (bool, default ``False``) — gate. When false the dispatcher never
  reaches this module.
* ``provider`` (str, default ``"hermes"``) — Paseo provider to run under.
* ``paseo_bin`` (str, default ``"paseo"``) — path/name of the paseo CLI.
* ``wait_timeout_slack_seconds`` (int, default ``300``) — added to the task's
  ``max_runtime_seconds`` to size the watcher's ``--timeout``.

Robustness contract: a Paseo outage must NEVER stall dispatch. Any failure of
the health check falls back automatically to ``_default_spawn`` (one WARNING).

CLI facts this module relies on (paseo v0.1.107, verified against the bundled
``@getpaseo/cli`` source):

* ``paseo status`` exits 0 when the daemon is reachable.
* ``paseo run -d --json ...`` emits a single JSON object on stdout with keys
  ``agentId``/``status``/``provider``/``cwd``/``title``; the created workspace
  id is logged to *stderr* ("Created workspace <id> ..." / "Using workspace
  <id>") — there is no ``workspaceId`` in the JSON, so it is parsed best-effort.
* ``paseo ls -g --json --label k=v`` filters non-archived agents by label and
  emits a JSON array of ``{id, shortId, name, provider, status, cwd, ...}``.
* ``paseo wait <id> [--timeout <seconds>]`` waits for the agent to become idle.
  IMPORTANT: the flag is ``--timeout`` (seconds), NOT ``--wait-timeout`` (that
  belongs to ``paseo run``). ``paseo wait`` exits **rc 0 for every agent state**
  it resolves — ``idle``, ``timeout`` (still running at deadline), ``permission``
  and ``error`` are all returned as data. It only exits nonzero (rc 1) when it
  cannot talk to the daemon / the wait itself fails. So the watcher exiting is
  the signal the dispatcher watches; a timeout does not surface as a distinct
  exit code and is instead handled by the dispatcher's stale-timeout reclaim.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Optional

_log = logging.getLogger(__name__)

# Agent statuses that mean "this agent is still ours to re-attach to" for the
# double-spawn guard. ``paseo ls`` already excludes archived agents; among the
# live ones, running/idle are the states a kanban worker legitimately sits in
# (idle == between turns / just finished, still attached). Anything else (e.g.
# ``error``) is treated as no usable agent so a fresh one is launched.
_REATTACHABLE_STATUSES = {"running", "idle"}

# stderr line emitted by ``paseo run`` announcing the workspace it used.
_WORKSPACE_RE = re.compile(r"(?:Using|Created) workspace (\S+)")

_STATUS_TIMEOUT_SECONDS = 5
_RUN_TIMEOUT_SECONDS = 60
_LS_TIMEOUT_SECONDS = 10


def _paseo_spawn_config() -> dict:
    """Return the ``kanban.paseo_spawn`` config sub-dict (never raises)."""
    try:
        from hermes_cli.config import load_config

        kanban_cfg = load_config().get("kanban") or {}
    except Exception:  # pragma: no cover - config load is defensive
        return {}
    return (kanban_cfg or {}).get("paseo_spawn") or {}


def paseo_spawn_enabled() -> bool:
    """Whether the dispatcher should use Paseo-agent workers by default."""
    return bool(_paseo_spawn_config().get("enabled", False))


def _truncate_title(title: Optional[str], limit: int = 60) -> str:
    text = (title or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _paseo_healthy(paseo_bin: str) -> bool:
    """Return True iff ``paseo status`` succeeds within a short timeout."""
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [paseo_bin, "status"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_STATUS_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def _find_reattachable_agent(paseo_bin: str, task_id: str) -> Optional[str]:
    """Return the id of an existing non-archived agent for ``task_id``.

    Uses ``paseo ls -g --json --label kanban_task=<id>`` so the guard is global
    (finds the agent regardless of the dispatcher's cwd) and idempotent across
    dispatcher recovery. Returns ``None`` on any error — a failed guard must not
    block the spawn (worst case: a redundant agent, never a stalled task).
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [paseo_bin, "ls", "-g", "--json", "--label", f"kanban_task={task_id}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_LS_TIMEOUT_SECONDS,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("kanban paseo_spawn: `paseo ls` failed for task %s (%s)", task_id, exc)
        return None
    if proc.returncode != 0:
        _log.warning(
            "kanban paseo_spawn: `paseo ls` exited %s for task %s: %s",
            proc.returncode,
            task_id,
            (proc.stderr or "").strip()[:300],
        )
        return None
    try:
        agents = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        _log.warning("kanban paseo_spawn: could not parse `paseo ls` JSON for task %s (%s)", task_id, exc)
        return None
    if not isinstance(agents, list):
        return None
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        if agent.get("status") in _REATTACHABLE_STATUSES and agent.get("id"):
            return str(agent["id"])
    return None


def _worker_env_delta(env: dict) -> dict:
    """Vars ``_build_worker_env`` added/changed vs the ambient environment.

    ``paseo run --env`` *adds* vars to the agent's provider process, which is
    spawned by the paseo daemon (it already has a base environment). Passing the
    full ``os.environ`` would both bloat the argv and leak unrelated dispatcher
    state; the meaningful "worker contract" is exactly the set of vars the
    builder set or overrode, so we diff against the ambient env.
    """
    base = os.environ
    return {k: v for k, v in env.items() if base.get(k) != v}


def _launch_agent(
    paseo_bin: str,
    provider: str,
    workspace: str,
    task,
    env: dict,
    labels: list[tuple[str, str]],
    log_f,
) -> tuple[str, Optional[str]]:
    """Run ``paseo run -d`` and return ``(agentId, workspaceId | None)``.

    Raises on failure (the dispatcher records it as a spawn failure, same as
    ``_default_spawn`` raising).
    """
    prompt = f"work kanban task {task.id}"
    title = f"{task.id} · {_truncate_title(getattr(task, 'title', None))}".strip(" ·")

    cmd = [
        paseo_bin,
        "run",
        "-d",
        "--json",
        "--provider",
        provider,
        "--cwd",
        workspace,
        "--title",
        title,
    ]
    for key, value in labels:
        cmd.extend(["--label", f"{key}={value}"])
    for key, value in _worker_env_delta(env).items():
        cmd.extend(["--env", f"{key}={value}"])
    cmd.append(prompt)

    log_f.write(f"[paseo_spawn] launching agent for task {task.id}\n".encode())
    log_f.flush()

    proc = subprocess.run(  # noqa: S603 - argv is a fixed list built above
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=_RUN_TIMEOUT_SECONDS,
        text=True,
    )
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            f"`paseo run` exited {proc.returncode} for task {task.id}: {stderr[:500]}"
        )
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"`paseo run` returned non-JSON for task {task.id}: {exc}: {(proc.stdout or '')[:300]}"
        )
    agent_id = data.get("agentId")
    if not agent_id:
        raise RuntimeError(f"`paseo run` returned no agentId for task {task.id}: {data!r}")

    workspace_id = None
    match = _WORKSPACE_RE.search(stderr)
    if match:
        workspace_id = match.group(1)

    log_f.write(
        f"[paseo_spawn] agent={agent_id} workspace={workspace_id or '?'}\n".encode()
    )
    log_f.flush()
    return str(agent_id), workspace_id


def _record_linkage_comment(
    task,
    agent_id: str,
    workspace_id: Optional[str],
    author: str,
    board: Optional[str],
) -> None:
    """Append a task comment linking the Paseo agent. Never raises."""
    try:
        from hermes_cli import kanban_db as kb

        machine = f"paseo_agent={agent_id}"
        if workspace_id:
            machine += f" paseo_workspace={workspace_id}"
        human = "Launched as Paseo agent — open this chat from Paseo (mobile/desktop) to watch or steer."
        body = f"{machine}\n{human}"
        with kb.connect_closing(board=board) as conn:
            kb.add_comment(conn, task.id, author or "kanban", body)
    except Exception as exc:  # never fail the spawn on a comment write
        _log.warning(
            "kanban paseo_spawn: could not record linkage comment for task %s (%s)",
            task.id,
            exc,
        )


def _spawn_watcher(
    paseo_bin: str,
    agent_id: str,
    timeout_seconds: Optional[int],
    log_f,
) -> int:
    """Popen a tracked ``paseo wait`` child; return its PID.

    The watcher's lifetime approximates the agent's run: it blocks until the
    agent goes idle (or the optional ``--timeout`` fires). Its stdout/stderr go
    to the shared per-task worker log. ``start_new_session`` detaches it so the
    dispatcher's process-group handling matches ``_default_spawn``.
    """
    cmd = [paseo_bin, "wait", agent_id, "--json"]
    if timeout_seconds and timeout_seconds > 0:
        # `paseo wait` uses `--timeout <seconds>` (NOT `--wait-timeout`).
        cmd.extend(["--timeout", str(int(timeout_seconds))])
    log_f.write(f"[paseo_spawn] watching agent {agent_id}\n".encode())
    log_f.flush()
    proc = subprocess.Popen(  # noqa: S603 - argv is a fixed list built above
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid


def spawn_via_paseo(task, workspace, *, board=None) -> Optional[int]:
    """Launch a kanban worker as a Paseo agent; return the watcher PID.

    Falls back to :func:`hermes_cli.kanban_db._default_spawn` (returning its
    PID) whenever the Paseo daemon is unreachable, so a Paseo outage degrades to
    the normal ``hermes chat`` worker instead of stalling dispatch.
    """
    from hermes_cli import kanban_db as kb

    cfg = _paseo_spawn_config()
    paseo_bin = cfg.get("paseo_bin") or "paseo"
    provider = cfg.get("provider") or "hermes"
    slack = kb._positive_int(cfg.get("wait_timeout_slack_seconds"), 300, minimum=0)

    # Health check — automatic fallback on any daemon trouble.
    if not _paseo_healthy(paseo_bin):
        _log.warning(
            "kanban paseo_spawn: `paseo status` failed; falling back to "
            "_default_spawn for task %s",
            task.id,
        )
        return kb._default_spawn(task, workspace, board=board)

    # Build the shared worker env contract (raises if task has no assignee,
    # exactly like _default_spawn).
    env = kb._build_worker_env(task, workspace, board=board)
    profile_arg = env.get("HERMES_PROFILE") or "kanban"

    # Per-task worker log, shared with the _default_spawn path (same dir +
    # rotation policy) so `hermes kanban log <id>` reads one file.
    log_dir = kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = kb.worker_log_rotation_config()
    kb._rotate_worker_log(log_path, rotate_bytes, backup_count)
    log_f = open(log_path, "ab")
    try:
        # Double-spawn guard: if a live agent for this task already exists, do
        # NOT create a second one — just re-attach a watcher. Makes dispatcher
        # recovery (watcher PID died but agent still alive) idempotent.
        existing = _find_reattachable_agent(paseo_bin, task.id)
        if existing:
            _log.info(
                "kanban paseo_spawn: re-attached to existing agent %s for task %s",
                existing,
                task.id,
            )
            log_f.write(
                f"[paseo_spawn] re-attached to existing agent {existing}\n".encode()
            )
            log_f.flush()
            agent_id = existing
        else:
            labels: list[tuple[str, str]] = [("kanban_task", task.id)]
            if getattr(task, "current_run_id", None) is not None:
                labels.append(("kanban_run", str(task.current_run_id)))
            board_slug = env.get("HERMES_KANBAN_BOARD")
            if board_slug:
                labels.append(("kanban_board", board_slug))
            agent_id, workspace_id = _launch_agent(
                paseo_bin, provider, workspace, task, env, labels, log_f
            )
            _record_linkage_comment(task, agent_id, workspace_id, profile_arg, board)

        # Size the watcher timeout: task max runtime + slack (unbounded when the
        # task has no max runtime — the watcher then simply lives until the
        # agent goes idle).
        max_runtime = getattr(task, "max_runtime_seconds", None)
        timeout_seconds = None
        if max_runtime:
            timeout_seconds = int(max_runtime) + int(slack)

        return _spawn_watcher(paseo_bin, agent_id, timeout_seconds, log_f)
    except Exception:
        log_f.close()
        raise

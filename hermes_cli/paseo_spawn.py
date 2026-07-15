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
* The dispatcher still receives a *PID* whose lifetime tracks the worker: a
  tracked watch-loop child (``python -m hermes_cli.paseo_spawn --watch ...``).
  IMPORTANT: the watcher's exit condition is the *kanban task's DB state*, NOT
  the Paseo agent's idle flag. ACP agents flap to ``idle`` between protocol
  turns (and on permission states), so ``paseo wait`` would exit rc 0 mid-task
  and the dispatcher would misclassify a healthy worker as a protocol
  violation (observed live on t_6811e170). The watch loop instead polls:

  - the kanban DB (read-only): task left this run's ``running`` state
    (done/blocked/review/archived, reclaimed, run superseded, or deleted)
    → **exit 0** — only the task's DB state decides success.
  - ``paseo inspect <agent> --json``: agent archived / deleted / errored /
    daemon unreachable for N *consecutive* polls → **exit 1** (worker
    genuinely gone; the dispatcher's crash recovery is then correct).
    Transient daemon blips reset on the next good poll.
  - a deadline (task ``max_runtime + slack``) → **exit 1** (the dispatcher's
    stale-timeout reclaim would fire anyway; the nonzero exit keeps the
    classification honest).

  While the agent works, ChingLing's env-driven auto-heartbeat keeps the task
  fresh — the watcher never exits 0 merely because the agent object says idle.

Config keys (under ``kanban.paseo_spawn`` in the profile ``config.yaml``):

* ``enabled`` (bool, default ``False``) — gate. When false the dispatcher never
  reaches this module.
* ``provider`` (str, default ``"hermes"``) — Paseo provider to run under.
* ``mode`` (str, default ``"dont_ask"``) — ACP session mode passed as
  ``paseo run --mode``. Workers must never pause on permission prompts; the
  hermes ACP adapter's valid mode ids are ``default`` / ``accept_edits`` /
  ``dont_ask`` (see ``acp_adapter/server.py`` — ``dont_ask`` maps the edit
  approval policy to ``session``, i.e. auto-allow). Set to an empty string /
  null to omit the flag and use the provider default.
* ``paseo_bin`` (str, default ``"paseo"``) — path/name of the paseo CLI.
* ``wait_timeout_slack_seconds`` (int, default ``300``) — added to the task's
  ``max_runtime_seconds`` to size the watch loop's deadline.

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
* Repo-worktree tasks (``workspace_kind == "worktree"``) run inside a
  registered Paseo workspace (``paseo run --workspace <id>``): the daemon
  classifies the git-worktree path as a project-linked worktree workspace
  (kind ``worktree``, projectId = the main repo root), so the agent groups
  under its project in the sidebar instead of minting a stray per-run
  "directory" workspace. An existing non-archived registration for the same
  cwd is reused (bare ``--cwd`` runs never dedupe). Scratch/dir tasks — and
  ANY failure in workspace resolution — fall back to plain ``--cwd``.
* ``paseo inspect <id> --json`` emits a single JSON object with capitalized
  keys (``Id``/``Status``/``Archived``/``ArchivedAt``/…). A missing agent or an
  unreachable daemon exits rc 1 with an ``{"error": …}`` payload on stderr.
* ``paseo wait`` is deliberately NOT used as the watcher: it resolves rc 0 as
  soon as the agent reports ``idle`` — which ACP providers flap to between
  protocol turns — and it also exits rc 0 for ``timeout`` / ``permission`` /
  ``error`` states (only daemon/connection failures are nonzero).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Optional

_log = logging.getLogger(__name__)

# Agent statuses that mean "this agent is ACTIVELY working" for the re-attach
# guard. A Paseo agent maps to a RUN, not to the task's lifetime: an idle
# labeled agent is a stale remnant of a finished run — its env carries the OLD
# run id / claim lock, so the kanban tools would reject anything it did, and no
# heartbeats flow (observed live: t_6811e170 runs 324/325 re-attached to an
# idle agent and timed out). Only an agent still mid-run is worth re-attaching
# to (the real recovery case: watcher died, agent still working). The CLI's
# agent-status vocabulary is running/idle/error (see ls.js statusOrder);
# busy/working are accepted defensively in case the daemon grows new active
# states. Everything else (idle, error, unknown) is stale → archive + fresh.
_ACTIVE_STATUSES = {"running", "busy", "working"}

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


def _list_task_agents(paseo_bin: str, task_id: str) -> list:
    """Return non-archived agents labeled ``kanban_task=<task_id>``.

    Uses ``paseo ls -g --json --label kanban_task=<id>`` so the guard is global
    (finds the agent regardless of the dispatcher's cwd) and idempotent across
    dispatcher recovery. Returns ``[]`` on any error — a failed guard must not
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
        return []
    if proc.returncode != 0:
        _log.warning(
            "kanban paseo_spawn: `paseo ls` exited %s for task %s: %s",
            proc.returncode,
            task_id,
            (proc.stderr or "").strip()[:300],
        )
        return []
    try:
        agents = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        _log.warning("kanban paseo_spawn: could not parse `paseo ls` JSON for task %s (%s)", task_id, exc)
        return []
    if not isinstance(agents, list):
        return []
    return [a for a in agents if isinstance(a, dict) and a.get("id")]


def _archive_agent(paseo_bin: str, agent_id: str, task_id: str) -> None:
    """Best-effort ``paseo archive <id>`` of a stale agent. Never raises.

    Only called for agents that are NOT actively working, so no ``--force``
    (which would interrupt a live run). Failure is logged and the spawn
    proceeds — a stray stale agent is cosmetic; a stalled task is not.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [paseo_bin, "archive", agent_id, "--json"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_LS_TIMEOUT_SECONDS,
            text=True,
        )
        if proc.returncode != 0:
            _log.warning(
                "kanban paseo_spawn: could not archive stale agent %s for task %s: %s",
                agent_id,
                task_id,
                (proc.stderr or "").strip()[:300],
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        _log.warning(
            "kanban paseo_spawn: could not archive stale agent %s for task %s (%s)",
            agent_id,
            task_id,
            exc,
        )


# ---------------------------------------------------------------------------
# Project-linked workspace resolution (repo-worktree tasks)
# ---------------------------------------------------------------------------

_WORKSPACE_CREATE_TIMEOUT_SECONDS = 30

# node -e script that asks the daemon to create a workspace backed by an
# existing directory. The daemon classifies the path itself: a git worktree
# registers as kind "worktree" with projectId = the main repo root (see the
# daemon's createLocalCheckoutWorkspace / classifyDirectoryForProjectMembership)
# — i.e. exactly the project-linked sidebar grouping we want. argv[1] = the
# client.js file URI is imported dynamically; argv[2] = the directory path.
_CREATE_WORKSPACE_SCRIPT = (
    "const {connectToDaemon} = await import(process.argv[1]);"
    "const c = await connectToDaemon({});"
    "try {"
    "  const r = await c.createWorkspace({source:{kind:'directory', path: process.argv[2]}});"
    "  if (!r.workspace) throw new Error(r.error ?? 'workspace create failed');"
    "  console.log(JSON.stringify({workspaceId: r.workspace.workspaceId ?? r.workspace.id ?? null}));"
    "} finally { await c.close().catch(() => {}); }"
)


def _paseo_home() -> "os.PathLike":
    from pathlib import Path

    return Path(os.environ.get("PASEO_HOME") or (Path.home() / ".paseo"))


def _paseo_client_js(paseo_bin: str):
    """Locate the paseo CLI's daemon-client module next to the resolved bin.

    The installed layout is ``<prefix>/lib/node_modules/@getpaseo/cli/bin/paseo``
    with the client at ``../dist/utils/client.js``. Returns None when it can't
    be found (→ caller falls back to plain ``--cwd``).
    """
    import shutil
    from pathlib import Path

    resolved = shutil.which(paseo_bin) or paseo_bin
    try:
        real = Path(resolved).resolve()
    except OSError:
        return None
    candidate = real.parent.parent / "dist" / "utils" / "client.js"
    return candidate if candidate.is_file() else None


def _find_registered_workspace(workspace_path: str) -> Optional[str]:
    """Return the id of a non-archived Paseo workspace whose cwd is the path.

    Reads the daemon's workspace registry file directly (same convention as
    the manual launcher) — cheap, read-only, and race-tolerant: a miss just
    means we create one. Returns None on any error.
    """
    import os.path

    registry = _paseo_home() / "projects" / "workspaces.json"
    try:
        with open(registry, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(entries, list):
        return None
    target = os.path.realpath(workspace_path)
    for item in entries:
        if not isinstance(item, dict) or item.get("archivedAt") is not None:
            continue
        cwd = item.get("cwd")
        if cwd and os.path.realpath(str(cwd)) == target and item.get("workspaceId"):
            return str(item["workspaceId"])
    return None


def _create_directory_workspace(paseo_bin: str, workspace_path: str) -> Optional[str]:
    """Create a daemon workspace backed by ``workspace_path``; return its id.

    Returns None on any failure (missing node, missing client.js, daemon
    error, unparsable output) — the caller falls back to plain ``--cwd``.
    """
    import shutil

    client_js = _paseo_client_js(paseo_bin)
    node = shutil.which("node")
    if client_js is None or node is None:
        _log.warning(
            "kanban paseo_spawn: cannot create project workspace "
            "(client.js=%s node=%s) — falling back to --cwd",
            client_js,
            node,
        )
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [
                node,
                "--input-type=module",
                "-e",
                _CREATE_WORKSPACE_SCRIPT,
                client_js.as_uri(),
                workspace_path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_WORKSPACE_CREATE_TIMEOUT_SECONDS,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        _log.warning("kanban paseo_spawn: workspace create failed for %s (%s)", workspace_path, exc)
        return None
    if proc.returncode != 0:
        _log.warning(
            "kanban paseo_spawn: workspace create exited %s for %s: %s",
            proc.returncode,
            workspace_path,
            (proc.stderr or "").strip()[:300],
        )
        return None
    try:
        data = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        _log.warning(
            "kanban paseo_spawn: unparsable workspace create output for %s (%s)",
            workspace_path,
            exc,
        )
        return None
    ws_id = data.get("workspaceId") if isinstance(data, dict) else None
    return str(ws_id) if ws_id else None


def _resolve_task_workspace_id(paseo_bin: str, task, workspace: str, log_f) -> Optional[str]:
    """Resolve the Paseo workspace to run a repo-worktree task in.

    Only ``workspace_kind == "worktree"`` tasks get a registered workspace —
    the daemon classifies their git-worktree path as a project-linked
    worktree workspace, so the agent groups under the repo project in the
    sidebar. Reusing an existing registration (by cwd) avoids the
    one-new-workspace-per-run duplication of bare ``--cwd`` runs. Scratch /
    dir tasks return None (plain ``--cwd``), as does ANY failure here — a
    workspace-registration problem must never block the spawn.
    """
    try:
        if getattr(task, "workspace_kind", None) != "worktree":
            return None
        existing = _find_registered_workspace(workspace)
        if existing:
            log_f.write(
                f"[paseo_spawn] using registered workspace {existing}\n".encode()
            )
            log_f.flush()
            return existing
        created = _create_directory_workspace(paseo_bin, workspace)
        if created:
            log_f.write(
                f"[paseo_spawn] registered project workspace {created}\n".encode()
            )
            log_f.flush()
        return created
    except Exception as exc:  # absolute fallback: never block the spawn
        _log.warning(
            "kanban paseo_spawn: workspace resolution failed for task %s (%s) — using --cwd",
            getattr(task, "id", "?"),
            exc,
        )
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
    mode: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Run ``paseo run -d`` and return ``(agentId, workspaceId | None)``.

    ``workspace_id``, when set, runs the agent inside that existing Paseo
    workspace (``--workspace``) so repo-worktree tasks group under their
    project in the sidebar; ``--cwd`` is always passed as well (with an
    explicit workspace, ``paseo run`` uses the ``--cwd`` value as the run
    cwd). Raises on failure (the dispatcher records it as a spawn failure,
    same as ``_default_spawn`` raising).
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
    if workspace_id:
        cmd.extend(["--workspace", workspace_id])
    if mode:
        # ACP session mode: `dont_ask` keeps dispatcher-spawned workers from
        # ever pausing on a permission prompt (matches manually-created
        # Paseo agents, which run with modeId "dont_ask").
        cmd.extend(["--mode", mode])
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

    resolved_workspace_id = workspace_id  # explicit --workspace wins
    if not resolved_workspace_id:
        match = _WORKSPACE_RE.search(stderr)
        if match:
            resolved_workspace_id = match.group(1)

    log_f.write(
        f"[paseo_spawn] agent={agent_id} workspace={resolved_workspace_id or '?'}\n".encode()
    )
    log_f.flush()
    return str(agent_id), resolved_workspace_id


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


# ---------------------------------------------------------------------------
# Watch loop (the tracked worker PID)
# ---------------------------------------------------------------------------

# Poll cadence and agent-gone threshold. 6 consecutive bad inspects at a 10s
# cadence ≈ 60s of sustained "agent gone / daemon down" before declaring the
# worker dead — long enough to ride out a paseo daemon restart.
WATCH_POLL_SECONDS = 10.0
WATCH_GONE_THRESHOLD = 6

# Exit codes of the watch loop (the PID the dispatcher tracks):
#   0 — the kanban task left this run's `running` state (terminal transition,
#       reclaim, run superseded, or task deleted). The dispatcher only flags a
#       protocol violation for tasks still `running`, so rc 0 here is always
#       clean.
#   1 — the agent is genuinely gone (archived/deleted/errored/unreachable for
#       WATCH_GONE_THRESHOLD consecutive polls) or the deadline passed while
#       the task is still running. Dispatcher crash recovery is then correct.
WATCH_EXIT_TASK_SETTLED = 0
WATCH_EXIT_WORKER_GONE = 1


def _read_task_run_state(db_path: str, task_id: str):
    """Read ``(status, current_run_id)`` for ``task_id`` — read-only, no writes.

    Opens its own short-lived read-only connection (URI ``mode=ro``) with a
    busy timeout so the watcher can never write to, lock, or initialize the
    kanban DB. Returns:

    * ``(status, current_run_id)`` when the row exists,
    * ``("__missing__", None)`` when the task row is gone (deleted), and
    * ``None`` on transient errors (locked/busy/unreadable) — the caller keeps
      looping; a permanently unreadable DB is bounded by the deadline.
    """
    import sqlite3
    from urllib.parse import quote

    try:
        uri = f"file:{quote(str(db_path))}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            conn.execute("PRAGMA busy_timeout=5000")
            row = conn.execute(
                "SELECT status, current_run_id FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if row is None:
        return ("__missing__", None)
    return (row[0], row[1])


def _inspect_agent_ok(paseo_bin: str, agent_id: str) -> bool:
    """One ``paseo inspect`` poll: True iff the agent is live and usable.

    "Usable" = inspect succeeds, agent is not archived, and its status is not
    ``error``. Any failure (nonzero rc, timeout, unparsable JSON, daemon down)
    returns False; the caller applies the consecutive-poll threshold so
    transient daemon blips don't kill the watcher.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv
            [paseo_bin, "inspect", agent_id, "--json"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_LS_TIMEOUT_SECONDS,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    # inspect --json uses capitalized keys (Status/Archived/ArchivedAt); accept
    # lowercase variants defensively in case the CLI schema changes.
    archived = data.get("Archived", data.get("archived"))
    if archived is None:
        archived = (data.get("ArchivedAt") or data.get("archivedAt")) is not None
    if archived:
        return False
    status = data.get("Status", data.get("status"))
    if status == "error":
        return False
    return True


def watch_worker(
    *,
    task_id: str,
    agent_id: str,
    db_path: str,
    run_id: Optional[int] = None,
    deadline: Optional[float] = None,
    paseo_bin: str = "paseo",
    poll_seconds: float = WATCH_POLL_SECONDS,
    gone_threshold: int = WATCH_GONE_THRESHOLD,
) -> int:
    """The watch loop body; returns the process exit code.

    The kanban task's DB state is the ONLY success signal: the loop exits 0
    exactly when the task is no longer ``running`` under ``run_id`` (terminal
    transition, reclaim, run superseded, or deleted). The Paseo agent flapping
    to ``idle`` between ACP turns never terminates the watcher. Agent-gone
    (``gone_threshold`` consecutive bad inspects) and deadline overrun exit 1.
    """
    import time

    consecutive_bad = 0
    while True:
        # (a) Task DB state — the authoritative success signal.
        state = _read_task_run_state(db_path, task_id)
        if state is not None:
            status, current_run = state
            if status != "running":
                _log.info(
                    "paseo watch: task %s left running (status=%s) — done",
                    task_id,
                    status,
                )
                return WATCH_EXIT_TASK_SETTLED
            if run_id is not None and current_run is not None and int(current_run) != int(run_id):
                _log.info(
                    "paseo watch: task %s run superseded (%s -> %s) — done",
                    task_id,
                    run_id,
                    current_run,
                )
                return WATCH_EXIT_TASK_SETTLED

        # (b) Agent liveness — only consecutive sustained failure counts.
        if _inspect_agent_ok(paseo_bin, agent_id):
            consecutive_bad = 0
        else:
            consecutive_bad += 1
            if consecutive_bad >= gone_threshold:
                _log.warning(
                    "paseo watch: agent %s gone/unreachable for %d consecutive "
                    "polls — declaring worker dead (task %s)",
                    agent_id,
                    consecutive_bad,
                    task_id,
                )
                return WATCH_EXIT_WORKER_GONE

        # (c) Deadline (task max_runtime + slack).
        if deadline is not None and time.time() >= deadline:
            _log.warning(
                "paseo watch: deadline passed for task %s (agent %s) — exiting",
                task_id,
                agent_id,
            )
            return WATCH_EXIT_WORKER_GONE

        time.sleep(poll_seconds)


def _spawn_watcher(
    task_id: str,
    agent_id: str,
    db_path: str,
    run_id: Optional[int],
    deadline: Optional[float],
    paseo_bin: str,
    log_f,
) -> int:
    """Popen the tracked watch-loop child; return its PID.

    Runs ``<sys.executable> -m hermes_cli.paseo_spawn --watch ...`` — the same
    interpreter as the dispatching gateway, so ``hermes_cli`` resolves to the
    deployed tree. Its stdout/stderr go to the shared per-task worker log.
    ``start_new_session`` detaches it so the dispatcher's process-group
    handling matches ``_default_spawn``.
    """
    import sys

    cmd = [
        sys.executable,
        "-m",
        "hermes_cli.paseo_spawn",
        "--watch",
        "--task",
        task_id,
        "--agent",
        agent_id,
        "--db",
        db_path,
        "--paseo-bin",
        paseo_bin,
    ]
    if run_id is not None:
        cmd.extend(["--run", str(int(run_id))])
    if deadline is not None:
        cmd.extend(["--deadline", str(int(deadline))])
    log_f.write(
        f"[paseo_spawn] watching agent {agent_id} (task-state watch loop)\n".encode()
    )
    log_f.flush()
    proc = subprocess.Popen(  # noqa: S603 - argv is a fixed list built above
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc.pid


def main(argv: Optional[list] = None) -> int:
    """CLI entrypoint: ``python -m hermes_cli.paseo_spawn --watch ...``."""
    import argparse

    parser = argparse.ArgumentParser(prog="hermes_cli.paseo_spawn")
    parser.add_argument("--watch", action="store_true", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--deadline", type=float, default=None)
    parser.add_argument("--paseo-bin", default="paseo")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [paseo_watch] %(levelname)s %(message)s",
    )
    # Test hook: override the poll cadence (seconds) without changing the CLI
    # contract the dispatcher spawns with.
    try:
        poll_seconds = float(os.environ.get("HERMES_PASEO_WATCH_POLL", WATCH_POLL_SECONDS))
    except ValueError:
        poll_seconds = WATCH_POLL_SECONDS
    return watch_worker(
        task_id=args.task,
        agent_id=args.agent,
        db_path=args.db,
        run_id=args.run,
        deadline=args.deadline,
        paseo_bin=args.paseo_bin,
        poll_seconds=poll_seconds,
    )


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
    # ACP session mode (default dont_ask so workers never pause on permission
    # prompts). An explicit empty string / null omits --mode entirely.
    mode = cfg.get("mode", "dont_ask")
    mode = str(mode).strip() if mode is not None else ""
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
        # Double-spawn guard, run-scoped: a Paseo agent maps to a RUN, not to
        # the task's lifetime.
        #
        # * An agent that is still ACTIVELY working (status running/busy/
        #   working) is the real recovery case — the watcher PID died but the
        #   agent is mid-run with a still-valid env. Re-attach a watcher only;
        #   never create a duplicate.
        # * Idle / errored labeled agents are STALE remnants of finished runs:
        #   their env carries the old HERMES_KANBAN_RUN_ID / CLAIM_LOCK, so
        #   the kanban tools would reject them and no heartbeats would flow.
        #   Archive them (best-effort) and create a FRESH agent with the new
        #   run's env — keeping at most one non-archived agent per task.
        agents = _list_task_agents(paseo_bin, task.id)
        active = next(
            (a for a in agents if a.get("status") in _ACTIVE_STATUSES),
            None,
        )
        if active is not None:
            agent_id = str(active["id"])
            _log.info(
                "kanban paseo_spawn: re-attached to actively working agent %s "
                "for task %s",
                agent_id,
                task.id,
            )
            log_f.write(
                f"[paseo_spawn] re-attached to existing agent {agent_id}\n".encode()
            )
            log_f.flush()
        else:
            for stale in agents:
                stale_id = str(stale["id"])
                _log.info(
                    "kanban paseo_spawn: archiving stale agent %s "
                    "(status=%s) for task %s before fresh spawn",
                    stale_id,
                    stale.get("status"),
                    task.id,
                )
                log_f.write(
                    f"[paseo_spawn] archiving stale agent {stale_id} "
                    f"(status={stale.get('status')})\n".encode()
                )
                log_f.flush()
                _archive_agent(paseo_bin, stale_id, task.id)
            labels: list[tuple[str, str]] = [("kanban_task", task.id)]
            if getattr(task, "current_run_id", None) is not None:
                labels.append(("kanban_run", str(task.current_run_id)))
            board_slug = env.get("HERMES_KANBAN_BOARD")
            if board_slug:
                labels.append(("kanban_board", board_slug))
            # Repo-worktree tasks run inside a registered, project-linked
            # Paseo workspace so they group under the project in the sidebar;
            # scratch/dir tasks (and any resolution failure) keep plain --cwd.
            run_workspace_id = _resolve_task_workspace_id(
                paseo_bin, task, workspace, log_f
            )
            agent_id, workspace_id = _launch_agent(
                paseo_bin, provider, workspace, task, env, labels, log_f,
                mode=mode or None,
                workspace_id=run_workspace_id,
            )
            _record_linkage_comment(task, agent_id, workspace_id, profile_arg, board)

        # Watch-loop deadline: task max runtime + slack (unbounded when the
        # task has no max runtime — the loop then lives until the task's DB
        # state settles or the agent is genuinely gone).
        import time

        max_runtime = getattr(task, "max_runtime_seconds", None)
        deadline = None
        if max_runtime:
            deadline = time.time() + int(max_runtime) + int(slack)

        run_id = getattr(task, "current_run_id", None)
        return _spawn_watcher(
            task.id,
            agent_id,
            env["HERMES_KANBAN_DB"],
            run_id,
            deadline,
            paseo_bin,
            log_f,
        )
    except Exception:
        log_f.close()
        raise


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    import sys

    sys.exit(main())

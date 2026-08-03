#!/usr/bin/env python3
"""Read-only Kanban health sentinel used by the developer cron watcher.

The Kanban DB is authoritative for task/run/watcher ownership. Paseo is an
optional execution surface: absence from Paseo is not an error when the current
run has a live direct worker. The script writes only its own dedupe state file.
"""
from __future__ import annotations

import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

DB_PATH = Path(os.environ.get("HERMES_KANBAN_DB", str(Path.home() / ".hermes" / "kanban.db")))
HEARTBEAT_STALE_S = int(os.environ.get("SENTINEL_HEARTBEAT_STALE_S", "600"))
STARTUP_GRACE_S = int(os.environ.get("SENTINEL_STARTUP_GRACE_S", "120"))
REATTACH_GRACE_S = int(os.environ.get("SENTINEL_REATTACH_GRACE_S", "120"))
APPROACHING_PCT = float(os.environ.get("SENTINEL_APPROACHING_PCT", "0.65"))
ACTION_COOLDOWN_S = int(os.environ.get("SENTINEL_ACTION_COOLDOWN_S", "3600"))
ESCALATE_AFTER = int(os.environ.get("SENTINEL_ESCALATE_AFTER", "2"))
BLOCKED_AWAITING_THRESH_S = int(os.environ.get("SENTINEL_BLOCKED_AWAITING_S", "2700"))
BLOCKED_DEDUPE_S = int(os.environ.get("SENTINEL_BLOCKED_DEDUPE_S", "86400"))
BUSY_TIMEOUT_MS = int(os.environ.get("HERMES_KANBAN_BUSY_TIMEOUT_MS", "5000"))
PASEO_BIN = os.environ.get("PASEO_BIN", "paseo")
_PROFILE_HOME = Path(__file__).resolve().parent.parent


def _default_max_runtime_s() -> int:
    env = os.environ.get("SENTINEL_DEFAULT_MAX_RUNTIME_S")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        text = (_PROFILE_HOME / "config.yaml").read_text(encoding="utf-8")
        match = re.search(
            r"^\s+default_max_runtime_seconds:\s*['\"]?(\d+)['\"]?\s*(?:#.*)?$",
            text,
            re.MULTILINE,
        )
        if match and int(match.group(1)) > 0:
            return int(match.group(1))
    except OSError:
        pass
    return 7200


DEFAULT_MAX_RUNTIME_S = _default_max_runtime_s()
STATE_PATH = Path(
    os.environ.get(
        "SENTINEL_STATE_PATH",
        str(_PROFILE_HOME / "cron" / "kanban_sentinel_state.json"),
    )
)
ACTIONABLE = {
    "stalled_no_heartbeat",
    "watcher_dead_agent_alive",
    "agent_dead_task_running",
    "approaching_timeout",
    "blocked_awaiting_human",
}
_SUGGESTED = {
    "stalled_no_heartbeat": "paseo_send_status_nudge",
    "approaching_timeout": "paseo_send_wrapup_nudge",
    "watcher_dead_agent_alive": "comment_watcher_reattach",
    "agent_dead_task_running": "comment_reclaim_evidence",
    "blocked_awaiting_human": "slack_ping_orchestration_unblock",
}


def _cooldown_for(verdict: str) -> int:
    return BLOCKED_DEDUPE_S if verdict == "blocked_awaiting_human" else ACTION_COOLDOWN_S


def _connect_ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def _run_paseo_ls(args: list[str]):
    try:
        proc = subprocess.run(
            [PASEO_BIN, "ls", "-g", "--json", *args],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    agents = data if isinstance(data, list) else data.get("agents", [])
    return agents if isinstance(agents, list) else None


def _load_paseo_agents():
    """Availability probe. None means unknown; an empty list is authoritative."""
    return _run_paseo_ls([])


def _load_task_agents(task_id: str, run_id, profile: str):
    """Query the exact task/run/profile labels and mark that match as authoritative."""
    if run_id is None:
        return []
    agents = _run_paseo_ls(
        [
            "--label",
            f"kanban_task={task_id}",
            "--label",
            f"kanban_run={int(run_id)}",
            "--label",
            f"kanban_profile={profile}",
        ]
    )
    if agents is None:
        return None
    out = []
    for item in agents:
        if isinstance(item, dict):
            copy = dict(item)
            copy["_sentinel_exact_labels"] = True
            out.append(copy)
    return out


def _has_label(agent: dict, key: str, value: str) -> bool:
    labels = agent.get("labels")
    if isinstance(labels, dict):
        return str(labels.get(key)) == value
    if isinstance(labels, list):
        for label in labels:
            if label == f"{key}={value}":
                return True
            if isinstance(label, dict) and str(label.get("key")) == key:
                return str(label.get("value")) == value
    return False


def _agent_for_task(agents, task_id: str, run_id, profile: str):
    if agents is None:
        return None
    candidates = []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        exact = agent.get("_sentinel_exact_labels") or (
            _has_label(agent, "kanban_task", task_id)
            and _has_label(agent, "kanban_run", str(run_id))
            and _has_label(agent, "kanban_profile", profile)
        )
        if exact:
            candidates.append(agent)
    candidates.sort(key=lambda item: 0 if item.get("status") in {"running", "busy", "working"} else 1)
    return candidates[0] if candidates else None


def _payload(row):
    if row is None or not row["payload"]:
        return {}
    try:
        value = json.loads(row["payload"])
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _current_linked_agent(conn, task_id: str, started_at: int | None):
    if started_at is None:
        return None
    rows = conn.execute(
        "SELECT body FROM task_comments WHERE task_id=? AND created_at>=? "
        "AND body LIKE '%paseo_agent=%' ORDER BY id DESC LIMIT 5",
        (task_id, int(started_at)),
    ).fetchall()
    for row in rows:
        match = re.search(r"(?:^|\s)paseo_agent=([^\s]+)", row["body"] or "")
        if match:
            return match.group(1)
    return None


def _classify(conn, task, now, agents):
    """Classify one running task from a consistent DB snapshot plus exact Paseo labels."""
    tid = task["id"]
    current_run_id = task["current_run_id"]
    run = None
    if current_run_id is not None:
        run = conn.execute(
            "SELECT * FROM task_runs WHERE id=? AND task_id=? AND status='running'",
            (int(current_run_id), tid),
        ).fetchone()
    run_valid = run is not None
    started = (run["started_at"] if run else None) or task["started_at"]
    run_age = max(0, now - int(started)) if started else 0
    max_rt = (
        (run["max_runtime_seconds"] if run else None)
        or task["max_runtime_seconds"]
        or DEFAULT_MAX_RUNTIME_S
    )
    elapsed_pct = round(run_age / max_rt, 3) if max_rt else 0.0

    event_hb = conn.execute(
        "SELECT MAX(created_at) AS value FROM task_events "
        "WHERE task_id=? AND kind='heartbeat' AND (run_id=? OR ? IS NULL)",
        (tid, current_run_id, current_run_id),
    ).fetchone()["value"]
    heartbeat_values = [task["last_heartbeat_at"], event_hb]
    if run is not None:
        heartbeat_values.append(run["last_heartbeat_at"])
    hb = max((int(value) for value in heartbeat_values if value), default=None)
    hb_age = now - hb if hb is not None else None
    hb_stale = hb_age is not None and hb_age > HEARTBEAT_STALE_S

    spawned = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='spawned' "
        "AND run_id=? ORDER BY id DESC LIMIT 1",
        (tid, current_run_id),
    ).fetchone() if current_run_id is not None else None
    spawned_pid = _payload(spawned).get("pid")
    pid_values = [task["worker_pid"], spawned_pid]
    if run is not None:
        pid_values.append(run["worker_pid"])
    known_pids = {int(value) for value in pid_values if value}
    worker_pid = int(
        spawned_pid
        or (run["worker_pid"] if run else 0)
        or task["worker_pid"]
        or 0
    ) or None
    mapping_consistent = len(known_pids) <= 1
    watcher_alive = bool(mapping_consistent and _pid_alive(worker_pid))

    profile = str((run["profile"] if run else None) or task["assignee"] or "")
    agent = _agent_for_task(agents, tid, current_run_id, profile)
    agent_status = str(agent.get("status") or "").lower() if agent else None
    agent_live = bool(agent and agent_status not in {"archived", "closed", "error"})
    linked_agent_id = _current_linked_agent(conn, tid, started)
    worker_mode = "paseo" if agent or linked_agent_id else ("direct" if agents is not None else "unknown")

    claim_values = [task["claim_expires"]]
    if run is not None:
        claim_values.append(run["claim_expires"])
    claim_expires = max((int(value) for value in claim_values if value), default=None)
    claim_expired = claim_expires is not None and claim_expires < now
    past_startup_grace = run_age > STARTUP_GRACE_S
    past_reattach_grace = run_age > REATTACH_GRACE_S

    verdict = "healthy"
    orphan_reason = None
    if not run_valid and past_startup_grace:
        verdict = "agent_dead_task_running"
        orphan_reason = "current_run_missing_or_not_running"
    elif not mapping_consistent and past_startup_grace and (claim_expired or hb_stale):
        verdict = "agent_dead_task_running"
        orphan_reason = "worker_mapping_inconsistent"
    elif not watcher_alive and agent_live and past_reattach_grace:
        verdict = "watcher_dead_agent_alive"
        orphan_reason = "watcher_gone_agent_exact_run_alive"
    elif not watcher_alive and not agent_live and past_startup_grace and (
        claim_expired or hb_stale
    ):
        verdict = "agent_dead_task_running"
        orphan_reason = "linked_agent_unreachable" if linked_agent_id else (
            "worker_gone_claim_expired"
            if claim_expired
            else "worker_gone_heartbeat_stale"
        )
    elif hb_stale and agent_live and agent_status == "idle":
        verdict = "stalled_no_heartbeat"
    elif elapsed_pct >= APPROACHING_PCT:
        verdict = "approaching_timeout"

    return {
        "task_id": tid,
        "assignee": task["assignee"],
        "verdict": verdict,
        "orphan_reason": orphan_reason,
        "current_run_id": current_run_id,
        "run_mapping_valid": run_valid,
        "heartbeat_age_s": hb_age,
        "elapsed_s": run_age,
        "max_runtime_s": max_rt,
        "elapsed_pct": elapsed_pct,
        "claim_expires": claim_expires,
        "claim_expired": claim_expired,
        "watcher_pid": worker_pid,
        "watcher_mapping_consistent": mapping_consistent,
        "watcher_alive": watcher_alive,
        "worker_mode": worker_mode,
        "linked_agent_id": linked_agent_id,
        "agent_short_id": (agent.get("shortId") or agent.get("id")) if agent else None,
        "agent_status": agent_status,
        "agent_name_hint": agent.get("name") if agent else None,
        "paseo_listing_available": agents is not None,
    }


def _human_activity_since(conn, tid, since_ts, block_event_id, assignee):
    try:
        comments = conn.execute(
            "SELECT COUNT(*) AS n FROM task_comments WHERE task_id=? AND created_at>? "
            "AND author IS NOT ? AND TRIM(COALESCE(author,'')) <> ''",
            (tid, since_ts, assignee),
        ).fetchone()["n"]
        if comments and int(comments) > 0:
            return True
        events = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=? AND id>? "
            "AND kind IN ('unblocked','promoted_manual')",
            (tid, block_event_id),
        ).fetchone()["n"]
        return bool(events and int(events) > 0)
    except sqlite3.Error:
        return True


def _classify_blocked(conn, task, now):
    tid = task["id"]
    blocked = conn.execute(
        "SELECT id, created_at, payload FROM task_events WHERE task_id=? "
        "AND kind='blocked' ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    if blocked is None:
        blocked = conn.execute(
            "SELECT id, created_at, payload FROM task_events WHERE task_id=? "
            "AND kind='created' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
    base = {
        "task_id": tid,
        "assignee": task["assignee"],
        "verdict": "healthy",
        "status": task["status"],
    }
    if blocked is None or not blocked["created_at"]:
        return base
    block_at = int(blocked["created_at"])
    block_age = now - block_at
    if block_age <= BLOCKED_AWAITING_THRESH_S:
        return base
    if _human_activity_since(conn, tid, block_at, blocked["id"], task["assignee"]):
        return base
    payload = _payload(blocked)
    base.update({
        "verdict": "blocked_awaiting_human",
        "block_event_id": blocked["id"],
        "block_kind": payload.get("kind"),
        "block_reason": str(payload.get("reason"))[:200] if payload.get("reason") else None,
        "block_age_s": block_age,
        "block_age_min": round(block_age / 60, 1),
    })
    return base


def _load_state():
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(state):
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, STATE_PATH)
    except OSError:
        pass


def main() -> int:
    now = int(time.time())
    if not DB_PATH.exists():
        return 0
    try:
        conn = _connect_ro(DB_PATH)
    except sqlite3.Error:
        return 0
    try:
        running = conn.execute(
            "SELECT id, assignee, status, started_at, last_heartbeat_at, "
            "current_run_id, worker_pid, max_runtime_seconds, claim_expires "
            "FROM tasks WHERE status='running'"
        ).fetchall()
        blocked = conn.execute(
            "SELECT id, assignee, status FROM tasks WHERE status='blocked'"
        ).fetchall()
        paseo_probe = _load_paseo_agents()
        verdicts = []
        for task in running:
            task_agents = None
            if paseo_probe is not None:
                run_profile = conn.execute(
                    "SELECT profile FROM task_runs WHERE id=? AND task_id=? "
                    "AND status='running'",
                    (task["current_run_id"], task["id"]),
                ).fetchone()
                profile = str(
                    (run_profile["profile"] if run_profile else None)
                    or task["assignee"]
                    or ""
                )
                task_agents = _load_task_agents(
                    task["id"], task["current_run_id"], profile
                )
            verdicts.append(_classify(conn, task, now, task_agents))
        verdicts.extend(_classify_blocked(conn, task, now) for task in blocked)
    except sqlite3.Error:
        conn.close()
        return 0
    conn.close()

    state = _load_state()
    live_ids = {item["task_id"] for item in verdicts}
    for task_id in list(state):
        if task_id not in live_ids:
            state.pop(task_id, None)

    anomalies = []
    for item in verdicts:
        if item["verdict"] == "healthy":
            state.pop(item["task_id"], None)
            continue
        previous = state.get(item["task_id"], {})
        last_at = int(previous.get("last_action_at", 0))
        count = int(previous.get("action_count", 0))
        cooldown = _cooldown_for(item["verdict"])
        allowed = item["verdict"] in ACTIONABLE and (now - last_at) >= cooldown
        item.update({
            "action_allowed": allowed,
            "action_count": count,
            "escalate": count >= ESCALATE_AFTER,
            "minutes_since_last_action": round((now - last_at) / 60, 1) if last_at else None,
            "suggested_action": _SUGGESTED.get(item["verdict"], "inspect"),
        })
        state[item["task_id"]] = {
            "last_action_at": now if allowed else last_at,
            "action_count": count + 1 if allowed else count,
            "last_verdict": item["verdict"],
        }
        anomalies.append(item)
    _save_state(state)
    if not anomalies:
        return 0
    print(json.dumps({
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "now_epoch": now,
        "host": socket.gethostname(),
        "heartbeat_stale_threshold_s": HEARTBEAT_STALE_S,
        "startup_grace_s": STARTUP_GRACE_S,
        "reattach_grace_s": REATTACH_GRACE_S,
        "action_cooldown_s": ACTION_COOLDOWN_S,
        "blocked_awaiting_threshold_s": BLOCKED_AWAITING_THRESH_S,
        "blocked_dedupe_s": BLOCKED_DEDUPE_S,
        "anomalies": anomalies,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

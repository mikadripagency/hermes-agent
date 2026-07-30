"""Process-local tool fence for dispatcher-spawned kanban workers."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional
from urllib.parse import quote

_TRANSITIONS = {"kanban_block", "kanban_complete"}
# ponytail: this gate is process-local; remote workers are fenced on their next
# DB check, not preempted mid-tool. Replace it with a DB lease if remote reclaim
# must cancel side effects that were already running when the claim advanced.
_GATE_CONDITION = threading.Condition()
_ACTIVE_TOOLS = 0
_TRANSITION_ACTIVE = False
_TRANSITIONS_WAITING = 0
_GATE_LOCAL = threading.local()


@contextmanager
def _tool_gate(*, transition: bool):
    """Allow parallel work but make lifecycle transitions exclusive."""
    global _ACTIVE_TOOLS, _TRANSITION_ACTIVE, _TRANSITIONS_WAITING
    with _GATE_CONDITION:
        if transition:
            _TRANSITIONS_WAITING += 1
            try:
                while _TRANSITION_ACTIVE or _ACTIVE_TOOLS:
                    _GATE_CONDITION.wait()
                _TRANSITION_ACTIVE = True
            finally:
                _TRANSITIONS_WAITING -= 1
        else:
            while _TRANSITION_ACTIVE or _TRANSITIONS_WAITING:
                _GATE_CONDITION.wait()
            _ACTIVE_TOOLS += 1
    try:
        yield
    finally:
        with _GATE_CONDITION:
            if transition:
                _TRANSITION_ACTIVE = False
            else:
                _ACTIVE_TOOLS -= 1
            _GATE_CONDITION.notify_all()


def worker_execution_fence_error() -> Optional[str]:
    """Return why the current kanban worker must not execute another tool."""
    task_id = str(os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not task_id:
        return None

    run_text = str(os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    db_path = str(os.environ.get("HERMES_KANBAN_DB") or "").strip()
    if not run_text or not db_path:
        return "kanban worker identity is incomplete; tool execution refused"
    try:
        run_id = int(run_text)
    except ValueError:
        return "kanban worker run id is invalid; tool execution refused"

    try:
        from hermes_cli.profiles import get_active_profile_name

        active_profile = str(get_active_profile_name() or "default").strip()
    except Exception:
        return "kanban worker profile is unavailable; tool execution refused"

    try:
        uri = f"file:{quote(db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT t.status, t.current_run_id, t.assignee, "
                "r.task_id AS run_task_id, r.profile AS run_profile, r.ended_at "
                "FROM tasks t LEFT JOIN task_runs r ON r.id = ? WHERE t.id = ?",
                (run_id, task_id),
            ).fetchone()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return "kanban worker identity could not be verified; tool execution refused"

    if (
        row is None
        or row["status"] != "running"
        or row["current_run_id"] is None
        or int(row["current_run_id"]) != run_id
        or row["run_task_id"] != task_id
        or row["ended_at"] is not None
    ):
        return f"stale kanban worker run {run_id} for {task_id}; tool execution refused"

    intended_profile = str(row["assignee"] or "").strip()
    run_profile = str(row["run_profile"] or "").strip()
    if not intended_profile or run_profile != intended_profile or active_profile != intended_profile:
        return (
            "kanban worker profile mismatch: "
            f"active={active_profile!r}, run={run_profile!r}, assignee={intended_profile!r}; "
            "tool execution refused"
        )
    return None


def fence_kanban_worker_tool(*, name_arg_index: int) -> Callable:
    """Fence tools and serialize lifecycle transitions against side effects.

    Ordinary calls retain parallel execution. A block/complete transition waits
    for active calls and prevents new calls from starting; calls queued behind
    it then recheck the authoritative run state and fail closed.
    """

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not os.environ.get("HERMES_KANBAN_TASK"):
                return function(*args, **kwargs)
            if getattr(_GATE_LOCAL, "depth", 0):
                return function(*args, **kwargs)

            name = (
                args[name_arg_index]
                if len(args) > name_arg_index
                else kwargs.get("function_name")
            )
            function_args = (
                args[name_arg_index + 1]
                if len(args) > name_arg_index + 1
                else kwargs.get("function_args")
            )
            if name == "tool_call" and isinstance(function_args, dict):
                name = function_args.get("name")
            with _tool_gate(transition=name in _TRANSITIONS):
                try:
                    error = worker_execution_fence_error()
                except Exception:
                    error = "kanban worker identity guard failed; tool execution refused"
                if error:
                    return json.dumps({"error": error}, ensure_ascii=False)
                _GATE_LOCAL.depth = 1
                try:
                    return function(*args, **kwargs)
                finally:
                    _GATE_LOCAL.depth = 0

        return wrapped

    return decorate

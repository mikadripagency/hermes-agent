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

from hermes_cli.kanban_run_lock import KanbanRunLockTimeout, task_run_lock

_TRANSITIONS = {"kanban_block", "kanban_complete"}
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
        from hermes_cli.profiles import get_active_profile_name, normalize_profile_name

        active_profile = normalize_profile_name(
            str(get_active_profile_name() or "default")
        )
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

    try:
        from hermes_cli.profiles import normalize_profile_name

        intended_profile = normalize_profile_name(str(row["assignee"] or ""))
        run_profile = normalize_profile_name(str(row["run_profile"] or ""))
    except (TypeError, ValueError):
        intended_profile = ""
        run_profile = ""
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
            transition = name in _TRANSITIONS
            task_id = str(os.environ.get("HERMES_KANBAN_TASK") or "").strip()
            db_path = str(os.environ.get("HERMES_KANBAN_DB") or "").strip()
            try:
                with _tool_gate(transition=transition), task_run_lock(
                    db_path, task_id, exclusive=transition,
                ):
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
            except (KanbanRunLockTimeout, OSError) as exc:
                return json.dumps({"error": str(exc)}, ensure_ascii=False)

        return wrapped

    return decorate


def fence_kanban_worker_tool_batch(
    *, message_arg_index: int, allow_transition: bool = True
) -> Callable:
    """Fence a sequential tool batch before middleware/hooks/checkpoints."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not os.environ.get("HERMES_KANBAN_TASK"):
                return function(*args, **kwargs)
            message = (
                args[message_arg_index]
                if len(args) > message_arg_index
                else kwargs.get("assistant_message")
            )
            calls = list(getattr(message, "tool_calls", None) or [])
            names = {
                getattr(getattr(call, "function", None), "name", None)
                for call in calls
            }
            messages = args[2] if len(args) > 2 else kwargs.get("messages")

            def reject(error: str):
                content = json.dumps({"error": error}, ensure_ascii=False)
                if isinstance(messages, list):
                    for call in calls:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": getattr(call, "id", "") or "",
                            "name": getattr(
                                getattr(call, "function", None), "name", "tool"
                            ),
                            "content": content,
                        })
                return None

            transition = bool(names & _TRANSITIONS)
            if transition and (not allow_transition or len(calls) != 1):
                return reject(
                    "Kanban lifecycle transitions must be called alone, outside "
                    "concurrent or mixed tool batches"
                )
            task_id = str(os.environ.get("HERMES_KANBAN_TASK") or "").strip()
            db_path = str(os.environ.get("HERMES_KANBAN_DB") or "").strip()
            try:
                with _tool_gate(transition=transition), task_run_lock(
                    db_path, task_id, exclusive=transition,
                ):
                    error = worker_execution_fence_error()
                    if error:
                        return reject(error)
                    _GATE_LOCAL.depth = 1
                    try:
                        return function(*args, **kwargs)
                    finally:
                        _GATE_LOCAL.depth = 0
            except (KanbanRunLockTimeout, OSError) as exc:
                return reject(str(exc))

        return wrapped

    return decorate

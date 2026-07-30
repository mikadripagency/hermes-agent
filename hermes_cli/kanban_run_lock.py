"""Cross-process run fences for Kanban worker effects and transitions."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class KanbanRunLockTimeout(RuntimeError):
    """Raised when a run transition cannot exclude an active worker effect."""


_LOCAL = threading.local()


def _lock_path(db_path: str, task_id: str) -> Path:
    safe_task_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in task_id)
    root = Path(db_path).resolve().parent / ".kanban-run-locks"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"{safe_task_id}.lock"


def _try_lock(file_obj, *, exclusive: bool) -> bool:
    if os.name == "nt":
        import msvcrt

        file_obj.seek(0)
        try:
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    try:
        fcntl.flock(file_obj.fileno(), mode | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock(file_obj) -> None:
    if os.name == "nt":
        import msvcrt

        file_obj.seek(0)
        msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)


@contextmanager
def task_run_lock(
    db_path: str,
    task_id: str,
    *,
    exclusive: bool,
    timeout: float = 5.0,
) -> Iterator[None]:
    """Hold a task-scoped shared effect lock or exclusive transition lock."""
    path = _lock_path(db_path, task_id)
    held = getattr(_LOCAL, "held", None)
    if held is None:
        held = _LOCAL.held = {}
    key = str(path)
    current = held.get(key)
    if current is not None:
        mode, depth = current
        if exclusive and mode != "exclusive":
            raise RuntimeError("cannot upgrade a shared Kanban run lock")
        held[key] = (mode, depth + 1)
        try:
            yield
        finally:
            mode, depth = held[key]
            if depth == 1:
                del held[key]
            else:
                held[key] = (mode, depth - 1)
        return

    file_obj = open(path, "a+b")
    if file_obj.tell() == 0:
        file_obj.write(b"0")
        file_obj.flush()
    deadline = time.monotonic() + max(0.0, timeout)
    try:
        while not _try_lock(file_obj, exclusive=exclusive):
            if time.monotonic() >= deadline:
                raise KanbanRunLockTimeout(
                    f"active Kanban worker effect for {task_id}; transition refused"
                )
            time.sleep(0.01)
        held[key] = ("exclusive" if exclusive or os.name == "nt" else "shared", 1)
        try:
            yield
        finally:
            del held[key]
            _unlock(file_obj)
    finally:
        file_obj.close()


def database_path(conn) -> str:
    """Return the on-disk main database path for an SQLite connection."""
    for _seq, name, path in conn.execute("PRAGMA database_list"):
        if name == "main" and path:
            return str(path)
    raise RuntimeError("Kanban run fencing requires an on-disk SQLite database")


def fence_task_transition(*, task_id_arg_index: int = 1):
    """Decorate a DB transition with the task's exclusive run lock."""
    from functools import wraps

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            conn = args[0] if args else kwargs.get("conn")
            task_id = (
                args[task_id_arg_index]
                if len(args) > task_id_arg_index
                else kwargs.get("task_id")
            )
            if conn is None or not task_id:
                return function(*args, **kwargs)
            with task_run_lock(
                database_path(conn), str(task_id), exclusive=True
            ):
                return function(*args, **kwargs)

        return wrapped

    return decorate

"""Guard against destructive raw ``sqlite3`` commands on live Hermes DBs.

On 2026-07-12 an agent ran ``REINDEX`` / ``VACUUM`` via the ``sqlite3`` CLI
directly against the live ``kanban.db`` and re-corrupted it. The gateway's own
database access goes through WAL-aware, serialized helpers
(``hermes_cli/kanban_db.py`` ``write_txn`` + auto-quarantine); a raw ``sqlite3``
process bypasses all of that and can checkpoint, rewrite, or lock a database
that another Hermes process is actively using.

This module detects a ``sqlite3`` invocation that targets a ``*.db`` (or its
``-wal`` / ``-shm`` / ``-journal`` sidecar) under ``~/.hermes`` and is NOT
clearly read-only, so ``tools/terminal_tool.py`` can hard-block it at
*execution* time — the same defence-in-depth shape as
``cron/lifecycle_guard.py``'s gateway-restart guard.

"Not clearly read-only" means the invocation either contains a write/DDL
keyword (``REINDEX``, ``VACUUM``, ``DROP``, ``DELETE``, ``UPDATE``, ``INSERT``,
``ALTER``, ``PRAGMA wal_checkpoint``, ``.recover``) or opens the database
without any SQL to run (an interactive REPL / read-write handle). An explicit
read-only URI (``file:...?mode=ro``) is the sanctioned safe form and is always
allowed, so ``SELECT`` / ``PRAGMA quick_check`` inspection still works.
"""

from __future__ import annotations

import re
import shlex

# A path token that lives under ~/.hermes and names a SQLite DB (or a sidecar).
_HERMES_DB_RE = re.compile(r"(?i)\.hermes[\w./\-]*\.db(?:-(?:wal|shm|journal))?\b")

# SQL that writes to / restructures the database. ``PRAGMA wal_checkpoint`` and
# ``.recover`` are included because both mutate or race a live WAL database.
_WRITE_DDL_RE = re.compile(
    r"(?i)(?:\breindex\b|\bvacuum\b|\bdrop\b|\bdelete\b|\bupdate\b|\binsert\b"
    r"|\balter\b|\breplace\b|\bcreate\b|pragma\s+wal_checkpoint|\.recover\b)"
)

# The one sanctioned read-only escape hatch: a ``file:...?mode=ro`` URI, which
# SQLite opens read-only at the engine level regardless of the SQL.
_MODE_RO_RE = re.compile(r"(?i)mode=ro\b")

# sqlite3 flags that consume the following token as a value; ``-cmd`` / ``-init``
# carry SQL, so their value must be scanned for write keywords.
_SQL_VALUE_FLAGS = {"-cmd", "-init"}
_OTHER_VALUE_FLAGS = {
    "-separator", "-nullvalue", "-newline", "-lookaside", "-mmap",
    "-maxsize", "-pagecache", "-vfs", "-deserialize",
}


def contains_destructive_sqlite_command(command: str) -> bool:
    """Return True if *command* runs ``sqlite3`` destructively on a Hermes DB.

    Conservative by design: when the SQL cannot be seen (interactive REPL,
    stdin-fed script) the invocation is treated as destructive, because it
    opens a read-write handle on a live database. An explicit ``mode=ro`` URI
    is always allowed.
    """
    if not command or "sqlite3" not in command:
        return False

    # Split into shell segments so a pipeline/compound line is judged per
    # command. Erring toward over-splitting only makes the guard more cautious.
    for seg in re.split(r"\|\||&&|[;\n|&]", command):
        if not re.search(r"\bsqlite3\b", seg):
            continue
        if not _HERMES_DB_RE.search(seg):
            continue  # not a Hermes DB — out of scope
        if _MODE_RO_RE.search(seg):
            continue  # explicit read-only URI is the sanctioned safe form

        try:
            tokens = shlex.split(seg, comments=False, posix=True)
        except ValueError:
            tokens = seg.split()

        idx = next(
            (i for i, t in enumerate(tokens)
             if t == "sqlite3" or t.endswith("/sqlite3")),
            None,
        )
        if idx is None:
            # sqlite3 appeared only as a substring we couldn't tokenize as a
            # command word; fall back to a keyword scan on the raw segment.
            if _WRITE_DDL_RE.search(seg):
                return True
            continue

        args = tokens[idx + 1:]
        positionals: list[str] = []
        sql_bearing: list[str] = []
        i = 0
        while i < len(args):
            a = args[i]
            if a.startswith("-"):
                if a in _SQL_VALUE_FLAGS and i + 1 < len(args):
                    sql_bearing.append(args[i + 1])
                    i += 2
                    continue
                if a in _OTHER_VALUE_FLAGS and i + 1 < len(args):
                    i += 2
                    continue
                i += 1
                continue
            positionals.append(a)
            i += 1

        # The first positional is the database path/URI; the rest are SQL.
        sql_args = [p for p in positionals if not _HERMES_DB_RE.search(p)]
        sql_args += sql_bearing
        combined_sql = " ".join(sql_args)

        if _WRITE_DDL_RE.search(combined_sql):
            return True
        if not sql_args:
            # A Hermes DB opened with no SQL to run: an interactive REPL or a
            # stdin-fed script — a read-write handle on a live database.
            return True

    return False


BLOCKED_MESSAGE = (
    "Blocked: refusing to run a destructive raw sqlite3 command against a live "
    "Hermes database under ~/.hermes. Direct sqlite3 writes/DDL (REINDEX, "
    "VACUUM, DROP, DELETE, UPDATE, INSERT, ALTER, PRAGMA wal_checkpoint, "
    ".recover) or an interactive/stdin session bypass Hermes' WAL-aware, "
    "serialized DB access and have re-corrupted the kanban DB before. "
    "For inspection, open it read-only, e.g. "
    "sqlite3 'file:<path>?mode=ro' 'PRAGMA quick_check;'. For repair, stop the "
    "gateway first and use the supervised path (`hermes doctor`); the gateway "
    "auto-quarantines a corrupt DB on next open."
)

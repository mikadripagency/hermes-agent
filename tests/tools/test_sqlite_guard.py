"""Tests for the destructive-sqlite3 guard (tools/sqlite_guard.py).

Regression: on 2026-07-12 an agent ran REINDEX/VACUUM via the sqlite3 CLI on
the live kanban.db and re-corrupted it. The guard hard-blocks destructive raw
sqlite3 against ~/.hermes/*.db while leaving read-only inspection allowed.
"""

import pytest

from tools.sqlite_guard import contains_destructive_sqlite_command


class TestBlocked:
    @pytest.mark.parametrize("cmd", [
        "sqlite3 ~/.hermes/kanban.db 'VACUUM;'",
        "sqlite3 ~/.hermes/kanban.db 'REINDEX;'",
        "sqlite3 /Users/me/.hermes/kanban.db 'DELETE FROM tasks'",
        "sqlite3 /Users/me/.hermes/state/hermes_state.db 'UPDATE tasks SET x=1'",
        "sqlite3 ~/.hermes/kanban.db 'DROP TABLE tasks'",
        "sqlite3 ~/.hermes/kanban.db 'ALTER TABLE tasks ADD COLUMN y TEXT'",
        "sqlite3 ~/.hermes/kanban.db 'INSERT INTO tasks VALUES (1)'",
        "sqlite3 ~/.hermes/kanban.db 'PRAGMA wal_checkpoint(TRUNCATE);'",
        "sqlite3 ~/.hermes/kanban.db '.recover' | sqlite3 new.db",
        "sqlite3 -cmd 'VACUUM' ~/.hermes/kanban.db",
        # WAL / sidecar targets
        "sqlite3 ~/.hermes/kanban.db-wal 'VACUUM'",
    ])
    def test_destructive_is_blocked(self, cmd):
        assert contains_destructive_sqlite_command(cmd), f"should block: {cmd!r}"

    @pytest.mark.parametrize("cmd", [
        # Interactive REPL / read-write handle with no SQL to run.
        "sqlite3 ~/.hermes/kanban.db",
        "sqlite3 /Users/me/.hermes/kanban.db",
        # Stdin-fed script: SQL is invisible, so treated as destructive.
        "cat fix.sql | sqlite3 ~/.hermes/kanban.db",
    ])
    def test_interactive_or_stdin_is_blocked(self, cmd):
        assert contains_destructive_sqlite_command(cmd), f"should block: {cmd!r}"

    def test_destructive_in_pipeline_is_blocked(self):
        assert contains_destructive_sqlite_command(
            "echo hi && sqlite3 ~/.hermes/kanban.db 'VACUUM'"
        )


class TestAllowed:
    @pytest.mark.parametrize("cmd", [
        # Sanctioned read-only URI form.
        "sqlite3 'file:/Users/me/.hermes/kanban.db?mode=ro' 'PRAGMA quick_check;'",
        "sqlite3 'file:/Users/me/.hermes/kanban.db?mode=ro' 'SELECT count(*) FROM tasks'",
        # A read-only SELECT/PRAGMA with explicit SQL (not interactive, no write).
        "sqlite3 ~/.hermes/kanban.db 'SELECT count(*) FROM tasks'",
        "sqlite3 ~/.hermes/kanban.db '.schema tasks'",
    ])
    def test_read_only_is_allowed(self, cmd):
        assert not contains_destructive_sqlite_command(cmd), f"should allow: {cmd!r}"

    @pytest.mark.parametrize("cmd", [
        # Non-Hermes databases are out of scope.
        "sqlite3 /tmp/scratch.db 'VACUUM'",
        "sqlite3 ./local.db 'DELETE FROM x'",
        "sqlite3 /var/data/app.db",
        # Not sqlite3 at all.
        "grep VACUUM ~/.hermes/notes.txt",
        "echo 'sqlite3 is a tool'",
    ])
    def test_out_of_scope_is_allowed(self, cmd):
        assert not contains_destructive_sqlite_command(cmd), f"should allow: {cmd!r}"

    def test_empty_command(self):
        assert not contains_destructive_sqlite_command("")

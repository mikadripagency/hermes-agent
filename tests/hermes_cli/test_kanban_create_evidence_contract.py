from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_delivery_create_requires_explicit_evidence_contract_without_writing_row(
    kanban_home,
):
    with kb.connect_closing() as conn:
        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("tasks", "task_events", "completion_deliveries")
        }

        with pytest.raises(ValueError, match="evidence contract"):
            kb.create_task(conn, title="ship safely", task_kind="delivery")

        after = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("tasks", "task_events", "completion_deliveries")
        }

    assert after == before


def test_delivery_create_normalizes_required_evidence(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="ship with proof",
            task_kind="delivery",
            required_evidence=[" regression_test ", "regression_test", "merge_receipt"],
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.required_evidence == ["regression_test", "merge_receipt"]
    assert task.evidence_contract_na_reason is None


def test_delivery_create_persists_explicit_na_reason_in_readback_and_event(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="analyze only",
            task_kind="delivery",
            evidence_contract_na_reason="  analysis-only output; no runtime gate  ",
        )
        task = kb.get_task(conn, task_id)
        created = next(event for event in kb.list_events(conn, task_id) if event.kind == "created")

    assert task is not None
    assert task.required_evidence is None
    assert task.evidence_contract_na_reason == "analysis-only output; no runtime gate"
    assert created.payload["evidence_contract_na_reason"] == task.evidence_contract_na_reason


def test_system_inbox_create_remains_exempt(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="process writer queue",
            task_kind="system_inbox",
            initial_status="blocked",
        )

    assert task_id.startswith("t_")


def test_legacy_python_create_caller_gets_auditable_compatibility_reason(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="legacy internal caller")
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.evidence_contract_na_reason == "legacy create_task caller (task_kind omitted)"


def test_delivery_contract_rejects_ambiguous_or_malformed_na_reason(kanban_home):
    malformed_reason: Any = {"reason": "not applicable"}
    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match="not both"):
            kb.create_task(
                conn,
                title="ambiguous",
                task_kind="delivery",
                required_evidence=["regression_test"],
                evidence_contract_na_reason="not applicable",
            )
        with pytest.raises(ValueError, match="must be a string"):
            kb.create_task(
                conn,
                title="wrong type",
                task_kind="delivery",
                evidence_contract_na_reason=malformed_reason,
            )


def test_cli_rejects_missing_delivery_contract_without_writing_row(kanban_home):
    with kb.connect_closing() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    output = kc.run_slash("create 'missing contract' --assignee alice")

    with kb.connect_closing() as conn:
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert "explicit evidence contract" in output
    assert after == before


def test_cli_accepts_and_reads_back_explicit_na_contract(kanban_home):
    output = kc.run_slash(
        "create 'analysis only' --assignee alice "
        "--evidence-na-reason 'no runtime side effects'"
    )
    match = re.search(r"(t_[a-f0-9]+)", output)
    assert match is not None

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, match.group(1))
    assert task is not None
    assert task.evidence_contract_na_reason == "no runtime side effects"

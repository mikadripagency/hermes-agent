from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
import shutil
import sqlite3
import threading
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


def test_repository_delivery_policy_adds_exact_merge_evidence(kanban_home, tmp_path):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="ship repository change",
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "repo"),
            required_evidence=["self_review", "merge_receipt"],
        )
        task = kb.get_task(conn, task_id)
        created = next(
            event for event in kb.list_events(conn, task_id)
            if event.kind == "created"
        )

    assert task is not None
    assert task.required_evidence == [
        "self_review", "merge_receipt", "delivery_ownership", "pr_merged",
    ]
    assert created.payload is not None
    assert created.payload["delivery_gates"] == ["merge"]


def test_repository_delivery_policy_rejects_na_contract_without_writing(
    kanban_home, tmp_path,
):
    with kb.connect_closing() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(
            ValueError,
            match=r"repository delivery.*pr_merged.*delivery_gates=\[\]",
        ):
            kb.create_task(
                conn,
                title="impossible repository delivery",
                workspace_kind="worktree",
                workspace_path=str(tmp_path / "repo"),
                evidence_contract_na_reason="proof lives in prose",
            )
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    assert after == before


def test_explicit_no_delivery_gates_preserves_no_code_worktree_lane(
    kanban_home, tmp_path,
):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="closure-only reconciliation",
            workspace_kind="worktree",
            workspace_path=str(tmp_path / "repo"),
            delivery_gates=[],
            evidence_contract_na_reason="no code, merge, deploy, or terminal delivery",
        )
        task = kb.get_task(conn, task_id)
        created = next(
            event for event in kb.list_events(conn, task_id)
            if event.kind == "created"
        )

    assert task is not None
    assert task.required_evidence is None
    assert task.evidence_contract_na_reason == (
        "no code, merge, deploy, or terminal delivery"
    )
    assert created.payload is not None
    assert created.payload["delivery_gates"] == []


def test_explicit_deploy_policy_adds_exact_merge_evidence(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="deploy through reviewed integration",
            delivery_gates=["deploy"],
            required_evidence=["merge_receipt"],
        )
        task = kb.get_task(conn, task_id)

    assert task is not None
    assert task.required_evidence == [
        "merge_receipt", "delivery_ownership", "pr_merged", "runtime_smoke",
    ]


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


def test_trusted_system_inbox_create_remains_exempt(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_system_inbox_task(
            conn,
            title="process writer queue",
            initial_status="blocked",
        )

    assert task_id.startswith("t_")


def test_python_create_without_contract_fails_closed(kanban_home):
    with kb.connect_closing() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        with pytest.raises(ValueError, match="explicit evidence contract"):
            kb.create_task(conn, title="new internal caller")
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    assert after == before


def test_legacy_python_idempotent_retry_resolves_pre_contract_row(kanban_home):
    with kb.connect_closing() as conn:
        conn.execute(
            "INSERT INTO tasks "
            "(id, title, status, task_kind, created_at, idempotency_key) "
            "VALUES ('t_legacy_retry', 'legacy', 'ready', 'delivery', 1, 'legacy-key')"
        )
        conn.commit()

        task_id = kb.create_task(
            conn,
            title="legacy retry",
            idempotency_key="legacy-key",
        )

    assert task_id == "t_legacy_retry"


def test_idempotent_retry_cannot_omit_an_existing_contract(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="contracted",
            idempotency_key="contracted-key",
            required_evidence=["regression_test"],
        )
        with pytest.raises(ValueError, match="must provide the existing task"):
            kb.create_task(
                conn,
                title="contracted retry",
                idempotency_key="contracted-key",
            )

    assert task_id.startswith("t_")


def test_copied_legacy_db_keeps_null_contract_and_idempotent_readback(
    kanban_home, tmp_path,
):
    source = tmp_path / "source.db"
    copied = tmp_path / "copied.db"
    conn = sqlite3.connect(source)
    conn.execute(
        "CREATE TABLE tasks ("
        "id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT, "
        "status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0, "
        "created_by TEXT, created_at INTEGER NOT NULL, started_at INTEGER, "
        "completed_at INTEGER, workspace_kind TEXT NOT NULL DEFAULT 'scratch', "
        "workspace_path TEXT, claim_lock TEXT, claim_expires INTEGER, "
        "idempotency_key TEXT)"
    )
    conn.execute(
        "CREATE TABLE task_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, "
        "kind TEXT NOT NULL, payload TEXT, created_at INTEGER NOT NULL)"
    )
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, status, created_at, idempotency_key) "
        "VALUES ('t_copied_legacy', 'legacy', 'ready', 1, 'copy-key')"
    )
    conn.commit()
    conn.close()
    shutil.copy2(source, copied)

    with kb.connect(copied) as conn:
        task = kb.get_task(conn, "t_copied_legacy")
        assert task is not None
        retry_id = kb.create_task(
            conn,
            title="legacy retry",
            idempotency_key="copy-key",
        )
        claimed = kb.claim_task(conn, task.id, claimer="copy-db-test")
        assert claimed is not None and claimed.current_run_id is not None
        completed = kb.complete_task(
            conn,
            task.id,
            summary=f"{task.id}/run {claimed.current_run_id} · legacy copy complete",
        )
        completed_task = kb.get_task(conn, task.id)

    assert task.required_evidence is None
    assert task.evidence_contract_na_reason is None
    assert retry_id == task.id
    assert completed is True
    assert completed_task is not None and completed_task.status == "done"


def test_concurrent_idempotent_creates_write_one_task_and_event(
    kanban_home, monkeypatch,
):
    barrier = threading.Barrier(2)
    original_new_task_id = kb._new_task_id

    def synchronized_new_task_id():
        task_id = original_new_task_id()
        barrier.wait(timeout=5)
        return task_id

    monkeypatch.setattr(kb, "_new_task_id", synchronized_new_task_id)

    def create_once():
        with kb.connect_closing() as conn:
            return kb.create_task(
                conn,
                title="one logical request",
                idempotency_key="concurrent-contract-create",
                required_evidence=["regression_test"],
            )

    with ThreadPoolExecutor(max_workers=2) as pool:
        task_ids = list(pool.map(lambda _: create_once(), range(2)))

    with kb.connect_closing() as conn:
        task_count = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key = ?",
            ("concurrent-contract-create",),
        ).fetchone()[0]
        event_count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'created'",
            (task_ids[0],),
        ).fetchone()[0]

    assert task_ids[0] == task_ids[1]
    assert task_count == 1
    assert event_count == 1


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


def test_cli_worktree_na_contract_requires_explicit_no_gate_policy(
    kanban_home, tmp_path,
):
    output = kc.run_slash(
        "create 'impossible repo lane' --assignee alice "
        f"--workspace worktree:{tmp_path / 'repo'} "
        "--evidence-na-reason 'proof only in prose'"
    )
    assert "N/A contract is impossible" in output

    accepted = kc.run_slash(
        "create 'closure-only lane' --assignee alice "
        f"--workspace worktree:{tmp_path / 'repo'} --no-delivery-gates "
        "--evidence-na-reason 'no code or merge side effect'"
    )
    assert "Created t_" in accepted


def test_cli_contract_amend_is_supported_and_audited(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="blocked legacy contract",
            assignee="developer",
            initial_status="blocked",
            evidence_contract_na_reason="legacy prose receipt",
        )

    output = kc.run_slash(
        f"contract-amend {task_id} --add-evidence pr_merged "
        "--add-evidence runtime_smoke --reason 'normal delivery gate'"
    )

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        amended = [
            event for event in kb.list_events(conn, task_id)
            if event.kind == "evidence_contract_amended"
        ]
    assert "EVIDENCE_CONTRACT_AMENDED" in output
    assert task is not None
    assert task.required_evidence == ["pr_merged", "runtime_smoke"]
    assert len(amended) == 1


def test_contract_amend_replaces_na_with_monotonic_audited_evidence(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="repair blocked repository delivery",
            assignee="developer",
            initial_status="blocked",
            evidence_contract_na_reason="legacy prose receipt",
        )
        event_id = kb.amend_evidence_contract(
            conn,
            task_id,
            add_required_evidence=["pr_merged", "runtime_smoke"],
            actor="developer",
            reason="normal merge/deploy gate requires exact machine receipts",
        )
        task = kb.get_task(conn, task_id)
        event = next(
            item for item in kb.list_events(conn, task_id)
            if item.kind == "evidence_contract_amended"
        )

    assert event_id > 0
    assert task is not None
    assert task.required_evidence == ["pr_merged", "runtime_smoke"]
    assert task.evidence_contract_na_reason is None
    assert event.payload == {
        "actor": "developer",
        "reason": "normal merge/deploy gate requires exact machine receipts",
        "added_required_evidence": ["pr_merged", "runtime_smoke"],
        "previous_required_evidence": None,
        "previous_evidence_contract_na_reason": "legacy prose receipt",
        "required_evidence": ["pr_merged", "runtime_smoke"],
    }


def test_contract_amend_refuses_active_or_terminal_task(kanban_home):
    with kb.connect_closing() as conn:
        active_id = kb.create_task(
            conn,
            title="active task",
            required_evidence=["pr_merged"],
        )
        claimed = kb.claim_task(conn, active_id, claimer="test")
        assert claimed is not None
        with pytest.raises(ValueError, match="blocked, todo, ready, or triage"):
            kb.amend_evidence_contract(
                conn,
                active_id,
                add_required_evidence=["runtime_smoke"],
                actor="developer",
                reason="too late while active",
            )

        done_id = kb.create_task(
            conn,
            title="completed task",
            evidence_contract_na_reason="no objective gate",
        )
        done_claim = kb.claim_task(conn, done_id, claimer="test")
        assert done_claim is not None and done_claim.current_run_id is not None
        assert kb.complete_task(
            conn,
            done_id,
            summary=f"{done_id}/run {done_claim.current_run_id} complete",
        )
        completed_events = [
            event.id for event in kb.list_events(conn, done_id)
            if event.kind == "completed"
        ]
        with pytest.raises(ValueError, match="blocked, todo, ready, or triage"):
            kb.amend_evidence_contract(
                conn,
                done_id,
                add_required_evidence=["pr_merged"],
                actor="developer",
                reason="must not rewrite completed receipt",
            )
        completed_events_after = [
            event.id for event in kb.list_events(conn, done_id)
            if event.kind == "completed"
        ]

    assert completed_events_after == completed_events


def test_contract_amend_rejects_unassigned_profile(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="owned blocked task",
            assignee="alice",
            initial_status="blocked",
            evidence_contract_na_reason="legacy prose receipt",
        )
        with pytest.raises(ValueError, match="not authorized"):
            kb.amend_evidence_contract(
                conn,
                task_id,
                add_required_evidence=["pr_merged"],
                actor="developer",
                reason="cross-profile attempt",
            )


def test_cli_cannot_select_system_inbox(kanban_home):
    with kb.connect_closing() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    output = kc.run_slash(
        "create 'untrusted inbox attempt' --assignee alice "
        "--task-kind system_inbox --evidence-na-reason test-only"
    )

    with kb.connect_closing() as conn:
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert "unrecognized arguments: --task-kind system_inbox" in output
    assert after == before

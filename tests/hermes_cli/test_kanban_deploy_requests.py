from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

REPO = "mikadripagency/hermes-agent"
REVIEWED = "a" * 40
INTEGRATION = "b" * 40


@pytest.fixture
def deploy_task(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="deploy",
            assignee="developer",
            required_evidence=["pr_merged"],
        )
        task = kb.claim_task(
            conn,
            task_id,
            claimer="worker",
            execution_profile="developer",
        )
        assert task and task.current_run_id
        run_id = task.current_run_id
        kb.record_review_receipt(
            conn,
            task_id,
            repository=REPO,
            reviewed_sha=REVIEWED,
            verdict="passed",
            findings=[],
            expected_run_id=run_id,
        )
        kb.bind_review_integration(
            conn,
            task_id,
            repository=REPO,
            reviewed_sha=REVIEWED,
            integration_sha=INTEGRATION,
            expected_run_id=run_id,
        )
    return task_id, run_id


def _events(conn, task_id):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id", (task_id,)
    ).fetchall()
    return [(row["kind"], json.loads(row["payload"] or "{}")) for row in rows]


def test_authorized_request_survives_worker_run_end_without_resubmission(deploy_task):
    task_id, run_id = deploy_task
    with kb.connect() as conn:
        request_id = kb.request_core_deploy(
            conn,
            task_id,
            repository=REPO,
            reviewed_sha=REVIEWED,
            integration_sha=INTEGRATION,
            expected_run_id=run_id,
        )
        conn.execute(
            "UPDATE tasks SET status='blocked', current_run_id=NULL, claim_lock=NULL, claim_expires=NULL WHERE id=?",
            (task_id,),
        )
        conn.commit()

        claim = kb.claim_core_deploy(
            conn,
            task_id,
            repository=REPO,
            reviewed_sha=REVIEWED,
            integration_sha=INTEGRATION,
            claimant="watchdog-a",
            lease_seconds=60,
        )

    assert claim["request_event_id"] == request_id
    assert claim["claim_id"]


def test_request_claim_and_ack_are_idempotent_and_task_durable(deploy_task):
    task_id, run_id = deploy_task
    with kb.connect() as conn:
        first = kb.request_core_deploy(
            conn, task_id, repository=REPO, reviewed_sha=REVIEWED,
            integration_sha=INTEGRATION, expected_run_id=run_id,
        )
        duplicate = kb.request_core_deploy(
            conn, task_id, repository=REPO, reviewed_sha=REVIEWED,
            integration_sha=INTEGRATION, expected_run_id=run_id,
        )
        assert duplicate == first

        claim = kb.claim_core_deploy(
            conn, task_id, repository=REPO, reviewed_sha=REVIEWED,
            integration_sha=INTEGRATION, claimant="watchdog-a", lease_seconds=60,
        )
        with pytest.raises(kb.CoreDeployRequestError, match="already claimed"):
            kb.claim_core_deploy(
                conn, task_id, repository=REPO, reviewed_sha=REVIEWED,
                integration_sha=INTEGRATION, claimant="watchdog-b", lease_seconds=60,
            )
        ack_event = kb.finish_core_deploy(
            conn, task_id, request_event_id=first, claim_id=claim["claim_id"],
            status="succeeded", receipt={"release_sha": INTEGRATION},
        )
        same_ack = kb.finish_core_deploy(
            conn, task_id, request_event_id=first, claim_id=claim["claim_id"],
            status="succeeded", receipt={"release_sha": INTEGRATION},
        )
        assert same_ack == ack_event

        kinds = [kind for kind, _ in _events(conn, task_id)]
    assert kinds.count("core_deploy_requested") == 1
    assert kinds.count("core_deploy_claimed") == 1
    assert kinds.count("core_deploy_succeeded") == 1


def test_changed_review_identity_fails_closed_with_audit_reason(deploy_task):
    task_id, run_id = deploy_task
    with kb.connect() as conn:
        kb.request_core_deploy(
            conn, task_id, repository=REPO, reviewed_sha=REVIEWED,
            integration_sha=INTEGRATION, expected_run_id=run_id,
        )
        with pytest.raises(kb.CoreDeployRequestError, match="identity mismatch"):
            kb.claim_core_deploy(
                conn, task_id, repository=REPO, reviewed_sha="c" * 40,
                integration_sha=INTEGRATION, claimant="watchdog", lease_seconds=60,
            )
        rejected = [payload for kind, payload in _events(conn, task_id)
                    if kind == "core_deploy_rejected"]
    assert rejected[-1]["reason"] == "request_identity_mismatch"

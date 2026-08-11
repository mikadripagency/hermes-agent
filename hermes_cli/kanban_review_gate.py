"""Fail-closed, SHA-scoped review receipts for repository deliveries."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Optional

_REVIEW_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class ReviewGateError(ValueError):
    """A repository delivery has no current, closed review receipt."""

    def __init__(self, task_id: str, reason: str):
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"review gate blocked for {task_id}: {reason}")


def review_gate_required(task) -> bool:
    """A merge evidence class is the immutable repository-lane contract."""
    return bool(task and "pr_merged" in (task.required_evidence or []))


def _validate_identity(repository: str, sha: str) -> tuple[str, str]:
    normalized_repository = str(repository or "").strip()
    normalized_sha = str(sha or "").strip().lower()
    if not _REPOSITORY_RE.fullmatch(normalized_repository):
        raise ValueError("repository must be owner/name")
    if not _REVIEW_SHA_RE.fullmatch(normalized_sha):
        raise ValueError("review SHA must be a full 40-character Git SHA")
    return normalized_repository, normalized_sha


def _events(conn: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT id, run_id, kind, payload FROM task_events "
        "WHERE task_id = ? AND kind IN "
        "('changes_requested', 'review_receipt', 'review_integration') "
        "ORDER BY id ASC",
        (task_id,),
    ).fetchall()


def _payload(row: sqlite3.Row) -> dict:
    try:
        value = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _require_current_run(
    conn: sqlite3.Connection, task_id: str, expected_run_id: Optional[int]
):
    from hermes_cli import kanban_db as kb

    task = kb.get_task(conn, task_id)
    if task is None:
        raise ReviewGateError(task_id, "task does not exist")
    if not review_gate_required(task):
        raise ReviewGateError(task_id, "task has no pr_merged evidence contract")
    if (
        task.status != "running"
        or task.current_run_id is None
        or expected_run_id is None
        or int(task.current_run_id) != int(expected_run_id)
    ):
        raise ReviewGateError(task_id, "current task/run identity does not match")
    return task


def _normalize_findings(findings) -> list[dict]:
    if not isinstance(findings, list):
        raise ValueError("findings must be a JSON list")
    normalized: list[dict] = []
    seen: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            raise ValueError("each finding must be an object")
        finding_id = str(finding.get("id") or "").strip()
        status = str(finding.get("status") or "").strip().casefold()
        proof = str(finding.get("proof") or "").strip()
        if not finding_id or finding_id in seen:
            raise ValueError("finding ids must be non-empty and unique")
        if status not in {"open", "closed"}:
            raise ValueError("finding status must be open or closed")
        if not proof:
            raise ValueError(f"finding {finding_id} requires proof")
        seen.add(finding_id)
        normalized.append({"id": finding_id, "status": status, "proof": proof})
    return normalized


def _open_findings(rows: list[sqlite3.Row], repository: str) -> dict[str, int]:
    open_findings: dict[str, int] = {}
    for row in rows:
        event_id = int(row["id"])
        payload = _payload(row)
        if row["kind"] == "changes_requested":
            open_findings[f"event:{event_id}"] = event_id
        elif row["kind"] == "review_receipt" and payload.get("repository") == repository:
            for finding in payload.get("findings") or []:
                finding_id = str(finding.get("id") or "")
                if finding.get("status") == "open":
                    open_findings[finding_id] = event_id
                elif finding.get("status") == "closed":
                    open_findings.pop(finding_id, None)
    return open_findings


def record_review_receipt(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    reviewed_sha: str,
    verdict: str,
    findings,
    expected_run_id: Optional[int],
) -> int:
    """Append a SHA-scoped review verdict for the authoritative current run."""
    from hermes_cli import kanban_db as kb

    repository, reviewed_sha = _validate_identity(repository, reviewed_sha)
    normalized_verdict = str(verdict or "").strip().casefold()
    if normalized_verdict not in {"passed", "changes_requested"}:
        raise ValueError("review verdict must be passed or changes_requested")
    normalized_findings = _normalize_findings(findings)
    open_in_receipt = [f["id"] for f in normalized_findings if f["status"] == "open"]
    if normalized_verdict == "passed" and open_in_receipt:
        raise ValueError("passed review receipts cannot contain open findings")
    if normalized_verdict == "changes_requested" and not open_in_receipt:
        raise ValueError("changes_requested receipts require open findings")

    with kb.write_txn(conn):
        _require_current_run(conn, task_id, expected_run_id)
        previous = _events(conn, task_id)
        unresolved = _open_findings(previous, repository)
        for finding in normalized_findings:
            if finding["status"] == "closed":
                unresolved.pop(finding["id"], None)
            else:
                unresolved[finding["id"]] = -1
        if normalized_verdict == "passed" and unresolved:
            finding_id = sorted(unresolved)[0]
            raise ReviewGateError(task_id, f"unresolved finding {finding_id}")
        previous_id = int(previous[-1]["id"]) if previous else None
        kb._append_event(
            conn,
            task_id,
            "review_receipt",
            {
                "repository": repository,
                "reviewed_sha": reviewed_sha,
                "verdict": normalized_verdict,
                "finding_status": "open" if open_in_receipt else "closed",
                "open_findings": len(unresolved),
                "findings": normalized_findings,
                "supersedes_event_id": previous_id,
            },
            run_id=int(expected_run_id or 0),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def assert_review_gate(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    final_sha: str,
    require_integration: bool = False,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Fail closed unless the current run's review covers the supplied SHA."""
    from hermes_cli import kanban_db as kb

    repository, final_sha = _validate_identity(repository, final_sha)
    if expected_run_id is None:
        task = kb.get_task(conn, task_id)
        if task is None:
            raise ReviewGateError(task_id, "task does not exist")
        if not review_gate_required(task):
            raise ReviewGateError(task_id, "task has no pr_merged evidence contract")
    else:
        _require_current_run(conn, task_id, expected_run_id)

    latest_blocking_id = 0
    latest_pass: Optional[tuple[int, dict]] = None
    integrations: list[tuple[int, dict]] = []
    for row in _events(conn, task_id):
        event_id = int(row["id"])
        payload = _payload(row)
        if row["kind"] == "changes_requested":
            latest_blocking_id = event_id
        elif (
            row["kind"] == "review_receipt"
            and payload.get("repository") == repository
            and (expected_run_id is None or row["run_id"] == expected_run_id)
        ):
            if payload.get("verdict") == "changes_requested" or payload.get("open_findings"):
                latest_blocking_id = event_id
            elif (
                payload.get("verdict") == "passed"
                and payload.get("finding_status") == "closed"
                and payload.get("open_findings") == 0
            ):
                latest_pass = (event_id, payload)
        elif (
            row["kind"] == "review_integration"
            and (expected_run_id is None or row["run_id"] == expected_run_id)
        ):
            integrations.append((event_id, payload))

    if latest_pass is None:
        reason = (
            "changes requested remain open"
            if latest_blocking_id
            else "missing review receipt for current run"
            if expected_run_id is not None
            else "missing review receipt"
        )
        raise ReviewGateError(task_id, reason)
    pass_id, pass_payload = latest_pass
    if pass_id <= latest_blocking_id:
        raise ReviewGateError(task_id, "changes requested remain open")
    reviewed_sha = str(pass_payload.get("reviewed_sha") or "")
    if not require_integration:
        if reviewed_sha != final_sha:
            raise ReviewGateError(
                task_id, "final head changed after review; incremental receipt required"
            )
        return True

    if not any(
        event_id > pass_id
        and payload.get("repository") == repository
        and payload.get("reviewed_sha") == reviewed_sha
        and payload.get("integration_sha") == final_sha
        for event_id, payload in integrations
    ):
        raise ReviewGateError(
            task_id, "final integration SHA is not bound to the reviewed head"
        )
    return True


def bind_review_integration(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    reviewed_sha: str,
    integration_sha: str,
    expected_run_id: Optional[int],
) -> bool:
    """Bind a verified merge commit to the exact reviewed PR head."""
    from hermes_cli import kanban_db as kb

    repository, reviewed_sha = _validate_identity(repository, reviewed_sha)
    _, integration_sha = _validate_identity(repository, integration_sha)
    with kb.write_txn(conn):
        _require_current_run(conn, task_id, expected_run_id)
        assert_review_gate(
            conn,
            task_id,
            repository=repository,
            final_sha=reviewed_sha,
            expected_run_id=expected_run_id,
        )
        kb._append_event(
            conn,
            task_id,
            "review_integration",
            {
                "repository": repository,
                "reviewed_sha": reviewed_sha,
                "integration_sha": integration_sha,
            },
            run_id=int(expected_run_id),
        )
    return True


def assert_deploy_review_gate(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    reviewed_sha: str,
    integration_sha: str,
    expected_run_id: Optional[int],
) -> bool:
    """Authorize a deploy only for the current run and its bound review."""
    _require_current_run(conn, task_id, expected_run_id)
    assert_review_gate(
        conn,
        task_id,
        repository=repository,
        final_sha=reviewed_sha,
        expected_run_id=expected_run_id,
    )
    assert_review_gate(
        conn,
        task_id,
        repository=repository,
        final_sha=integration_sha,
        require_integration=True,
        expected_run_id=expected_run_id,
    )
    return True


def assert_completion_review_gate(
    conn: sqlite3.Connection,
    task,
    metadata: Optional[dict],
    expected_run_id: Optional[int],
) -> None:
    if not review_gate_required(task):
        return
    _require_current_run(conn, task.id, expected_run_id)
    evidence = metadata.get("evidence") if isinstance(metadata, dict) else None
    merged = evidence.get("pr_merged") if isinstance(evidence, dict) else None
    if not isinstance(merged, dict):
        raise ReviewGateError(task.id, "pr_merged evidence must be a structured receipt")
    try:
        assert_review_gate(
            conn,
            task.id,
            repository=merged.get("repository"),
            final_sha=merged.get("head_sha"),
            expected_run_id=expected_run_id,
        )
        assert_review_gate(
            conn,
            task.id,
            repository=merged.get("repository"),
            final_sha=merged.get("merge_sha"),
            require_integration=True,
            expected_run_id=expected_run_id,
        )
    except ValueError as exc:
        if isinstance(exc, ReviewGateError):
            raise
        raise ReviewGateError(task.id, str(exc)) from exc

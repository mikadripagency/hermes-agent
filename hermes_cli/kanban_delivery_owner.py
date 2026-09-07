"""Fail-closed ownership for overlapping repository delivery lanes."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import PurePosixPath
from typing import Iterable, Optional

_REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_VALID_SEAM_SOURCES = {"declared", "diff"}
_CHANGELOG_SEAM = "CHANGELOG.md"
_ACTIVE_STATUSES = {"running", "review"}


class DeliveryOwnershipError(ValueError):
    """A repository delivery lane is missing ownership or overlaps another."""

    def __init__(self, task_id: str, reason: str):
        self.task_id = task_id
        self.reason = reason
        super().__init__(f"delivery ownership blocked for {task_id}: {reason}")


def _payload(raw: object) -> dict:
    if not isinstance(raw, (str, bytes, bytearray)):
        raw = "{}"
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _normalize_repository(repository: str) -> str:
    value = str(repository or "").strip()
    if not _REPOSITORY_RE.fullmatch(value):
        raise ValueError("repository must be owner/name")
    return value


def _normalize_branch(branch: str) -> str:
    value = str(branch or "").strip()
    if not _BRANCH_RE.fullmatch(value) or ".." in value or "//" in value:
        raise ValueError("branch must be a non-empty safe git ref name")
    return value


def _normalize_pr_number(pr_number: int) -> int:
    try:
        value = int(pr_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("pr_number must be a positive integer") from exc
    if value < 1:
        raise ValueError("pr_number must be a positive integer")
    return value


def _normalize_seams(seams: Iterable[str]) -> list[str]:
    normalized: set[str] = set()
    for seam in seams:
        value = str(seam or "").strip().replace("\\", "/")
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or value.startswith("./")
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError(f"invalid repository seam: {seam!r}")
        normalized.add(path.as_posix())
    if not normalized:
        raise ValueError("at least one repository seam is required")
    return sorted(normalized)


def _normalize_seam_source(seam_source: str) -> str:
    value = str(seam_source or "").strip().casefold()
    if value not in _VALID_SEAM_SOURCES:
        raise ValueError("seam_source must be declared or diff")
    return value


def _require_current_repository_run(
    conn: sqlite3.Connection,
    task_id: str,
    expected_run_id: Optional[int],
    *,
    branch: Optional[str] = None,
):
    from hermes_cli import kanban_db as kb

    task = kb.get_task(conn, task_id)
    if task is None:
        raise DeliveryOwnershipError(task_id, "task does not exist")
    if "pr_merged" not in (task.required_evidence or []):
        raise DeliveryOwnershipError(task_id, "task has no pr_merged evidence contract")
    if (
        task.status != "running"
        or task.current_run_id is None
        or expected_run_id is None
        or int(task.current_run_id) != int(expected_run_id)
    ):
        raise DeliveryOwnershipError(task_id, "current task/run identity does not match")
    if task.workspace_kind != "worktree":
        raise DeliveryOwnershipError(task_id, "repository owner must use a worktree")
    if not task.branch_name:
        raise DeliveryOwnershipError(task_id, "repository owner has no task branch")
    if branch is not None and task.branch_name != branch:
        raise DeliveryOwnershipError(
            task_id,
            f"branch {branch!r} does not match task branch {task.branch_name!r}",
        )
    return task


def _active_claims(
    conn: sqlite3.Connection, repository: str
) -> dict[str, tuple[dict, sqlite3.Row]]:
    rows = conn.execute(
        "SELECT e.id, e.task_id, e.kind, e.payload, t.status, t.branch_name, "
        "t.assignee, t.current_run_id FROM task_events e "
        "JOIN tasks t ON t.id = e.task_id "
        "WHERE e.kind IN ('delivery_owner_claimed', 'delivery_superseded') "
        "ORDER BY e.id ASC"
    ).fetchall()
    claims: dict[str, tuple[dict, sqlite3.Row]] = {}
    for row in rows:
        if row["kind"] == "delivery_superseded":
            claims.pop(row["task_id"], None)
            continue
        payload = _payload(row["payload"])
        if payload.get("repository") == repository:
            claims[row["task_id"]] = (payload, row)
    return {
        task_id: claim
        for task_id, claim in claims.items()
        if claim[1]["status"] in _ACTIVE_STATUSES
        or (
            claim[1]["status"] == "blocked"
            and str(claim[1]["branch_name"] or "").strip()
        )
    }


def _latest_owner_event(conn: sqlite3.Connection, task_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT id, kind, payload, run_id FROM task_events "
        "WHERE task_id = ? AND kind IN "
        "('delivery_owner_claimed', 'delivery_superseded') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()


def _conflicts(
    claims: dict[str, tuple[dict, sqlite3.Row]],
    *,
    task_id: str,
    seams: list[str],
) -> list[tuple[str, dict, list[str]]]:
    candidate = set(seams)
    found: list[tuple[str, dict, list[str]]] = []
    for owner_task_id, (payload, _row) in claims.items():
        if owner_task_id == task_id:
            continue
        overlap = sorted(candidate.intersection(payload.get("seams") or []))
        if overlap:
            found.append((owner_task_id, payload, overlap))
    return found


def _claim_payload(
    *,
    repository: str,
    branch: str,
    pr_number: int,
    seams: list[str],
    seam_source: str,
    changelog_overlap_reason: Optional[str],
    changelog_overlap_task_ids: list[str],
    supersedes_task_id: Optional[str] = None,
) -> dict:
    return {
        "repository": repository,
        "branch": branch,
        "pr_number": pr_number,
        "seams": seams,
        "seam_source": seam_source,
        "changelog_overlap_reason": changelog_overlap_reason,
        "changelog_overlap_task_ids": changelog_overlap_task_ids,
        "supersedes_task_id": supersedes_task_id,
    }


def claim_delivery_ownership(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    branch: str,
    pr_number: int,
    seams: Iterable[str],
    seam_source: str,
    expected_run_id: Optional[int],
    changelog_overlap_reason: Optional[str] = None,
) -> int:
    """Atomically claim repository seams for one current task/run."""
    from hermes_cli import kanban_db as kb

    repository = _normalize_repository(repository)
    branch = _normalize_branch(branch)
    pr_number = _normalize_pr_number(pr_number)
    normalized_seams = _normalize_seams(seams)
    seam_source = _normalize_seam_source(seam_source)
    reason = str(changelog_overlap_reason or "").strip() or None

    with kb.write_txn(conn):
        latest = _latest_owner_event(conn, task_id)
        if latest is not None and latest["kind"] == "delivery_superseded":
            raise DeliveryOwnershipError(task_id, "task was superseded")
        _require_current_repository_run(
            conn, task_id, expected_run_id, branch=branch
        )
        conflicts = _conflicts(
            _active_claims(conn, repository), task_id=task_id, seams=normalized_seams
        )
        non_changelog = [
            (owner_id, payload, overlap)
            for owner_id, payload, overlap in conflicts
            if overlap != [_CHANGELOG_SEAM]
        ]
        if non_changelog:
            owner_id, payload, overlap = non_changelog[0]
            raise DeliveryOwnershipError(
                task_id,
                f"overlaps owner {owner_id} branch={payload.get('branch')} "
                f"pr={payload.get('pr_number')} seams={','.join(overlap)}",
            )
        if conflicts and not reason:
            owner_id, _payload_value, _overlap = conflicts[0]
            raise DeliveryOwnershipError(
                task_id,
                f"CHANGELOG-only overlap with owner {owner_id} requires an audited reason",
            )
        payload = _claim_payload(
            repository=repository,
            branch=branch,
            pr_number=pr_number,
            seams=normalized_seams,
            seam_source=seam_source,
            changelog_overlap_reason=reason,
            changelog_overlap_task_ids=sorted(owner_id for owner_id, _, _ in conflicts),
        )
        kb._append_event(
            conn,
            task_id,
            "delivery_owner_claimed",
            payload,
            run_id=int(expected_run_id),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def assert_delivery_ownership(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    branch: Optional[str] = None,
    pr_number: Optional[int] = None,
    seams: Optional[Iterable[str]] = None,
    expected_run_id: Optional[int],
) -> bool:
    """Fail closed unless this current run is the sole owner of the seams."""
    repository = _normalize_repository(repository)
    normalized_branch = _normalize_branch(branch) if branch is not None else None
    normalized_pr = _normalize_pr_number(pr_number) if pr_number is not None else None
    normalized_seams = _normalize_seams(seams) if seams is not None else None
    _require_current_repository_run(
        conn, task_id, expected_run_id, branch=normalized_branch
    )
    latest = _latest_owner_event(conn, task_id)
    if latest is None:
        raise DeliveryOwnershipError(task_id, "missing delivery ownership claim")
    if latest["kind"] == "delivery_superseded":
        raise DeliveryOwnershipError(task_id, "task was superseded")
    if latest["run_id"] != expected_run_id:
        raise DeliveryOwnershipError(task_id, "ownership claim belongs to a stale run")
    payload = _payload(latest["payload"])
    if payload.get("repository") != repository:
        raise DeliveryOwnershipError(task_id, "repository does not match ownership claim")
    if normalized_branch is not None and payload.get("branch") != normalized_branch:
        raise DeliveryOwnershipError(task_id, "branch does not match ownership claim")
    if normalized_pr is not None and payload.get("pr_number") != normalized_pr:
        raise DeliveryOwnershipError(task_id, "PR does not match ownership claim")
    owned_seams = _normalize_seams(payload.get("seams") or [])
    if normalized_seams is not None and not set(normalized_seams).issubset(owned_seams):
        raise DeliveryOwnershipError(task_id, "diff contains unowned repository seams")
    conflicts = _conflicts(
        _active_claims(conn, repository), task_id=task_id, seams=owned_seams
    )
    for owner_id, _other, overlap in conflicts:
        if overlap != [_CHANGELOG_SEAM] or not payload.get("changelog_overlap_reason"):
            raise DeliveryOwnershipError(
                task_id,
                f"ownership is no longer exclusive; overlaps owner {owner_id}",
            )
    return True


def repository_has_active_delivery_owner(
    conn: sqlite3.Connection, repository: str
) -> bool:
    """Return whether a repository has any non-superseded active owner."""
    return bool(_active_claims(conn, _normalize_repository(repository)))


def supersede_delivery_owner(
    conn: sqlite3.Connection,
    source_task_id: str,
    target_task_id: str,
    *,
    repository: str,
    branch: str,
    pr_number: int,
    seams: Iterable[str],
    seam_source: str,
    actor: str,
    reason: str,
    expected_target_run_id: Optional[int],
) -> int:
    """Atomically transfer ownership and evidence to one replacement task."""
    from hermes_cli import kanban_db as kb

    if source_task_id == target_task_id:
        raise ValueError("source and target tasks must differ")
    repository = _normalize_repository(repository)
    branch = _normalize_branch(branch)
    pr_number = _normalize_pr_number(pr_number)
    normalized_seams = _normalize_seams(seams)
    seam_source = _normalize_seam_source(seam_source)
    actor = str(actor or "").strip()
    reason = str(reason or "").strip()
    if not actor or not reason:
        raise ValueError("supersession requires an actor and reason")

    with kb.write_txn(conn):
        target = _require_current_repository_run(
            conn,
            target_task_id,
            expected_target_run_id,
            branch=branch,
        )
        source = kb.get_task(conn, source_task_id)
        source_event = _latest_owner_event(conn, source_task_id)
        if source is None or source_event is None or source_event["kind"] != "delivery_owner_claimed":
            raise DeliveryOwnershipError(source_task_id, "source is not an active owner")
        source_payload = _payload(source_event["payload"])
        if source_payload.get("repository") != repository:
            raise DeliveryOwnershipError(source_task_id, "source repository does not match")
        if source.status not in {"running", "review", "blocked"}:
            raise DeliveryOwnershipError(source_task_id, "source is not an active owner")
        if actor not in {"default", source.assignee, target.assignee}:
            raise DeliveryOwnershipError(
                source_task_id, f"profile {actor!r} cannot supersede this owner"
            )
        overlap = sorted(
            set(source_payload.get("seams") or []).intersection(normalized_seams)
        )
        if not overlap:
            raise DeliveryOwnershipError(
                source_task_id, "replacement does not overlap the source seams"
            )
        source_required = list(source.required_evidence or [])
        target_required = list(target.required_evidence or [])
        reconciled = target_required + [
            name for name in source_required if name not in target_required
        ]
        source_run_id = source.current_run_id
        if source_run_id is not None:
            conn.execute(
                "UPDATE task_runs SET status = 'superseded', outcome = 'superseded', "
                "summary = ?, ended_at = ?, claim_lock = NULL, claim_expires = NULL, "
                "worker_pid = NULL WHERE id = ? AND ended_at IS NULL",
                (f"superseded by {target_task_id}: {reason}", int(time.time()), source_run_id),
            )
        conn.execute(
            "UPDATE tasks SET status = 'blocked', block_kind = 'dependency', "
            "branch_name = NULL, claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, current_run_id = NULL WHERE id = ?",
            (source_task_id,),
        )
        conn.execute(
            "UPDATE tasks SET required_evidence = ?, evidence_contract_na_reason = NULL "
            "WHERE id = ?",
            (json.dumps(reconciled), target_task_id),
        )
        kb._append_event(
            conn,
            source_task_id,
            "delivery_superseded",
            {
                "replacement_task_id": target_task_id,
                "actor": actor,
                "reason": reason,
                "repository": repository,
                "overlap": overlap,
                "source_required_evidence": source_required,
                "reconciled_required_evidence": reconciled,
            },
            run_id=source_run_id,
        )
        superseded_event_id = int(
            conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        )
        kb._append_event(
            conn,
            target_task_id,
            "delivery_owner_claimed",
            _claim_payload(
                repository=repository,
                branch=branch,
                pr_number=pr_number,
                seams=normalized_seams,
                seam_source=seam_source,
                changelog_overlap_reason=None,
                changelog_overlap_task_ids=[],
                supersedes_task_id=source_task_id,
            ),
            run_id=int(expected_target_run_id),
        )
        kb._append_event(
            conn,
            target_task_id,
            "delivery_supersession_accepted",
            {
                "source_task_id": source_task_id,
                "actor": actor,
                "reason": reason,
                "repository": repository,
                "required_evidence": reconciled,
            },
            run_id=int(expected_target_run_id),
        )
        claims = _active_claims(conn, repository)
        remaining = [
            owner_id
            for owner_id, (payload, _row) in claims.items()
            if set(payload.get("seams") or []).intersection(normalized_seams)
        ]
        if remaining != [target_task_id]:
            raise DeliveryOwnershipError(
                target_task_id,
                f"supersession left multiple owners: {','.join(sorted(remaining))}",
            )
        return superseded_event_id


def delivery_push_allowed(
    *, local_ref: str, remote_ref: str, remote_branch: str
) -> bool:
    """Protected branches accept PR merges, never direct git pushes."""
    del local_ref
    branch = _normalize_branch(remote_branch)
    return str(remote_ref or "").strip() != f"refs/heads/{branch}"

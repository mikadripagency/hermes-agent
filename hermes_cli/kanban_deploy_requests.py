"""Durable, task-scoped core deployment request lifecycle."""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
from typing import Any, Optional

from hermes_cli.kanban_review_gate import (
    ReviewGateError,
    assert_deploy_review_gate,
    assert_review_gate,
)

_EVENT_KINDS = (
    "core_deploy_requested",
    "core_deploy_claimed",
    "core_deploy_succeeded",
    "core_deploy_failed",
    "core_deploy_rejected",
)


class CoreDeployRequestError(ValueError):
    """A core deployment request cannot advance safely."""


def _payload(row: sqlite3.Row) -> dict[str, Any]:
    try:
        value = json.loads(row["payload"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _rows(conn: sqlite3.Connection, task_id: str) -> list[sqlite3.Row]:
    placeholders = ",".join("?" for _ in _EVENT_KINDS)
    return conn.execute(
        f"SELECT id, kind, payload FROM task_events WHERE task_id=? "
        f"AND kind IN ({placeholders}) ORDER BY id",
        (task_id, *_EVENT_KINDS),
    ).fetchall()


def _identity(repository: str, reviewed_sha: str, integration_sha: str) -> dict[str, str]:
    return {
        "repository": str(repository or "").strip(),
        "reviewed_sha": str(reviewed_sha or "").strip().lower(),
        "integration_sha": str(integration_sha or "").strip().lower(),
    }


def _state(rows: list[sqlite3.Row]) -> tuple[dict[int, dict], dict[int, tuple[int, str, dict]]]:
    requests: dict[int, dict] = {}
    states: dict[int, tuple[int, str, dict]] = {}
    for row in rows:
        event_id = int(row["id"])
        payload = _payload(row)
        if row["kind"] == "core_deploy_requested":
            requests[event_id] = payload
            continue
        request_event_id = payload.get("request_event_id")
        if isinstance(request_event_id, int):
            states[request_event_id] = (event_id, row["kind"], payload)
    return requests, states


def request_core_deploy(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    reviewed_sha: str,
    integration_sha: str,
    expected_run_id: Optional[int],
) -> int:
    """Authorize exactly one request for a reviewed integration identity."""
    from hermes_cli import kanban_db as kb

    identity = _identity(repository, reviewed_sha, integration_sha)
    with kb.write_txn(conn):
        assert_deploy_review_gate(
            conn,
            task_id,
            **identity,
            expected_run_id=expected_run_id,
        )
        requests, _ = _state(_rows(conn, task_id))
        for event_id, payload in requests.items():
            if _identity(
                payload.get("repository", ""),
                payload.get("reviewed_sha", ""),
                payload.get("integration_sha", ""),
            ) == identity:
                return event_id
        kb._append_event(
            conn,
            task_id,
            "core_deploy_requested",
            {**identity, "authorized_run_id": int(expected_run_id or 0)},
            run_id=int(expected_run_id or 0),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def claim_core_deploy(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    repository: str,
    reviewed_sha: str,
    integration_sha: str,
    claimant: str,
    lease_seconds: int = 300,
) -> dict[str, Any]:
    """Claim a durable request independently of the originating worker run."""
    from hermes_cli import kanban_db as kb

    identity = _identity(repository, reviewed_sha, integration_sha)
    if not str(claimant or "").strip():
        raise CoreDeployRequestError("claimant is required")
    if lease_seconds <= 0:
        raise CoreDeployRequestError("lease_seconds must be positive")

    rejection: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    with kb.write_txn(conn):
        rows = _rows(conn, task_id)
        requests, states = _state(rows)
        matching = [
            event_id
            for event_id, payload in requests.items()
            if _identity(
                payload.get("repository", ""),
                payload.get("reviewed_sha", ""),
                payload.get("integration_sha", ""),
            ) == identity
        ]
        if not matching:
            rejection = "request_identity_mismatch" if requests else "request_missing"
        else:
            request_event_id = matching[-1]
            try:
                assert_review_gate(
                    conn,
                    task_id,
                    repository=identity["repository"],
                    final_sha=identity["reviewed_sha"],
                )
                assert_review_gate(
                    conn,
                    task_id,
                    repository=identity["repository"],
                    final_sha=identity["integration_sha"],
                    require_integration=True,
                )
            except (ReviewGateError, ValueError) as exc:
                rejection = f"review_gate_changed:{exc}"
            if rejection is None:
                state = states.get(request_event_id)
                now = int(time.time())
                if state and state[1] == "core_deploy_succeeded":
                    raise CoreDeployRequestError("request already succeeded")
                if (
                    state
                    and state[1] == "core_deploy_claimed"
                    and int(state[2].get("claim_expires_at") or 0) > now
                ):
                    raise CoreDeployRequestError("request already claimed")
                claim_id = secrets.token_hex(16)
                kb._append_event(
                    conn,
                    task_id,
                    "core_deploy_claimed",
                    {
                        "request_event_id": request_event_id,
                        "claim_id": claim_id,
                        "claimant": str(claimant).strip(),
                        "claim_expires_at": now + lease_seconds,
                    },
                )
                result = {"request_event_id": request_event_id, "claim_id": claim_id}
        if rejection is not None:
            kb._append_event(
                conn,
                task_id,
                "core_deploy_rejected",
                {"reason": rejection, **identity},
            )
    if rejection is not None:
        if rejection == "request_identity_mismatch":
            raise CoreDeployRequestError("request identity mismatch")
        raise CoreDeployRequestError(rejection)
    assert result is not None
    return result


def finish_core_deploy(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    request_event_id: int,
    claim_id: str,
    status: str,
    receipt: Optional[dict[str, Any]] = None,
) -> int:
    """Durably acknowledge a claimed request; repeated identical ack is safe."""
    from hermes_cli import kanban_db as kb

    normalized_status = str(status or "").strip().casefold()
    if normalized_status not in {"succeeded", "failed"}:
        raise CoreDeployRequestError("status must be succeeded or failed")
    if not isinstance(receipt, dict):
        raise CoreDeployRequestError("receipt must be a JSON object")

    with kb.write_txn(conn):
        requests, states = _state(_rows(conn, task_id))
        if request_event_id not in requests:
            raise CoreDeployRequestError("request does not exist")
        state = states.get(request_event_id)
        terminal_kind = f"core_deploy_{normalized_status}"
        if state and state[1] == terminal_kind and state[2].get("claim_id") == claim_id:
            return state[0]
        if not state or state[1] != "core_deploy_claimed":
            raise CoreDeployRequestError("request has no active claim")
        if state[2].get("claim_id") != claim_id:
            raise CoreDeployRequestError("claim identity mismatch")
        kb._append_event(
            conn,
            task_id,
            terminal_kind,
            {
                "request_event_id": request_event_id,
                "claim_id": claim_id,
                "receipt": receipt,
            },
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

#!/usr/bin/env python3
"""Reject direct pushes to a repository's protected delivery branch."""

from __future__ import annotations

import argparse
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli.kanban_delivery_owner import delivery_push_allowed

_MARKER = "# hermes-kanban-delivery-pre-push"


def _hook_path(repo: Path) -> Path:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "not a git repository")
    common = Path(proc.stdout.strip())
    if not common.is_absolute():
        common = (repo / common).resolve()
    return common / "hooks" / "pre-push"


def install(repo: Path, protected_branch: str) -> Path:
    """Install the idempotent repo-wide hook without replacing foreign hooks."""
    hook = _hook_path(repo.resolve())
    if hook.exists() and _MARKER not in hook.read_text(encoding="utf-8"):
        raise RuntimeError(f"refusing to replace existing hook: {hook}")
    hook.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "#!/bin/sh\n"
        f"{_MARKER}\n"
        f"exec {shlex.quote(sys.executable)} "
        f"{shlex.quote(str(Path(__file__).resolve()))} "
        f"--protected-branch {shlex.quote(protected_branch)}\n"
    )
    hook.write_text(content, encoding="utf-8")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook


def check_updates(protected_branch: str) -> int:
    """Read Git's pre-push protocol and reject before the network side effect."""
    for raw in sys.stdin:
        fields = raw.split()
        if len(fields) != 4:
            print("delivery gate: malformed pre-push input", file=sys.stderr)
            return 1
        local_ref, _local_sha, remote_ref, _remote_sha = fields
        if not delivery_push_allowed(
            local_ref=local_ref,
            remote_ref=remote_ref,
            remote_branch=protected_branch,
        ):
            task_id = os.environ.get("HERMES_KANBAN_TASK", "missing")
            print(
                f"delivery gate: direct push to {protected_branch} rejected for "
                f"task {task_id}; protected delivery requires the reviewed PR merge gate",
                file=sys.stderr,
            )
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protected-branch", required=True)
    parser.add_argument("--install-repo", type=Path)
    args = parser.parse_args()
    if args.install_repo is not None:
        print(install(args.install_repo, args.protected_branch))
        return 0
    return check_updates(args.protected_branch)


if __name__ == "__main__":
    raise SystemExit(main())

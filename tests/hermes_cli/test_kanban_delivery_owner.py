"""Single-writer ownership contracts for repository delivery lanes."""

from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb

pytestmark = pytest.mark.usefixtures(
    "explicit_delivery_contract_for_kanban_fixtures",
    "claimed_completion_for_kanban_fixtures",
)


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _repository_task(conn, *, title: str, branch: str, evidence=None):
    task_id = kb.create_task(
        conn,
        title=title,
        assignee="developer",
        workspace_kind="worktree",
        workspace_path="/tmp/repository",
        branch_name=branch,
        required_evidence=evidence or ["pr_merged"],
    )
    task = kb.claim_task(conn, task_id, claimer=f"test:{branch}")
    assert task is not None and task.current_run_id is not None
    return task


@pytest.mark.parametrize(
    "repository",
    ("Pryapus/Drip-Research-Hub", "mikadripagency/hermes-agent"),
)
def test_competing_delivery_is_rejected_before_ownership_side_effect(
    kanban_home, repository
):
    with kb.connect() as conn:
        owner = _repository_task(conn, title="canonical", branch="delivery/canonical")
        rival = _repository_task(conn, title="recovery", branch="delivery/recovery")
        kb.claim_delivery_ownership(
            conn,
            owner.id,
            repository=repository,
            branch=owner.branch_name,
            pr_number=361,
            seams=["apps/reports/page.tsx", "CHANGELOG.md"],
            seam_source="diff",
            expected_run_id=owner.current_run_id,
        )
        before = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?",
            (rival.id,),
        ).fetchone()[0]

        with pytest.raises(kb.DeliveryOwnershipError, match=owner.id):
            kb.claim_delivery_ownership(
                conn,
                rival.id,
                repository=repository,
                branch=rival.branch_name,
                pr_number=362,
                seams=["apps/reports/page.tsx"],
                seam_source="declared",
                expected_run_id=rival.current_run_id,
            )

        after = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?",
            (rival.id,),
        ).fetchone()[0]
        assert after == before
        kb.record_review_receipt(
            conn,
            rival.id,
            repository=repository,
            reviewed_sha="a" * 40,
            verdict="passed",
            findings=[],
            expected_run_id=rival.current_run_id,
        )
        with pytest.raises(kb.DeliveryOwnershipError, match="missing delivery ownership"):
            kb.assert_review_gate(
                conn,
                rival.id,
                repository=repository,
                final_sha="a" * 40,
                expected_run_id=rival.current_run_id,
            )
        assert kb.assert_delivery_ownership(
            conn,
            owner.id,
            repository=repository,
            branch=owner.branch_name,
            pr_number=361,
            seams=["apps/reports/page.tsx", "CHANGELOG.md"],
            expected_run_id=owner.current_run_id,
        )


def test_changelog_only_overlap_requires_an_audited_reason(kanban_home):
    repository = "Pryapus/Drip-Research-Hub"
    with kb.connect() as conn:
        first = _repository_task(conn, title="first", branch="delivery/first")
        second = _repository_task(conn, title="second", branch="delivery/second")
        kb.claim_delivery_ownership(
            conn,
            first.id,
            repository=repository,
            branch=first.branch_name,
            pr_number=1,
            seams=["CHANGELOG.md"],
            seam_source="diff",
            expected_run_id=first.current_run_id,
        )
        with pytest.raises(kb.DeliveryOwnershipError, match="CHANGELOG"):
            kb.claim_delivery_ownership(
                conn,
                second.id,
                repository=repository,
                branch=second.branch_name,
                pr_number=2,
                seams=["CHANGELOG.md"],
                seam_source="diff",
                expected_run_id=second.current_run_id,
            )

        event_id = kb.claim_delivery_ownership(
            conn,
            second.id,
            repository=repository,
            branch=second.branch_name,
            pr_number=2,
            seams=["CHANGELOG.md"],
            seam_source="diff",
            changelog_overlap_reason="separate release notes; rebase before merge",
            expected_run_id=second.current_run_id,
        )
        payload = json.loads(
            conn.execute(
                "SELECT payload FROM task_events WHERE id = ?", (event_id,)
            ).fetchone()[0]
        )
        assert payload["changelog_overlap_reason"] == (
            "separate release notes; rebase before merge"
        )
        assert payload["changelog_overlap_task_ids"] == [first.id]


def test_supersession_reconciles_contract_and_leaves_one_owner(kanban_home):
    repository = "Pryapus/Drip-Research-Hub"
    with kb.connect() as conn:
        source = _repository_task(
            conn,
            title="old owner",
            branch="delivery/old",
            evidence=["pr_merged", "authenticated_production_e2e"],
        )
        target = _repository_task(
            conn,
            title="replacement",
            branch="delivery/new",
            evidence=["pr_merged", "runtime_smoke"],
        )
        kb.claim_delivery_ownership(
            conn,
            source.id,
            repository=repository,
            branch=source.branch_name,
            pr_number=361,
            seams=["apps/reports/page.tsx"],
            seam_source="diff",
            expected_run_id=source.current_run_id,
        )

        event_id = kb.supersede_delivery_owner(
            conn,
            source.id,
            target.id,
            repository=repository,
            branch=target.branch_name,
            pr_number=362,
            seams=["apps/reports/page.tsx"],
            seam_source="diff",
            actor="developer",
            reason="owner selected the replacement implementation",
            expected_target_run_id=target.current_run_id,
        )

        old = kb.get_task(conn, source.id)
        new = kb.get_task(conn, target.id)
        assert old is not None and old.status == "blocked"
        assert old.current_run_id is None and old.branch_name is None
        assert new is not None
        assert new.required_evidence == [
            "pr_merged",
            "runtime_smoke",
            "delivery_ownership",
            "authenticated_production_e2e",
        ]
        superseded = conn.execute(
            "SELECT kind, payload FROM task_events WHERE id = ?", (event_id,)
        ).fetchone()
        assert superseded["kind"] == "delivery_superseded"
        assert json.loads(superseded["payload"])["replacement_task_id"] == target.id
        assert kb.assert_delivery_ownership(
            conn,
            target.id,
            repository=repository,
            branch=target.branch_name,
            pr_number=362,
            seams=["apps/reports/page.tsx"],
            expected_run_id=target.current_run_id,
        )
        with pytest.raises(kb.DeliveryOwnershipError, match="superseded"):
            kb.claim_delivery_ownership(
                conn,
                source.id,
                repository=repository,
                branch="delivery/old",
                pr_number=361,
                seams=["apps/reports/page.tsx"],
                seam_source="diff",
                expected_run_id=source.current_run_id,
            )


def test_new_run_cannot_reuse_stale_ownership(kanban_home):
    repository = "mikadripagency/hermes-agent"
    with kb.connect() as conn:
        owner = _repository_task(conn, title="owner", branch="delivery/owner")
        assert owner.branch_name is not None
        kb.claim_delivery_ownership(
            conn,
            owner.id,
            repository=repository,
            branch=owner.branch_name,
            pr_number=42,
            seams=["hermes_cli/kanban.py"],
            seam_source="diff",
            expected_run_id=owner.current_run_id,
        )
        assert kb.block_task(
            conn,
            owner.id,
            reason="transient handoff",
            kind="transient",
            expected_run_id=owner.current_run_id,
        )
        assert kb.unblock_task(conn, owner.id)
        current = kb.claim_task(conn, owner.id, claimer="test:current")
        assert current is not None and current.current_run_id != owner.current_run_id

        with pytest.raises(kb.DeliveryOwnershipError, match="stale run"):
            kb.assert_delivery_ownership(
                conn,
                owner.id,
                repository=repository,
                branch=owner.branch_name,
                pr_number=42,
                seams=["hermes_cli/kanban.py"],
                expected_run_id=current.current_run_id,
            )


def test_protected_branch_push_is_never_a_supported_delivery_path():
    assert not kb.delivery_push_allowed(
        local_ref="refs/heads/main",
        remote_ref="refs/heads/main",
        remote_branch="main",
    )
    assert not kb.delivery_push_allowed(
        local_ref="refs/heads/deploy",
        remote_ref="refs/heads/deploy",
        remote_branch="deploy",
    )
    assert kb.delivery_push_allowed(
        local_ref="refs/heads/delivery/task",
        remote_ref="refs/heads/delivery/task",
        remote_branch="main",
    )


def test_cli_derives_seams_from_the_repository_diff(tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "base"], check=True)
    base = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    (repo / "src").mkdir()
    (repo / "src" / "feature.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "src/feature.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "feature"], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()

    seams, source = kc._delivery_seams(
        Namespace(seam=None, seams_from_diff=(str(repo), base, head))
    )

    assert seams == ["src/feature.py"]
    assert source == "diff"


def test_pre_push_hook_rejects_protected_ref_before_git_push():
    script = Path(__file__).parents[2] / "scripts" / "kanban_delivery_pre_push.py"
    blocked = subprocess.run(
        [sys.executable, str(script), "--protected-branch", "main"],
        input=f"refs/heads/main {'a' * 40} refs/heads/main {'b' * 40}\n",
        text=True,
        capture_output=True,
        check=False,
    )
    allowed = subprocess.run(
        [sys.executable, str(script), "--protected-branch", "main"],
        input=f"refs/heads/delivery/task {'a' * 40} refs/heads/delivery/task {'b' * 40}\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert blocked.returncode == 1
    assert "requires the reviewed PR merge gate" in blocked.stderr
    assert allowed.returncode == 0


def test_delivery_owner_cli_claims_and_checks_the_current_lane(kanban_home):
    with kb.connect() as conn:
        task = _repository_task(
            conn,
            title="CLI owner",
            branch="delivery/cli-owner",
        )

    claimed = kc.run_slash(
        f"delivery-own {task.id} --repository mikadripagency/hermes-agent "
        f"--branch {task.branch_name} --pr-number 42 --seam hermes_cli/kanban.py"
    )
    gated = kc.run_slash(
        f"delivery-gate {task.id} --repository mikadripagency/hermes-agent "
        f"--branch {task.branch_name} --pr-number 42 --seam hermes_cli/kanban.py"
    )

    assert "DELIVERY_OWNER_CLAIMED" in claimed
    assert "DELIVERY_OWNER_GATE_PASS" in gated

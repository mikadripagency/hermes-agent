"""Tests for the updates.pin_branch guard in ``hermes update``.

These exercise the branch-resolution logic that decides whether the updater
switches to the default target branch or stays pinned to the current
deployment/fork branch. They use a real temporary git repo (no network, no
real update) plus config monkeypatching — the whole point of the guard is to
never switch a pinned checkout to main and destroy local fixes.
"""

import subprocess

import pytest

from hermes_cli import main as cli_main
from hermes_cli.main import (
    _PinnedBranchError,
    _branch_upstream,
    _resolve_update_remote,
    _update_branch_is_pinned,
)

GIT = ["git"]


def _run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo_with_upstream(tmp_path):
    """Local repo whose ``deploy`` branch tracks ``fork/deploy``.

    Mirrors this machine's layout: a pinned deployment branch with a
    configured non-origin upstream remote.
    """
    upstream = tmp_path / "upstream.git"
    upstream.mkdir()
    _run(GIT + ["init", "--bare", "-b", "deploy"], upstream)

    work = tmp_path / "work"
    work.mkdir()
    _run(GIT + ["init", "-b", "main"], work)
    _run(GIT + ["config", "user.email", "t@t.t"], work)
    _run(GIT + ["config", "user.name", "t"], work)
    (work / "f.txt").write_text("hi\n")
    _run(GIT + ["add", "."], work)
    _run(GIT + ["commit", "-m", "init"], work)
    _run(GIT + ["remote", "add", "fork", str(upstream)], work)
    _run(GIT + ["checkout", "-b", "deploy"], work)
    _run(GIT + ["push", "-u", "fork", "deploy"], work)
    return work


class TestBranchUpstream:
    def test_resolves_configured_upstream(self, repo_with_upstream):
        assert _branch_upstream(GIT, repo_with_upstream, "deploy") == (
            "fork",
            "deploy",
            "fork/deploy",
        )

    def test_returns_none_without_upstream(self, repo_with_upstream, tmp_path):
        _run(GIT + ["checkout", "-b", "orphan"], repo_with_upstream)
        assert _branch_upstream(GIT, repo_with_upstream, "orphan") is None


class TestResolveUpdateRemote:
    def test_default_when_not_pinned(self, repo_with_upstream):
        # Not pinned: keep historical origin/<target> behavior even though the
        # checkout is on `deploy` (the caller will switch to main as before).
        assert _resolve_update_remote(
            GIT, repo_with_upstream, "main", "deploy", pinned=False
        ) == ("origin", "main", "origin/main")

    def test_pinned_uses_current_branch_upstream(self, repo_with_upstream):
        assert _resolve_update_remote(
            GIT, repo_with_upstream, "main", "deploy", pinned=True
        ) == ("fork", "deploy", "fork/deploy")

    def test_pinned_on_target_stays_default(self, repo_with_upstream):
        # Already on the target branch → nothing to pin against.
        assert _resolve_update_remote(
            GIT, repo_with_upstream, "main", "main", pinned=True
        ) == ("origin", "main", "origin/main")

    def test_pinned_detached_head_aborts(self, repo_with_upstream):
        # Detached HEAD under pinning must refuse: switching to origin/main
        # would abandon the checked-out (fix-carrying) commit. Regression guard
        # for the 2026-07-15 incident where the updater ran upstream code.
        with pytest.raises(_PinnedBranchError) as exc:
            _resolve_update_remote(
                GIT, repo_with_upstream, "main", "HEAD", pinned=True
            )
        assert "refusing to switch to main" in str(exc.value)
        assert exc.value.current_branch == "HEAD"

    def test_detached_head_not_pinned_stays_default(self, repo_with_upstream):
        # With pinning OFF, detached HEAD keeps the historical default (the
        # caller switches to the target branch as before).
        assert _resolve_update_remote(
            GIT, repo_with_upstream, "main", "HEAD", pinned=False
        ) == ("origin", "main", "origin/main")

    def test_pinned_without_upstream_raises(self, repo_with_upstream):
        _run(GIT + ["checkout", "-b", "orphan"], repo_with_upstream)
        with pytest.raises(_PinnedBranchError) as exc:
            _resolve_update_remote(
                GIT, repo_with_upstream, "main", "orphan", pinned=True
            )
        assert "refusing to switch to main" in str(exc.value)
        assert exc.value.current_branch == "orphan"


class TestPinBranchConfig:
    def test_default_is_false(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {},
        )
        assert _update_branch_is_pinned() is False

    def test_reads_true(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"updates": {"pin_branch": True}},
        )
        assert _update_branch_is_pinned() is True

    def test_schema_default_present_and_false(self):
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["updates"]["pin_branch"] is False

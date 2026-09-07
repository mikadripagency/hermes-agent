"""Delegated reviewer subprocesses must not inherit Kanban authority."""

from __future__ import annotations

import os
from unittest.mock import patch

from tools.delegate_tool import _run_single_child, _strip_blocked_tools
from tools.environments.local import (
    _make_run_env,
    _sanitize_subprocess_env,
    hermes_subprocess_env,
)


_KANBAN_AUTHORITY_ENV = {
    "HERMES_KANBAN_TASK": "t_parent",
    "HERMES_KANBAN_RUN_ID": "710",
    "HERMES_KANBAN_CLAIM_LOCK": "parent-claim",
    "HERMES_KANBAN_DB": "/live/kanban.db",
    "HERMES_KANBAN_BOARD": "production",
}


class _CapturingChild:
    def __init__(self, subagent_id: str, nested: "_CapturingChild | None" = None):
        self._subagent_id = subagent_id
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self._delegate_saved_tool_names = []
        self._credential_pool = None
        self._current_task_id = None
        self._active_children: list[object] = []
        self._active_children_lock = None
        self.tool_progress_callback = None
        self.model = "test/model"
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_reasoning_tokens = 0
        self.nested = nested
        self.captured: dict[str, dict[str, str]] = {}
        self.nested_result = None
        self.closed = False

    def get_activity_summary(self):
        return {"api_call_count": 1, "max_iterations": 2, "current_tool": None}

    def run_conversation(self, **_kwargs):
        self.captured = {
            "foreground": _make_run_env({}),
            "background": _sanitize_subprocess_env(dict(os.environ)),
            "non_terminal": hermes_subprocess_env(inherit_credentials=False),
        }
        if self.nested is not None:
            self.nested_result = _run_single_child(
                task_index=1,
                goal="nested reviewer",
                child=self.nested,
                parent_agent=self,
            )
        return {
            "final_response": "review complete",
            "completed": True,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

    def close(self):
        self.closed = True


class _Parent:
    _current_task_id = None
    _active_children: list[object] = []
    _active_children_lock = None

    @staticmethod
    def _touch_activity(_description):
        return None


def test_reviewer_and_nested_subreviewer_process_env_is_sanitized():
    inherited = {
        **_KANBAN_AUTHORITY_ENV,
        "HERMES_HOME": "/profiles/developer",
        "HERMES_SESSION_SOURCE": "cli",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    nested = _CapturingChild("sa-1-nested")
    reviewer = _CapturingChild("sa-0-reviewer", nested=nested)

    with patch.dict(os.environ, inherited, clear=True):
        parent_env = _make_run_env({})
        result = _run_single_child(
            task_index=0,
            goal="review",
            child=reviewer,
            parent_agent=_Parent(),
        )
        restored_env = _make_run_env({})

    assert result["status"] == "completed"
    assert reviewer.nested_result is not None
    assert reviewer.nested_result["status"] == "completed"
    assert reviewer.closed and nested.closed
    for name, value in _KANBAN_AUTHORITY_ENV.items():
        assert parent_env[name] == value
        assert restored_env[name] == value
    for child in (reviewer, nested):
        for launched_env in child.captured.values():
            assert launched_env["_HERMES_DISPOSABLE_AGENT"] == "1"
            for name in _KANBAN_AUTHORITY_ENV:
                assert name not in launched_env


def test_reviewer_cannot_inherit_in_process_kanban_tools():
    assert _strip_blocked_tools(["terminal", "kanban"]) == ["terminal"]

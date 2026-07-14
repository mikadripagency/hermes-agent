"""Tests for agent/system_prompt.py — context-file cwd wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _captured_context_cwd(agent):
    """The cwd build_system_prompt_parts hands to build_context_files_prompt."""
    captured = {}

    def fake_context_files(cwd=None, skip_soul=False, context_length=None):
        captured["cwd"] = cwd
        return ""

    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", side_effect=fake_context_files),
    ):
        build_system_prompt_parts(agent)
    return captured["cwd"]


class TestContextFileCwd:
    def test_none_when_terminal_cwd_unset(self, monkeypatch):
        # Unset → None, so discovery falls back to the launch dir inside
        # build_context_files_prompt (the local-CLI #19242 contract).
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert _captured_context_cwd(_make_agent()) is None

    def test_configured_dir_when_terminal_cwd_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        assert _captured_context_cwd(_make_agent()) == tmp_path


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


def _init_code_repo(path):
    """A git repo that actually holds code — the coding posture requires a source
    file (or manifest), not a bare ``.git`` (a prose/notes repo stays general)."""
    import subprocess

    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    (path / "main.py").write_text("print('hi')\n")


class TestCodingContextBlock:
    def test_injected_when_active(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        stable = _stable_prompt(agent)
        assert "coding agent" in stable
        assert "Workspace" in stable

    def test_absent_when_off(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=["read_file"], platform="cli")
        # Drive the real path: force the resolved mode to "off" via config.
        with patch("agent.coding_context._coding_mode", return_value="off"):
            stable = _stable_prompt(agent)
        assert "coding agent" not in stable

    def test_absent_without_tools(self, monkeypatch, tmp_path):
        _init_code_repo(tmp_path)
        monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
        agent = _make_agent(valid_tool_names=[], platform="cli")
        assert "coding agent" not in _stable_prompt(agent)


class TestKanbanGuidance:
    """The full worker protocol is worker-only; ordinary chat with the kanban
    toolset gets a short orchestrator blurb instead (token cost regression)."""

    def test_worker_context_gets_full_protocol(self, monkeypatch):
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc123")
        agent = _make_agent(
            valid_tool_names=["kanban_show"], _kanban_worker_guidance=None,
        )
        prompt = _stable_prompt(agent)
        assert "Kanban task execution protocol" in prompt
        assert "kanban_heartbeat" in prompt

    def test_orchestrator_context_gets_short_blurb_only(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        agent = _make_agent(
            valid_tool_names=["kanban_show"], _kanban_worker_guidance=None,
        )
        prompt = _stable_prompt(agent)
        assert "Kanban orchestration" in prompt
        # The ~5KB worker protocol must NOT leak into ordinary chat sessions.
        assert "Kanban task execution protocol" not in prompt

    def test_no_kanban_tools_no_guidance(self, monkeypatch):
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        agent = _make_agent(valid_tool_names=[], _kanban_worker_guidance=None)
        prompt = _stable_prompt(agent)
        assert "Kanban orchestration" not in prompt
        assert "Kanban task execution protocol" not in prompt

    def test_preresolved_guidance_used_verbatim(self):
        from agent.prompt_builder import KANBAN_ORCHESTRATOR_GUIDANCE
        agent = _make_agent(
            valid_tool_names=["kanban_show"],
            _kanban_worker_guidance=KANBAN_ORCHESTRATOR_GUIDANCE,
        )
        assert "Kanban orchestration" in _stable_prompt(agent)


def test_orchestrator_guidance_is_much_shorter_and_has_routing_verbs():
    from agent.prompt_builder import KANBAN_GUIDANCE, KANBAN_ORCHESTRATOR_GUIDANCE
    # The whole point is a smaller prompt: well under half the worker protocol.
    assert len(KANBAN_ORCHESTRATOR_GUIDANCE) < len(KANBAN_GUIDANCE) / 2
    for verb in ("kanban_create", "kanban_list", "kanban_show", "kanban_comment"):
        assert verb in KANBAN_ORCHESTRATOR_GUIDANCE

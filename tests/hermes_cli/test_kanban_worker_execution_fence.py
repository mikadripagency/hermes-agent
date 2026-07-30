from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "developer").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _delivery(conn, *, assignee: str = "developer") -> str:
    return kb.create_task(
        conn,
        title="profile-fenced delivery",
        assignee=assignee,
        evidence_contract_na_reason="runtime identity regression",
    )


def test_claim_rejects_execution_profile_mismatch_before_run_creation(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = _delivery(conn)

        assert kb.claim_task(conn, task_id, execution_profile="default") is None

        task = kb.get_task(conn, task_id)
        runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)
        ).fetchone()[0]
        rejected = [event for event in kb.list_events(conn, task_id) if event.kind == "claim_rejected"]

    assert task is not None and task.status == "ready"
    assert task.claim_lock is None and task.current_run_id is None
    assert runs == 0
    assert rejected[-1].payload == {
        "reason": "execution_profile_mismatch",
        "assignee": "developer",
        "execution_profile": "default",
    }


def test_claim_materializes_authoritative_execution_profile(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = _delivery(conn)

        task = kb.claim_task(conn, task_id, execution_profile="developer")

        assert task is not None and task.current_run_id is not None
        run = kb.get_run(conn, task.current_run_id)

    assert run is not None and run.profile == "developer"


def test_review_claim_uses_same_execution_profile_fence(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))

        assert kb.claim_review_task(
            conn, task_id, execution_profile="default"
        ) is None
        task = kb.get_task(conn, task_id)
        runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)
        ).fetchone()[0]

    assert task is not None and task.status == "review"
    assert task.current_run_id is None and runs == 0


def test_legacy_wrong_profile_route_conflict_cannot_rollback_containment_block(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kb, "_resolve_orchestration_completion_route", lambda: ("slack", "C-ORCH")
    )
    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None
        with kb.write_txn(conn):
            # Historical pre-fence shape: the durable task/route owner is
            # developer, but the already-running worker was stamped default.
            conn.execute(
                "UPDATE task_runs SET profile='default' WHERE id=?",
                (task.current_run_id,),
            )
            conn.execute(
                "INSERT INTO kanban_notify_subs "
                "(task_id, platform, chat_id, thread_id, notifier_profile, created_at) "
                "VALUES (?, 'slack', 'C-ORCH', '', 'developer', 1)",
                (task_id,),
            )

        assert kb.block_task(
            conn,
            task_id,
            reason="owner capability required",
            kind="needs_input",
            expected_run_id=task.current_run_id,
        )

        blocked = kb.get_task(conn, task_id)
        route = conn.execute(
            "SELECT notifier_profile FROM kanban_notify_subs WHERE task_id=?",
            (task_id,),
        ).fetchone()
        deliveries = conn.execute(
            "SELECT state, notifier_profile FROM completion_deliveries WHERE task_id=?",
            (task_id,),
        ).fetchall()

    assert blocked is not None and blocked.status == "blocked"
    assert route["notifier_profile"] == "developer"
    assert [(row["state"], row["notifier_profile"]) for row in deliveries] == [
        ("pending", "developer")
    ]


def test_stale_run_terminal_tool_is_rejected_before_side_effect(
    kanban_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        stale = kb.claim_task(conn, task_id, execution_profile="developer")
        assert stale is not None and stale.current_run_id is not None
        stale_run_id = stale.current_run_id
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', ended_at=2 "
                "WHERE id=?",
                (stale_run_id,),
            )
            replacement = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, 'developer', 'running', 3)",
                (task_id,),
            )
            conn.execute(
                "UPDATE tasks SET current_run_id=? WHERE id=?",
                (replacement.lastrowid, task_id),
            )

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    marker = tmp_path / "stale-side-effect"

    from model_tools import handle_function_call

    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": f"python3 -c \"from pathlib import Path; Path({str(marker)!r}).touch()\""},
        )
    )

    assert "stale kanban worker run" in result["error"]
    assert not marker.exists()


def test_wrong_profile_terminal_tool_is_rejected_before_side_effect(
    kanban_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    marker = tmp_path / "wrong-profile-side-effect"

    from model_tools import handle_function_call

    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": f"python3 -c \"from pathlib import Path; Path({str(marker)!r}).touch()\""},
        )
    )

    assert "kanban worker profile mismatch" in result["error"]
    assert not marker.exists()


def test_current_intended_profile_can_execute_tool(
    kanban_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    marker = tmp_path / "current-worker-side-effect"

    from model_tools import handle_function_call

    result = json.loads(
        handle_function_call(
            "terminal",
            {"command": f"python3 -c \"from pathlib import Path; Path({str(marker)!r}).touch()\""},
        )
    )

    assert result["exit_code"] == 0
    assert marker.exists()


@pytest.mark.parametrize(
    ("transition_name", "transition_args"),
    [("kanban_block", {}), ("tool_call", {"name": "kanban_block"})],
)
def test_concurrent_transition_serializes_before_side_effect(
    kanban_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transition_name: str,
    transition_args: dict,
) -> None:
    import threading

    from agent.kanban_worker_fence import fence_kanban_worker_tool

    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    transitioned = threading.Event()
    marker = tmp_path / "raced-side-effect"
    results = {}

    @fence_kanban_worker_tool(name_arg_index=0)
    def transition(_function_name: str, _function_args: dict) -> str:
        with kb.connect_closing() as conn:
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task_id,))
        transitioned.set()
        return "blocked"

    @fence_kanban_worker_tool(name_arg_index=0)
    def side_effect(_function_name: str) -> str:
        marker.touch()
        return "ran"

    first = threading.Thread(
        target=lambda: results.setdefault(
            "transition", transition(transition_name, transition_args)
        )
    )
    first.start()
    assert transitioned.wait(timeout=5)
    second = threading.Thread(
        target=lambda: results.setdefault("side_effect", side_effect("terminal"))
    )
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert results["transition"] == "blocked"
    assert "stale kanban worker run" in json.loads(results["side_effect"])["error"]
    assert not marker.exists()


def test_sequential_path_rejects_stale_run_before_middleware(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from agent.tool_executor import execute_tool_calls_sequential

    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id + 1))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="memory", arguments='{"action":"add"}'),
    )
    assistant_message = SimpleNamespace(tool_calls=[tool_call])
    messages = []

    execute_tool_calls_sequential(object(), assistant_message, messages, "", 0)

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert "stale kanban worker run" in messages[0]["content"]


def test_concurrent_path_rejects_stale_run_before_middleware(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from agent import tool_executor

    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id + 1))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    middleware_ran = False

    def middleware(*args, **kwargs):
        nonlocal middleware_ran
        middleware_ran = True
        return kwargs["function_args"], []

    monkeypatch.setattr(
        tool_executor, "_apply_tool_request_middleware_for_agent", middleware
    )
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments='{"path":"x"}'),
    )
    messages = []
    agent = MagicMock()
    agent._interrupt_requested = False

    tool_executor.execute_tool_calls_concurrent(
        agent, SimpleNamespace(tool_calls=[tool_call]), messages, "", 0
    )

    assert middleware_ran is False
    assert len(messages) == 1
    assert "stale kanban worker run" in messages[0]["content"]


def test_sequential_mixed_transition_batch_is_rejected_before_any_effect(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from agent.tool_executor import execute_tool_calls_sequential

    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    assistant_message = SimpleNamespace(tool_calls=[
        SimpleNamespace(
            id="complete",
            function=SimpleNamespace(name="kanban_complete", arguments="{}"),
        ),
        SimpleNamespace(
            id="effect",
            function=SimpleNamespace(name="terminal", arguments='{"command":"true"}'),
        ),
    ])
    messages = []

    execute_tool_calls_sequential(object(), assistant_message, messages, "", 0)

    assert len(messages) == 2
    assert all("must be called alone" in message["content"] for message in messages)
    with kb.connect_closing() as conn:
        current = kb.get_task(conn, task_id)
    assert current is not None and current.status == "running"


def test_run_transition_waits_until_admitted_effect_finishes(
    kanban_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from agent.kanban_worker_fence import fence_kanban_worker_tool

    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    admitted = threading.Event()
    release = threading.Event()
    marker = tmp_path / "leased-effect"
    results = {}

    @fence_kanban_worker_tool(name_arg_index=0)
    def side_effect(_function_name: str) -> str:
        admitted.set()
        assert release.wait(timeout=5)
        marker.touch()
        return "ran"

    effect_thread = threading.Thread(
        target=lambda: results.setdefault("effect", side_effect("terminal"))
    )
    effect_thread.start()
    assert admitted.wait(timeout=5)

    def transition() -> None:
        with kb.connect_closing() as conn:
            results["blocked"] = kb.block_task(
                conn,
                task_id,
                reason="takeover",
                kind="capability",
                expected_run_id=task.current_run_id,
            )

    transition_thread = threading.Thread(target=transition)
    transition_thread.start()
    transition_thread.join(timeout=0.1)
    with kb.connect_closing() as conn:
        still_running = kb.get_task(conn, task_id)

    assert transition_thread.is_alive()
    assert still_running is not None and still_running.status == "running"
    assert not marker.exists()

    release.set()
    effect_thread.join(timeout=5)
    transition_thread.join(timeout=5)

    assert results == {"effect": "ran", "blocked": True}
    assert marker.exists()
    with kb.connect_closing() as conn:
        blocked_task = kb.get_task(conn, task_id)
        assert blocked_task is not None and blocked_task.status == "blocked"


@pytest.mark.live_system_guard_bypass
def test_kanban_worker_rejects_background_processes(
    kanban_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os
    import time

    from tools.terminal_tool import terminal_tool

    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setattr(os, "getpgrp", os.getpid)

    result = json.loads(
        terminal_tool(
            "python3 -c 'import os,time; os.setsid(); time.sleep(30)'",
            task_id="test",
            workdir=str(tmp_path),
            background=True,
        )
    )

    assert result["status"] == "blocked"
    assert "cannot detach" in result["error"]

    detached = json.loads(
        terminal_tool(
            "python3 -c 'import os,time; os.setsid(); time.sleep(30)'",
            task_id="test",
            workdir=str(tmp_path),
        )
    )
    assert detached["status"] == "blocked"
    assert "cannot detach" in detached["error"]

    from tools.code_execution_tool import execute_code

    nested = json.loads(execute_code("import os; os.fork()"))
    assert nested["status"] == "blocked"
    assert "cannot spawn or detach" in nested["error"]

    marker = tmp_path / "escaped"
    script = tmp_path / "detach.py"
    script.write_text(
        "from os import fork,setsid,close,_exit\n"
        "from time import sleep\n"
        "p=fork()\n"
        "if p: _exit(0)\n"
        "setsid(); close(1); close(2); sleep(3)\n"
        f"open({str(marker)!r},'w').write('escaped')\n",
        encoding="utf-8",
    )
    indirect = json.loads(
        terminal_tool(f"python3 {script}", task_id="test", workdir=str(tmp_path))
    )
    assert indirect["exit_code"] == -1
    assert "cannot detach" in indirect["error"]
    time.sleep(0.4)
    assert not marker.exists()


def test_non_transition_tools_remain_parallel(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from agent.kanban_worker_fence import fence_kanban_worker_tool

    with kb.connect_closing() as conn:
        task_id = _delivery(conn)
        task = kb.claim_task(conn, task_id, execution_profile="developer")
        assert task is not None and task.current_run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kanban_home / "kanban.db"))
    monkeypatch.setenv("HERMES_PROFILE", "developer")
    monkeypatch.setenv("HERMES_HOME", str(kanban_home / "profiles" / "developer"))
    rendezvous = threading.Barrier(2, timeout=5)
    results = []

    @fence_kanban_worker_tool(name_arg_index=0)
    def read_tool(_function_name: str) -> str:
        rendezvous.wait()
        return "ok"

    threads = [
        threading.Thread(target=lambda: results.append(read_tool("read_file")))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert results == ["ok", "ok"]

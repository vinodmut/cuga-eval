"""Tests for AppWorld external agent adapters."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.appworld.agents.base import APPWORLD_AGENT_PROMPT, AppWorldInvokeResult
from benchmarks.appworld.agents.factory import EXTERNAL_AGENT_NAMES, create_appworld_agent
from benchmarks.appworld.agents.final_answer import (
    format_appworld_final_answer,
    maybe_format_appworld_final_answer,
    skip_final_answer_format,
)
from benchmarks.appworld.agents.tool_loop import (
    extract_final_answer,
    extract_tool_request,
    run_tool_react_loop,
)

pytestmark = pytest.mark.sanity


class _MockTool:
    def __init__(self, name: str, result: Any = "ok"):
        self.name = name
        self.description = f"Mock {name}"
        self._result = result

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        return {"tool": self.name, "args": args, "result": self._result}


@pytest.mark.parametrize("agent_name", sorted(EXTERNAL_AGENT_NAMES))
def test_factory_creates_agent(agent_name: str):
    tools = [_MockTool("supervisor_login")]
    agent = create_appworld_agent(agent_name, tools=tools, max_steps=3)
    assert agent is not None
    assert hasattr(agent, "invoke")


def test_appworld_invoke_result_shape():
    result = AppWorldInvokeResult(
        answer="done",
        tool_calls=[{"name": "t1", "arguments": {}, "result": "ok"}],
        react_steps=2,
    )
    assert result.answer == "done"
    assert len(result.tool_calls) == 1
    assert result.react_steps == 2
    assert result.error is None


def test_shared_prompt_not_empty():
    assert "Never invent or guess values" in APPWORLD_AGENT_PROMPT
    assert "Pagination" in APPWORLD_AGENT_PROMPT
    assert "page_index" in APPWORLD_AGENT_PROMPT
    assert "filter parameters" in APPWORLD_AGENT_PROMPT


def test_extract_tool_request():
    text = 'Thought\n```json\n{"action": "tool", "tool_name": "login", "args": {"x": 1}}\n```'
    parsed = extract_tool_request(text)
    assert parsed == ("login", {"x": 1})


def test_extract_final_answer():
    assert extract_final_answer("Reasoning\nFinal Answer: hello world") == "hello world"


@pytest.mark.asyncio
async def test_format_appworld_final_answer_strips_markdown_count():
    intent = "How many priority-1 unread email threads are in my Gmail inbox?"
    raw = "**1**"

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="answer: 1"))

    formatted = await format_appworld_final_answer(intent, raw, llm=mock_llm)
    assert formatted == "1"
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_format_appworld_final_answer_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("APPWORLD_SKIP_FINAL_ANSWER_FORMAT", "1")
    assert skip_final_answer_format() is True
    result = await maybe_format_appworld_final_answer("intent", "**1**")
    assert result == "**1**"


@pytest.mark.asyncio
async def test_run_tool_react_loop_returns_result_shape():
    tools = [_MockTool("fetch_data")]

    call_count = 0

    async def fake_llm_two_step(convo, invoke_callbacks=None):
        nonlocal call_count
        del convo, invoke_callbacks
        call_count += 1
        if call_count == 1:
            return '```json\n{"action": "tool", "tool_name": "fetch_data", "args": {}}\n```'
        return "Final Answer: completed"

    result = await run_tool_react_loop(
        tools=tools,
        system_prompt="test",
        intent="do task",
        user_context="ctx",
        call_llm=fake_llm_two_step,
        max_steps=3,
    )
    assert isinstance(result, AppWorldInvokeResult)
    assert result.answer == "completed"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "fetch_data"


@pytest.mark.asyncio
async def test_deepagents_adapter_invoke():
    import sys

    from langchain_core.messages import AIMessage

    tools = [_MockTool("t1")]

    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Final answer here")]})

    mock_deepagents = MagicMock()
    mock_deepagents.create_deep_agent = MagicMock(return_value=mock_agent)

    with patch.dict(sys.modules, {"deepagents": mock_deepagents}):
        with patch("benchmarks.appworld.agents.deepagents.create_eval_llm", return_value=MagicMock()):
            from benchmarks.appworld.agents.deepagents import DeepAgentsAppWorldAgent

            agent = DeepAgentsAppWorldAgent(tools=tools)
            agent._agent = None
            result = await agent.invoke(
                intent="test task",
                thread_id="thread-1",
                user_context="supervisor info",
            )

    assert isinstance(result, AppWorldInvokeResult)
    assert result.answer == "Final answer here"


@pytest.mark.asyncio
async def test_hermes_adapter_runs_native_cli_and_parses_session(tmp_path, monkeypatch):
    import json

    from benchmarks.appworld.agents.hermes import HermesAppWorldAgent

    binary = tmp_path / "hermes"
    binary.touch()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    session = {
        "id": "session-1",
        "started_at": 10,
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 50,
        "cache_write_tokens": 10,
        "api_call_count": 2,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "tool_call", "arguments": '{"calls": []}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-2",
                        "function": {
                            "name": "tool_call",
                            "arguments": json.dumps(
                                {
                                    "calls": [
                                        {
                                            "name": "mcp__appworld__supervisor__complete_task",
                                            "arguments": {"status": "success", "answer": "15"},
                                        }
                                    ]
                                }
                            ),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "ok"},
            {"role": "assistant", "content": "15 with extra explanation"},
        ],
    }

    async def fake_process(*args, cwd, env, timeout):
        del cwd, env, timeout
        if args[0] == "sessions":
            from pathlib import Path

            Path(args[2]).write_text(json.dumps(session) + "\n", encoding="utf-8")
            return 0, "exported"
        assert args[:2] == ("--yolo", "chat")
        assert args[args.index("--toolsets") + 1] == "appworld"
        assert args[args.index("--max-turns") + 1] == "25"
        return 0, "session_id: session-1\n15\n"

    agent = HermesAppWorldAgent(
        tools=[_MockTool("ignored")],
        hermes_binary=binary,
        runs_dir=tmp_path / "runs",
    )
    agent._run_process = AsyncMock(side_effect=fake_process)
    result = await agent.invoke(intent="count emails", thread_id="t1", user_context="ctx")

    assert result.answer == "15"
    assert result.react_steps == 3
    assert [call["name"] for call in result.tool_calls] == ["tool_call", "tool_call"]
    assert result.metrics["total_tokens"] == 120
    assert result.metrics["total_llm_calls"] == 2
    run_config = json.loads((tmp_path / "runs" / "t1" / "home" / "config.yaml").read_text())
    assert run_config["toolsets"] == ["appworld"]
    assert run_config["mcp_servers"]["appworld"]["url"].endswith("/mcp/")


def test_factory_unknown_agent_raises():
    with pytest.raises(ValueError, match="Unknown external agent"):
        create_appworld_agent("unknown", tools=[])


async def test_stub_template_runs_a_tool_and_returns_an_answer():
    """The copy-me template in stub.py must actually work end to end.

    A template nobody runs rots. This drives StubAppWorldAgent with a scripted
    model: one tool call, then a final answer. If this breaks, every adapter
    copied from it starts broken too.
    """
    from benchmarks.appworld.agents.stub import StubAppWorldAgent

    replies = iter(
        [
            '```json\n{"action": "tool", "tool_name": "supervisor_login", "args": {}}\n```',
            "Final Answer: logged in",
        ]
    )

    agent = StubAppWorldAgent(tools=[_MockTool("supervisor_login")], max_steps=4)
    agent._llm = MagicMock(ainvoke=AsyncMock(side_effect=lambda *a, **k: MagicMock(content=next(replies))))

    result = await agent.invoke(intent="log in", thread_id="t1", user_context="")

    assert result.answer == "logged in"
    assert [call["name"] for call in result.tool_calls] == ["supervisor_login"]
    assert result.error is None


async def test_stub_template_forwards_callbacks_to_the_model():
    """Dropping the callbacks silently zeroes the agent's token and cost columns."""
    from benchmarks.appworld.agents.stub import StubAppWorldAgent

    ainvoke = AsyncMock(return_value=MagicMock(content="Final Answer: done"))
    agent = StubAppWorldAgent(tools=[_MockTool("noop")], max_steps=2)
    agent._llm = MagicMock(ainvoke=ainvoke)
    sentinel = object()

    await agent.invoke(intent="x", thread_id="t", config={"callbacks": [sentinel]})

    assert ainvoke.await_args.kwargs["config"]["callbacks"] == [sentinel]

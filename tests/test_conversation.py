"""Tests for multi-turn conversation management."""

import asyncio

import pytest

from water.agents.conversation import (
    ConversationManager,
    ConversationState,
    Turn,
    create_conversation_task,
)
from water.agents.context import ContextManager
from water.agents.llm import MockProvider
from water.core.flow import Flow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleTurn:
    """A single user message produces one assistant turn."""

    def test_single_turn(self):
        provider = MockProvider(default_response="Hello there!")
        mgr = ConversationManager(provider=provider, system_prompt="You are helpful.")
        state = ConversationState()

        turn = _run(mgr.send("Hi", state))

        assert turn.role == "assistant"
        assert turn.content == "Hello there!"
        assert len(state.history) == 2  # user + assistant


class TestMultiTurnPreservesHistory:
    """Multi-turn dialogue sends full history to the provider."""

    def test_history_sent_to_provider(self):
        provider = MockProvider(responses=["R1", "R2", "R3"])
        mgr = ConversationManager(provider=provider, system_prompt="sys")
        state = ConversationState()

        _run(mgr.send("msg1", state))
        _run(mgr.send("msg2", state))
        _run(mgr.send("msg3", state))

        # The third call should have received: system + prior turns + user msg3
        # At the time of the 3rd send, history has: msg1, R1, msg2, R2, then msg3 is appended
        # before calling the provider — so 5 non-system messages.
        last_call = provider.call_history[-1]
        assert last_call[0] == {"role": "system", "content": "sys"}
        roles = [m["role"] for m in last_call[1:]]
        assert roles == ["user", "assistant", "user", "assistant", "user"]


class TestMaxHistoryTruncation:
    """History is truncated when it exceeds max_history."""

    def test_truncation(self):
        provider = MockProvider(default_response="ok")
        mgr = ConversationManager(provider=provider, max_history=4)
        state = ConversationState()

        # Send 5 messages -> 10 turns, should be truncated to 4
        for i in range(5):
            _run(mgr.send(f"msg{i}", state))

        assert len(state.history) == 4
        # The kept turns should be the most recent ones
        assert state.history[0].role == "user"
        assert state.history[0].content == "msg3"


class TestClear:
    """clear() resets history and slots."""

    def test_clear_resets(self):
        provider = MockProvider(default_response="ok")
        mgr = ConversationManager(provider=provider)
        state = ConversationState()
        state.slots["name"] = "Alice"

        _run(mgr.send("hi", state))
        assert len(state.history) > 0

        mgr.clear(state)

        assert len(state.history) == 0
        assert state.slots == {}


class TestGetHistory:
    """get_history returns properly formatted messages."""

    def test_with_max_turns(self):
        provider = MockProvider(default_response="ok")
        mgr = ConversationManager(provider=provider, system_prompt="sys")
        state = ConversationState()

        _run(mgr.send("a", state))
        _run(mgr.send("b", state))

        # Only last 2 turns
        msgs = mgr.get_history(state, max_turns=2)
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert len(msgs) == 3  # system + 2 turns

    def test_without_system_prompt(self):
        provider = MockProvider(default_response="ok")
        mgr = ConversationManager(provider=provider)
        state = ConversationState()

        _run(mgr.send("hi", state))
        msgs = mgr.get_history(state)
        assert msgs[0]["role"] == "user"


class TestCreateConversationTask:
    """create_conversation_task produces a usable Task."""

    def test_task_in_flow(self):
        provider = MockProvider(responses=["first reply", "second reply"])
        task = create_conversation_task(
            id="chat",
            provider=provider,
            system_prompt="You are a bot.",
        )
        flow = Flow(id="conv-flow").then(task).register()

        result1 = _run(flow.run({"message": "hello", "conversation_id": "c1"}))
        assert result1["response"] == "first reply"
        assert result1["conversation_id"] == "c1"
        assert result1["turn_count"] == 2  # user + assistant

        result2 = _run(flow.run({"message": "follow up", "conversation_id": "c1"}))
        assert result2["response"] == "second reply"
        assert result2["turn_count"] == 4  # 2 user + 2 assistant

    def test_requires_provider(self):
        with pytest.raises(ValueError, match="requires a provider"):
            create_conversation_task(id="bad")


class TestConcurrentConversations:
    """Multiple conversation IDs are tracked independently."""

    def test_independent_states(self):
        provider = MockProvider(default_response="ok")
        task = create_conversation_task(
            id="multi", provider=provider, system_prompt="sys"
        )
        flow = Flow(id="multi-flow").then(task).register()

        _run(flow.run({"message": "a", "conversation_id": "conv-a"}))
        _run(flow.run({"message": "b", "conversation_id": "conv-b"}))
        result_a = _run(flow.run({"message": "c", "conversation_id": "conv-a"}))
        result_b = _run(flow.run({"message": "d", "conversation_id": "conv-b"}))

        assert result_a["turn_count"] == 4  # 2 messages * 2 turns each
        assert result_b["turn_count"] == 4


class TestContextManagerIntegration:
    """ConversationManager delegates to ContextManager when provided."""

    def test_uses_context_manager(self):
        provider = MockProvider(default_response="ok")
        ctx = ContextManager(max_tokens=200, strategy="sliding_window")
        mgr = ConversationManager(
            provider=provider, system_prompt="sys", context_manager=ctx
        )
        state = ConversationState()

        # Send enough messages that windowing would kick in
        for i in range(20):
            _run(mgr.send(f"message {i} " * 10, state))

        # The context manager should have been used (total_tokens_used > 0)
        assert ctx.total_tokens_used > 0


class TestTurnDataclass:
    """Turn auto-fills timestamp."""

    def test_auto_timestamp(self):
        t = Turn(role="user", content="hi")
        assert t.timestamp != ""

    def test_explicit_timestamp(self):
        t = Turn(role="user", content="hi", timestamp="2024-01-01T00:00:00Z")
        assert t.timestamp == "2024-01-01T00:00:00Z"


class TestConversationStateDefaults:
    """ConversationState auto-generates id and created_at."""

    def test_auto_id(self):
        s = ConversationState()
        assert len(s.conversation_id) > 0

    def test_explicit_id(self):
        s = ConversationState(conversation_id="my-conv")
        assert s.conversation_id == "my-conv"

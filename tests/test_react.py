"""Tests for the ReAct agentic loop (water.agents.react)."""

import json
import pytest
from water.agents.react import create_agentic_task
from water.agents.tools import Tool


# ---------------------------------------------------------------------------
# Mock providers simulating OpenAI / Anthropic tool-calling responses
# ---------------------------------------------------------------------------

class MockToolProvider:
    """Provider that returns a tool call on the first turn and text on the second."""

    def __init__(self):
        self.call_count = 0
        self.received_messages = []

    async def complete(self, **kwargs):
        self.call_count += 1
        self.received_messages.append(kwargs.get("messages", []))
        if self.call_count == 1:
            return {
                "content": "Let me look that up.",
                "tool_calls": [{
                    "id": "call_001",
                    "function": {
                        "name": "greet",
                        "arguments": {"name": "World"},
                    },
                }],
            }
        return {"content": "The answer is: Hello World!", "tool_calls": []}


class MockMultiToolProvider:
    """Provider that returns multiple tool calls in a single turn."""

    def __init__(self):
        self.call_count = 0
        self.received_messages = []

    async def complete(self, **kwargs):
        self.call_count += 1
        self.received_messages.append(kwargs.get("messages", []))
        if self.call_count == 1:
            return {
                "content": "Let me use two tools.",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "function": {"name": "greet", "arguments": {"name": "Alice"}},
                    },
                    {
                        "id": "call_b",
                        "function": {"name": "greet", "arguments": {"name": "Bob"}},
                    },
                ],
            }
        return {"content": "Done with both.", "tool_calls": []}


def make_greet_tool():
    return Tool(
        name="greet",
        description="Greet someone by name.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        execute=lambda name: f"Hello {name}!",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agentic_loop_basic():
    """Tool call → observe → final text response."""
    provider = MockToolProvider()
    tool = make_greet_tool()

    task = create_agentic_task(
        id="test_agent",
        provider=provider,
        tools=[tool],
        max_iterations=5,
    )

    result = await task.execute({"input_data": {"prompt": "Say hi"}}, None)
    assert result["response"] == "The answer is: Hello World!"
    assert result["iterations"] == 2
    assert len(result["tool_history"]) == 1
    assert result["tool_history"][0]["tool"] == "greet"


@pytest.mark.asyncio
async def test_tool_arguments_stay_as_dicts():
    """Tool call arguments must be dicts (not JSON strings) in the message history."""
    provider = MockToolProvider()
    tool = make_greet_tool()

    task = create_agentic_task(
        id="test_args",
        provider=provider,
        tools=[tool],
        max_iterations=5,
    )

    await task.execute({"input_data": {"prompt": "Go"}}, None)

    # On the second call, the messages should include the assistant's tool_calls
    second_call_messages = provider.received_messages[1]
    assistant_msg = [m for m in second_call_messages if m["role"] == "assistant"][0]
    tc_args = assistant_msg["tool_calls"][0]["function"]["arguments"]
    assert isinstance(tc_args, dict), f"Expected dict, got {type(tc_args).__name__}: {tc_args}"
    assert tc_args == {"name": "World"}


@pytest.mark.asyncio
async def test_multi_tool_calls_produce_individual_tool_messages():
    """Multiple tool calls in one turn produce individual tool result messages."""
    provider = MockMultiToolProvider()
    tool = make_greet_tool()

    task = create_agentic_task(
        id="test_multi",
        provider=provider,
        tools=[tool],
        max_iterations=5,
    )

    result = await task.execute({"input_data": {"prompt": "Greet two people"}}, None)

    # Second LLM call should have tool result messages
    second_call_messages = provider.received_messages[1]
    tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == "call_a"
    assert tool_msgs[1]["tool_call_id"] == "call_b"
    assert result["iterations"] == 2
    assert len(result["tool_history"]) == 2


@pytest.mark.asyncio
async def test_string_arguments_are_parsed_to_dicts():
    """If a provider returns arguments as a JSON string, react.py parses them."""

    class StringArgsProvider:
        def __init__(self):
            self.call_count = 0

        async def complete(self, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                return {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_str",
                        "function": {
                            "name": "greet",
                            "arguments": '{"name": "StringTest"}',
                        },
                    }],
                }
            return {"content": "Done", "tool_calls": []}

    provider = StringArgsProvider()
    tool = make_greet_tool()
    task = create_agentic_task(id="test_str_args", provider=provider, tools=[tool])

    result = await task.execute({"input_data": {"prompt": "Go"}}, None)
    assert result["tool_history"][0]["arguments"] == {"name": "StringTest"}


@pytest.mark.asyncio
async def test_on_tool_call_rejection():
    """on_tool_call returning False should reject the tool call."""
    provider = MockToolProvider()
    tool = make_greet_tool()

    task = create_agentic_task(
        id="test_reject",
        provider=provider,
        tools=[tool],
        max_iterations=5,
        on_tool_call=lambda name, args: False,
    )

    result = await task.execute({"input_data": {"prompt": "Go"}}, None)
    assert result["tool_history"][0]["result"]["success"] is False
    assert "rejected" in result["tool_history"][0]["result"]["error"]

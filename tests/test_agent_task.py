"""Tests for LLM-Native Agent Tasks."""

import pytest
import asyncio
from typing import Dict, Any
from pydantic import BaseModel

from water import Flow, create_task
from water.agents.llm import (
    create_agent_task,
    MockProvider,
    CustomProvider,
    LLMProvider,
    AgentInput,
    AgentOutput,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TopicInput(BaseModel):
    topic: str


class TopicOutput(BaseModel):
    response: str
    topic: str


class SummaryOutput(BaseModel):
    summary: str


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_agent_task_basic():
    """Create an agent task with a mock provider and execute it."""
    mock = MockProvider(default_response="Hello from LLM")

    task = create_agent_task(
        id="basic_agent",
        description="A basic agent task",
        prompt_template="Say hello about {topic}",
        provider_instance=mock,
    )

    assert task.id == "basic_agent"
    assert task.description == "A basic agent task"

    # Execute directly
    result = await task.execute(
        {"input_data": {"topic": "water"}}, None
    )
    assert result["response"] == "Hello from LLM"
    assert result["topic"] == "water"

    # Verify the provider received the formatted prompt
    assert len(mock.call_history) == 1
    assert mock.call_history[0][-1]["content"] == "Say hello about water"


@pytest.mark.asyncio
async def test_agent_task_prompt_formatting():
    """Template variables are correctly filled from input_data."""
    mock = MockProvider(default_response="ok")

    task = create_agent_task(
        id="fmt",
        prompt_template="Translate '{text}' to {language}",
        provider_instance=mock,
    )

    result = await task.execute(
        {"input_data": {"text": "hello", "language": "French"}}, None
    )

    sent_msg = mock.call_history[0][-1]["content"]
    assert sent_msg == "Translate 'hello' to French"
    assert result["response"] == "ok"
    assert result["text"] == "hello"
    assert result["language"] == "French"


@pytest.mark.asyncio
async def test_agent_task_system_prompt():
    """System prompt is included as the first message."""
    mock = MockProvider(default_response="yes")

    task = create_agent_task(
        id="sys",
        prompt_template="Do something",
        system_prompt="You are a helpful assistant.",
        provider_instance=mock,
    )

    await task.execute({"input_data": {}}, None)

    messages = mock.call_history[0]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"


@pytest.mark.asyncio
async def test_agent_task_output_parser():
    """Custom output_parser transforms the LLM response text."""
    mock = MockProvider(default_response='{"summary": "TL;DR it works"}')

    import json

    def my_parser(text: str) -> dict:
        return json.loads(text)

    task = create_agent_task(
        id="parsed",
        prompt_template="Summarise {topic}",
        provider_instance=mock,
        output_parser=my_parser,
        output_schema=SummaryOutput,
    )

    result = await task.execute({"input_data": {"topic": "AI"}}, None)
    assert result == {"summary": "TL;DR it works"}


@pytest.mark.asyncio
async def test_agent_task_in_flow():
    """Agent task works inside a Flow."""
    mock = MockProvider(default_response="Flow-generated answer")

    agent = create_agent_task(
        id="flow_agent",
        prompt_template="Answer about {topic}",
        provider_instance=mock,
        input_schema=TopicInput,
        output_schema=TopicOutput,
    )

    flow = Flow(id="agent_flow", description="Agent in a flow")
    flow.then(agent).register()

    result = await flow.run({"topic": "testing"})
    assert result["response"] == "Flow-generated answer"
    assert result["topic"] == "testing"


@pytest.mark.asyncio
async def test_agent_task_with_regular_task():
    """Mix agent tasks and regular tasks in a flow."""
    mock = MockProvider(default_response="LLM says hi")

    class InputSchema(BaseModel):
        topic: str

    class MiddleSchema(BaseModel):
        response: str
        topic: str

    class FinalSchema(BaseModel):
        result: str

    # Regular task that processes the LLM output
    def postprocess(params, context):
        data = params["input_data"]
        return {"result": f"Processed: {data['response']}"}

    agent = create_agent_task(
        id="agent_step",
        prompt_template="Talk about {topic}",
        provider_instance=mock,
        input_schema=InputSchema,
        output_schema=MiddleSchema,
    )

    post = create_task(
        id="post_step",
        input_schema=MiddleSchema,
        output_schema=FinalSchema,
        execute=postprocess,
    )

    flow = Flow(id="mixed_flow", description="Agent + regular tasks")
    flow.then(agent).then(post).register()

    result = await flow.run({"topic": "water"})
    assert result["result"] == "Processed: LLM says hi"


@pytest.mark.asyncio
async def test_agent_task_mock_provider():
    """MockProvider returns configured responses in order."""
    mock = MockProvider(responses=["first", "second", "third"])

    task = create_agent_task(
        id="multi",
        prompt_template="{prompt}",
        provider_instance=mock,
    )

    r1 = await task.execute({"input_data": {"prompt": "a"}}, None)
    r2 = await task.execute({"input_data": {"prompt": "b"}}, None)
    r3 = await task.execute({"input_data": {"prompt": "c"}}, None)

    assert r1["response"] == "first"
    assert r2["response"] == "second"
    assert r3["response"] == "third"

    # Cycling behaviour
    r4 = await task.execute({"input_data": {"prompt": "d"}}, None)
    assert r4["response"] == "first"


@pytest.mark.asyncio
async def test_agent_task_custom_provider():
    """CustomProvider wraps a user-provided async callable."""

    async def my_llm(messages, **kwargs):
        user_msg = messages[-1]["content"]
        return f"Echo: {user_msg}"

    custom = CustomProvider(fn=my_llm)

    task = create_agent_task(
        id="custom",
        prompt_template="Hello {name}",
        provider_instance=custom,
    )

    result = await task.execute({"input_data": {"name": "World"}}, None)
    assert result["response"] == "Echo: Hello World"


@pytest.mark.asyncio
async def test_agent_task_retry():
    """Retries work on agent tasks (retry_count is passed through)."""
    call_count = 0

    async def flaky_llm(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("Temporary failure")
        return "success after retries"

    custom = CustomProvider(fn=flaky_llm)

    task = create_agent_task(
        id="retry_agent",
        prompt_template="Do it",
        provider_instance=custom,
        retry_count=3,
    )

    # The task itself has retry_count=3; execute it inside a flow so
    # the execution engine handles retries.
    flow = Flow(id="retry_flow", description="Retry test")
    flow.then(task).register()

    result = await flow.run({})
    assert result["response"] == "success after retries"
    assert call_count == 3


@pytest.mark.asyncio
async def test_agent_task_no_template_uses_prompt_key():
    """When no prompt_template is set, the 'prompt' key is used."""
    mock = MockProvider(default_response="got it")

    task = create_agent_task(
        id="no_tpl",
        provider_instance=mock,
    )

    result = await task.execute(
        {"input_data": {"prompt": "Tell me a joke"}}, None
    )

    sent = mock.call_history[0][-1]["content"]
    assert sent == "Tell me a joke"
    assert result["response"] == "got it"


@pytest.mark.asyncio
async def test_agent_task_missing_template_variable():
    """Missing template variable raises a clear error."""
    mock = MockProvider(default_response="nope")

    task = create_agent_task(
        id="bad_tpl",
        prompt_template="Hello {missing_var}",
        provider_instance=mock,
    )

    with pytest.raises(ValueError, match="not found in input data"):
        await task.execute({"input_data": {"other": "value"}}, None)


@pytest.mark.asyncio
async def test_agent_task_returns_usage_data():
    """Verify that token usage data is preserved in the task output."""

    # We create a custom mock that returns 'usage' in its response,
    # simulating how OpenAIProvider or AnthropicProvider behaves.
    class UsageMockProvider(LLMProvider):
        async def complete(self, messages, **kwargs) -> dict:
            return {
                "text": "Here is some text.",
                "usage": {"input_tokens": 100, "output_tokens": 50}
            }

    task = create_agent_task(
        id="usage_test_agent",
        prompt_template="Hello {name}",
        provider_instance=UsageMockProvider(),
    )

    result = await task.execute({"input_data": {"name": "World"}}, None)

    # Assert the response text is there
    assert result["response"] == "Here is some text."

    # Assert the token usage made it into the final result dictionary!
    assert "usage" in result, "Usage data was stripped from the task result!"
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 50

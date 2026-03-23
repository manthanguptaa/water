"""Tests for structured output support (water.agents.structured)."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from water.agents.llm import MockProvider
from water.agents.structured import (
    _extract_json,
    _schema_to_prompt,
    create_structured_task,
)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    reasoning: str


class SimpleResult(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# _schema_to_prompt
# ---------------------------------------------------------------------------

class TestSchemaToPrompt:
    def test_contains_field_names(self):
        prompt = _schema_to_prompt(SentimentResult)
        assert '"sentiment"' in prompt
        assert '"confidence"' in prompt
        assert '"reasoning"' in prompt

    def test_contains_json_schema(self):
        prompt = _schema_to_prompt(SentimentResult)
        assert "JSON Schema:" in prompt
        # Should contain valid JSON
        schema_start = prompt.index("JSON Schema:\n") + len("JSON Schema:\n")
        schema_text = prompt[schema_start:]
        parsed = json.loads(schema_text)
        assert "properties" in parsed

    def test_instruction_text(self):
        prompt = _schema_to_prompt(SimpleResult)
        assert "You MUST respond with valid JSON" in prompt


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_raw_json(self):
        raw = '{"name": "test", "value": 42}'
        assert _extract_json(raw) == raw

    def test_json_in_markdown_fence(self):
        text = 'Here is the result:\n```json\n{"name": "test", "value": 42}\n```'
        extracted = _extract_json(text)
        data = json.loads(extracted)
        assert data == {"name": "test", "value": 42}

    def test_json_in_plain_fence(self):
        text = '```\n{"name": "test", "value": 42}\n```'
        extracted = _extract_json(text)
        data = json.loads(extracted)
        assert data == {"name": "test", "value": 42}

    def test_whitespace_stripped(self):
        text = '  \n  {"a": 1}  \n  '
        assert _extract_json(text) == '{"a": 1}'


# ---------------------------------------------------------------------------
# Successful parsing
# ---------------------------------------------------------------------------

class TestSuccessfulParsing:
    def test_valid_json_response(self):
        valid_json = json.dumps({
            "sentiment": "positive",
            "confidence": 0.95,
            "reasoning": "The text expresses strong approval.",
        })
        mock = MockProvider(default_response=valid_json)

        task = create_structured_task(
            id="test_sentiment",
            provider_instance=mock,
            model_cls=SentimentResult,
            prompt_template="Analyze: {text}",
        )

        result = asyncio.get_event_loop().run_until_complete(
            task.execute({"text": "I love this!"}, None)
        )

        assert result["sentiment"] == "positive"
        assert result["confidence"] == 0.95
        assert result["reasoning"] == "The text expresses strong approval."

    def test_json_in_markdown_fence_response(self):
        fenced = '```json\n{"name": "water", "value": 7}\n```'
        mock = MockProvider(default_response=fenced)

        task = create_structured_task(
            id="test_fenced",
            provider_instance=mock,
            model_cls=SimpleResult,
            prompt_template="Extract: {prompt}",
        )

        result = asyncio.get_event_loop().run_until_complete(
            task.execute({"prompt": "anything"}, None)
        )

        assert result["name"] == "water"
        assert result["value"] == 7


# ---------------------------------------------------------------------------
# Retry on invalid JSON
# ---------------------------------------------------------------------------

class TestRetryInvalidJson:
    def test_retry_then_succeed(self):
        """First response is not valid JSON, second is."""
        valid = json.dumps({"name": "ok", "value": 1})
        mock = MockProvider(responses=["not json at all!!!", valid])

        task = create_structured_task(
            id="test_retry_json",
            provider_instance=mock,
            model_cls=SimpleResult,
            max_retries=3,
        )

        result = asyncio.get_event_loop().run_until_complete(
            task.execute({"prompt": "go"}, None)
        )

        assert result["name"] == "ok"
        assert result["value"] == 1
        # Provider should have been called twice
        assert len(mock.call_history) == 2


# ---------------------------------------------------------------------------
# Retry on schema violation
# ---------------------------------------------------------------------------

class TestRetrySchemaViolation:
    def test_wrong_fields_then_correct(self):
        """First response is valid JSON but wrong fields; second is correct."""
        wrong = json.dumps({"wrong_field": "oops"})
        correct = json.dumps({"name": "fixed", "value": 42})
        mock = MockProvider(responses=[wrong, correct])

        task = create_structured_task(
            id="test_retry_schema",
            provider_instance=mock,
            model_cls=SimpleResult,
            max_retries=3,
        )

        result = asyncio.get_event_loop().run_until_complete(
            task.execute({"prompt": "go"}, None)
        )

        assert result["name"] == "fixed"
        assert result["value"] == 42
        assert len(mock.call_history) == 2

    def test_wrong_type_then_correct(self):
        """First response has wrong type for a field; second is correct."""
        wrong_type = json.dumps({"name": "ok", "value": "not_an_int"})
        correct = json.dumps({"name": "ok", "value": 10})
        mock = MockProvider(responses=[wrong_type, correct])

        task = create_structured_task(
            id="test_retry_type",
            provider_instance=mock,
            model_cls=SimpleResult,
            max_retries=3,
        )

        result = asyncio.get_event_loop().run_until_complete(
            task.execute({"prompt": "go"}, None)
        )

        assert result["name"] == "ok"
        assert result["value"] == 10


# ---------------------------------------------------------------------------
# Max retries exceeded
# ---------------------------------------------------------------------------

class TestMaxRetriesExceeded:
    def test_raises_after_max_retries(self):
        """All responses are invalid — should raise ValueError."""
        bad = "this is not json"
        mock = MockProvider(default_response=bad)

        task = create_structured_task(
            id="test_exhausted",
            provider_instance=mock,
            model_cls=SimpleResult,
            max_retries=2,
        )

        with pytest.raises(ValueError, match="Structured output validation failed after 3 attempts"):
            asyncio.get_event_loop().run_until_complete(
                task.execute({"prompt": "go"}, None)
            )

        # Should have been called max_retries + 1 = 3 times
        assert len(mock.call_history) == 3

    def test_zero_retries_fails_immediately(self):
        """With max_retries=0, a single bad response raises."""
        mock = MockProvider(default_response="bad")

        task = create_structured_task(
            id="test_zero_retry",
            provider_instance=mock,
            model_cls=SimpleResult,
            max_retries=0,
        )

        with pytest.raises(ValueError, match="Structured output validation failed after 1 attempts"):
            asyncio.get_event_loop().run_until_complete(
                task.execute({"prompt": "go"}, None)
            )

        assert len(mock.call_history) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_model_cls_required(self):
        with pytest.raises(ValueError, match="requires a model_cls"):
            create_structured_task(
                id="no_model",
                provider_instance=MockProvider(),
            )

    def test_unsupported_mode(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            create_structured_task(
                id="bad_mode",
                provider_instance=MockProvider(),
                model_cls=SimpleResult,
                mode="tool_use",
            )

    def test_system_prompt_includes_schema(self):
        """The augmented system prompt should contain schema instructions."""
        valid = json.dumps({"name": "x", "value": 1})
        mock = MockProvider(default_response=valid)

        task = create_structured_task(
            id="test_sys_prompt",
            provider_instance=mock,
            model_cls=SimpleResult,
            system_prompt="You are a helpful assistant.",
        )

        asyncio.get_event_loop().run_until_complete(
            task.execute({"prompt": "go"}, None)
        )

        # Check that the system message sent to the provider contains schema
        system_msg = mock.call_history[0][0]
        assert system_msg["role"] == "system"
        assert "You are a helpful assistant." in system_msg["content"]
        assert "JSON Schema:" in system_msg["content"]

"""Tests for water.guardrails.retry – RetryWithFeedback strategy."""

import asyncio
import time
from typing import List

import pytest

from water.guardrails.base import GuardrailResult
from water.guardrails.retry import RetryContext, RetryWithFeedback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _violation(reason: str, name: str = "test") -> GuardrailResult:
    return GuardrailResult(passed=False, reason=reason, guardrail_name=name)


def _pass() -> GuardrailResult:
    return GuardrailResult(passed=True)


# ---------------------------------------------------------------------------
# format_feedback
# ---------------------------------------------------------------------------

class TestFormatFeedback:
    def test_single_violation(self):
        strategy = RetryWithFeedback()
        feedback = strategy.format_feedback([_violation("too long")])
        assert "too long" in feedback
        assert "rejected" in feedback

    def test_multiple_violations(self):
        strategy = RetryWithFeedback()
        violations = [_violation("too long"), _violation("contains PII")]
        feedback = strategy.format_feedback(violations)
        assert "too long" in feedback
        assert "contains PII" in feedback
        # reasons should be joined with "; "
        assert "too long; contains PII" in feedback

    def test_empty_violations(self):
        strategy = RetryWithFeedback()
        assert strategy.format_feedback([]) == ""

    def test_custom_template(self):
        tpl = "FIX: {{reason}}"
        strategy = RetryWithFeedback(feedback_template=tpl)
        feedback = strategy.format_feedback([_violation("bad word")])
        assert feedback == "FIX: bad word"


# ---------------------------------------------------------------------------
# execute_with_retry
# ---------------------------------------------------------------------------

class TestExecuteWithRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self):
        """If the first execution passes all guardrails, return immediately."""
        strategy = RetryWithFeedback(backoff_factor=0)
        call_count = 0

        async def execute_fn(params, ctx):
            nonlocal call_count
            call_count += 1
            return {"output": "hello"}

        def check_fn(result):
            return [_pass()]

        result = await strategy.execute_with_retry(execute_fn, check_fn, {})
        assert result == {"output": "hello"}
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self):
        """Fail once then succeed on the second attempt."""
        strategy = RetryWithFeedback(max_retries=3, backoff_factor=0)
        attempts = []

        async def execute_fn(params, ctx):
            attempts.append(params.copy())
            if len(attempts) == 1:
                return {"output": "bad"}
            return {"output": "good"}

        def check_fn(result):
            if result["output"] == "bad":
                return [_violation("content was bad")]
            return [_pass()]

        result = await strategy.execute_with_retry(execute_fn, check_fn, {"prompt": "hi"})
        assert result == {"output": "good"}
        assert len(attempts) == 2
        # The second call should contain feedback
        assert "feedback" in attempts[1]

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        """All attempts fail -- result should have __retry_exhausted."""
        strategy = RetryWithFeedback(max_retries=2, backoff_factor=0)
        call_count = 0

        async def execute_fn(params, ctx):
            nonlocal call_count
            call_count += 1
            return {"output": "still bad"}

        def check_fn(result):
            return [_violation("always fails")]

        result = await strategy.execute_with_retry(execute_fn, check_fn, {})
        assert result["__retry_exhausted"] is True
        assert len(result["__violations"]) == 3  # initial + 2 retries
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_backoff_factor(self):
        """Verify that increasing delays are applied between retries."""
        strategy = RetryWithFeedback(max_retries=2, backoff_factor=0.1)

        async def execute_fn(params, ctx):
            return {"output": "bad"}

        def check_fn(result):
            return [_violation("nope")]

        start = time.monotonic()
        await strategy.execute_with_retry(execute_fn, check_fn, {})
        elapsed = time.monotonic() - start

        # backoff_factor=0.1 -> delays of 0.1*1 + 0.1*2 = 0.3 seconds
        assert elapsed >= 0.25

    @pytest.mark.asyncio
    async def test_zero_backoff_no_delay(self):
        strategy = RetryWithFeedback(max_retries=2, backoff_factor=0)

        async def execute_fn(params, ctx):
            return {"output": "bad"}

        def check_fn(result):
            return [_violation("nope")]

        start = time.monotonic()
        await strategy.execute_with_retry(execute_fn, check_fn, {})
        elapsed = time.monotonic() - start
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """The on_retry callback should be invoked before each retry."""
        contexts_seen: List[RetryContext] = []

        async def on_retry(ctx: RetryContext):
            contexts_seen.append(ctx)

        strategy = RetryWithFeedback(max_retries=2, backoff_factor=0, on_retry=on_retry)

        async def execute_fn(params, ctx):
            return {"output": "bad"}

        def check_fn(result):
            return [_violation("nope")]

        await strategy.execute_with_retry(execute_fn, check_fn, {"x": 1})
        assert len(contexts_seen) == 2
        assert contexts_seen[0].attempt == 1
        assert contexts_seen[1].attempt == 2
        assert contexts_seen[0].original_input == {"x": 1}


# ---------------------------------------------------------------------------
# RetryContext
# ---------------------------------------------------------------------------

class TestRetryContext:
    def test_fields(self):
        ctx = RetryContext(
            attempt=2,
            max_retries=5,
            violations=[_violation("bad")],
            original_input={"prompt": "hello"},
            feedback="fix it",
        )
        assert ctx.attempt == 2
        assert ctx.max_retries == 5
        assert len(ctx.violations) == 1
        assert ctx.original_input == {"prompt": "hello"}
        assert ctx.feedback == "fix it"

    def test_default_feedback(self):
        ctx = RetryContext(
            attempt=1,
            max_retries=3,
            violations=[],
            original_input={},
        )
        assert ctx.feedback == ""


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_negative_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            RetryWithFeedback(max_retries=-1)

"""Tests for constructor validation across public classes.

Verifies that constructors raise ValueError for invalid arguments.
Covers: Task, create_task, FlowScheduler.schedule, FallbackChain, StreamManager.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel

from water.core.task import Task, create_task
from water.utils.scheduler import FlowScheduler
from water.agents.fallback import FallbackChain
from water.agents.llm import LLMProvider
from water.resilience.circuit_breaker import CircuitBreaker
from water.integrations.streaming import StreamManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class InputModel(BaseModel):
    text: str


class OutputModel(BaseModel):
    result: str


def dummy_execute(inputs, context):
    return OutputModel(result="ok")


def _make_task(**overrides):
    defaults = dict(
        input_schema=InputModel,
        output_schema=OutputModel,
        execute=dummy_execute,
    )
    defaults.update(overrides)
    return Task(**defaults)


class FakeProvider(LLMProvider):
    async def complete(self, messages, **kwargs):
        return {"content": "hi"}


# ---------------------------------------------------------------------------
# Task validation
# ---------------------------------------------------------------------------

class TestTaskValidation:
    def test_retry_count_negative(self):
        with pytest.raises(ValueError, match="retry_count must be >= 0, got -1"):
            _make_task(retry_count=-1)

    def test_retry_count_zero_ok(self):
        t = _make_task(retry_count=0)
        assert t.retry_count == 0

    def test_retry_delay_negative(self):
        with pytest.raises(ValueError, match="retry_delay must be >= 0, got -0.5"):
            _make_task(retry_delay=-0.5)

    def test_retry_delay_zero_ok(self):
        t = _make_task(retry_delay=0.0)
        assert t.retry_delay == 0.0

    def test_retry_backoff_negative(self):
        with pytest.raises(ValueError, match="retry_backoff must be >= 0, got -1"):
            _make_task(retry_backoff=-1)

    def test_retry_backoff_zero_ok(self):
        t = _make_task(retry_backoff=0)
        assert t.retry_backoff == 0

    def test_timeout_zero(self):
        with pytest.raises(ValueError, match="timeout must be > 0, got 0"):
            _make_task(timeout=0)

    def test_timeout_negative(self):
        with pytest.raises(ValueError, match="timeout must be > 0, got -5"):
            _make_task(timeout=-5)

    def test_timeout_positive_ok(self):
        t = _make_task(timeout=1.0)
        assert t.timeout == 1.0

    def test_timeout_none_ok(self):
        t = _make_task(timeout=None)
        assert t.timeout is None

    def test_rate_limit_zero(self):
        with pytest.raises(ValueError, match="rate_limit must be > 0, got 0"):
            _make_task(rate_limit=0)

    def test_rate_limit_negative(self):
        with pytest.raises(ValueError, match="rate_limit must be > 0, got -1"):
            _make_task(rate_limit=-1)

    def test_rate_limit_positive_ok(self):
        t = _make_task(rate_limit=5.0)
        assert t.rate_limit == 5.0

    def test_rate_limit_none_ok(self):
        t = _make_task(rate_limit=None)
        assert t.rate_limit is None


class TestCreateTaskValidation:
    def test_retry_count_negative(self):
        with pytest.raises(ValueError, match="retry_count must be >= 0"):
            create_task(
                input_schema=InputModel,
                output_schema=OutputModel,
                execute=dummy_execute,
                retry_count=-1,
            )

    def test_timeout_zero(self):
        with pytest.raises(ValueError, match="timeout must be > 0"):
            create_task(
                input_schema=InputModel,
                output_schema=OutputModel,
                execute=dummy_execute,
                timeout=0,
            )


# ---------------------------------------------------------------------------
# FlowScheduler.schedule validation
# ---------------------------------------------------------------------------

class TestSchedulerValidation:
    def test_assert_replaced_with_valueerror(self):
        """The old assert for cron_expr is now a proper ValueError."""
        scheduler = FlowScheduler()
        flow = MagicMock()
        # Both None should raise ValueError (existing check)
        with pytest.raises(ValueError, match="Either cron_expr or interval_seconds must be provided"):
            scheduler.schedule(flow, input_data={})

    def test_interval_seconds_negative(self):
        scheduler = FlowScheduler()
        flow = MagicMock()
        with pytest.raises(ValueError, match="interval_seconds must be > 0, got -10"):
            scheduler.schedule(flow, input_data={}, interval_seconds=-10)

    def test_interval_seconds_zero(self):
        scheduler = FlowScheduler()
        flow = MagicMock()
        with pytest.raises(ValueError, match="interval_seconds must be > 0, got 0"):
            scheduler.schedule(flow, input_data={}, interval_seconds=0)

    def test_interval_seconds_positive_ok(self):
        scheduler = FlowScheduler()
        flow = MagicMock()
        job_id = scheduler.schedule(flow, input_data={}, interval_seconds=60)
        assert job_id is not None


# ---------------------------------------------------------------------------
# FallbackChain validation
# ---------------------------------------------------------------------------

class TestFallbackChainValidation:
    def test_circuit_breaker_key_out_of_range(self):
        providers = [FakeProvider()]
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        with pytest.raises(ValueError, match="circuit_breakers key 5 is not a valid provider index"):
            FallbackChain(providers=providers, circuit_breakers={5: cb})

    def test_circuit_breaker_key_negative(self):
        providers = [FakeProvider()]
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        with pytest.raises(ValueError, match="circuit_breakers key -1 is not a valid provider index"):
            FallbackChain(providers=providers, circuit_breakers={-1: cb})

    def test_circuit_breaker_key_valid_ok(self):
        providers = [FakeProvider(), FakeProvider()]
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
        chain = FallbackChain(providers=providers, circuit_breakers={0: cb, 1: cb})
        assert len(chain.circuit_breakers) == 2


# ---------------------------------------------------------------------------
# StreamManager validation
# ---------------------------------------------------------------------------

class TestStreamManagerValidation:
    def test_max_queue_size_negative(self):
        with pytest.raises(ValueError, match="max_queue_size must be >= 0, got -1"):
            StreamManager(max_queue_size=-1)

    def test_max_queue_size_zero_ok(self):
        sm = StreamManager(max_queue_size=0)
        assert sm.max_queue_size == 0

    def test_max_queue_size_positive_ok(self):
        sm = StreamManager(max_queue_size=10)
        assert sm.max_queue_size == 10

"""Tests for FallbackChain (water.agents.fallback)."""

import asyncio
from typing import Optional

import pytest

from water.agents.llm import LLMProvider, MockProvider
from water.agents.fallback import FallbackChain, ProviderMetrics
from water.resilience.circuit_breaker import CircuitBreaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FailingProvider(LLMProvider):
    """A provider that always raises an exception."""

    def __init__(self, error: Optional[Exception] = None) -> None:
        self.error = error or RuntimeError("provider failed")

    async def complete(self, messages, **kwargs) -> dict:
        raise self.error


class SlowProvider(LLMProvider):
    """A provider that sleeps before returning, to simulate latency."""

    def __init__(self, delay: float, response: str = "slow") -> None:
        self.delay = delay
        self.response = response

    async def complete(self, messages, **kwargs) -> dict:
        await asyncio.sleep(self.delay)
        return {"text": self.response}


MESSAGES = [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# first_success strategy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_success_first_provider_succeeds():
    """When the first provider succeeds, its response is returned."""
    chain = FallbackChain(
        providers=[MockProvider("primary"), MockProvider("secondary")],
        strategy="first_success",
    )
    result = await chain.complete(MESSAGES)
    assert result["text"] == "primary"


@pytest.mark.asyncio
async def test_first_success_first_fails_second_succeeds():
    """When the first provider fails, the chain falls back to the second."""
    chain = FallbackChain(
        providers=[FailingProvider(), MockProvider("backup")],
        strategy="first_success",
    )
    result = await chain.complete(MESSAGES)
    assert result["text"] == "backup"


@pytest.mark.asyncio
async def test_all_providers_fail_raises_last_error():
    """When every provider fails, the last exception is re-raised."""
    chain = FallbackChain(
        providers=[
            FailingProvider(RuntimeError("err1")),
            FailingProvider(ValueError("err2")),
        ],
        strategy="first_success",
    )
    with pytest.raises(ValueError, match="err2"):
        await chain.complete(MESSAGES)


# ---------------------------------------------------------------------------
# round_robin strategy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_round_robin_distributes_calls():
    """Round-robin cycles through providers across successive calls."""
    p0 = MockProvider("p0")
    p1 = MockProvider("p1")
    p2 = MockProvider("p2")
    chain = FallbackChain(providers=[p0, p1, p2], strategy="round_robin")

    results = []
    for _ in range(6):
        r = await chain.complete(MESSAGES)
        results.append(r["text"])

    # Should cycle: p0, p1, p2, p0, p1, p2
    assert results == ["p0", "p1", "p2", "p0", "p1", "p2"]


@pytest.mark.asyncio
async def test_round_robin_skips_failing_provider():
    """Round-robin falls back to the next provider if the primary fails."""
    chain = FallbackChain(
        providers=[FailingProvider(), MockProvider("ok"), MockProvider("also_ok")],
        strategy="round_robin",
    )
    # First call: starts at index 0 (fails) -> falls to 1
    r1 = await chain.complete(MESSAGES)
    assert r1["text"] == "ok"

    # Second call: starts at index 1 (ok)
    r2 = await chain.complete(MESSAGES)
    assert r2["text"] == "ok"


# ---------------------------------------------------------------------------
# lowest_latency strategy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lowest_latency_prefers_faster_provider():
    """After warming up, lowest_latency should prefer the faster provider."""
    fast = MockProvider("fast")
    slow = SlowProvider(delay=0.05, response="slow")

    chain = FallbackChain(providers=[slow, fast], strategy="lowest_latency")

    # Warm-up: call twice so both providers get measured.
    # First call: both have 0 data so order is [0, 1]; index 0 (slow) runs.
    await chain.complete(MESSAGES)
    # Second call: index 0 has latency ~0.05; index 1 has 0.0 (unmeasured)
    # so index 1 comes first.
    await chain.complete(MESSAGES)

    # Now both have been measured. Fast provider should be preferred.
    result = await chain.complete(MESSAGES)
    assert result["text"] == "fast"


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_metrics_tracking():
    """Metrics should accurately count calls and failures."""
    chain = FallbackChain(
        providers=[FailingProvider(), MockProvider("ok")],
        strategy="first_success",
    )
    await chain.complete(MESSAGES)
    await chain.complete(MESSAGES)

    metrics = chain.get_metrics()
    # Provider 0: 2 calls, 2 failures
    assert metrics[0]["calls"] == 2
    assert metrics[0]["failures"] == 2
    assert metrics[0]["avg_latency"] == 0.0  # no successes
    # Provider 1: 2 calls, 0 failures
    assert metrics[1]["calls"] == 2
    assert metrics[1]["failures"] == 0
    assert metrics[1]["avg_latency"] > 0.0


@pytest.mark.asyncio
async def test_reset_metrics():
    """reset_metrics should zero out all counters."""
    chain = FallbackChain(providers=[MockProvider("a")], strategy="first_success")
    await chain.complete(MESSAGES)
    chain.reset_metrics()

    metrics = chain.get_metrics()
    assert metrics[0]["calls"] == 0
    assert metrics[0]["failures"] == 0
    assert metrics[0]["avg_latency"] == 0.0


# ---------------------------------------------------------------------------
# Single provider & empty list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_provider_chain():
    """A chain with one provider should behave like the provider itself."""
    chain = FallbackChain(providers=[MockProvider("only")])
    result = await chain.complete(MESSAGES)
    assert result["text"] == "only"


def test_empty_provider_list_raises():
    """An empty provider list should raise ValueError at construction time."""
    with pytest.raises(ValueError, match="at least one provider"):
        FallbackChain(providers=[])


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker_skips_open_provider():
    """Providers with an open circuit breaker are skipped entirely."""
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
    cb.record_failure()  # trips the breaker immediately

    chain = FallbackChain(
        providers=[MockProvider("blocked"), MockProvider("available")],
        strategy="first_success",
        circuit_breakers={0: cb},
    )
    result = await chain.complete(MESSAGES)
    assert result["text"] == "available"
    # Blocked provider should have 0 calls
    assert chain.metrics[0].calls == 0


@pytest.mark.asyncio
async def test_all_providers_circuit_broken_raises():
    """When all providers are circuit-broken, a RuntimeError is raised."""
    cb0 = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
    cb0.record_failure()
    cb1 = CircuitBreaker(failure_threshold=1, recovery_timeout=9999)
    cb1.record_failure()

    chain = FallbackChain(
        providers=[MockProvider("a"), MockProvider("b")],
        strategy="first_success",
        circuit_breakers={0: cb0, 1: cb1},
    )
    with pytest.raises(RuntimeError, match="All providers are unavailable"):
        await chain.complete(MESSAGES)


# ---------------------------------------------------------------------------
# ProviderMetrics dataclass
# ---------------------------------------------------------------------------

def test_provider_metrics_avg_latency():
    """avg_latency should compute correctly from calls, failures, and total_latency."""
    m = ProviderMetrics(calls=10, failures=2, total_latency=4.0)
    # 8 successful calls, 4.0 total -> 0.5 avg
    assert m.avg_latency == pytest.approx(0.5)

    m2 = ProviderMetrics(calls=3, failures=3)
    assert m2.avg_latency == 0.0

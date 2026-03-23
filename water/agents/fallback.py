"""
Model Fallback Chain for Water.

Provides a FallbackChain that implements LLMProvider and tries multiple
providers using configurable strategies: first_success, round_robin, or
lowest_latency. Tracks per-provider metrics and optionally integrates
with CircuitBreaker to skip providers in a failed state.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from water.agents.llm import LLMProvider
from water.resilience.circuit_breaker import CircuitBreaker


# ---------------------------------------------------------------------------
# Per-provider metrics
# ---------------------------------------------------------------------------

@dataclass
class ProviderMetrics:
    """Tracks call count, failure count, and cumulative latency for a provider."""

    calls: int = 0
    failures: int = 0
    total_latency: float = 0.0

    @property
    def avg_latency(self) -> float:
        """Average latency per successful call, or 0.0 if no successful calls."""
        successful = self.calls - self.failures
        if successful <= 0:
            return 0.0
        return self.total_latency / successful


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

_VALID_STRATEGIES = {"first_success", "round_robin", "lowest_latency"}


class FallbackChain(LLMProvider):
    """
    An LLMProvider that delegates to a list of providers using a fallback
    strategy.

    Strategies
    ----------
    * **first_success** -- try providers in list order; return the first
      successful response.
    * **round_robin** -- cycle through providers, falling back to the next
      on failure.
    * **lowest_latency** -- sort providers by average latency and try them
      in that order, falling back on failure.

    Parameters
    ----------
    providers:
        Ordered list of LLMProvider instances.
    strategy:
        One of ``"first_success"``, ``"round_robin"``, or
        ``"lowest_latency"``.
    circuit_breakers:
        Optional mapping of provider index to a
        :class:`~water.resilience.circuit_breaker.CircuitBreaker`.
        When supplied, providers whose breaker is open are skipped.
    """

    def __init__(
        self,
        providers: List[LLMProvider],
        strategy: str = "first_success",
        circuit_breakers: Optional[Dict[int, CircuitBreaker]] = None,
    ) -> None:
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {sorted(_VALID_STRATEGIES)}"
            )

        self.providers = providers
        self.strategy = strategy
        self.circuit_breakers: Dict[int, CircuitBreaker] = circuit_breakers or {}

        # Validate circuit_breaker keys are valid provider indices
        for key in self.circuit_breakers:
            if key < 0 or key >= len(providers):
                raise ValueError(
                    f"circuit_breakers key {key} is not a valid provider index. "
                    f"Must be in range 0..{len(providers) - 1}"
                )
        self.metrics: Dict[int, ProviderMetrics] = {
            i: ProviderMetrics() for i in range(len(providers))
        }
        self._rr_index: int = 0

    # -- internal helpers ---------------------------------------------------

    def _provider_order(self) -> List[int]:
        """Return provider indices in the order dictated by the strategy."""
        n = len(self.providers)

        if self.strategy == "first_success":
            return list(range(n))

        if self.strategy == "round_robin":
            indices = [(self._rr_index + i) % n for i in range(n)]
            self._rr_index = (self._rr_index + 1) % n
            return indices

        if self.strategy == "lowest_latency":
            # Sort by avg_latency; providers with no data (0.0) go first so
            # they get a chance to be measured.
            order = sorted(
                range(n),
                key=lambda i: (
                    self.metrics[i].avg_latency
                    if (self.metrics[i].calls - self.metrics[i].failures) > 0
                    else 0.0
                ),
            )
            return order

        # Fallback (should not be reachable)
        return list(range(n))  # pragma: no cover

    def _is_available(self, index: int) -> bool:
        """Check whether a provider is available (circuit breaker aware)."""
        cb = self.circuit_breakers.get(index)
        if cb is None:
            return True
        return cb.can_execute()

    async def _try_provider(
        self, index: int, messages: List[Dict[str, str]], **kwargs: Any
    ) -> dict:
        """Attempt a single provider call, updating metrics and breakers."""
        provider = self.providers[index]
        m = self.metrics[index]
        m.calls += 1

        start = time.monotonic()
        try:
            result = await provider.complete(messages, **kwargs)
        except Exception:
            m.failures += 1
            cb = self.circuit_breakers.get(index)
            if cb is not None:
                cb.record_failure()
            raise

        elapsed = time.monotonic() - start
        m.total_latency += elapsed

        cb = self.circuit_breakers.get(index)
        if cb is not None:
            cb.record_success()

        return result

    # -- public API ---------------------------------------------------------

    async def complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> dict:
        """Try providers according to the configured strategy.

        Returns the first successful response. If every provider fails,
        re-raises the exception from the last attempted provider.
        """
        order = self._provider_order()
        last_error: Optional[Exception] = None

        for idx in order:
            if not self._is_available(idx):
                continue
            try:
                return await self._try_provider(idx, messages, **kwargs)
            except Exception as exc:
                last_error = exc

        if last_error is not None:
            raise last_error

        # All providers were skipped by circuit breakers
        raise RuntimeError("All providers are unavailable")

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Return a list of per-provider metric dicts."""
        results: List[Dict[str, Any]] = []
        for i, m in sorted(self.metrics.items()):
            results.append(
                {
                    "provider_index": i,
                    "calls": m.calls,
                    "failures": m.failures,
                    "avg_latency": m.avg_latency,
                }
            )
        return results

    def reset_metrics(self) -> None:
        """Reset all per-provider metrics to zero."""
        for m in self.metrics.values():
            m.calls = 0
            m.failures = 0
            m.total_latency = 0.0

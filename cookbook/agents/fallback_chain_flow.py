"""
Cookbook: Model Fallback Chain
=============================

Demonstrates the FallbackChain LLM provider, which wraps multiple providers
and tries them in order using configurable strategies.

Scenarios shown:
  1. Basic first_success fallback (one failing, one succeeding provider).
  2. Round-robin distribution across providers.
  3. Metrics inspection after a series of calls.
"""

import asyncio

from water.agents.llm import LLMProvider, OpenAIProvider
from water.agents.fallback import FallbackChain


# ---------------------------------------------------------------------------
# A provider that always fails (simulates an unreachable API)
# ---------------------------------------------------------------------------

class UnreachableProvider(LLMProvider):
    """Simulates a provider whose API is down."""

    async def complete(self, messages, **kwargs) -> dict:
        raise ConnectionError("API unreachable")


# ---------------------------------------------------------------------------
# 1. Basic first_success fallback
# ---------------------------------------------------------------------------

async def demo_first_success():
    print("=== first_success strategy ===")
    chain = FallbackChain(
        providers=[
            UnreachableProvider(),          # primary -- always fails
            OpenAIProvider(model="gpt-4o-mini", temperature=0.3),  # secondary -- real LLM
        ],
        strategy="first_success",
    )

    result = await chain.complete([{"role": "user", "content": "Hello"}])
    print(f"Response: {result['text']}")
    # -> "backup-response" because the primary failed

    metrics = chain.get_metrics()
    for m in metrics:
        print(f"  Provider {m['provider_index']}: "
              f"calls={m['calls']} failures={m['failures']} "
              f"avg_latency={m['avg_latency']:.4f}s")
    print()


# ---------------------------------------------------------------------------
# 2. Round-robin distribution
# ---------------------------------------------------------------------------

async def demo_round_robin():
    print("=== round_robin strategy ===")
    chain = FallbackChain(
        providers=[
            OpenAIProvider(model="gpt-4o-mini", temperature=0.0),
            OpenAIProvider(model="gpt-4o-mini", temperature=0.5),
            OpenAIProvider(model="gpt-4o-mini", temperature=1.0),
        ],
        strategy="round_robin",
    )

    for i in range(6):
        result = await chain.complete([{"role": "user", "content": f"Call {i}"}])
        print(f"  Call {i}: {result['text']}")
    # Calls are distributed: A, B, C, A, B, C
    print()


# ---------------------------------------------------------------------------
# 3. Metrics inspection
# ---------------------------------------------------------------------------

async def demo_metrics():
    print("=== Metrics inspection ===")
    chain = FallbackChain(
        providers=[
            UnreachableProvider(),
            OpenAIProvider(model="gpt-4o-mini", temperature=0.3),
        ],
        strategy="first_success",
    )

    for _ in range(5):
        await chain.complete([{"role": "user", "content": "test"}])

    print("After 5 calls:")
    for m in chain.get_metrics():
        print(f"  Provider {m['provider_index']}: "
              f"calls={m['calls']} failures={m['failures']} "
              f"avg_latency={m['avg_latency']:.6f}s")

    chain.reset_metrics()
    print("After reset_metrics():")
    for m in chain.get_metrics():
        print(f"  Provider {m['provider_index']}: "
              f"calls={m['calls']} failures={m['failures']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await demo_first_success()
    await demo_round_robin()
    await demo_metrics()


if __name__ == "__main__":
    asyncio.run(main())

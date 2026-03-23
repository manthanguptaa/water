"""
Cookbook: Guardrail Retry with Feedback Loop
============================================

Demonstrates how to use ``RetryWithFeedback`` to automatically re-execute
an agent task when guardrails detect problems, feeding the violation details
back so the next attempt can self-correct.
"""

import asyncio
from typing import List
from water.guardrails.base import Guardrail, GuardrailResult
from water.guardrails.retry import RetryContext, RetryWithFeedback


# ---------------------------------------------------------------------------
# 1. Basic retry with a simple content filter
# ---------------------------------------------------------------------------

class NoProfanityGuardrail(Guardrail):
    """Rejects output that contains placeholder 'bad words'."""

    BLOCKED_WORDS = {"badword", "offensive"}

    def validate(self, data, context=None):
        text = data.get("output", "").lower()
        for word in self.BLOCKED_WORDS:
            if word in text:
                return GuardrailResult(
                    passed=False,
                    reason=f"Output contains blocked word: '{word}'",
                )
        return GuardrailResult(passed=True)


async def basic_retry_example():
    """Run a simulated agent call that fails once, then succeeds."""

    print("=== Basic Retry Example ===\n")

    call_count = 0

    async def fake_agent_call(params, context):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"output": "Here is my answer with a badword in it."}
        # On retry the agent 'fixes' itself thanks to the feedback
        return {"output": "Here is my corrected, clean answer."}

    guardrail = NoProfanityGuardrail(name="no_profanity", action="retry")

    def check(result):
        return [guardrail.validate(result)]

    strategy = RetryWithFeedback(
        max_retries=3,
        backoff_factor=0,  # no delay for the demo
    )

    result = await strategy.execute_with_retry(
        execute_fn=fake_agent_call,
        check_fn=check,
        params={"prompt": "Tell me a joke"},
    )

    print(f"  Attempts: {call_count}")
    print(f"  Final output: {result['output']}")
    print()


# ---------------------------------------------------------------------------
# 2. Custom feedback templates
# ---------------------------------------------------------------------------

async def custom_template_example():
    """Show how a custom feedback template changes the injected message."""

    print("=== Custom Feedback Template ===\n")

    strategy = RetryWithFeedback(
        max_retries=2,
        feedback_template=(
            "[SYSTEM] The following issues were found: {{reason}}. "
            "Rewrite your response to address them."
        ),
        backoff_factor=0,
    )

    violations = [
        GuardrailResult(passed=False, reason="response too short"),
        GuardrailResult(passed=False, reason="missing citation"),
    ]
    feedback = strategy.format_feedback(violations)
    print(f"  Feedback: {feedback}")
    print()


# ---------------------------------------------------------------------------
# 3. Integration with an agent task (simulated)
# ---------------------------------------------------------------------------

async def agent_task_integration():
    """Simulate end-to-end: prompt -> agent -> guardrails -> retry loop."""

    print("=== Agent Task Integration ===\n")

    attempt_log: List[RetryContext] = []

    async def log_retry(ctx: RetryContext):
        attempt_log.append(ctx)
        print(f"  [retry] attempt {ctx.attempt}/{ctx.max_retries} "
              f"-- feedback: {ctx.feedback[:60]}...")

    class LengthGuardrail(Guardrail):
        """Require the output to be at least 20 characters."""

        def validate(self, data, context=None):
            text = data.get("output", "")
            if len(text) < 20:
                return GuardrailResult(
                    passed=False,
                    reason=f"Output too short ({len(text)} chars, need >= 20)",
                )
            return GuardrailResult(passed=True)

    guardrail = LengthGuardrail(name="min_length", action="retry")
    call_count = 0

    async def agent(params, context):
        nonlocal call_count
        call_count += 1
        if "feedback" in params:
            # Agent sees the feedback and produces a longer answer
            return {"output": "This is a much longer and more detailed answer."}
        return {"output": "Short."}

    strategy = RetryWithFeedback(
        max_retries=3,
        backoff_factor=0,
        on_retry=log_retry,
    )

    result = await strategy.execute_with_retry(
        execute_fn=agent,
        check_fn=lambda r: [guardrail.validate(r)],
        params={"prompt": "Explain water framework"},
    )

    exhausted = result.get("__retry_exhausted", False)
    print(f"  Total calls: {call_count}")
    print(f"  Exhausted: {exhausted}")
    print(f"  Final output: {result['output']}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await basic_retry_example()
    await custom_template_example()
    await agent_task_integration()


if __name__ == "__main__":
    asyncio.run(main())

"""
Benchmark Flow — Run tool-use and instruction benchmarks across mock providers.

Demonstrates using BenchmarkRunner to compare two providers and print a
leaderboard.  Uses MockProvider so no API keys are required.

Usage:
    python cookbook/agents/benchmark_flow.py
"""

import asyncio
import json
import tempfile
import os

from water.agents.llm import MockProvider
from water.bench import BenchmarkRunner, ToolUseBenchmark, InstructionBenchmark


async def main():
    # --- Set up two mock providers ---
    # "good" provider returns well-formed tool call JSON and follows instructions
    good_tool_response = json.dumps({
        "tool_name": "get_weather",
        "arguments": {"city": "Paris"},
    })
    good_provider = MockProvider(default_response=good_tool_response)

    # "bad" provider returns free-text that doesn't match expected formats
    bad_provider = MockProvider(default_response="I'm not sure how to help with that.")

    # --- Run benchmarks ---
    runner = BenchmarkRunner(
        providers={
            "good_model": good_provider,
            "bad_model": bad_provider,
        },
        benchmarks=[
            ToolUseBenchmark(),
            InstructionBenchmark(),
        ],
    )

    print("Running benchmarks...")
    report = await runner.run()

    # --- Print leaderboard ---
    print("\n=== Leaderboard ===\n")
    print(report.leaderboard())

    # --- Print summary ---
    print("\n=== Summary ===\n")
    summary = report.summary()
    print(json.dumps(summary, indent=2))

    # --- Export to JSON ---
    export_path = os.path.join(tempfile.gettempdir(), "benchmark_results.json")
    report.export(export_path)
    print(f"\nResults exported to: {export_path}")


if __name__ == "__main__":
    asyncio.run(main())

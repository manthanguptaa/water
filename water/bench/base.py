"""
Agent benchmarking framework for Water.

Provides standardized benchmarks for evaluating agent capabilities
across different LLM providers: tool-use accuracy, instruction following,
latency, and more.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkCase:
    """Single benchmark test case."""
    id: str
    prompt: str
    expected: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark case against a provider."""
    case_id: str
    provider_name: str
    output: Any
    score: float  # 0.0 to 1.0
    latency_ms: float
    error: Optional[str] = None


class Benchmark(ABC):
    """Abstract base class for benchmarks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""
        ...

    @abstractmethod
    def generate_cases(self) -> List[BenchmarkCase]:
        """Generate the list of benchmark cases."""
        ...

    @abstractmethod
    def evaluate(self, output: Any, expected: Any) -> float:
        """
        Score a provider output against the expected result.

        Args:
            output: Raw text output from the provider.
            expected: Expected output from the benchmark case.

        Returns:
            Score between 0.0 and 1.0.
        """
        ...


class BenchmarkReport:
    """Collects benchmark results and produces leaderboards and summaries."""

    def __init__(self, results: Optional[List[BenchmarkResult]] = None) -> None:
        self.results: List[BenchmarkResult] = results or []

    def leaderboard(self) -> str:
        """
        Render a text table showing provider scores per benchmark.

        Returns:
            Formatted leaderboard string.
        """
        if not self.results:
            return "No results."

        # Group results by provider and benchmark prefix
        provider_scores: Dict[str, Dict[str, List[float]]] = {}
        benchmarks_seen: List[str] = []

        for r in self.results:
            # Extract benchmark name from case_id (format: "benchmark_name/case_id")
            parts = r.case_id.split("/", 1)
            bench_name = parts[0] if len(parts) > 1 else "default"

            if bench_name not in benchmarks_seen:
                benchmarks_seen.append(bench_name)

            provider_scores.setdefault(r.provider_name, {})
            provider_scores[r.provider_name].setdefault(bench_name, []).append(r.score)

        # Build table
        col_width = max(14, *(len(b) for b in benchmarks_seen)) + 2
        provider_width = max(16, *(len(p) for p in provider_scores)) + 2

        header = f"{'Provider':<{provider_width}}"
        for b in benchmarks_seen:
            header += f"| {b:<{col_width}}"
        header += f"| {'Avg':<8}| {'Avg Latency':<14}"

        sep = "-" * len(header)
        lines = [sep, header, sep]

        for provider, bench_map in sorted(provider_scores.items()):
            row = f"{provider:<{provider_width}}"
            all_scores: List[float] = []
            for b in benchmarks_seen:
                scores = bench_map.get(b, [])
                if scores:
                    avg = sum(scores) / len(scores)
                    all_scores.extend(scores)
                    row += f"| {avg * 100:>{col_width - 2}.1f}% "
                else:
                    row += f"| {'N/A':>{col_width - 1}} "

            overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
            row += f"| {overall_avg * 100:>5.1f}% "

            # Average latency for this provider
            provider_results = [r for r in self.results if r.provider_name == provider]
            avg_latency = (
                sum(r.latency_ms for r in provider_results) / len(provider_results)
                if provider_results
                else 0.0
            )
            row += f"| {avg_latency:>10.1f} ms "

            lines.append(row)

        lines.append(sep)
        return "\n".join(lines)

    def summary(self) -> Dict[str, Any]:
        """Structured summary of results."""
        if not self.results:
            return {"total_cases": 0, "providers": {}}

        providers: Dict[str, Any] = {}
        for r in self.results:
            entry = providers.setdefault(r.provider_name, {
                "total": 0,
                "score_sum": 0.0,
                "latency_sum": 0.0,
                "errors": 0,
            })
            entry["total"] += 1
            entry["score_sum"] += r.score
            entry["latency_sum"] += r.latency_ms
            if r.error:
                entry["errors"] += 1

        for name, data in providers.items():
            total = data["total"]
            data["avg_score"] = data["score_sum"] / total if total else 0.0
            data["avg_latency_ms"] = data["latency_sum"] / total if total else 0.0

        return {
            "total_cases": len(self.results),
            "providers": {
                name: {
                    "total_cases": d["total"],
                    "avg_score": round(d["avg_score"], 4),
                    "avg_latency_ms": round(d["avg_latency_ms"], 2),
                    "errors": d["errors"],
                }
                for name, d in providers.items()
            },
        }

    def export(self, path: str) -> None:
        """Export results to a JSON file."""
        data = {
            "summary": self.summary(),
            "results": [
                {
                    "case_id": r.case_id,
                    "provider_name": r.provider_name,
                    "output": r.output,
                    "score": r.score,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across multiple providers.

    Args:
        providers: Mapping of provider name to LLMProvider instance.
        benchmarks: List of Benchmark instances to run.
    """

    def __init__(
        self,
        providers: Dict[str, Any],
        benchmarks: List[Benchmark],
    ) -> None:
        self.providers = providers
        self.benchmarks = benchmarks

    async def run(self) -> BenchmarkReport:
        """
        Run all benchmarks across all providers and collect results.

        Returns:
            BenchmarkReport containing all results.
        """
        results: List[BenchmarkResult] = []

        for benchmark in self.benchmarks:
            cases = benchmark.generate_cases()

            for provider_name, provider in self.providers.items():
                for case in cases:
                    case_id = f"{benchmark.name}/{case.id}"
                    start = time.perf_counter()
                    error: Optional[str] = None
                    output: Any = None
                    score: float = 0.0

                    try:
                        messages = [{"role": "user", "content": case.prompt}]
                        response = await provider.complete(messages)
                        output = response.get("text", "")
                        score = benchmark.evaluate(output, case.expected)
                    except Exception as exc:
                        error = str(exc)
                        score = 0.0

                    elapsed_ms = (time.perf_counter() - start) * 1000.0

                    results.append(BenchmarkResult(
                        case_id=case_id,
                        provider_name=provider_name,
                        output=output,
                        score=score,
                        latency_ms=elapsed_ms,
                        error=error,
                    ))

        return BenchmarkReport(results=results)

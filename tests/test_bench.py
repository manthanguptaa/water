"""Tests for the agent benchmarking suite."""

import asyncio
import json
import os
import tempfile

import pytest

from water.bench.base import (
    Benchmark,
    BenchmarkCase,
    BenchmarkReport,
    BenchmarkResult,
    BenchmarkRunner,
)
from water.bench.tool_use import ToolUseBenchmark
from water.bench.instruction import InstructionBenchmark
from water.agents.llm import MockProvider


# ---------------------------------------------------------------------------
# Dataclass creation
# ---------------------------------------------------------------------------

class TestBenchmarkCase:
    def test_create_minimal(self):
        case = BenchmarkCase(id="c1", prompt="Hello", expected="world")
        assert case.id == "c1"
        assert case.prompt == "Hello"
        assert case.expected == "world"
        assert case.metadata == {}

    def test_create_with_metadata(self):
        case = BenchmarkCase(id="c2", prompt="P", expected=42, metadata={"tag": "x"})
        assert case.metadata == {"tag": "x"}


class TestBenchmarkResult:
    def test_create(self):
        r = BenchmarkResult(
            case_id="tool_use/weather",
            provider_name="mock",
            output="hello",
            score=0.75,
            latency_ms=12.3,
        )
        assert r.case_id == "tool_use/weather"
        assert r.score == 0.75
        assert r.error is None

    def test_create_with_error(self):
        r = BenchmarkResult(
            case_id="x", provider_name="p", output=None,
            score=0.0, latency_ms=0.0, error="boom",
        )
        assert r.error == "boom"


# ---------------------------------------------------------------------------
# ToolUseBenchmark
# ---------------------------------------------------------------------------

class TestToolUseBenchmark:
    def test_generate_cases(self):
        bench = ToolUseBenchmark()
        cases = bench.generate_cases()
        assert len(cases) >= 8
        assert all(isinstance(c, BenchmarkCase) for c in cases)
        assert bench.name == "tool_use"

    def test_evaluate_perfect(self):
        bench = ToolUseBenchmark()
        output = json.dumps({"tool_name": "get_weather", "arguments": {"city": "Paris"}})
        expected = {"tool_name": "get_weather", "arguments": {"city": "Paris"}}
        score = bench.evaluate(output, expected)
        assert score == 1.0

    def test_evaluate_correct_tool_wrong_args(self):
        bench = ToolUseBenchmark()
        output = json.dumps({"tool_name": "get_weather", "arguments": {"city": "London"}})
        expected = {"tool_name": "get_weather", "arguments": {"city": "Paris"}}
        score = bench.evaluate(output, expected)
        assert score == 0.5  # tool name correct, args wrong

    def test_evaluate_wrong_tool(self):
        bench = ToolUseBenchmark()
        output = json.dumps({"tool_name": "search_web", "arguments": {"city": "Paris"}})
        expected = {"tool_name": "get_weather", "arguments": {"city": "Paris"}}
        score = bench.evaluate(output, expected)
        assert score < 1.0
        # wrong tool but args partially match — only args score contributes
        assert score <= 0.5

    def test_evaluate_unparseable(self):
        bench = ToolUseBenchmark()
        score = bench.evaluate("not json at all", {"tool_name": "x", "arguments": {}})
        assert score == 0.0

    def test_evaluate_dict_input(self):
        bench = ToolUseBenchmark()
        output = {"tool_name": "get_weather", "arguments": {"city": "Paris"}}
        expected = {"tool_name": "get_weather", "arguments": {"city": "Paris"}}
        score = bench.evaluate(output, expected)
        assert score == 1.0

    def test_evaluate_markdown_json(self):
        bench = ToolUseBenchmark()
        output = '```json\n{"tool_name": "get_weather", "arguments": {"city": "Paris"}}\n```'
        expected = {"tool_name": "get_weather", "arguments": {"city": "Paris"}}
        score = bench.evaluate(output, expected)
        assert score == 1.0


# ---------------------------------------------------------------------------
# InstructionBenchmark
# ---------------------------------------------------------------------------

class TestInstructionBenchmark:
    def test_generate_cases(self):
        bench = InstructionBenchmark()
        cases = bench.generate_cases()
        assert len(cases) >= 8
        assert bench.name == "instruction"

    def test_evaluate_json(self):
        bench = InstructionBenchmark()
        assert bench.evaluate('["apple", "banana", "cherry"]', {"check_type": "is_json"}) == 1.0
        assert bench.evaluate("not json", {"check_type": "is_json"}) == 0.0

    def test_evaluate_bullet_count(self):
        bench = InstructionBenchmark()
        text = "- one\n- two\n- three"
        assert bench.evaluate(text, {"check_type": "bullet_count", "count": 3}) == 1.0
        assert bench.evaluate("- one\n- two", {"check_type": "bullet_count", "count": 3}) < 1.0

    def test_evaluate_max_words(self):
        bench = InstructionBenchmark()
        short = "This is short."
        long = " ".join(["word"] * 60)
        assert bench.evaluate(short, {"check_type": "max_words", "max_words": 50}) == 1.0
        assert bench.evaluate(long, {"check_type": "max_words", "max_words": 50}) == 0.0

    def test_evaluate_contains_word(self):
        bench = InstructionBenchmark()
        assert bench.evaluate("Dogs show loyalty.", {"check_type": "contains_word", "word": "loyalty"}) == 1.0
        assert bench.evaluate("Dogs are great.", {"check_type": "contains_word", "word": "loyalty"}) == 0.0

    def test_evaluate_all_uppercase(self):
        bench = InstructionBenchmark()
        assert bench.evaluate("HELLO WORLD", {"check_type": "all_uppercase"}) == 1.0
        assert bench.evaluate("Hello World", {"check_type": "all_uppercase"}) == 0.0

    def test_evaluate_numbered_list(self):
        bench = InstructionBenchmark()
        text = "1. Python\n2. Java\n3. Go\n4. Rust\n5. C++"
        assert bench.evaluate(text, {"check_type": "numbered_list", "count": 5}) == 1.0

    def test_evaluate_contains_words(self):
        bench = InstructionBenchmark()
        text = "The blue waves crash on the shore."
        assert bench.evaluate(text, {"check_type": "contains_words", "words": ["waves", "blue"]}) == 1.0
        assert bench.evaluate("The blue sky.", {"check_type": "contains_words", "words": ["waves", "blue"]}) == 0.5

    def test_evaluate_excludes_letter(self):
        bench = InstructionBenchmark()
        assert bench.evaluate("A cat sat.", {"check_type": "excludes_letter", "letter": "e"}) == 1.0
        assert bench.evaluate("The cat.", {"check_type": "excludes_letter", "letter": "e"}) == 0.0

    def test_evaluate_exact_word_count(self):
        bench = InstructionBenchmark()
        ten_words = "one two three four five six seven eight nine ten"
        assert bench.evaluate(ten_words, {"check_type": "exact_word_count", "count": 10}) == 1.0
        assert bench.evaluate("too short", {"check_type": "exact_word_count", "count": 10}) == 0.0

    def test_evaluate_empty_output(self):
        bench = InstructionBenchmark()
        assert bench.evaluate("", {"check_type": "is_json"}) == 0.0


# ---------------------------------------------------------------------------
# BenchmarkRunner with MockProvider
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    @pytest.mark.asyncio
    async def test_run_with_mock_provider(self):
        # A mock that always returns a valid tool call JSON
        good_response = json.dumps({"tool_name": "get_weather", "arguments": {"city": "Paris"}})
        good_provider = MockProvider(default_response=good_response)
        bad_provider = MockProvider(default_response="I don't know")

        runner = BenchmarkRunner(
            providers={"good": good_provider, "bad": bad_provider},
            benchmarks=[ToolUseBenchmark()],
        )
        report = await runner.run()

        assert isinstance(report, BenchmarkReport)
        assert len(report.results) > 0

        # Good provider should score higher on average
        good_scores = [r.score for r in report.results if r.provider_name == "good"]
        bad_scores = [r.score for r in report.results if r.provider_name == "bad"]
        assert sum(good_scores) / len(good_scores) > sum(bad_scores) / len(bad_scores)

    @pytest.mark.asyncio
    async def test_run_measures_latency(self):
        provider = MockProvider(default_response='{"tool_name": "x", "arguments": {}}')
        runner = BenchmarkRunner(
            providers={"mock": provider},
            benchmarks=[ToolUseBenchmark()],
        )
        report = await runner.run()
        for r in report.results:
            assert r.latency_ms >= 0.0

    @pytest.mark.asyncio
    async def test_run_handles_provider_error(self):
        class FailProvider:
            async def complete(self, messages, **kwargs):
                raise RuntimeError("provider exploded")

        runner = BenchmarkRunner(
            providers={"fail": FailProvider()},
            benchmarks=[ToolUseBenchmark()],
        )
        report = await runner.run()
        for r in report.results:
            assert r.error is not None
            assert r.score == 0.0

    @pytest.mark.asyncio
    async def test_run_multiple_benchmarks(self):
        provider = MockProvider(default_response='["a","b","c"]')
        runner = BenchmarkRunner(
            providers={"mock": provider},
            benchmarks=[ToolUseBenchmark(), InstructionBenchmark()],
        )
        report = await runner.run()
        case_ids = [r.case_id for r in report.results]
        assert any(cid.startswith("tool_use/") for cid in case_ids)
        assert any(cid.startswith("instruction/") for cid in case_ids)


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    def test_leaderboard_no_results(self):
        report = BenchmarkReport()
        assert report.leaderboard() == "No results."

    def test_leaderboard_with_results(self):
        results = [
            BenchmarkResult("tool_use/a", "providerA", "x", 0.9, 10.0),
            BenchmarkResult("tool_use/b", "providerA", "x", 0.8, 12.0),
            BenchmarkResult("tool_use/a", "providerB", "x", 0.5, 20.0),
            BenchmarkResult("tool_use/b", "providerB", "x", 0.6, 22.0),
        ]
        report = BenchmarkReport(results=results)
        lb = report.leaderboard()
        assert "providerA" in lb
        assert "providerB" in lb
        assert "tool_use" in lb

    def test_summary(self):
        results = [
            BenchmarkResult("tool_use/a", "p1", "x", 0.9, 10.0),
            BenchmarkResult("tool_use/b", "p1", "x", 0.7, 15.0),
            BenchmarkResult("tool_use/a", "p2", "x", 0.5, 20.0, error="oops"),
        ]
        report = BenchmarkReport(results=results)
        s = report.summary()
        assert s["total_cases"] == 3
        assert "p1" in s["providers"]
        assert "p2" in s["providers"]
        assert s["providers"]["p1"]["avg_score"] == 0.8
        assert s["providers"]["p2"]["errors"] == 1

    def test_export(self):
        results = [
            BenchmarkResult("tool_use/a", "mock", "out", 0.9, 5.0),
        ]
        report = BenchmarkReport(results=results)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.export(path)
            with open(path) as f:
                data = json.load(f)
            assert "summary" in data
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["score"] == 0.9
        finally:
            os.unlink(path)

    def test_summary_empty(self):
        report = BenchmarkReport()
        s = report.summary()
        assert s["total_cases"] == 0
        assert s["providers"] == {}

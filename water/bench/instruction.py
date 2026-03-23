"""
Instruction-following benchmark.

Tests whether an LLM can comply with specific formatting and content
constraints in its output.
"""

import json
import re
from typing import Any, List

from water.bench.base import Benchmark, BenchmarkCase


class InstructionBenchmark(Benchmark):
    """
    Benchmark for instruction-following compliance.

    Each case provides a prompt with a specific constraint and a
    ``check_type`` in metadata describing how to verify compliance.
    """

    @property
    def name(self) -> str:
        return "instruction"

    def generate_cases(self) -> List[BenchmarkCase]:
        return [
            BenchmarkCase(
                id="respond_json",
                prompt="List three fruits. Respond ONLY with valid JSON: a JSON array of strings.",
                expected={"check_type": "is_json"},
            ),
            BenchmarkCase(
                id="three_bullets",
                prompt=(
                    "Give me three benefits of exercise. "
                    "Respond with exactly 3 bullet points, each starting with '- '."
                ),
                expected={"check_type": "bullet_count", "count": 3},
            ),
            BenchmarkCase(
                id="under_50_words",
                prompt="Explain quantum computing in under 50 words.",
                expected={"check_type": "max_words", "max_words": 50},
            ),
            BenchmarkCase(
                id="include_word",
                prompt=(
                    "Write a sentence about dogs. "
                    "You must include the word 'loyalty' in your response."
                ),
                expected={"check_type": "contains_word", "word": "loyalty"},
            ),
            BenchmarkCase(
                id="all_caps",
                prompt="Say 'hello world' but in ALL CAPS.",
                expected={"check_type": "all_uppercase"},
            ),
            BenchmarkCase(
                id="numbered_list",
                prompt=(
                    "List 5 programming languages. "
                    "Respond with a numbered list (1. ... 2. ... etc)."
                ),
                expected={"check_type": "numbered_list", "count": 5},
            ),
            BenchmarkCase(
                id="single_sentence",
                prompt="What is the capital of France? Answer in exactly one sentence.",
                expected={"check_type": "sentence_count", "count": 1},
            ),
            BenchmarkCase(
                id="include_two_words",
                prompt=(
                    "Describe the ocean. "
                    "Your response must include both 'waves' and 'blue'."
                ),
                expected={"check_type": "contains_words", "words": ["waves", "blue"]},
            ),
            BenchmarkCase(
                id="no_letter_e",
                prompt="Write a sentence about a cat without using the letter 'e'.",
                expected={"check_type": "excludes_letter", "letter": "e"},
            ),
            BenchmarkCase(
                id="exactly_10_words",
                prompt="Write a sentence that is exactly 10 words long.",
                expected={"check_type": "exact_word_count", "count": 10},
            ),
        ]

    def evaluate(self, output: Any, expected: Any) -> float:
        """
        Score instruction compliance.

        Returns 1.0 if the constraint is satisfied, 0.0 otherwise.
        Some check types return partial scores.
        """
        if not isinstance(output, str) or not output.strip():
            return 0.0

        text = output.strip()
        check = expected.get("check_type", "")

        if check == "is_json":
            return _check_json(text)

        if check == "bullet_count":
            return _check_bullet_count(text, expected.get("count", 3))

        if check == "max_words":
            return _check_max_words(text, expected.get("max_words", 50))

        if check == "contains_word":
            word = expected.get("word", "")
            return 1.0 if word.lower() in text.lower() else 0.0

        if check == "all_uppercase":
            # Check that alphabetical characters are uppercase
            alpha = "".join(c for c in text if c.isalpha())
            return 1.0 if alpha and alpha == alpha.upper() else 0.0

        if check == "numbered_list":
            return _check_numbered_list(text, expected.get("count", 5))

        if check == "sentence_count":
            return _check_sentence_count(text, expected.get("count", 1))

        if check == "contains_words":
            words = expected.get("words", [])
            if not words:
                return 1.0
            found = sum(1 for w in words if w.lower() in text.lower())
            return found / len(words)

        if check == "excludes_letter":
            letter = expected.get("letter", "")
            return 1.0 if letter.lower() not in text.lower() else 0.0

        if check == "exact_word_count":
            target = expected.get("count", 10)
            actual = len(text.split())
            return 1.0 if actual == target else 0.0

        return 0.0


def _check_json(text: str) -> float:
    """Return 1.0 if text is valid JSON."""
    try:
        json.loads(text)
        return 1.0
    except (json.JSONDecodeError, TypeError):
        pass
    # Try stripping markdown fences
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                json.loads(block)
                return 1.0
            except (json.JSONDecodeError, TypeError):
                continue
    return 0.0


def _check_bullet_count(text: str, count: int) -> float:
    """Return 1.0 if text has exactly ``count`` bullet lines starting with '- '."""
    bullets = [line for line in text.splitlines() if line.strip().startswith("- ")]
    if len(bullets) == count:
        return 1.0
    # Partial credit
    if bullets:
        return max(0.0, 1.0 - abs(len(bullets) - count) / count)
    return 0.0


def _check_max_words(text: str, max_words: int) -> float:
    """Return 1.0 if text is under max_words."""
    word_count = len(text.split())
    return 1.0 if word_count <= max_words else 0.0


def _check_numbered_list(text: str, count: int) -> float:
    """Return 1.0 if text contains numbered items 1..count."""
    found = 0
    for i in range(1, count + 1):
        pattern = rf"^\s*{i}[\.\)]\s"
        if re.search(pattern, text, re.MULTILINE):
            found += 1
    return found / count if count > 0 else 1.0


def _check_sentence_count(text: str, count: int) -> float:
    """Return 1.0 if text has exactly ``count`` sentences."""
    # Simple sentence splitting on . ! ?
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if len(sentences) == count:
        return 1.0
    if sentences:
        return max(0.0, 1.0 - abs(len(sentences) - count) / max(count, 1))
    return 0.0

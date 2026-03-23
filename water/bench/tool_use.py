"""
Tool-use accuracy benchmark.

Tests whether an LLM can correctly identify which tool to call and
extract the right arguments from a user prompt.
"""

import json
from typing import Any, Dict, List

from water.bench.base import Benchmark, BenchmarkCase


class ToolUseBenchmark(Benchmark):
    """
    Benchmark for tool selection and argument extraction.

    Each case contains a user prompt and the expected tool name + arguments.
    The LLM is asked to respond with a JSON object containing ``tool_name``
    and ``arguments``.  Scoring awards 0.5 for correct tool name and 0.5
    for correct arguments.
    """

    @property
    def name(self) -> str:
        return "tool_use"

    def generate_cases(self) -> List[BenchmarkCase]:
        return [
            BenchmarkCase(
                id="weather_paris",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- get_weather(city: str) — Get current weather for a city\n"
                    "- search_web(query: str) — Search the web\n\n"
                    "User: What's the weather in Paris?"
                ),
                expected={"tool_name": "get_weather", "arguments": {"city": "Paris"}},
            ),
            BenchmarkCase(
                id="search_python",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- get_weather(city: str) — Get current weather for a city\n"
                    "- search_web(query: str) — Search the web\n\n"
                    "User: Search for Python tutorials"
                ),
                expected={"tool_name": "search_web", "arguments": {"query": "Python tutorials"}},
            ),
            BenchmarkCase(
                id="send_email",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- send_email(to: str, subject: str, body: str) — Send an email\n"
                    "- create_calendar_event(title: str, date: str) — Create an event\n\n"
                    "User: Email alice@example.com about the meeting tomorrow"
                ),
                expected={
                    "tool_name": "send_email",
                    "arguments": {"to": "alice@example.com"},
                },
                metadata={"partial_args": True},
            ),
            BenchmarkCase(
                id="calendar_event",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- send_email(to: str, subject: str, body: str) — Send an email\n"
                    "- create_calendar_event(title: str, date: str) — Create an event\n\n"
                    "User: Schedule a team standup for March 15"
                ),
                expected={
                    "tool_name": "create_calendar_event",
                    "arguments": {"title": "team standup", "date": "March 15"},
                },
            ),
            BenchmarkCase(
                id="calculate_sum",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- calculate(expression: str) — Evaluate a math expression\n"
                    "- convert_units(value: float, from_unit: str, to_unit: str) — Convert units\n\n"
                    "User: What is 42 * 17?"
                ),
                expected={"tool_name": "calculate", "arguments": {"expression": "42 * 17"}},
            ),
            BenchmarkCase(
                id="convert_temp",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- calculate(expression: str) — Evaluate a math expression\n"
                    "- convert_units(value: float, from_unit: str, to_unit: str) — Convert units\n\n"
                    "User: Convert 100 degrees Fahrenheit to Celsius"
                ),
                expected={
                    "tool_name": "convert_units",
                    "arguments": {"value": 100, "from_unit": "Fahrenheit", "to_unit": "Celsius"},
                },
            ),
            BenchmarkCase(
                id="read_file",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- read_file(path: str) — Read a file's contents\n"
                    "- write_file(path: str, content: str) — Write content to a file\n"
                    "- list_directory(path: str) — List files in a directory\n\n"
                    "User: Show me the contents of /etc/hosts"
                ),
                expected={"tool_name": "read_file", "arguments": {"path": "/etc/hosts"}},
            ),
            BenchmarkCase(
                id="list_dir",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- read_file(path: str) — Read a file's contents\n"
                    "- write_file(path: str, content: str) — Write content to a file\n"
                    "- list_directory(path: str) — List files in a directory\n\n"
                    "User: What files are in the /tmp directory?"
                ),
                expected={"tool_name": "list_directory", "arguments": {"path": "/tmp"}},
            ),
            BenchmarkCase(
                id="translate_text",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- translate(text: str, target_language: str) — Translate text\n"
                    "- summarize(text: str, max_length: int) — Summarize text\n\n"
                    "User: Translate 'Hello world' to Spanish"
                ),
                expected={
                    "tool_name": "translate",
                    "arguments": {"text": "Hello world", "target_language": "Spanish"},
                },
            ),
            BenchmarkCase(
                id="summarize_text",
                prompt=(
                    "You have access to tools. Respond ONLY with a JSON object "
                    '{"tool_name": "<name>", "arguments": {<args>}}.\n\n'
                    "Available tools:\n"
                    "- translate(text: str, target_language: str) — Translate text\n"
                    "- summarize(text: str, max_length: int) — Summarize text\n\n"
                    "User: Summarize this article in 100 words or less: "
                    "The quick brown fox jumps over the lazy dog."
                ),
                expected={
                    "tool_name": "summarize",
                    "arguments": {"text": "The quick brown fox jumps over the lazy dog."},
                },
                metadata={"partial_args": True},
            ),
        ]

    def evaluate(self, output: Any, expected: Any) -> float:
        """
        Score tool use output.

        Awards 0.5 for correct tool name and 0.5 for matching arguments.
        Parses the output as JSON; returns 0.0 if unparseable.
        """
        parsed = _parse_tool_output(output)
        if parsed is None:
            return 0.0

        score = 0.0

        # Tool name match (0.5 points)
        expected_name = expected.get("tool_name", "")
        actual_name = parsed.get("tool_name", "")
        if actual_name == expected_name:
            score += 0.5

        # Arguments match (0.5 points)
        expected_args = expected.get("arguments", {})
        actual_args = parsed.get("arguments", {})
        if expected_args and actual_args:
            matched = 0
            total = len(expected_args)
            for key, val in expected_args.items():
                actual_val = actual_args.get(key)
                if actual_val is not None and _fuzzy_match(actual_val, val):
                    matched += 1
            score += 0.5 * (matched / total) if total > 0 else 0.5
        elif not expected_args and not actual_args:
            score += 0.5

        return score


def _parse_tool_output(output: Any) -> Any:
    """Try to parse a JSON tool call from LLM output."""
    if isinstance(output, dict):
        return output
    if not isinstance(output, str):
        return None
    text = output.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting JSON from markdown code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _fuzzy_match(actual: Any, expected: Any) -> bool:
    """Case-insensitive, type-tolerant comparison."""
    if actual == expected:
        return True
    # Compare string representations case-insensitively
    return str(actual).lower().strip() == str(expected).lower().strip()

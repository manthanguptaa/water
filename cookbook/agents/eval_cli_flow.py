"""
Cookbook: Setting up eval configs for the ``water eval`` CLI.

This example shows how to create eval config files (JSON and YAML)
and run them via the Water CLI.

Usage
-----
1. Create a JSON eval config (see ``write_sample_json_config``).
2. Run it::

       water eval run eval_config.json --format text
       water eval run eval_config.json --format json --output report.json

3. Compare two runs for regressions::

       water eval compare baseline.json current.json

4. Discover eval configs in a directory::

       water eval list ./configs/
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) Programmatic config creation
# ---------------------------------------------------------------------------

SAMPLE_JSON_CONFIG = {
    "suite": "greeting_eval",
    "flow": "cookbook.eval_cli_flow:greeting_flow",
    "evaluators": [
        {"type": "exact_match", "key": "greeting"},
        {"type": "contains", "field": "greeting", "substrings": ["Hello"]},
        {"type": "semantic_similarity", "key": "greeting", "threshold": 0.6},
    ],
    "cases": [
        {
            "name": "basic_greet",
            "input": {"name": "Alice"},
            "expected": {"greeting": "Hello, Alice!"},
            "tags": ["smoke"],
        },
        {
            "name": "empty_name",
            "input": {"name": ""},
            "expected": {"greeting": "Hello, !"},
            "tags": ["edge"],
        },
    ],
}

SAMPLE_YAML_CONFIG = """\
suite: greeting_eval
flow: cookbook.eval_cli_flow:greeting_flow

evaluators:
  - type: exact_match
    key: greeting
  - type: contains
    field: greeting
    substrings:
      - Hello
  - type: semantic_similarity
    key: greeting
    threshold: 0.6

cases:
  - name: basic_greet
    input:
      name: Alice
    expected:
      greeting: "Hello, Alice!"
    tags:
      - smoke
  - name: empty_name
    input:
      name: ""
    expected:
      greeting: "Hello, !"
    tags:
      - edge
"""


def write_sample_json_config(directory: str = ".") -> Path:
    """Write a sample JSON eval config to *directory*."""
    p = Path(directory) / "eval_config.json"
    p.write_text(json.dumps(SAMPLE_JSON_CONFIG, indent=2))
    print(f"Wrote {p}")
    return p


def write_sample_yaml_config(directory: str = ".") -> Path:
    """Write a sample YAML eval config to *directory*."""
    p = Path(directory) / "eval_config.yaml"
    p.write_text(SAMPLE_YAML_CONFIG)
    print(f"Wrote {p}")
    return p


# ---------------------------------------------------------------------------
# 2) A minimal flow to evaluate against
# ---------------------------------------------------------------------------

from pydantic import BaseModel
from water.core.flow import Flow
from water.core.task import Task, create_task


class GreetInput(BaseModel):
    name: str = "World"

class GreetOutput(BaseModel):
    greeting: str


def greet(params, ctx):
    name = params.get("input_data", params).get("name", "World")
    return {"greeting": f"Hello, {name}!"}


greet_task = create_task(
    id="greet",
    input_schema=GreetInput,
    output_schema=GreetOutput,
    execute=greet,
)

greeting_flow = Flow(id="greeting_flow", description="Simple greeting flow")
greeting_flow.then(greet_task).register()


# ---------------------------------------------------------------------------
# 3) Run when invoked directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio
    from water.eval.config import EvalConfig, build_evaluators, build_cases
    from water.eval.suite import EvalSuite

    config = EvalConfig.from_dict(SAMPLE_JSON_CONFIG)
    evaluators = build_evaluators(config.evaluators)
    cases = build_cases(config.cases)
    suite = EvalSuite(flow=greeting_flow, evaluators=evaluators, cases=cases, name=config.suite_name)
    report = asyncio.run(suite.run())
    print(report.summary())
    print()
    print(report.to_json(indent=2))

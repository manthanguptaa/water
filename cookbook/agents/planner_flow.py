"""
cookbook/planner_flow.py -- Dynamic Task Injection with PlannerAgent

Demonstrates how an LLM-driven planner can dynamically choose and
sequence tasks at runtime to satisfy a natural-language goal.

Usage:
    python cookbook/planner_flow.py
"""

import asyncio
import json

from pydantic import BaseModel

from water.core.task import Task
from water.agents.llm import OpenAIProvider
from water.agents.planner import (
    PlannerAgent,
    TaskRegistry,
    create_planner_task,
)


# ---------------------------------------------------------------------------
# 1. Define some mock domain tasks
# ---------------------------------------------------------------------------

class TextInput(BaseModel):
    text: str = ""


class TextOutput(BaseModel):
    text: str = ""


async def fetch_data(params, ctx):
    """Simulate fetching data from an API."""
    url = params.get("url", "https://example.com/data")
    print(f"  [fetch_data] Fetching from {url}")
    return {"raw_data": f"data_from_{url}", "url": url}


async def clean_data(params, ctx):
    """Simulate cleaning raw data."""
    raw = params.get("raw_data", "")
    cleaned = raw.upper().replace("_", " ")
    print(f"  [clean_data] Cleaned: {cleaned}")
    return {"cleaned_data": cleaned}


async def summarise(params, ctx):
    """Simulate summarising cleaned data."""
    cleaned = params.get("cleaned_data", "nothing")
    summary = f"Summary of [{cleaned[:40]}...]"
    print(f"  [summarise] {summary}")
    return {"summary": summary}


async def save_report(params, ctx):
    """Simulate saving a report."""
    summary = params.get("summary", "")
    print(f"  [save_report] Report saved with summary: {summary}")
    return {"saved": True, "report_id": "RPT-001"}


def _make_task(name, fn, desc):
    return Task(
        id=name,
        description=desc,
        input_schema=TextInput,
        output_schema=TextOutput,
        execute=fn,
    )


# ---------------------------------------------------------------------------
# 2. Build the registry
# ---------------------------------------------------------------------------

def build_registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register("fetch_data", _make_task("fetch_data", fetch_data, "Fetch data from a URL"))
    registry.register("clean_data", _make_task("clean_data", clean_data, "Clean and normalise raw data"))
    registry.register("summarise", _make_task("summarise", summarise, "Produce a short summary"))
    registry.register("save_report", _make_task("save_report", save_report, "Save the final report"))
    return registry


# ---------------------------------------------------------------------------
# 3. Run the planner with a MockProvider
# ---------------------------------------------------------------------------

async def demo_planner_agent():
    """Use PlannerAgent directly."""
    print("=" * 60)
    print("Demo 1: PlannerAgent.plan_and_execute")
    print("=" * 60)

    provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.0)
    registry = build_registry()
    agent = PlannerAgent(provider=provider, task_registry=registry)

    result = await agent.plan_and_execute(
        goal="Generate a sales report from the API",
    )

    print("\nFinal accumulated data:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nExecution history:")
    for entry in agent.execution_history:
        print(f"  Step {entry['step']}: {entry['task']} -> {entry['status']}")


async def demo_planner_task():
    """Wrap PlannerAgent as a reusable Task via create_planner_task."""
    print("\n" + "=" * 60)
    print("Demo 2: create_planner_task (composable Task)")
    print("=" * 60)

    provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.0)
    registry = build_registry()

    planner_task = create_planner_task(
        id="inventory_planner",
        description="Dynamically plan and execute inventory analysis",
        provider=provider,
        task_registry=registry,
    )

    output = await planner_task.execute({"goal": "Analyse current inventory"}, None)
    print(f"\nPlanner task output keys: {list(output.keys())}")
    print(f"Plan reasoning: {output['plan']['reasoning']}")
    print(f"Steps executed: {len(output['history'])}")
    for entry in output["history"]:
        print(f"  Step {entry['step']}: {entry['task']} -> {entry['status']}")


# ---------------------------------------------------------------------------
# 4. Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(demo_planner_agent())
    asyncio.run(demo_planner_task())

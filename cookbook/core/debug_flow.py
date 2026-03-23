"""
Cookbook: Debugging a Flow with FlowDebugger
============================================

Demonstrates how to use the programmatic debugger to step through a flow,
inspect state at each task, set breakpoints, modify input mid-execution,
and review history after completion.
"""

import asyncio
from typing import Any, Dict

from pydantic import BaseModel

from water import Flow, create_task
from water.debug import Breakpoint, FlowDebugger


# ---------------------------------------------------------------------------
# 1. Define schemas and simple tasks
# ---------------------------------------------------------------------------

class ValIn(BaseModel):
    value: int


class ValOut(BaseModel):
    value: int


def add_one(params: Dict[str, Any], ctx) -> Dict[str, Any]:
    d = params["input_data"]
    return {"value": d["value"] + 1}


def multiply_two(params: Dict[str, Any], ctx) -> Dict[str, Any]:
    d = params["input_data"]
    return {"value": d["value"] * 2}


def subtract_three(params: Dict[str, Any], ctx) -> Dict[str, Any]:
    d = params["input_data"]
    return {"value": d["value"] - 3}


# ---------------------------------------------------------------------------
# 2. Build a 3-task pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Flow:
    t1 = create_task(id="add_one", execute=add_one, input_schema=ValIn, output_schema=ValOut, validate_schema=False)
    t2 = create_task(id="multiply_two", execute=multiply_two, input_schema=ValIn, output_schema=ValOut, validate_schema=False)
    t3 = create_task(id="subtract_three", execute=subtract_three, input_schema=ValIn, output_schema=ValOut, validate_schema=False)
    flow = Flow(id="debug_demo", description="Debug demo pipeline")
    flow.then(t1).then(t2).then(t3).register()
    return flow


# ---------------------------------------------------------------------------
# 3. Step through every task
# ---------------------------------------------------------------------------

async def demo_step_through():
    print("=== Step-through (all tasks) ===")
    flow = build_pipeline()
    debugger = FlowDebugger(flow)

    async for step in debugger.step_through({"value": 5}):
        print(f"  Step {step.step_number}: {step.task_id}")
        print(f"    Input:  {step.input_data}")
        # After iteration continues, the task runs and output is recorded.

    print("\n  Final history:")
    for s in debugger.get_history():
        print(f"    {s.task_id}: {s.input_data} -> {s.output_data} [{s.status}]")


# ---------------------------------------------------------------------------
# 4. Use breakpoints
# ---------------------------------------------------------------------------

async def demo_breakpoints():
    print("\n=== Breakpoint on 'multiply_two' ===")
    flow = build_pipeline()
    debugger = FlowDebugger(
        flow,
        breakpoints=[Breakpoint(task_id="multiply_two")],
    )

    async for step in debugger.step_through({"value": 10}):
        print(f"  Paused at: {step.task_id}, input = {step.input_data}")
        # Override the input to multiply_two
        step.modify_input({"value": 100})
        print(f"  Modified input to: {{'value': 100}}")

    print("\n  History after breakpoint override:")
    for s in debugger.get_history():
        print(f"    {s.task_id}: {s.input_data} -> {s.output_data}")


# ---------------------------------------------------------------------------
# 5. Inspect history and rewind
# ---------------------------------------------------------------------------

async def demo_rewind():
    print("\n=== Rewind / Time-Travel ===")
    flow = build_pipeline()
    debugger = FlowDebugger(flow)

    async for step in debugger.step_through({"value": 7}):
        pass  # run to completion

    snapshot = debugger.rewind(2)
    print(f"  Rewound to step {snapshot.step_number} ({snapshot.task_id}):")
    print(f"    Input was:  {snapshot.input_data}")
    print(f"    Output was: {snapshot.output_data}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    await demo_step_through()
    await demo_breakpoints()
    await demo_rewind()


if __name__ == "__main__":
    asyncio.run(main())

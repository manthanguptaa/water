"""Tests for the Agentic Debugging & Time-Travel feature (Issue #45)."""

import pytest
from typing import Any, Dict

from pydantic import BaseModel

from water import Flow, create_task
from water.debug import Breakpoint, DebugStep, FlowDebugger


# ---------------------------------------------------------------------------
# Schemas and task helpers
# ---------------------------------------------------------------------------

class ValIn(BaseModel):
    value: int


class ValOut(BaseModel):
    value: int


def _add_one(params: Dict[str, Any], ctx) -> Dict[str, Any]:
    d = params["input_data"]
    return {"value": d["value"] + 1}


def _multiply_two(params: Dict[str, Any], ctx) -> Dict[str, Any]:
    d = params["input_data"]
    return {"value": d["value"] * 2}


def _subtract_three(params: Dict[str, Any], ctx) -> Dict[str, Any]:
    d = params["input_data"]
    return {"value": d["value"] - 3}


def _make_flow(flow_id: str = "debug_test") -> Flow:
    """Build a 3-task pipeline: +1 -> *2 -> -3."""
    t1 = create_task(
        id="add_one", execute=_add_one,
        input_schema=ValIn, output_schema=ValOut,
        description="add one", validate_schema=False,
    )
    t2 = create_task(
        id="multiply_two", execute=_multiply_two,
        input_schema=ValIn, output_schema=ValOut,
        description="multiply by two", validate_schema=False,
    )
    t3 = create_task(
        id="subtract_three", execute=_subtract_three,
        input_schema=ValIn, output_schema=ValOut,
        description="subtract three", validate_schema=False,
    )
    flow = Flow(id=flow_id, description="debug test flow")
    flow.then(t1).then(t2).then(t3).register()
    return flow


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_step_through_all_tasks():
    """Step through a 3-task flow and verify each step yields the correct
    task_id and input data."""
    flow = _make_flow("step_all")
    debugger = FlowDebugger(flow)

    steps = []
    async for step in debugger.step_through({"value": 5}):
        steps.append((step.step_number, step.task_id, step.input_data.copy()))

    assert len(steps) == 3
    assert steps[0] == (1, "add_one", {"value": 5})
    assert steps[1] == (2, "multiply_two", {"value": 6})
    assert steps[2] == (3, "subtract_three", {"value": 12})


@pytest.mark.asyncio
async def test_breakpoint_on_specific_task():
    """With a breakpoint on 'multiply_two', the debugger should only pause
    at that task."""
    flow = _make_flow("bp_task")
    debugger = FlowDebugger(
        flow,
        breakpoints=[Breakpoint(task_id="multiply_two")],
    )

    paused_tasks = []
    async for step in debugger.step_through({"value": 10}):
        paused_tasks.append(step.task_id)

    assert paused_tasks == ["multiply_two"]


@pytest.mark.asyncio
async def test_conditional_breakpoint():
    """A conditional breakpoint pauses only when the condition is met."""
    flow = _make_flow("bp_cond")
    # Pause only when value > 10
    debugger = FlowDebugger(
        flow,
        breakpoints=[Breakpoint(condition=lambda d: d.get("value", 0) > 10)],
    )

    paused_tasks = []
    async for step in debugger.step_through({"value": 5}):
        paused_tasks.append(step.task_id)

    # add_one: input value=5 (no pause)
    # multiply_two: input value=6 (no pause)
    # subtract_three: input value=12 (pause! 12 > 10)
    assert paused_tasks == ["subtract_three"]


@pytest.mark.asyncio
async def test_get_history():
    """After a full step-through, get_history returns all 3 steps."""
    flow = _make_flow("history")
    debugger = FlowDebugger(flow)

    async for step in debugger.step_through({"value": 1}):
        pass  # just consume all steps

    history = debugger.get_history()
    assert len(history) == 3

    # Verify all steps completed
    for s in history:
        assert s.status == "completed"
        assert s.output_data is not None

    # Verify outputs: 1 -> +1=2, 2 -> *2=4, 4 -> -3=1
    assert history[0].output_data == {"value": 2}
    assert history[1].output_data == {"value": 4}
    assert history[2].output_data == {"value": 1}


@pytest.mark.asyncio
async def test_rewind_returns_correct_snapshot():
    """rewind(step_number) returns the correct state snapshot."""
    flow = _make_flow("rewind")
    debugger = FlowDebugger(flow)

    async for step in debugger.step_through({"value": 3}):
        pass

    # Rewind to step 2 (multiply_two)
    snapshot = debugger.rewind(2)
    assert snapshot.step_number == 2
    assert snapshot.task_id == "multiply_two"
    assert snapshot.input_data == {"value": 4}  # 3 + 1 = 4
    assert snapshot.output_data == {"value": 8}  # 4 * 2 = 8
    assert snapshot.status == "completed"


@pytest.mark.asyncio
async def test_rewind_invalid_step():
    """rewind with an invalid step number raises ValueError."""
    flow = _make_flow("rewind_err")
    debugger = FlowDebugger(flow)

    async for step in debugger.step_through({"value": 0}):
        pass

    with pytest.raises(ValueError, match="No step with number 99"):
        debugger.rewind(99)


@pytest.mark.asyncio
async def test_modify_input():
    """modify_input changes the data passed to the task."""
    flow = _make_flow("modify")
    debugger = FlowDebugger(flow)

    async for step in debugger.step_through({"value": 5}):
        if step.task_id == "multiply_two":
            # Override input: instead of 6 (5+1), use 100
            step.modify_input({"value": 100})

    history = debugger.get_history()
    # multiply_two got 100, so output = 200
    assert history[1].output_data == {"value": 200}
    # subtract_three got 200, so output = 197
    assert history[2].output_data == {"value": 197}


@pytest.mark.asyncio
async def test_breakpoint_task_and_condition():
    """A breakpoint with both task_id and condition only triggers when both match."""
    flow = _make_flow("bp_both")
    debugger = FlowDebugger(
        flow,
        breakpoints=[
            Breakpoint(task_id="multiply_two", condition=lambda d: d.get("value", 0) > 100),
        ],
    )

    paused_tasks = []
    async for step in debugger.step_through({"value": 5}):
        paused_tasks.append(step.task_id)

    # multiply_two input is 6, which is NOT > 100, so no pause
    assert paused_tasks == []


@pytest.mark.asyncio
async def test_multiple_breakpoints():
    """Multiple breakpoints each trigger independently."""
    flow = _make_flow("bp_multi")
    debugger = FlowDebugger(
        flow,
        breakpoints=[
            Breakpoint(task_id="add_one"),
            Breakpoint(task_id="subtract_three"),
        ],
    )

    paused_tasks = []
    async for step in debugger.step_through({"value": 5}):
        paused_tasks.append(step.task_id)

    assert paused_tasks == ["add_one", "subtract_three"]

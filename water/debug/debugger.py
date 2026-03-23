"""
Agentic Debugging & Time-Travel for Water flows.

Provides a programmatic debugger that wraps flow execution with step-by-step
control, breakpoints, state inspection, input modification, and time-travel
(rewind to any previous step and re-execute from that point).

The debugger works as middleware injected into the flow, intercepting
before_task / after_task hooks to pause execution and yield control back
to the caller.
"""

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from water.middleware.base import Middleware

__all__ = [
    "Breakpoint",
    "DebugStep",
    "FlowDebugger",
]


@dataclass
class Breakpoint:
    """A breakpoint that causes the debugger to pause execution.

    Attributes:
        task_id: If set, pause when execution reaches the task with this ID.
        condition: If set, pause only when this callable returns True for the
            current input data dict.  When both ``task_id`` and ``condition``
            are set, both must match for the breakpoint to trigger.
    """

    task_id: Optional[str] = None
    condition: Optional[Callable[[dict], bool]] = None

    def matches(self, task_id: str, data: dict) -> bool:
        """Return True if this breakpoint should trigger."""
        if self.task_id is not None and self.task_id != task_id:
            return False
        if self.condition is not None and not self.condition(data):
            return False
        # At least one criterion must be specified
        return self.task_id is not None or self.condition is not None


@dataclass
class DebugStep:
    """Snapshot of execution state at a single task boundary.

    Yielded by :meth:`FlowDebugger.step_through` before each task executes.
    The caller can inspect ``input_data`` and optionally call
    :meth:`modify_input` to change the data before the task runs.

    Attributes:
        step_number: 1-based ordinal within the debug session.
        task_id: The ID of the task about to execute.
        input_data: The data dict the task will receive.
        output_data: Populated after the task completes (None while pending).
        status: One of ``"pending"``, ``"completed"``, or ``"error"``.
        error: Error message if the task raised an exception.
    """

    step_number: int
    task_id: str
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending / completed / error
    error: Optional[str] = None

    # -- internal plumbing (not part of public snapshot) --
    _modified_input: Optional[Dict[str, Any]] = field(
        default=None, repr=False, compare=False
    )

    def modify_input(self, new_data: Dict[str, Any]) -> None:
        """Replace the input data that will be passed to the task."""
        self._modified_input = new_data


class _DebugMiddleware(Middleware):
    """Internal middleware that pauses execution at each task for the debugger."""

    def __init__(self, debugger: "FlowDebugger") -> None:
        super().__init__(order=-1000)  # run first
        self._debugger = debugger

    async def before_task(self, task_id: str, data: dict, context: Any) -> dict:
        """Signal the debugger that a task is about to run, then wait."""
        return await self._debugger._on_before_task(task_id, data)

    async def after_task(
        self, task_id: str, data: dict, result: dict, context: Any
    ) -> dict:
        """Record the task output in the current debug step."""
        self._debugger._on_after_task(task_id, result)
        return result


class FlowDebugger:
    """Step-through debugger for Water flows with breakpoint and time-travel support.

    Usage::

        debugger = FlowDebugger(flow, breakpoints=[Breakpoint(task_id="analyse")])
        async for step in debugger.step_through({"query": "hello"}):
            print(step.task_id, step.input_data)
            if step.task_id == "analyse":
                step.modify_input({"query": "override"})

    Parameters
    ----------
    flow:
        A **registered** ``Flow`` instance.
    breakpoints:
        Optional list of :class:`Breakpoint` instances.  When no breakpoints
        are provided the debugger pauses at *every* task (full step-through
        mode).
    """

    def __init__(self, flow: Any, breakpoints: Optional[List[Breakpoint]] = None):
        self._flow = flow
        self._breakpoints: List[Breakpoint] = breakpoints or []
        self._history: List[DebugStep] = []
        self._step_counter: int = 0

        # Synchronisation primitives for pause/resume between the flow
        # execution task and the caller iterating via step_through.
        self._step_ready = asyncio.Event()  # middleware sets when step ready
        self._step_consumed = asyncio.Event()  # caller sets after inspecting
        self._current_step: Optional[DebugStep] = None
        self._flow_done = asyncio.Event()
        self._flow_error: Optional[Exception] = None
        self._flow_result: Optional[Dict[str, Any]] = None

        # Inject debug middleware
        self._middleware = _DebugMiddleware(self)
        self._flow.middleware.append(self._middleware)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def step_through(self, input_data: Dict[str, Any]) -> AsyncIterator[DebugStep]:
        """Run the flow, yielding a :class:`DebugStep` before each task.

        The caller can inspect each step and optionally call
        ``step.modify_input(...)`` before continuing iteration. Iteration
        drives execution forward.
        """
        self._history.clear()
        self._step_counter = 0
        self._flow_done.clear()
        self._flow_error = None
        self._flow_result = None

        # Launch the flow in a background task
        loop = asyncio.get_event_loop()
        flow_task = loop.create_task(self._run_flow(input_data))

        try:
            while True:
                # Wait for either the next step or the flow to finish.
                step_ready_task = loop.create_task(self._step_ready.wait())
                flow_done_task = loop.create_task(self._flow_done.wait())

                done, pending = await asyncio.wait(
                    {step_ready_task, flow_done_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for p in pending:
                    p.cancel()
                    try:
                        await p
                    except asyncio.CancelledError:
                        pass

                if self._step_ready.is_set() and self._current_step is not None:
                    self._step_ready.clear()
                    yield self._current_step
                    # Signal the middleware to continue
                    self._step_consumed.set()
                elif self._flow_done.is_set():
                    break
        finally:
            # Clean up: ensure the flow task completes
            if not flow_task.done():
                flow_task.cancel()
                try:
                    await flow_task
                except asyncio.CancelledError:
                    pass

            # Remove our middleware so the flow can be reused
            if self._middleware in self._flow.middleware:
                self._flow.middleware.remove(self._middleware)

            if self._flow_error is not None:
                raise self._flow_error

    def get_history(self) -> List[DebugStep]:
        """Return all recorded debug steps (completed or not)."""
        return list(self._history)

    def rewind(self, step_number: int) -> DebugStep:
        """Return a deep copy of the debug step at the given step number.

        This snapshot contains the ``input_data`` that was passed to the task
        at that step, enabling the caller to re-execute the flow from that
        point using :class:`~water.core.replay.ReplayEngine`.

        Raises:
            ValueError: If the step number is out of range.
        """
        for step in self._history:
            if step.step_number == step_number:
                return DebugStep(
                    step_number=step.step_number,
                    task_id=step.task_id,
                    input_data=copy.deepcopy(step.input_data),
                    output_data=copy.deepcopy(step.output_data) if step.output_data else None,
                    status=step.status,
                    error=step.error,
                )
        raise ValueError(
            f"No step with number {step_number}. "
            f"Valid range: 1..{len(self._history)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers (called by _DebugMiddleware)
    # ------------------------------------------------------------------

    async def _on_before_task(self, task_id: str, data: dict) -> dict:
        """Called by the debug middleware before each task."""
        self._step_counter += 1
        step = DebugStep(
            step_number=self._step_counter,
            task_id=task_id,
            input_data=copy.deepcopy(data),
        )
        self._history.append(step)

        should_pause = self._should_pause(task_id, data)

        if should_pause:
            self._current_step = step
            self._step_consumed.clear()
            self._step_ready.set()
            # Wait for the caller to consume this step
            await self._step_consumed.wait()

            # If the caller modified input, use the new data
            if step._modified_input is not None:
                step.input_data = copy.deepcopy(step._modified_input)
                return step._modified_input

        return data

    def _on_after_task(self, task_id: str, result: dict) -> None:
        """Called by the debug middleware after each task."""
        # Update the most recent step for this task
        for step in reversed(self._history):
            if step.task_id == task_id and step.status == "pending":
                step.output_data = copy.deepcopy(result)
                step.status = "completed"
                break

    def _should_pause(self, task_id: str, data: dict) -> bool:
        """Determine whether execution should pause at this task."""
        if not self._breakpoints:
            # No breakpoints = step-through every task
            return True
        return any(bp.matches(task_id, data) for bp in self._breakpoints)

    async def _run_flow(self, input_data: Dict[str, Any]) -> None:
        """Execute the flow in the background."""
        try:
            self._flow_result = await self._flow.run(input_data)
        except Exception as exc:
            logger.exception("Flow execution failed during debugging")
            self._flow_error = exc
        finally:
            self._flow_done.set()

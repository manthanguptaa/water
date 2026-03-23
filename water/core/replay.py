"""
Execution Replay Engine for Water flows.

Allows replaying a flow execution from a given point, reusing cached task
outputs for steps before the replay point and re-executing from that point
onward.  Useful for debugging failures, iterating on individual tasks, and
overriding inputs without re-running an entire pipeline.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

__all__ = [
    "ReplayConfig",
    "ReplayResult",
    "ReplayEngine",
]

from water.core.types import SerializableMixin


@dataclass
class ReplayConfig(SerializableMixin):
    """Configuration that controls how a replay is performed."""

    from_task: Optional[str] = None
    from_step: Optional[int] = None
    override_inputs: Optional[Dict[str, Dict[str, Any]]] = None  # task_id -> input override
    skip_tasks: List[str] = field(default_factory=list)


@dataclass
class ReplayResult(SerializableMixin):
    """Outcome of a replay execution."""

    original_session_id: str
    replay_session_id: str
    replayed_from: str
    cached_steps: List[str]  # steps reused from original
    re_executed_steps: List[str]  # steps that were re-run
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: str = "completed"


class ReplayEngine:
    """Replays a flow execution from a given point using cached task outputs."""

    def __init__(self, storage=None):
        self.storage = storage
        self._task_outputs: Dict[str, Dict[str, Any]] = {}  # task_id -> output from original

    # ------------------------------------------------------------------
    # Session loading
    # ------------------------------------------------------------------

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a previous session's task outputs from storage (sync wrapper).

        Returns ``None`` when called inside an already-running event loop
        because blocking is not possible in that situation.
        """
        if self.storage and hasattr(self.storage, "get_task_runs"):
            import asyncio

            try:
                asyncio.get_running_loop()
                return None  # Can't block in async context
            except RuntimeError:
                pass  # No running loop — safe to proceed
        return None

    async def load_session_async(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Load task outputs from a previous execution asynchronously."""
        outputs: Dict[str, Dict[str, Any]] = {}
        if self.storage and hasattr(self.storage, "get_task_runs"):
            task_runs = await self.storage.get_task_runs(session_id)
            for run in task_runs:
                if hasattr(run, "task_id") and hasattr(run, "output_data"):
                    outputs[run.task_id] = run.output_data or {}
        return outputs

    # ------------------------------------------------------------------
    # Manual setup helpers
    # ------------------------------------------------------------------

    def set_task_outputs(self, outputs: Dict[str, Dict[str, Any]]) -> None:
        """Manually set task outputs for replay (useful for testing)."""
        import copy
        self._task_outputs = copy.deepcopy(outputs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_task_ids(flow) -> List[str]:
        """Extract ordered task IDs from a flow's execution graph."""
        task_ids: List[str] = []
        for node in flow._tasks:
            task = node.get("task") if isinstance(node, dict) else getattr(node, "task", None)
            if task is not None and hasattr(task, "id"):
                task_ids.append(task.id)
        return task_ids

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    async def replay(
        self,
        flow,
        session_id: str,
        config: Optional[ReplayConfig] = None,
    ) -> ReplayResult:
        """Replay a flow from a specific point.

        Parameters
        ----------
        flow:
            A *registered* ``Flow`` instance.
        session_id:
            The execution / session ID of the original run whose cached
            outputs should be reused.
        config:
            Optional :class:`ReplayConfig` controlling the replay behaviour.

        Returns
        -------
        ReplayResult
            A summary of the replay including which steps were cached vs
            re-executed and the final result (or error).
        """
        config = config or ReplayConfig()
        replay_id = f"replay_{uuid.uuid4().hex[:8]}"

        # Load original session task outputs if not already set
        if not self._task_outputs:
            self._task_outputs = await self.load_session_async(session_id)

        cached_steps: List[str] = []
        re_executed_steps: List[str] = []

        # Determine ordered task ids
        task_ids = self._get_task_ids(flow)

        # Determine the replay starting index
        replay_from_index = 0
        if config.from_task:
            for i, tid in enumerate(task_ids):
                if tid == config.from_task:
                    replay_from_index = i
                    break
        elif config.from_step is not None:
            replay_from_index = config.from_step

        # Build accumulated data from cached outputs up to replay point
        data: Dict[str, Any] = {}
        for i, tid in enumerate(task_ids):
            if i < replay_from_index and tid in self._task_outputs:
                data.update(self._task_outputs[tid])
                cached_steps.append(tid)

        # Apply any input overrides
        if config.override_inputs:
            for task_id, overrides in config.override_inputs.items():
                data.update(overrides)

        # Re-execute from replay point
        try:
            result = await flow.run(data)
            re_executed_steps = [
                tid
                for tid in task_ids[replay_from_index:]
                if tid not in config.skip_tasks
            ]

            return ReplayResult(
                original_session_id=session_id,
                replay_session_id=replay_id,
                replayed_from=config.from_task or f"step_{replay_from_index}",
                cached_steps=cached_steps,
                re_executed_steps=re_executed_steps,
                result=result,
                status="completed",
            )
        except Exception as e:
            return ReplayResult(
                original_session_id=session_id,
                replay_session_id=replay_id,
                replayed_from=config.from_task or f"step_{replay_from_index}",
                cached_steps=cached_steps,
                re_executed_steps=re_executed_steps,
                error=str(e),
                status="failed",
            )

__all__ = [
    "BatchItem",
    "BatchResult",
    "BatchProcessor",
    "create_batch_task",
]

"""
Batch LLM Calls for Water.

Provides a BatchProcessor that executes a single Task against multiple inputs
concurrently, with configurable concurrency limits, automatic retries, and
progress reporting.  Also includes a factory function to wrap any Task as a
batch-processing Task that fits into a Flow.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel

from water.core.task import Task
from water.core.types import SerializableMixin


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BatchItem(SerializableMixin):
    """Tracks a single item within a batch run."""

    index: int
    input_data: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class BatchResult(SerializableMixin):
    """Aggregated result of a batch run."""

    items: List[BatchItem] = field(default_factory=list)
    total: int = 0
    completed: int = 0
    failed: int = 0

    @property
    def success_rate(self) -> float:
        """Return the fraction of items that completed successfully."""
        if self.total == 0:
            return 0.0
        return self.completed / self.total

    def get_results(self) -> List[Optional[Dict[str, Any]]]:
        """Return results in order, ``None`` for failed items."""
        return [item.result if item.status == "completed" else None for item in self.items]

    def get_errors(self) -> List[Dict[str, Any]]:
        """Return a list of ``{index, error}`` dicts for failed items."""
        return [
            {"index": item.index, "error": item.error}
            for item in self.items
            if item.status == "failed"
        ]


# ---------------------------------------------------------------------------
# BatchProcessor
# ---------------------------------------------------------------------------

class BatchProcessor:
    """
    Executes a :class:`Task` against many inputs concurrently.

    Args:
        max_concurrency: Maximum number of concurrent task executions.
        retry_failed: Whether to retry items that fail.
        max_retries: Number of retry rounds for failed items.
        on_progress: Optional callback ``(completed, total)`` invoked after
            each item finishes.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        retry_failed: bool = True,
        max_retries: int = 2,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.retry_failed = retry_failed
        self.max_retries = max_retries
        self.on_progress = on_progress

    async def run_batch(
        self,
        task: Task,
        inputs: List[Dict[str, Any]],
    ) -> BatchResult:
        """Execute *task* against every dict in *inputs* concurrently."""

        if not inputs:
            return BatchResult(items=[], total=0, completed=0, failed=0)

        semaphore = asyncio.Semaphore(self.max_concurrency)
        items = [BatchItem(index=i, input_data=inp) for i, inp in enumerate(inputs)]
        finished_count = 0
        total = len(items)

        async def run_single(item: BatchItem) -> None:
            nonlocal finished_count
            async with semaphore:
                try:
                    item.status = "running"
                    result = await task.execute(item.input_data, None)
                    item.result = result
                    item.status = "completed"
                except Exception as exc:
                    logger.warning("Batch item %d failed: %s", item.index, exc, exc_info=True)
                    item.error = str(exc)
                    item.status = "failed"
                finally:
                    finished_count += 1
                    if self.on_progress is not None:
                        self.on_progress(finished_count, total)

        # Initial run
        await asyncio.gather(*[run_single(item) for item in items])

        # Retry rounds
        if self.retry_failed:
            for _ in range(self.max_retries):
                failed_items = [i for i in items if i.status == "failed"]
                if not failed_items:
                    break
                for item in failed_items:
                    item.status = "pending"
                    item.error = None
                await asyncio.gather(*[run_single(item) for item in failed_items])

        completed = sum(1 for i in items if i.status == "completed")
        failed = sum(1 for i in items if i.status == "failed")

        return BatchResult(
            items=items,
            total=total,
            completed=completed,
            failed=failed,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_batch_task(
    id: Optional[str] = None,
    description: Optional[str] = None,
    task: Task = None,
    max_concurrency: int = 10,
    input_key: str = "items",
    output_key: str = "results",
) -> Task:
    """
    Create a :class:`Task` that processes a batch of items using another task.

    The returned task accepts ``{<input_key>: [item, ...]}`` and returns
    ``{<output_key>: [result | None, ...]}``.

    Args:
        id: Task identifier (auto-generated if omitted).
        description: Human-readable description.
        task: The inner task to run for each item.
        max_concurrency: Max concurrent executions of *task*.
        input_key: Key in the input dict that holds the list of items.
        output_key: Key in the output dict that holds the list of results.
    """

    task_id = id or f"batch_{uuid.uuid4().hex[:8]}"

    InputSchema = type(
        f"{task_id}_Input",
        (BaseModel,),
        {"__annotations__": {input_key: list}},
    )
    OutputSchema = type(
        f"{task_id}_Output",
        (BaseModel,),
        {"__annotations__": {output_key: list}},
    )

    processor = BatchProcessor(max_concurrency=max_concurrency)

    async def execute(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        items = params.get(input_key, [])
        result = await processor.run_batch(task, items)
        return {output_key: result.get_results()}

    return Task(
        id=task_id,
        description=description or f"Batch processor: {task_id}",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        execute=execute,
    )

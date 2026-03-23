import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


class FlowEvent:
    """Represents a single event emitted during flow execution."""

    def __init__(
        self,
        event_type: str,
        flow_id: str,
        data: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
        execution_id: Optional[str] = None,
    ) -> None:
        self.event_type = event_type
        self.flow_id = flow_id
        self.task_id = task_id
        self.execution_id = execution_id
        self.data = data or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "flow_id": self.flow_id,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return f"FlowEvent({self.event_type}, flow={self.flow_id}, task={self.task_id})"


class EventEmitter:
    """
    Async event emitter for real-time flow execution updates.

    Consumers can iterate over events using `async for event in emitter.subscribe()`.
    """

    def __init__(self) -> None:
        self._queues: List[asyncio.Queue] = []
        self._closed = False

    async def emit(self, event: FlowEvent) -> None:
        """Push an event to all subscribers."""
        for queue in self._queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Event queue full, dropping event: {event}")

    def subscribe(self, max_queue_size: int = 1000) -> "EventSubscription":
        """
        Create a new subscription to receive events.

        Returns:
            An async iterable of FlowEvent objects.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._queues.append(queue)
        return EventSubscription(queue, self)

    def _unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        if queue in self._queues:
            self._queues.remove(queue)

    async def close(self) -> None:
        """Signal all subscribers that no more events will be emitted."""
        self._closed = True
        sentinel = None
        for queue in self._queues:
            try:
                queue.put_nowait(sentinel)
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._queues)


class EventSubscription:
    """Async iterable subscription to flow events."""

    def __init__(self, queue: asyncio.Queue, emitter: EventEmitter) -> None:
        self._queue = queue
        self._emitter = emitter

    async def __aiter__(self) -> AsyncIterator[FlowEvent]:
        try:
            while True:
                event = await self._queue.get()
                if event is None:  # Sentinel for close
                    break
                yield event
        finally:
            self._emitter._unsubscribe(self._queue)

    async def get(self, timeout: Optional[float] = None) -> Optional[FlowEvent]:
        """Get the next event, optionally with a timeout."""
        try:
            if timeout:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    def close(self) -> None:
        """Unsubscribe from further events."""
        self._emitter._unsubscribe(self._queue)

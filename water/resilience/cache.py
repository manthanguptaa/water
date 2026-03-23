"""
Task caching/memoization support for Water flows.

Provides a cache abstraction and an in-memory implementation so that
expensive task executions can be skipped when the same input is seen again.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class TaskCache(ABC):
    """Abstract base class for task result caches."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key*, or None on miss."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store *value* under *key* with an optional TTL in seconds."""
        ...

    @abstractmethod
    def has(self, key: str) -> bool:
        """Return True if *key* is present and not expired."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the cache."""
        ...


class InMemoryCache(TaskCache):
    """Simple dict-backed cache with optional per-entry TTL."""

    def __init__(self) -> None:
        # _store maps key -> (value, expire_at | None)
        self._store: Dict[str, Tuple[Any, Optional[float]]] = {}

    # -- internal helpers --------------------------------------------------

    def _is_expired(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return True
        _, expire_at = entry
        if expire_at is not None and time.monotonic() >= expire_at:
            del self._store[key]
            return True
        return False

    # -- public API --------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        if self._is_expired(key):
            return None
        value, _ = self._store[key]
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expire_at = (time.monotonic() + ttl) if ttl is not None else None
        self._store[key] = (value, expire_at)

    def has(self, key: str) -> bool:
        return not self._is_expired(key)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_SENTINEL = object()


def cache_key(task_id: str, data: Any) -> str:
    """Create a deterministic hash from a task id and its input data."""
    payload = json.dumps({"task_id": task_id, "data": data}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

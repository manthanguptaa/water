"""
Task caching/memoization support for Water flows.

Provides a cache abstraction and an in-memory implementation so that
expensive task executions can be skipped when the same input is seen again.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
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
    """Simple dict-backed cache with optional per-entry TTL and max size.

    Parameters
    ----------
    max_size:
        Maximum number of entries. When exceeded, expired entries are removed
        first, then the oldest (least recently used) entries are evicted.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        # _store maps key -> (value, expire_at | None)
        self._store: OrderedDict[str, Tuple[Any, Optional[float]]] = OrderedDict()

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
        # Mark as recently used
        self._store.move_to_end(key)
        value, _ = self._store[key]
        return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expire_at = (time.monotonic() + ttl) if ttl is not None else None
        # If key already exists, remove it first so it moves to end
        if key in self._store:
            del self._store[key]
        self._store[key] = (value, expire_at)
        # Enforce max_size
        if len(self._store) > self.max_size:
            self.cleanup()
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def has(self, key: str) -> bool:
        return not self._is_expired(key)

    def clear(self) -> None:
        self._store.clear()

    def cleanup(self) -> None:
        """Remove all expired entries from the cache."""
        now = time.monotonic()
        expired_keys = [
            k for k, (_, expire_at) in self._store.items()
            if expire_at is not None and now >= expire_at
        ]
        for k in expired_keys:
            del self._store[k]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_SENTINEL = object()


def cache_key(task_id: str, data: Any) -> str:
    """Create a deterministic hash from a task id and its input data."""
    payload = json.dumps({"task_id": task_id, "data": data}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

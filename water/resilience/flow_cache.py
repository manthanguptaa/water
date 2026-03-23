__all__ = [
    "FlowCacheBackend",
    "InMemoryFlowCache",
    "CacheEntry",
    "CacheStats",
    "FlowCache",
]

"""
Flow-level output caching for Water flows.

Provides an async cache abstraction designed for caching entire flow outputs
(as opposed to individual task results). Includes an in-memory backend,
configurable TTL, custom key functions, and hit/miss statistics tracking.
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod

from water.core.types import SerializableMixin


class FlowCacheBackend(ABC):
    """Abstract async backend for flow-level caching."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Return the cached value for *key*, or ``None`` on miss."""
        ...

    @abstractmethod
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Store *value* under *key* with an optional TTL in seconds."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove *key* from the cache. Return ``True`` if it existed."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all entries from the cache."""
        ...


class InMemoryFlowCache(FlowCacheBackend):
    """Simple dict-backed async cache with optional per-entry TTL."""

    def __init__(self) -> None:
        # _store maps key -> {"value": ..., "expires_at": float | None}
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _is_expired(self, key: str) -> bool:
        entry = self._store.get(key)
        if entry is None:
            return True
        expires_at = entry.get("expires_at")
        if expires_at is not None and time.monotonic() >= expires_at:
            del self._store[key]
            return True
        return False

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self._is_expired(key):
            return None
        self._store.move_to_end(key)
        return self._store[key]["value"]

    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        expires_at = (time.monotonic() + ttl) if ttl is not None else None
        if key in self._store:
            del self._store[key]
        self._store[key] = {"value": value, "expires_at": expires_at}

    async def evict_oldest(self) -> None:
        """Remove the oldest entry from the cache."""
        if self._store:
            self._store.popitem(last=False)

    async def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def clear(self) -> None:
        self._store.clear()


@dataclass
class CacheEntry(SerializableMixin):
    """Represents a single cached flow result."""

    key: str
    value: Dict[str, Any]
    created_at: float
    expires_at: Optional[float] = None
    hit_count: int = 0


@dataclass
class CacheStats:
    """Tracks cache hit/miss statistics."""

    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a float between 0.0 and 1.0."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class FlowCache:
    """High-level flow output cache with stats tracking.

    Parameters
    ----------
    backend:
        The storage backend (defaults to :class:`InMemoryFlowCache`).
    ttl:
        Default time-to-live in seconds for cached entries.
    max_size:
        Maximum number of entries. When exceeded, the oldest entry is evicted.
    key_fn:
        Optional callable that takes flow input data and returns a cache key
        string.  When ``None`` a SHA-256 hash of the JSON-serialised input is
        used.
    """

    def __init__(
        self,
        backend: Optional[FlowCacheBackend] = None,
        ttl: int = 3600,
        max_size: int = 1000,
        key_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        self.backend = backend or InMemoryFlowCache()
        self.ttl = ttl
        self.max_size = max_size
        self.key_fn = key_fn
        self.stats = CacheStats()

    def _generate_key(self, input_data: Dict[str, Any]) -> str:
        """Produce a deterministic cache key for the given input."""
        if self.key_fn:
            return self.key_fn(input_data)
        payload = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Look up a cached result for *input_data*.

        Returns the cached value on hit (incrementing ``stats.hits``), or
        ``None`` on miss (incrementing ``stats.misses``).
        """
        key = self._generate_key(input_data)
        result = await self.backend.get(key)
        if result is not None:
            self.stats.hits += 1
            return result
        self.stats.misses += 1
        return None

    async def set(self, input_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache a flow result for *input_data*."""
        key = self._generate_key(input_data)
        await self.backend.set(key, result, ttl=self.ttl)
        self.stats.size += 1
        # Enforce max_size by evicting oldest entries
        while self.stats.size > self.max_size:
            if isinstance(self.backend, InMemoryFlowCache):
                await self.backend.evict_oldest()
            self.stats.size -= 1

    async def invalidate(self, input_data: Dict[str, Any]) -> bool:
        """Remove the cached entry for *input_data*.

        Returns ``True`` if an entry was removed.
        """
        key = self._generate_key(input_data)
        deleted = await self.backend.delete(key)
        if deleted:
            self.stats.size = max(0, self.stats.size - 1)
        return deleted

    async def clear(self) -> None:
        """Clear all cached entries and reset the size counter."""
        await self.backend.clear()
        self.stats.size = 0

    def get_stats(self) -> CacheStats:
        """Return a snapshot of the current cache statistics."""
        return self.stats

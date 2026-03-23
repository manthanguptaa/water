"""Tests for flow-level output caching (water.resilience.flow_cache)."""

import pytest
import time

from water.resilience.flow_cache import (
    FlowCache,
    FlowCacheBackend,
    InMemoryFlowCache,
    CacheStats,
    CacheEntry,
)


# ---------------------------------------------------------------------------
# InMemoryFlowCache backend tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_backend_get_set():
    """Basic get/set round-trip on the in-memory backend."""
    backend = InMemoryFlowCache()
    await backend.set("k1", {"answer": 42})
    result = await backend.get("k1")
    assert result == {"answer": 42}


@pytest.mark.asyncio
async def test_in_memory_backend_get_missing_key():
    """Getting a non-existent key returns None."""
    backend = InMemoryFlowCache()
    assert await backend.get("no_such_key") is None


@pytest.mark.asyncio
async def test_in_memory_backend_delete():
    """Deleting a key removes it and returns True; missing key returns False."""
    backend = InMemoryFlowCache()
    await backend.set("k1", {"v": 1})
    assert await backend.delete("k1") is True
    assert await backend.get("k1") is None
    assert await backend.delete("k1") is False


@pytest.mark.asyncio
async def test_in_memory_backend_clear():
    """Clear removes all entries."""
    backend = InMemoryFlowCache()
    await backend.set("a", {"v": 1})
    await backend.set("b", {"v": 2})
    await backend.clear()
    assert await backend.get("a") is None
    assert await backend.get("b") is None


@pytest.mark.asyncio
async def test_in_memory_backend_ttl_expiration():
    """An entry set with a short TTL expires after the deadline."""
    backend = InMemoryFlowCache()
    await backend.set("ttl_key", {"v": 1}, ttl=0.05)

    # Should still be present immediately
    assert await backend.get("ttl_key") == {"v": 1}

    time.sleep(0.06)

    # Should have expired
    assert await backend.get("ttl_key") is None


# ---------------------------------------------------------------------------
# FlowCache key generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flow_cache_default_key_generation():
    """Default key generation produces the same hash for identical inputs."""
    fc = FlowCache()
    key1 = fc._generate_key({"city": "London"})
    key2 = fc._generate_key({"city": "London"})
    key3 = fc._generate_key({"city": "Paris"})
    assert key1 == key2
    assert key1 != key3


@pytest.mark.asyncio
async def test_flow_cache_custom_key_fn():
    """A custom key_fn is used instead of the default hash."""
    fc = FlowCache(key_fn=lambda d: d.get("id", "unknown"))

    await fc.set({"id": "req-1", "payload": "abc"}, {"result": "ok"})
    result = await fc.get({"id": "req-1", "payload": "abc"})
    assert result == {"result": "ok"}

    # Same key even with different payload because key_fn only uses "id"
    result2 = await fc.get({"id": "req-1", "payload": "xyz"})
    assert result2 == {"result": "ok"}


# ---------------------------------------------------------------------------
# FlowCache hit / miss / stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flow_cache_hit_tracking():
    """A cache hit increments stats.hits."""
    fc = FlowCache()
    await fc.set({"x": 1}, {"y": 2})
    await fc.get({"x": 1})
    assert fc.stats.hits == 1
    assert fc.stats.misses == 0


@pytest.mark.asyncio
async def test_flow_cache_miss_tracking():
    """A cache miss increments stats.misses."""
    fc = FlowCache()
    await fc.get({"x": 999})
    assert fc.stats.misses == 1
    assert fc.stats.hits == 0


@pytest.mark.asyncio
async def test_hit_rate_calculation():
    """hit_rate returns the correct ratio of hits to total lookups."""
    stats = CacheStats(hits=3, misses=7)
    assert stats.hit_rate == pytest.approx(0.3)


def test_hit_rate_zero_total():
    """hit_rate is 0.0 when no lookups have been performed."""
    stats = CacheStats()
    assert stats.hit_rate == 0.0


# ---------------------------------------------------------------------------
# FlowCache invalidation and clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flow_cache_invalidate():
    """Invalidating an entry removes it and adjusts size."""
    fc = FlowCache()
    await fc.set({"a": 1}, {"b": 2})
    assert fc.stats.size == 1

    removed = await fc.invalidate({"a": 1})
    assert removed is True
    assert fc.stats.size == 0

    # Getting it now should miss
    assert await fc.get({"a": 1}) is None
    assert fc.stats.misses == 1


@pytest.mark.asyncio
async def test_flow_cache_clear():
    """Clearing the cache resets size and removes all entries."""
    fc = FlowCache()
    await fc.set({"a": 1}, {"r": 1})
    await fc.set({"b": 2}, {"r": 2})
    assert fc.stats.size == 2

    await fc.clear()
    assert fc.stats.size == 0
    assert await fc.get({"a": 1}) is None


# ---------------------------------------------------------------------------
# CacheStats / CacheEntry dataclass basics
# ---------------------------------------------------------------------------


def test_cache_stats_properties():
    """CacheStats defaults and computed hit_rate."""
    stats = CacheStats()
    assert stats.hits == 0
    assert stats.misses == 0
    assert stats.size == 0
    assert stats.hit_rate == 0.0

    stats.hits = 50
    stats.misses = 50
    assert stats.hit_rate == pytest.approx(0.5)


def test_cache_entry_fields():
    """CacheEntry stores expected fields."""
    entry = CacheEntry(
        key="abc",
        value={"result": 42},
        created_at=1000.0,
        expires_at=2000.0,
        hit_count=3,
    )
    assert entry.key == "abc"
    assert entry.value == {"result": 42}
    assert entry.hit_count == 3


# ---------------------------------------------------------------------------
# FlowCache max_size tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flow_cache_max_size_tracked():
    """max_size is stored and enforced — oldest entries are evicted."""
    fc = FlowCache(max_size=2)
    assert fc.max_size == 2

    await fc.set({"i": 1}, {"o": 1})
    await fc.set({"i": 2}, {"o": 2})
    await fc.set({"i": 3}, {"o": 3})  # Exceeds max_size — oldest evicted
    assert fc.stats.size == 2


@pytest.mark.asyncio
async def test_get_stats_returns_stats():
    """get_stats() returns the same CacheStats instance."""
    fc = FlowCache()
    await fc.set({"x": 1}, {"y": 1})
    await fc.get({"x": 1})
    await fc.get({"z": 99})

    stats = fc.get_stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.size == 1

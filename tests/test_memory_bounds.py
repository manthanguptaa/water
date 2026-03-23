"""Tests for bounded in-memory storage and cache classes (issue #97)."""

import asyncio
import time
from unittest.mock import patch

import pytest

from water.storage.base import InMemoryStorage, FlowSession, TaskRun
from water.resilience.cache import InMemoryCache
from water.resilience.flow_cache import FlowCache, InMemoryFlowCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(execution_id: str, flow_id: str = "flow1") -> FlowSession:
    return FlowSession(flow_id=flow_id, input_data={"k": execution_id}, execution_id=execution_id)


def _make_task_run(execution_id: str, task_id: str, run_id: str) -> TaskRun:
    return TaskRun(execution_id=execution_id, task_id=task_id, node_index=0, id=run_id)


# ---------------------------------------------------------------------------
# InMemoryStorage
# ---------------------------------------------------------------------------

class TestInMemoryStorageBounds:
    """InMemoryStorage should evict oldest sessions when max_sessions is exceeded."""

    @pytest.mark.asyncio
    async def test_evicts_oldest_session(self) -> None:
        storage = InMemoryStorage(max_sessions=3)
        for i in range(5):
            await storage.save_session(_make_session(f"s{i}"))

        # Only the 3 most recent should remain
        assert len(storage._sessions) == 3
        assert await storage.get_session("s0") is None
        assert await storage.get_session("s1") is None
        # s2, s3, s4 survive (s2 was accessed by get_session so it moved to end)
        assert await storage.get_session("s4") is not None

    @pytest.mark.asyncio
    async def test_lru_ordering_on_get(self) -> None:
        storage = InMemoryStorage(max_sessions=3)
        await storage.save_session(_make_session("s0"))
        await storage.save_session(_make_session("s1"))
        await storage.save_session(_make_session("s2"))

        # Access s0 so it becomes most recently used
        await storage.get_session("s0")

        # Add two more, which should evict s1 and s2 (oldest), not s0
        await storage.save_session(_make_session("s3"))
        await storage.save_session(_make_session("s4"))

        assert await storage.get_session("s0") is not None
        assert await storage.get_session("s1") is None
        assert await storage.get_session("s2") is None

    @pytest.mark.asyncio
    async def test_evicts_task_runs_with_session(self) -> None:
        storage = InMemoryStorage(max_sessions=2)
        await storage.save_session(_make_session("s0"))
        await storage.save_task_run(_make_task_run("s0", "t1", "r1"))
        await storage.save_session(_make_session("s1"))
        await storage.save_session(_make_session("s2"))

        # s0 should be evicted along with its task runs
        assert await storage.get_session("s0") is None
        assert "s0" not in storage._task_runs

    @pytest.mark.asyncio
    async def test_caps_task_runs_per_session(self) -> None:
        storage = InMemoryStorage(max_task_runs_per_session=3)
        await storage.save_session(_make_session("s0"))
        for i in range(10):
            await storage.save_task_run(_make_task_run("s0", "task", f"r{i}"))

        runs = await storage.get_task_runs("s0")
        assert len(runs) == 3
        # Most recent runs should be kept
        assert runs[0].id == "r7"
        assert runs[2].id == "r9"


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------

class TestInMemoryCacheBounds:
    """InMemoryCache should evict when max_size is exceeded."""

    def test_evicts_oldest_when_full(self) -> None:
        cache = InMemoryCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)

        assert cache.get("a") is None  # evicted
        assert cache.get("d") == 4

    def test_lru_ordering_on_get(self) -> None:
        cache = InMemoryCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access 'a' to make it recently used
        cache.get("a")

        cache.set("d", 4)
        # 'b' should be evicted (oldest unused), not 'a'
        assert cache.get("b") is None
        assert cache.get("a") == 1

    def test_cleanup_removes_expired(self) -> None:
        cache = InMemoryCache(max_size=100)
        cache.set("keep", "yes", ttl=9999)
        cache.set("expire1", "no", ttl=0.0)
        cache.set("expire2", "no", ttl=0.0)

        # Let entries expire
        time.sleep(0.01)
        cache.cleanup()

        assert cache.has("keep")
        assert not cache.has("expire1")
        assert not cache.has("expire2")
        assert len(cache._store) == 1

    def test_evicts_expired_before_oldest(self) -> None:
        cache = InMemoryCache(max_size=3)
        cache.set("a", 1, ttl=0.0)  # will expire
        cache.set("b", 2)
        cache.set("c", 3)

        time.sleep(0.01)
        cache.set("d", 4)

        # 'a' expired and was cleaned up; 'b', 'c', 'd' remain
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("d") == 4


# ---------------------------------------------------------------------------
# FlowCache
# ---------------------------------------------------------------------------

class TestFlowCacheBounds:
    """FlowCache should enforce max_size."""

    @pytest.mark.asyncio
    async def test_enforces_max_size(self) -> None:
        cache = FlowCache(ttl=3600, max_size=3)
        for i in range(5):
            await cache.set({"i": i}, {"result": i})

        assert cache.stats.size == 3

        # Oldest entries should be evicted
        assert await cache.get({"i": 0}) is None
        assert await cache.get({"i": 1}) is None
        assert await cache.get({"i": 4}) is not None

    @pytest.mark.asyncio
    async def test_lru_ordering(self) -> None:
        cache = FlowCache(ttl=3600, max_size=3)
        await cache.set({"i": 0}, {"result": 0})
        await cache.set({"i": 1}, {"result": 1})
        await cache.set({"i": 2}, {"result": 2})

        # Access i=0 so it becomes recently used
        await cache.get({"i": 0})

        # Add two more, evicting i=1 and i=2
        await cache.set({"i": 3}, {"result": 3})
        await cache.set({"i": 4}, {"result": 4})

        assert await cache.get({"i": 0}) is not None
        assert await cache.get({"i": 1}) is None
        assert await cache.get({"i": 2}) is None

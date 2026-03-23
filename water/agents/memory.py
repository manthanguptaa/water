__all__ = [
    "MemoryLayer",
    "MemoryEntry",
    "MemoryBackend",
    "InMemoryBackend",
    "FileBackend",
    "MemoryManager",
    "create_memory_tools",
]

"""
Layered Memory Hierarchy for Water Agents.

Provides a priority-ordered memory system (ORG > PROJECT > USER > SESSION >
AUTO_LEARNED) that agents can read from and write to during ReAct loops.
Multiple storage backends are supported and memories can be surfaced as
system prompt context or manipulated via agent tools.
"""

import abc
import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

from water.agents.tools import Tool


# ---------------------------------------------------------------------------
# Memory Layer (ordered by priority, ORG highest)
# ---------------------------------------------------------------------------

class MemoryLayer(enum.Enum):
    ORG = "org"
    PROJECT = "project"
    USER = "user"
    SESSION = "session"
    AUTO_LEARNED = "auto_learned"


_LAYER_PRIORITY: List[MemoryLayer] = [
    MemoryLayer.ORG,
    MemoryLayer.PROJECT,
    MemoryLayer.USER,
    MemoryLayer.SESSION,
    MemoryLayer.AUTO_LEARNED,
]


# ---------------------------------------------------------------------------
# Memory Entry
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    key: str
    value: Any
    layer: MemoryLayer
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None

    @property
    def expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "layer": self.layer.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "ttl": self.ttl,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            layer=MemoryLayer(data["layer"]),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            ttl=data.get("ttl"),
        )


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class MemoryBackend(abc.ABC):
    """Abstract storage backend for memory entries."""

    @abc.abstractmethod
    async def get(self, key: str, layer: MemoryLayer) -> Optional[MemoryEntry]:
        ...

    @abc.abstractmethod
    async def set(self, entry: MemoryEntry) -> None:
        ...

    @abc.abstractmethod
    async def delete(self, key: str, layer: MemoryLayer) -> None:
        ...

    @abc.abstractmethod
    async def search(
        self, query: str, layer: Optional[MemoryLayer] = None, limit: int = 10
    ) -> List[MemoryEntry]:
        ...

    @abc.abstractmethod
    async def list_all(self, layer: Optional[MemoryLayer] = None) -> List[MemoryEntry]:
        ...


# ---------------------------------------------------------------------------
# In-Memory Backend
# ---------------------------------------------------------------------------

class InMemoryBackend(MemoryBackend):
    """Simple dict-backed memory backend. Data lives only in-process."""

    def __init__(self) -> None:
        self._store: Dict[tuple, MemoryEntry] = {}

    async def get(self, key: str, layer: MemoryLayer) -> Optional[MemoryEntry]:
        entry = self._store.get((layer, key))
        if entry is not None and entry.expired:
            del self._store[(layer, key)]
            return None
        return entry

    async def set(self, entry: MemoryEntry) -> None:
        self._store[(entry.layer, entry.key)] = entry

    async def delete(self, key: str, layer: MemoryLayer) -> None:
        self._store.pop((layer, key), None)

    async def search(
        self, query: str, layer: Optional[MemoryLayer] = None, limit: int = 10
    ) -> List[MemoryEntry]:
        query_lower = query.lower()
        results: List[MemoryEntry] = []
        for (entry_layer, _), entry in self._store.items():
            if entry.expired:
                continue
            if layer is not None and entry_layer != layer:
                continue
            if query_lower in entry.key.lower() or query_lower in str(entry.value).lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    async def list_all(self, layer: Optional[MemoryLayer] = None) -> List[MemoryEntry]:
        return [
            e for e in self._store.values()
            if not e.expired and (layer is None or e.layer == layer)
        ]


# ---------------------------------------------------------------------------
# File Backend
# ---------------------------------------------------------------------------

class FileBackend(MemoryBackend):
    """
    Stores each layer as a separate JSON file inside a directory.

    Files are loaded lazily on first access and written back on every mutation.
    """

    def __init__(self, directory: str) -> None:
        self._dir = directory
        os.makedirs(directory, exist_ok=True)
        self._cache: Dict[MemoryLayer, Dict[str, MemoryEntry]] = {}

    def _path_for_layer(self, layer: MemoryLayer) -> str:
        return os.path.join(self._dir, f"{layer.value}.json")

    def _load_layer(self, layer: MemoryLayer) -> Dict[str, MemoryEntry]:
        if layer in self._cache:
            return self._cache[layer]
        path = self._path_for_layer(layer)
        entries: Dict[str, MemoryEntry] = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            for item in data:
                entry = MemoryEntry.from_dict(item)
                if not entry.expired:
                    entries[entry.key] = entry
        self._cache[layer] = entries
        return entries

    def _save_layer(self, layer: MemoryLayer) -> None:
        entries = self._cache.get(layer, {})
        path = self._path_for_layer(layer)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in entries.values()], f, indent=2)

    async def get(self, key: str, layer: MemoryLayer) -> Optional[MemoryEntry]:
        entries = self._load_layer(layer)
        entry = entries.get(key)
        if entry is not None and entry.expired:
            del entries[key]
            self._save_layer(layer)
            return None
        return entry

    async def set(self, entry: MemoryEntry) -> None:
        entries = self._load_layer(entry.layer)
        entries[entry.key] = entry
        self._save_layer(entry.layer)

    async def delete(self, key: str, layer: MemoryLayer) -> None:
        entries = self._load_layer(layer)
        if key in entries:
            del entries[key]
            self._save_layer(layer)

    async def search(
        self, query: str, layer: Optional[MemoryLayer] = None, limit: int = 10
    ) -> List[MemoryEntry]:
        query_lower = query.lower()
        results: List[MemoryEntry] = []
        layers = [layer] if layer is not None else list(MemoryLayer)
        for lyr in layers:
            for entry in self._load_layer(lyr).values():
                if entry.expired:
                    continue
                if query_lower in entry.key.lower() or query_lower in str(entry.value).lower():
                    results.append(entry)
                    if len(results) >= limit:
                        return results
        return results

    async def list_all(self, layer: Optional[MemoryLayer] = None) -> List[MemoryEntry]:
        layers = [layer] if layer is not None else list(MemoryLayer)
        results: List[MemoryEntry] = []
        for lyr in layers:
            for entry in self._load_layer(lyr).values():
                if not entry.expired:
                    results.append(entry)
        return results


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Unified interface over one or more memory backends.

    Pass either a mapping of layers to backends, or a single default backend
    that will be used for every layer.
    """

    def __init__(
        self,
        backends: Optional[Dict[MemoryLayer, MemoryBackend]] = None,
        default_backend: Optional[MemoryBackend] = None,
    ) -> None:
        if backends is None and default_backend is None:
            default_backend = InMemoryBackend()
        self._backends: Dict[MemoryLayer, MemoryBackend] = backends or {}
        self._default: Optional[MemoryBackend] = default_backend

    def _backend_for(self, layer: MemoryLayer) -> MemoryBackend:
        backend = self._backends.get(layer, self._default)
        if backend is None:
            raise ValueError(f"No backend configured for layer {layer.value}")
        return backend

    async def add(
        self,
        key: str,
        value: Any,
        layer: MemoryLayer,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            key=key,
            value=value,
            layer=layer,
            metadata=metadata or {},
            ttl=ttl,
        )
        await self._backend_for(layer).set(entry)
        return entry

    async def get(
        self, key: str, layer: Optional[MemoryLayer] = None
    ) -> Optional[MemoryEntry]:
        if layer is not None:
            return await self._backend_for(layer).get(key, layer)
        # Search all layers in priority order.
        for lyr in _LAYER_PRIORITY:
            try:
                entry = await self._backend_for(lyr).get(key, lyr)
            except ValueError:
                logger.debug("No backend configured for memory layer '%s'", lyr.value)
                continue
            if entry is not None:
                return entry
        return None

    async def search(
        self, query: str, layer: Optional[MemoryLayer] = None, limit: int = 10
    ) -> List[MemoryEntry]:
        if layer is not None:
            return await self._backend_for(layer).search(query, layer, limit)
        # Aggregate across all layers, preserving priority order.
        results: List[MemoryEntry] = []
        seen_keys: set = set()
        for lyr in _LAYER_PRIORITY:
            try:
                backend = self._backend_for(lyr)
            except ValueError:
                logger.debug("No backend configured for memory layer '%s' during search", lyr.value)
                continue
            entries = await backend.search(query, lyr, limit - len(results))
            for entry in entries:
                ident = (entry.layer, entry.key)
                if ident not in seen_keys:
                    seen_keys.add(ident)
                    results.append(entry)
                    if len(results) >= limit:
                        return results
        return results

    async def delete(self, key: str, layer: MemoryLayer) -> None:
        await self._backend_for(layer).delete(key, layer)

    async def get_all(self, layer: Optional[MemoryLayer] = None) -> List[MemoryEntry]:
        if layer is not None:
            return await self._backend_for(layer).list_all(layer)
        results: List[MemoryEntry] = []
        for lyr in _LAYER_PRIORITY:
            try:
                backend = self._backend_for(lyr)
            except ValueError:
                logger.debug("No backend configured for memory layer '%s' during get_all", lyr.value)
                continue
            results.extend(await backend.list_all(lyr))
        return results

    def to_system_prompt(self) -> str:
        """Format all non-expired memories into a system prompt section grouped by layer.

        This is a synchronous helper intended to be called after memories have
        already been loaded.  It pulls from backend caches where available,
        falling back to an empty list for backends that require async loading.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside a running event loop; collect from caches directly.
            entries = self._collect_from_caches()
        else:
            entries = asyncio.run(self.get_all())

        if not entries:
            return ""

        grouped: Dict[MemoryLayer, List[MemoryEntry]] = {}
        for entry in entries:
            if not entry.expired:
                grouped.setdefault(entry.layer, []).append(entry)

        if not grouped:
            return ""

        lines = ["# Memories"]
        for lyr in _LAYER_PRIORITY:
            layer_entries = grouped.get(lyr)
            if not layer_entries:
                continue
            lines.append(f"\n## {lyr.value.upper()}")
            for entry in layer_entries:
                lines.append(f"- **{entry.key}**: {entry.value}")
        return "\n".join(lines)

    def _collect_from_caches(self) -> List[MemoryEntry]:
        """Best-effort synchronous collection from backend caches."""
        results: List[MemoryEntry] = []
        for lyr in _LAYER_PRIORITY:
            try:
                backend = self._backend_for(lyr)
            except ValueError:
                logger.debug("No backend configured for memory layer '%s' during cache collection", lyr.value)
                continue
            # InMemoryBackend
            if isinstance(backend, InMemoryBackend):
                results.extend(
                    e for e in backend._store.values()
                    if not e.expired and e.layer == lyr
                )
            # FileBackend (uses its lazy cache)
            elif isinstance(backend, FileBackend):
                for entry in backend._load_layer(lyr).values():
                    if not entry.expired:
                        results.append(entry)
        return results


# ---------------------------------------------------------------------------
# Agent tools for self-managed memory
# ---------------------------------------------------------------------------

def create_memory_tools(manager: MemoryManager) -> List[Tool]:
    """Create tools that let an agent store, recall, and list memories."""

    async def memory_store(key: str, value: str) -> str:
        entry = await manager.add(key, value, MemoryLayer.AUTO_LEARNED)
        return f"Stored memory '{key}' in AUTO_LEARNED layer."

    async def memory_recall(query: str, limit: int = 5) -> List[Dict[str, Any]]:
        entries = await manager.search(query, limit=limit)
        return [
            {"key": e.key, "value": e.value, "layer": e.layer.value}
            for e in entries
        ]

    async def memory_list(layer: Optional[str] = None) -> List[Dict[str, Any]]:
        mem_layer: Optional[MemoryLayer] = None
        if layer is not None:
            mem_layer = MemoryLayer(layer)
        entries = await manager.get_all(mem_layer)
        return [
            {"key": e.key, "value": e.value, "layer": e.layer.value}
            for e in entries
        ]

    store_tool = Tool(
        name="memory_store",
        description="Store a key-value pair in the agent's learned memory.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "A short identifier for this memory."},
                "value": {"type": "string", "description": "The content to remember."},
            },
            "required": ["key", "value"],
        },
        execute=memory_store,
    )

    recall_tool = Tool(
        name="memory_recall",
        description="Search memories by a text query. Returns matching entries across all layers.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text to match against memory keys and values."},
                "limit": {"type": "integer", "description": "Max results to return.", "default": 5},
            },
            "required": ["query"],
        },
        execute=memory_recall,
    )

    list_tool = Tool(
        name="memory_list",
        description="List all memories, optionally filtered by layer.",
        input_schema={
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "description": "Filter to a specific layer (org, project, user, session, auto_learned). Omit for all.",
                    "enum": [l.value for l in MemoryLayer],
                },
            },
            "required": [],
        },
        execute=memory_list,
    )

    return [store_tool, recall_tool, list_tool]

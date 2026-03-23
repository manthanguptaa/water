import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from water.storage.base import FlowSession, FlowStatus, StorageBackend, TaskRun

logger = logging.getLogger(__name__)


class RedisStorage(StorageBackend):
    """Redis storage backend for flow execution data.

    Requires the ``redis`` package (``pip install redis``).
    Sessions are stored as JSON strings keyed by
    ``{prefix}:session:{execution_id}``, and task runs are stored under
    ``{prefix}:taskrun:{execution_id}:{task_run_id}``.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "water",
        socket_timeout: float = 30.0,
    ) -> None:
        try:
            import redis.asyncio as aioredis  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'redis' package is required for RedisStorage. "
                "Install it with: pip install redis"
            )

        self._redis_url = redis_url
        self._prefix = prefix
        self.socket_timeout = socket_timeout
        self._redis: Any = None

    async def _get_client(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self._redis_url, decode_responses=True, socket_timeout=self.socket_timeout)
        return self._redis

    def _session_key(self, execution_id: str) -> str:
        return f"{self._prefix}:session:{execution_id}"

    def _session_index_key(self, flow_id: str) -> str:
        return f"{self._prefix}:sessions_by_flow:{flow_id}"

    def _all_sessions_key(self) -> str:
        return f"{self._prefix}:all_sessions"

    def _taskrun_key(self, execution_id: str, task_run_id: str) -> str:
        return f"{self._prefix}:taskrun:{execution_id}:{task_run_id}"

    def _taskruns_index_key(self, execution_id: str) -> str:
        return f"{self._prefix}:taskruns:{execution_id}"

    # ---- StorageBackend interface ----

    async def save_session(self, session: FlowSession) -> None:
        session.updated_at = datetime.now(timezone.utc)
        client = await self._get_client()
        data = json.dumps(session.to_dict())
        key = self._session_key(session.execution_id)

        pipe = client.pipeline()
        pipe.set(key, data)
        # Maintain indices so we can list / filter sessions later.
        pipe.sadd(self._all_sessions_key(), session.execution_id)
        pipe.sadd(self._session_index_key(session.flow_id), session.execution_id)
        await pipe.execute()

    async def get_session(self, execution_id: str) -> Optional[FlowSession]:
        client = await self._get_client()
        data = await client.get(self._session_key(execution_id))
        if data is None:
            return None
        return FlowSession.from_dict(json.loads(data))

    async def list_sessions(self, flow_id: Optional[str] = None) -> List[FlowSession]:
        client = await self._get_client()

        if flow_id:
            exec_ids = await client.smembers(self._session_index_key(flow_id))
        else:
            exec_ids = await client.smembers(self._all_sessions_key())

        if not exec_ids:
            return []

        keys = [self._session_key(eid) for eid in exec_ids]
        values = await client.mget(keys)

        sessions: List[FlowSession] = []
        for val in values:
            if val is not None:
                sessions.append(FlowSession.from_dict(json.loads(val)))

        # Return newest first, matching SQLiteStorage behaviour.
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    async def save_task_run(self, task_run: TaskRun) -> None:
        client = await self._get_client()
        data = json.dumps(task_run.to_dict())
        key = self._taskrun_key(task_run.execution_id, task_run.id)

        pipe = client.pipeline()
        pipe.set(key, data)
        pipe.sadd(self._taskruns_index_key(task_run.execution_id), task_run.id)
        await pipe.execute()

    async def get_task_runs(self, execution_id: str) -> List[TaskRun]:
        client = await self._get_client()
        run_ids = await client.smembers(self._taskruns_index_key(execution_id))
        if not run_ids:
            return []

        keys = [self._taskrun_key(execution_id, rid) for rid in run_ids]
        values = await client.mget(keys)

        runs: List[TaskRun] = []
        for val in values:
            if val is not None:
                runs.append(TaskRun.from_dict(json.loads(val)))

        runs.sort(key=lambda r: (r.node_index, r.started_at or datetime.min))
        return runs

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from water.storage.base import FlowSession, FlowStatus, StorageBackend, TaskRun

logger = logging.getLogger(__name__)


class PostgresStorage(StorageBackend):
    """PostgreSQL storage backend for flow execution data.

    Requires the ``asyncpg`` package (``pip install asyncpg``).
    Call :meth:`initialize` once before use to create the required tables.
    """

    def __init__(self, dsn: str, command_timeout: float = 30.0) -> None:
        try:
            import asyncpg  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'asyncpg' package is required for PostgresStorage. "
                "Install it with: pip install asyncpg"
            )

        self._dsn = dsn
        self.command_timeout = command_timeout
        self._pool: Any = None

    async def initialize(self) -> None:
        """Create the database tables if they don't exist and set up
        the connection pool."""
        import asyncpg

        self._pool = await asyncpg.create_pool(dsn=self._dsn, command_timeout=self.command_timeout)

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS flow_sessions (
                    execution_id TEXT PRIMARY KEY,
                    flow_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    current_node_index INTEGER NOT NULL DEFAULT 0,
                    current_data TEXT NOT NULL,
                    context_state TEXT NOT NULL DEFAULT '{}',
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS task_runs (
                    id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    node_index INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    input_data TEXT,
                    output_data TEXT,
                    error TEXT,
                    started_at TEXT,
                    completed_at TEXT
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_runs_execution
                ON task_runs(execution_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_flow_id
                ON flow_sessions(flow_id)
            """)

    async def _get_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError(
                "PostgresStorage has not been initialized. "
                "Call 'await storage.initialize()' before use."
            )
        return self._pool

    # ---- StorageBackend interface ----

    async def save_session(self, session: FlowSession) -> None:
        session.updated_at = datetime.now(timezone.utc)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO flow_sessions
                    (execution_id, flow_id, status, input_data, current_node_index,
                     current_data, context_state, result, error, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (execution_id) DO UPDATE SET
                    flow_id = EXCLUDED.flow_id,
                    status = EXCLUDED.status,
                    input_data = EXCLUDED.input_data,
                    current_node_index = EXCLUDED.current_node_index,
                    current_data = EXCLUDED.current_data,
                    context_state = EXCLUDED.context_state,
                    result = EXCLUDED.result,
                    error = EXCLUDED.error,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at
                """,
                session.execution_id,
                session.flow_id,
                session.status.value,
                json.dumps(session.input_data),
                session.current_node_index,
                json.dumps(session.current_data),
                json.dumps(session.context_state),
                json.dumps(session.result) if session.result else None,
                session.error,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            )

    async def get_session(self, execution_id: str) -> Optional[FlowSession]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM flow_sessions WHERE execution_id = $1",
                execution_id,
            )
            if row is None:
                return None
            return self._row_to_session(row)

    async def list_sessions(self, flow_id: Optional[str] = None) -> List[FlowSession]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if flow_id:
                rows = await conn.fetch(
                    "SELECT * FROM flow_sessions WHERE flow_id = $1 ORDER BY created_at DESC",
                    flow_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM flow_sessions ORDER BY created_at DESC"
                )
            return [self._row_to_session(row) for row in rows]

    async def save_task_run(self, task_run: TaskRun) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO task_runs
                    (id, execution_id, task_id, node_index, status,
                     input_data, output_data, error, started_at, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    execution_id = EXCLUDED.execution_id,
                    task_id = EXCLUDED.task_id,
                    node_index = EXCLUDED.node_index,
                    status = EXCLUDED.status,
                    input_data = EXCLUDED.input_data,
                    output_data = EXCLUDED.output_data,
                    error = EXCLUDED.error,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at
                """,
                task_run.id,
                task_run.execution_id,
                task_run.task_id,
                task_run.node_index,
                task_run.status,
                json.dumps(task_run.input_data) if task_run.input_data else None,
                json.dumps(task_run.output_data) if task_run.output_data else None,
                task_run.error,
                task_run.started_at.isoformat() if task_run.started_at else None,
                task_run.completed_at.isoformat() if task_run.completed_at else None,
            )

    async def get_task_runs(self, execution_id: str) -> List[TaskRun]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM task_runs WHERE execution_id = $1 ORDER BY node_index, started_at",
                execution_id,
            )
            return [self._row_to_task_run(row) for row in rows]

    # ---- Helpers ----

    @staticmethod
    def _row_to_session(row: Any) -> FlowSession:
        return FlowSession(
            execution_id=row["execution_id"],
            flow_id=row["flow_id"],
            status=FlowStatus(row["status"]),
            input_data=json.loads(row["input_data"]),
            current_node_index=row["current_node_index"],
            current_data=json.loads(row["current_data"]),
            context_state=json.loads(row["context_state"]),
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_task_run(row: Any) -> TaskRun:
        return TaskRun(
            id=row["id"],
            execution_id=row["execution_id"],
            task_id=row["task_id"],
            node_index=row["node_index"],
            status=row["status"],
            input_data=json.loads(row["input_data"]) if row["input_data"] else None,
            output_data=json.loads(row["output_data"]) if row["output_data"] else None,
            error=row["error"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

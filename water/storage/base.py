import json
import uuid
import sqlite3
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FlowStatus(str, Enum):
    """Status of a flow execution session."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class FlowSession:
    """Represents a single execution of a flow."""

    def __init__(
        self,
        flow_id: str,
        input_data: Dict[str, Any],
        execution_id: Optional[str] = None,
        status: FlowStatus = FlowStatus.PENDING,
        current_node_index: int = 0,
        current_data: Optional[Dict[str, Any]] = None,
        context_state: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> None:
        self.execution_id = execution_id or f"exec_{uuid.uuid4().hex[:8]}"
        self.flow_id = flow_id
        self.status = status
        self.input_data = input_data
        self.current_node_index = current_node_index
        self.current_data = current_data if current_data is not None else input_data
        self.context_state = context_state or {}
        self.result = result
        self.error = error
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "flow_id": self.flow_id,
            "status": self.status.value,
            "input_data": self.input_data,
            "current_node_index": self.current_node_index,
            "current_data": self.current_data,
            "context_state": self.context_state,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FlowSession":
        return cls(
            execution_id=data["execution_id"],
            flow_id=data["flow_id"],
            status=FlowStatus(data["status"]),
            input_data=data["input_data"],
            current_node_index=data["current_node_index"],
            current_data=data["current_data"],
            context_state=data.get("context_state", {}),
            result=data.get("result"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class TaskRun:
    """Represents a single execution of a task within a flow session."""

    def __init__(
        self,
        execution_id: str,
        task_id: str,
        node_index: int,
        status: str = "pending",
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        id: Optional[str] = None,
    ) -> None:
        self.id = id or f"run_{uuid.uuid4().hex[:8]}"
        self.execution_id = execution_id
        self.task_id = task_id
        self.node_index = node_index
        self.status = status
        self.input_data = input_data
        self.output_data = output_data
        self.error = error
        self.started_at = started_at
        self.completed_at = completed_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "node_index": self.node_index,
            "status": self.status,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskRun":
        return cls(
            id=data["id"],
            execution_id=data["execution_id"],
            task_id=data["task_id"],
            node_index=data["node_index"],
            status=data["status"],
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )


class StorageBackend(ABC):
    """Abstract base class for flow execution storage backends."""

    @abstractmethod
    async def save_session(self, session: FlowSession) -> None:
        """Save or update a flow session."""

    @abstractmethod
    async def get_session(self, execution_id: str) -> Optional[FlowSession]:
        """Get a flow session by execution ID."""

    @abstractmethod
    async def list_sessions(self, flow_id: Optional[str] = None) -> List[FlowSession]:
        """List flow sessions, optionally filtered by flow ID."""

    @abstractmethod
    async def save_task_run(self, task_run: TaskRun) -> None:
        """Save a task run record."""

    @abstractmethod
    async def get_task_runs(self, execution_id: str) -> List[TaskRun]:
        """Get all task runs for a given execution."""


class InMemoryStorage(StorageBackend):
    """In-memory storage backend for development and testing."""

    def __init__(self) -> None:
        self._sessions: Dict[str, FlowSession] = {}
        self._task_runs: Dict[str, List[TaskRun]] = {}

    async def save_session(self, session: FlowSession) -> None:
        session.updated_at = datetime.now(timezone.utc)
        self._sessions[session.execution_id] = session

    async def get_session(self, execution_id: str) -> Optional[FlowSession]:
        return self._sessions.get(execution_id)

    async def list_sessions(self, flow_id: Optional[str] = None) -> List[FlowSession]:
        sessions = list(self._sessions.values())
        if flow_id:
            sessions = [s for s in sessions if s.flow_id == flow_id]
        return sessions

    async def save_task_run(self, task_run: TaskRun) -> None:
        if task_run.execution_id not in self._task_runs:
            self._task_runs[task_run.execution_id] = []
        # Update existing or append
        runs = self._task_runs[task_run.execution_id]
        for i, existing in enumerate(runs):
            if existing.id == task_run.id:
                runs[i] = task_run
                return
        runs.append(task_run)

    async def get_task_runs(self, execution_id: str) -> List[TaskRun]:
        return list(self._task_runs.get(execution_id, []))


class SQLiteStorage(StorageBackend):
    """SQLite storage backend for persistent flow execution data."""

    def __init__(self, db_path: str = "water_flows.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            conn.execute("""
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
                    completed_at TEXT,
                    FOREIGN KEY (execution_id) REFERENCES flow_sessions(execution_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_runs_execution
                ON task_runs(execution_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_flow_id
                ON flow_sessions(flow_id)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def save_session(self, session: FlowSession) -> None:
        session.updated_at = datetime.now(timezone.utc)
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO flow_sessions
                (execution_id, flow_id, status, input_data, current_node_index,
                 current_data, context_state, result, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
                ),
            )
            conn.commit()

    async def get_session(self, execution_id: str) -> Optional[FlowSession]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM flow_sessions WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
            if not row:
                return None
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

    async def list_sessions(self, flow_id: Optional[str] = None) -> List[FlowSession]:
        with self._get_conn() as conn:
            if flow_id:
                rows = conn.execute(
                    "SELECT * FROM flow_sessions WHERE flow_id = ? ORDER BY created_at DESC",
                    (flow_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM flow_sessions ORDER BY created_at DESC"
                ).fetchall()

            return [
                FlowSession(
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
                for row in rows
            ]

    async def save_task_run(self, task_run: TaskRun) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_runs
                (id, execution_id, task_id, node_index, status,
                 input_data, output_data, error, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
                ),
            )
            conn.commit()

    async def get_task_runs(self, execution_id: str) -> List[TaskRun]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM task_runs WHERE execution_id = ? ORDER BY node_index, started_at",
                (execution_id,),
            ).fetchall()

            return [
                TaskRun(
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
                for row in rows
            ]

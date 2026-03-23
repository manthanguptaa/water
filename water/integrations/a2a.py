"""
Agent-to-Agent (A2A) Protocol Support for Water.

Implements Google's A2A protocol for inter-agent communication.
Enables agents to discover each other, negotiate capabilities,
and collaborate across trust boundaries.
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# A2A Data Models
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    """A2A task lifecycle states."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"


@dataclass
class AgentSkill:
    """A capability that an agent advertises."""
    id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
        }


@dataclass
class AgentCard:
    """
    JSON metadata describing an agent's capabilities, endpoint, and auth.

    Served at ``/.well-known/agent.json`` per the A2A spec.
    """
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    skills: List[AgentSkill] = field(default_factory=list)
    auth_schemes: List[str] = field(default_factory=lambda: ["none"])
    protocols: List[str] = field(default_factory=lambda: ["a2a/1.0"])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "skills": [s.to_dict() for s in self.skills],
            "authentication": {"schemes": self.auth_schemes},
            "protocols": self.protocols,
        }


class MessagePart:
    """A part of an A2A message (text, file, or data)."""

    def __init__(self, kind: str, content: Any, mime_type: str = "text/plain"):
        self.kind = kind  # "text", "file", "data"
        self.content = content
        self.mime_type = mime_type

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "content": self.content,
            "mimeType": self.mime_type,
        }

    @classmethod
    def text(cls, content: str) -> "MessagePart":
        return cls(kind="text", content=content)

    @classmethod
    def data(cls, content: dict, mime_type: str = "application/json") -> "MessagePart":
        return cls(kind="data", content=content, mime_type=mime_type)


@dataclass
class A2AMessage:
    """A message in the A2A protocol."""
    role: str  # "user" or "agent"
    parts: List[MessagePart] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "A2AMessage":
        parts = [
            MessagePart(kind=p["kind"], content=p["content"], mime_type=p.get("mimeType", "text/plain"))
            for p in d.get("parts", [])
        ]
        return cls(role=d["role"], parts=parts, timestamp=d.get("timestamp", ""))


@dataclass
class A2ATask:
    """Represents an A2A task with lifecycle management."""
    id: str
    state: TaskState = TaskState.SUBMITTED
    messages: List[A2AMessage] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "state": self.state.value,
            "messages": [m.to_dict() for m in self.messages],
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d


# ---------------------------------------------------------------------------
# A2A Server — expose a Water flow as an A2A-compliant agent
# ---------------------------------------------------------------------------

class A2AServer:
    """
    Expose any Water flow as an A2A-compliant agent.

    Other agents can discover this agent via ``/.well-known/agent.json``
    and interact through the standard A2A task lifecycle.
    """

    def __init__(
        self,
        flow: Any,
        name: str,
        description: str = "",
        url: str = "http://localhost:8000",
        skills: Optional[List[AgentSkill]] = None,
        auth_schemes: Optional[List[str]] = None,
        version: str = "1.0.0",
    ):
        self.flow = flow
        self.card = AgentCard(
            name=name,
            description=description,
            url=url,
            skills=skills or [],
            auth_schemes=auth_schemes or ["none"],
            version=version,
        )
        self._tasks: Dict[str, A2ATask] = {}

    def get_agent_card(self) -> dict:
        """Return the agent card for discovery."""
        return self.card.to_dict()

    async def handle_request(self, request: dict) -> dict:
        """
        Handle an incoming A2A JSON-RPC 2.0 request.

        Supports: tasks/send, tasks/get, tasks/cancel, agent/info
        """
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", str(uuid.uuid4()))

        handlers = {
            "agent/info": self._handle_agent_info,
            "tasks/send": self._handle_tasks_send,
            "tasks/get": self._handle_tasks_get,
            "tasks/cancel": self._handle_tasks_cancel,
        }

        handler = handlers.get(method)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        try:
            result = await handler(params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            logger.exception("A2A request handler failed for method '%s'", method)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)},
            }

    async def _handle_agent_info(self, params: dict) -> dict:
        return self.card.to_dict()

    async def _handle_tasks_send(self, params: dict) -> dict:
        task_id = params.get("id", str(uuid.uuid4()))
        messages = params.get("messages", [])

        task = A2ATask(id=task_id)
        task.messages = [A2AMessage.from_dict(m) for m in messages]
        task.state = TaskState.WORKING
        self._tasks[task_id] = task

        # Extract input from messages
        input_data = self._extract_input(task.messages)

        try:
            result = await self.flow.run(input_data)
            task.state = TaskState.COMPLETED
            task.result = result
            task.updated_at = datetime.now(timezone.utc).isoformat()

            # Add agent response message
            response_parts = [MessagePart.data(result)]
            task.messages.append(A2AMessage(role="agent", parts=response_parts))
        except Exception as e:
            logger.exception("A2A task '%s' failed during flow execution", task_id)
            task.state = TaskState.FAILED
            task.error = str(e)
            task.updated_at = datetime.now(timezone.utc).isoformat()

        return task.to_dict()

    async def _handle_tasks_get(self, params: dict) -> dict:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        return task.to_dict()

    async def _handle_tasks_cancel(self, params: dict) -> dict:
        task_id = params.get("id", "")
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        task.state = TaskState.CANCELED
        task.updated_at = datetime.now(timezone.utc).isoformat()
        return task.to_dict()

    @staticmethod
    def _extract_input(messages: List[A2AMessage]) -> dict:
        """Extract input data from A2A messages."""
        for msg in messages:
            if msg.role == "user":
                for part in msg.parts:
                    if part.kind == "data" and isinstance(part.content, dict):
                        return part.content
                    if part.kind == "text":
                        return {"prompt": part.content}
        return {}

    def add_routes(self, app: Any) -> None:
        """
        Add A2A routes to a FastAPI app.

        Registers:
        - GET /.well-known/agent.json
        - POST /a2a
        """
        from fastapi import Request
        from fastapi.responses import JSONResponse

        server = self

        @app.get("/.well-known/agent.json")
        async def agent_card():
            return JSONResponse(content=server.get_agent_card())

        @app.post("/a2a")
        async def a2a_endpoint(request: Request):
            body = await request.json()
            response = await server.handle_request(body)
            return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# A2A Client — call external A2A agents from within a Water flow
# ---------------------------------------------------------------------------

class A2AClient:
    """
    Call external A2A agents from within a Water flow.

    Handles agent discovery, task submission, and result retrieval.
    """

    def __init__(
        self,
        agent_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.agent_url = agent_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self._agent_card: Optional[dict] = None

    async def discover(self) -> dict:
        """Fetch the agent card from the remote agent."""
        import urllib.request
        import urllib.error

        url = f"{self.agent_url}/.well-known/agent.json"
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                self._agent_card = json.loads(resp.read().decode())
                return self._agent_card
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to discover agent at {url}: {e}")

    async def send_task(
        self,
        messages: Optional[List[A2AMessage]] = None,
        input_data: Optional[dict] = None,
        task_id: Optional[str] = None,
    ) -> A2ATask:
        """
        Send a task to the remote A2A agent.

        Args:
            messages: A2A messages to send.
            input_data: Convenience — auto-wrapped as a data message.
            task_id: Optional task ID (auto-generated if omitted).

        Returns:
            The A2ATask with the agent's response.
        """
        import urllib.request
        import urllib.error

        if messages is None:
            messages = []
        if input_data is not None:
            messages.append(
                A2AMessage(role="user", parts=[MessagePart.data(input_data)])
            )

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/send",
            "params": {
                "id": task_id or str(uuid.uuid4()),
                "messages": [m.to_dict() for m in messages],
            },
        }

        url = f"{self.agent_url}/a2a"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to send task to {url}: {e}")

        if "error" in response:
            raise RuntimeError(
                f"A2A error: {response['error'].get('message', 'Unknown error')}"
            )

        result = response.get("result", {})
        task = A2ATask(
            id=result.get("id", ""),
            state=TaskState(result.get("state", "submitted")),
        )
        task.result = result.get("result")
        task.error = result.get("error")
        return task

    async def get_task(self, task_id: str) -> A2ATask:
        """Get the status of a previously submitted task."""
        import urllib.request
        import urllib.error

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id},
        }

        url = f"{self.agent_url}/a2a"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to get task from {url}: {e}")

        if "error" in response:
            raise RuntimeError(
                f"A2A error: {response['error'].get('message', 'Unknown error')}"
            )

        result = response.get("result", {})
        task = A2ATask(
            id=result.get("id", ""),
            state=TaskState(result.get("state", "submitted")),
        )
        task.result = result.get("result")
        task.error = result.get("error")
        return task

    async def cancel_task(self, task_id: str) -> A2ATask:
        """Cancel a previously submitted task."""
        import urllib.request
        import urllib.error

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tasks/cancel",
            "params": {"id": task_id},
        }

        url = f"{self.agent_url}/a2a"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to cancel task at {url}: {e}")

        result = response.get("result", {})
        return A2ATask(
            id=result.get("id", ""),
            state=TaskState(result.get("state", "canceled")),
        )


# ---------------------------------------------------------------------------
# Factory — wrap an A2A call as a standard Water task
# ---------------------------------------------------------------------------

def create_a2a_task(
    id: str,
    client: A2AClient,
    description: Optional[str] = None,
    input_key: str = "prompt",
    retry_count: int = 0,
    timeout: Optional[float] = None,
) -> Any:
    """
    Create a Water Task that delegates to an external A2A agent.

    Args:
        id: Task identifier.
        client: A2AClient pointed at the remote agent.
        description: Human-readable description.
        input_key: Key in input data to send as text (default: "prompt").
        retry_count: Number of retries on failure.
        timeout: Timeout in seconds.

    Returns:
        A Task instance ready to be added to a Flow.
    """
    from water.core.task import Task

    class A2AInput(BaseModel):
        prompt: str = ""

    class A2AOutput(BaseModel):
        response: dict = {}

    async def execute(params: dict, context: Any) -> dict:
        input_data = params.get("input_data", params)

        # Build message from input
        if isinstance(input_data, dict) and len(input_data) > 1:
            messages = [A2AMessage(role="user", parts=[MessagePart.data(input_data)])]
        else:
            text = str(input_data.get(input_key, ""))
            messages = [A2AMessage(role="user", parts=[MessagePart.text(text)])]

        task = await client.send_task(messages=messages)

        if task.state == TaskState.FAILED:
            raise RuntimeError(f"A2A task failed: {task.error}")

        return task.result if task.result else {"response": "no result"}

    return Task(
        id=id,
        description=description or f"A2A task: {id}",
        input_schema=A2AInput,
        output_schema=A2AOutput,
        execute=execute,
        retry_count=retry_count,
        timeout=timeout,
    )

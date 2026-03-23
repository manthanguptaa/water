import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from water.core.task import Task


class RiskLevel(IntEnum):
    """Risk levels for approval gates, ordered from lowest to highest."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TimeoutAction(str, Enum):
    """Actions to take when an approval request times out."""
    DENY = "deny"
    APPROVE = "approve"
    ESCALATE = "escalate"


class ApprovalDenied(Exception):
    """Raised when an approval request is denied or times out with deny policy."""

    def __init__(self, request_id: str, reason: str = None):
        self.request_id = request_id
        self.reason = reason
        msg = f"Approval denied for request '{request_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


@dataclass
class ApprovalPolicy:
    """Policy controlling how approval gates behave."""
    auto_approve_below: RiskLevel = RiskLevel.LOW
    timeout: float = 300.0
    timeout_action: TimeoutAction = TimeoutAction.DENY
    escalation_channel: Optional[str] = None
    require_reason: bool = False
    max_auto_approvals: Optional[int] = None  # None = unlimited


@dataclass
class ApprovalRequest:
    """Represents a single approval request."""
    request_id: str
    task_id: str
    execution_id: str
    action_description: str
    risk_level: RiskLevel
    data_summary: Dict[str, Any]
    created_at: datetime
    status: str = "pending"  # "pending", "approved", "denied", "escalated", "timed_out"
    decided_by: Optional[str] = None  # "auto", "human:<name>", "timeout"
    reason: Optional[str] = None


class ApprovalGate:
    """
    Manages approval requests with auto-approve, escalation, and timeout policies.

    Uses asyncio.Future for pending approvals. Low-risk actions can be
    auto-approved synchronously based on policy.
    """

    def __init__(self, policy: ApprovalPolicy = None):
        self.policy = policy or ApprovalPolicy()
        self._pending: Dict[str, asyncio.Future] = {}
        self._requests: Dict[str, ApprovalRequest] = {}
        self._auto_approval_count: int = 0

    async def request_approval(
        self,
        task_id: str,
        execution_id: str,
        action_description: str,
        risk_level: RiskLevel,
        data_summary: Dict[str, Any],
    ) -> ApprovalRequest:
        """
        Request approval for an action.

        Auto-approves if the risk level is at or below the policy threshold
        and the auto-approval limit has not been reached. Otherwise, waits
        for a human decision or times out according to policy.
        """
        request_id = f"approval_{uuid.uuid4().hex[:8]}"
        request = ApprovalRequest(
            request_id=request_id,
            task_id=task_id,
            execution_id=execution_id,
            action_description=action_description,
            risk_level=risk_level,
            data_summary=data_summary,
            created_at=datetime.now(timezone.utc),
        )
        self._requests[request_id] = request

        # Check auto-approve eligibility
        can_auto = risk_level <= self.policy.auto_approve_below
        if can_auto and self.policy.max_auto_approvals is not None:
            can_auto = self._auto_approval_count < self.policy.max_auto_approvals

        if can_auto:
            request.status = "approved"
            request.decided_by = "auto"
            self._auto_approval_count += 1
            return request

        # Need human approval — create a future and wait
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future

        try:
            await asyncio.wait_for(future, timeout=self.policy.timeout)
        except asyncio.TimeoutError:
            # Handle timeout according to policy
            del self._pending[request_id]
            if self.policy.timeout_action == TimeoutAction.APPROVE:
                request.status = "approved"
                request.decided_by = "timeout"
                return request
            elif self.policy.timeout_action == TimeoutAction.ESCALATE:
                request.status = "escalated"
                request.decided_by = "timeout"
                return request
            else:  # TimeoutAction.DENY is the default
                request.status = "timed_out"
                request.decided_by = "timeout"
                raise ApprovalDenied(
                    request_id,
                    reason=f"Timed out after {self.policy.timeout}s",
                )

        # Future was resolved by approve() or deny()
        self._pending.pop(request_id, None)
        if request.status == "denied":
            raise ApprovalDenied(request_id, reason=request.reason)

        return request

    def approve(self, request_id: str, decided_by: str = "human") -> None:
        """Approve a pending request."""
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"No request with id: {request_id}")
        request.status = "approved"
        request.decided_by = f"human:{decided_by}" if decided_by != "auto" else "auto"
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(True)

    def deny(self, request_id: str, reason: str = None, decided_by: str = "human") -> None:
        """Deny a pending request."""
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"No request with id: {request_id}")
        request.status = "denied"
        request.decided_by = f"human:{decided_by}" if decided_by != "auto" else "auto"
        request.reason = reason
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(False)

    def get_pending(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [
            req for req in self._requests.values()
            if req.status == "pending"
        ]

    def get_history(self) -> List[ApprovalRequest]:
        """Get all approval requests (including resolved)."""
        return list(self._requests.values())


def create_approval_task(
    id: str = None,
    description: str = None,
    action_description: str = "Execute action",
    risk_level: RiskLevel = RiskLevel.MEDIUM,
    gate: ApprovalGate = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    summary_fn: Optional[Callable] = None,
) -> Task:
    """
    Factory function to create a Task that requires approval before proceeding.

    Args:
        id: Task identifier.
        description: Task description.
        action_description: Human-readable description of the action requiring approval.
        risk_level: Risk level of this action.
        gate: ApprovalGate instance. A default one is created if not provided.
        input_schema: Pydantic model for task input.
        output_schema: Pydantic model for task output.
        summary_fn: Optional callable(input_data) -> dict to extract a summary
                     from input data. Defaults to first 3 keys.

    Returns:
        A Task instance that gates execution behind approval.
    """
    task_id = id or f"approval_{uuid.uuid4().hex[:8]}"
    approval_gate = gate or ApprovalGate()

    if not input_schema:
        input_schema = type(
            f"{task_id}_Input",
            (BaseModel,),
            {"__annotations__": {"data": Dict[str, Any]}},
        )
    if not output_schema:
        output_schema = type(
            f"{task_id}_Output",
            (BaseModel,),
            {"__annotations__": {"data": Dict[str, Any]}},
        )

    async def execute(params, context):
        data = params["input_data"]

        # Build data summary
        if summary_fn:
            data_summary = summary_fn(data)
        else:
            # Default: first 3 keys
            keys = list(data.keys())[:3]
            data_summary = {k: data[k] for k in keys}

        request = await approval_gate.request_approval(
            task_id=context.task_id,
            execution_id=context.execution_id,
            action_description=action_description,
            risk_level=risk_level,
            data_summary=data_summary,
        )

        # If we get here, the request was approved
        return data

    return Task(
        id=task_id,
        description=description or f"Approval gate: {action_description}",
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
    )

import asyncio
import uuid
from typing import Type, Optional, Dict, Any, Callable
from pydantic import BaseModel

from water.core.task import Task
from water.core.exceptions import WaterError


class HumanInputRequired(Exception):
    """Raised when a human-in-the-loop task needs input before continuing."""

    def __init__(self, task_id: str, prompt: str, execution_id: str, request_id: str):
        self.task_id = task_id
        self.prompt = prompt
        self.execution_id = execution_id
        self.request_id = request_id
        super().__init__(
            f"Human input required for task '{task_id}': {prompt} "
            f"(request_id: {request_id})"
        )


class HumanInputManager:
    """
    Manages pending human input requests.

    When a HumanTask is executed, it registers a pending request here.
    External code (API handler, CLI, etc.) can then provide the response,
    which unblocks the waiting task.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, asyncio.Future] = {}
        self._prompts: Dict[str, str] = {}

    def create_request(self, request_id: str, prompt: str) -> asyncio.Future:
        """Create a pending human input request."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        self._prompts[request_id] = prompt
        return future

    def provide_input(self, request_id: str, data: Dict[str, Any]) -> None:
        """Provide human input for a pending request."""
        future = self._pending.get(request_id)
        if not future:
            raise ValueError(f"No pending request with id: {request_id}")
        if future.done():
            raise ValueError(f"Request {request_id} already resolved")
        future.set_result(data)

    def get_pending(self) -> Dict[str, str]:
        """Get all pending requests as {request_id: prompt}."""
        return {
            rid: prompt
            for rid, prompt in self._prompts.items()
            if rid in self._pending and not self._pending[rid].done()
        }

    def cancel(self, request_id: str) -> None:
        """Cancel a pending request."""
        future = self._pending.get(request_id)
        if future and not future.done():
            future.cancel()
        self._pending.pop(request_id, None)
        self._prompts.pop(request_id, None)


def create_human_task(
    id: Optional[str] = None,
    description: Optional[str] = None,
    prompt: str = "Please provide input",
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    human_input_manager: Optional[HumanInputManager] = None,
    timeout: Optional[float] = None,
    transform: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = None,
) -> Task:
    """
    Create a task that pauses execution and waits for human input.

    Args:
        id: Task identifier
        description: Task description
        prompt: Message shown to the human operator
        input_schema: Pydantic model for the data flowing into this task
        output_schema: Pydantic model for the data after human input is merged
        human_input_manager: Manager to coordinate input requests. If None,
                             raises HumanInputRequired immediately.
        timeout: How long to wait for human input (seconds). None = wait forever.
        transform: Optional function(input_data, human_response) -> output_data.
                   If not provided, human response is merged into input data.

    Returns:
        A Task instance that waits for human input during execution.
    """
    if not input_schema:
        input_schema = type(
            f"{id or 'human'}_Input",
            (BaseModel,),
            {"__annotations__": {"data": Dict[str, Any]}},
        )
    if not output_schema:
        output_schema = type(
            f"{id or 'human'}_Output",
            (BaseModel,),
            {"__annotations__": {"data": Dict[str, Any]}},
        )

    manager = human_input_manager

    async def execute(params, context):
        data = params["input_data"]
        request_id = f"human_{uuid.uuid4().hex[:8]}"

        if not manager:
            raise HumanInputRequired(
                task_id=context.task_id,
                prompt=prompt,
                execution_id=context.execution_id,
                request_id=request_id,
            )

        future = manager.create_request(request_id, prompt)

        try:
            if timeout:
                human_response = await asyncio.wait_for(future, timeout=timeout)
            else:
                human_response = await future
        except asyncio.TimeoutError:
            manager.cancel(request_id)
            raise TimeoutError(
                f"Human input for task '{context.task_id}' timed out after {timeout}s"
            )

        if transform:
            return transform(data, human_response)
        else:
            return {**data, **human_response}

    return Task(
        id=id or f"human_{uuid.uuid4().hex[:8]}",
        description=description or f"Human input: {prompt}",
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
    )

from typing import Any, Type, Callable, Optional, Dict
from pydantic import BaseModel
from water.core.exceptions import WaterError
import uuid

# Import here to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from water.core.types import InputData, OutputData
    from water.core.context import ExecutionContext


__all__ = [
    "Task",
    "create_task",
]


class Task:
    """
    A single executable unit within a Water flow.

    Tasks define input/output schemas using Pydantic models and contain
    an execute function that processes data. Tasks can be synchronous
    or asynchronous.
    """

    def __init__(
        self,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        execute: Callable[[Dict[str, 'InputData'], 'ExecutionContext'], 'OutputData'],
        id: Optional[str] = None,
        description: Optional[str] = None,
        retry_count: int = 0,
        retry_delay: float = 0.0,
        retry_backoff: float = 1.0,
        timeout: Optional[float] = None,
        validate_schema: bool = False,
        rate_limit: Optional[float] = None,
        cache: Optional[Any] = None,
        circuit_breaker: Optional[Any] = None,
    ) -> None:
        """
        Initialize a new Task.

        Args:
            input_schema: Pydantic BaseModel class defining expected input structure
            output_schema: Pydantic BaseModel class defining output structure
            execute: Function that processes input data and returns output
            id: Unique identifier for the task. Auto-generated if not provided.
            description: Human-readable description of the task's purpose
            retry_count: Number of retry attempts on failure (0 = no retries)
            retry_delay: Initial delay in seconds between retries
            retry_backoff: Multiplier applied to delay after each retry (1.0 = fixed, 2.0 = exponential)
            timeout: Optional timeout in seconds for task execution
            validate_schema: If True, validate input/output against schemas at runtime
            rate_limit: Optional max executions per second (e.g., 5.0 for 5 calls/sec)
            cache: Optional TaskCache instance for memoizing task results
            circuit_breaker: Optional CircuitBreaker instance for protecting external calls

        Raises:
            WaterError: If schemas are not Pydantic BaseModel classes or execute is not callable
        """
        self.id: str = id if id else f"task_{uuid.uuid4().hex[:8]}"
        self.description: str = description if description else f"Task {self.id}"

        # Validate schemas are Pydantic BaseModel classes
        if not input_schema or not (isinstance(input_schema, type) and issubclass(input_schema, BaseModel)):
            raise WaterError("input_schema must be a Pydantic BaseModel class")
        if not output_schema or not (isinstance(output_schema, type) and issubclass(output_schema, BaseModel)):
            raise WaterError("output_schema must be a Pydantic BaseModel class")
        if not execute or not callable(execute):
            raise WaterError("Task must have a callable execute function")

        self.input_schema = input_schema
        self.output_schema = output_schema
        self.execute = execute
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.validate_schema = validate_schema
        self.rate_limit = rate_limit
        self.cache = cache
        self.circuit_breaker = circuit_breaker


def create_task(
    id: Optional[str] = None,
    description: Optional[str] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    execute: Optional[Callable[[Dict[str, 'InputData'], 'ExecutionContext'], 'OutputData']] = None,
    retry_count: int = 0,
    retry_delay: float = 0.0,
    retry_backoff: float = 1.0,
    timeout: Optional[float] = None,
    validate_schema: bool = False,
    rate_limit: Optional[float] = None,
    cache: Optional[Any] = None,
    circuit_breaker: Optional[Any] = None,
) -> Task:
    """
    Factory function to create a Task instance.
    """
    return Task(
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
        id=id,
        description=description,
        retry_count=retry_count,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
        timeout=timeout,
        validate_schema=validate_schema,
        rate_limit=rate_limit,
        cache=cache,
        circuit_breaker=circuit_breaker,
    )

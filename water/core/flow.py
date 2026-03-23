from typing import Any, List, Optional, Tuple, Dict, Type
import asyncio
import inspect
import logging
import uuid

from pydantic import BaseModel


__all__ = [
    "Flow",
]


logger = logging.getLogger(__name__)

from water.core.engine import ExecutionEngine, NodeType, FlowPausedError, FlowStoppedError
from water.middleware.hooks import HookManager
from water.core.types import (
    InputData,
    OutputData,
    ConditionFunction,
    ExecutionNode
)

class Flow:
    """
    A workflow orchestrator that allows building and executing complex data processing pipelines.

    Flows support sequential execution, parallel processing, conditional branching, and loops.
    All flows must be registered before execution. Optionally accepts a storage backend
    to enable pause, stop, and resume of workflows.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        description: Optional[str] = None,
        storage: Optional[Any] = None,
        version: Optional[str] = None,
        strict_contracts: bool = False,
        max_concurrency: int = 10,
    ) -> None:
        """
        Initialize a new Flow.

        Args:
            id: Unique identifier for the flow. Auto-generated if not provided.
            description: Human-readable description of the flow's purpose.
            storage: Optional storage backend for persistence and pause/resume support.
            version: Optional version string for tracking flow schema changes.
            strict_contracts: If True, register() raises ValueError on contract violations
                              instead of just logging warnings.
        """
        self.id: str = id if id else f"flow_{uuid.uuid4().hex[:8]}"
        self.description: str = description if description else f"Flow {self.id}"
        self.version: Optional[str] = version
        self.strict_contracts: bool = strict_contracts
        self.max_concurrency: int = max_concurrency
        self._tasks: List[ExecutionNode] = []
        self._registered: bool = False
        self.metadata: Dict[str, Any] = {}
        self.storage = storage
        self.hooks = HookManager()
        self.events: Optional[Any] = None
        self.telemetry: Optional[Any] = None
        self.checkpoint: Optional[Any] = None
        self.middleware: List[Any] = []
        self.dlq: Optional[Any] = None
        self.secrets: Optional[Any] = None
        self._services: Dict[str, Any] = {}

    def _validate_registration_state(self) -> None:
        """Ensure flow is not registered when adding tasks."""
        if self._registered:
            raise RuntimeError("Cannot add tasks after registration")

    def _validate_task(self, task: Any) -> None:
        """Validate that a task is not None."""
        if task is None:
            raise ValueError("Task cannot be None")

    @staticmethod
    def _coerce_task(task: Any) -> Any:
        """Convert a Flow to a Task if needed."""
        if isinstance(task, Flow):
            return task.as_task()
        return task

    def _validate_condition(self, condition: ConditionFunction) -> None:
        """Validate that a condition function is not async."""
        if inspect.iscoroutinefunction(condition):
            raise ValueError("Branch conditions cannot be async functions")

    def _validate_loop_condition(self, condition: ConditionFunction) -> None:
        """Validate that a loop condition function is not async."""
        if inspect.iscoroutinefunction(condition):
            raise ValueError("Loop conditions cannot be async functions")

    def use(self, middleware: Any) -> 'Flow':
        """
        Add a middleware to the flow.

        Middleware ``before_task`` / ``after_task`` hooks are called around
        every task execution, in the order they were added.

        Args:
            middleware: A Middleware instance (or any object with before_task/after_task).

        Returns:
            Self for method chaining.
        """
        self.middleware.append(middleware)
        return self

    def inject(self, name: str, service: Any) -> 'Flow':
        """
        Register a shared service for dependency injection into tasks.

        Injected services are available to all tasks via
        ``context.get_service(name)`` during flow execution.

        Args:
            name: Unique name to identify the service
            service: The service instance to inject

        Returns:
            Self for method chaining
        """
        self._services[name] = service
        return self

    def set_metadata(self, key: str, value: Any) -> 'Flow':
        """
        Set metadata for this flow.

        Args:
            key: The metadata key
            value: The metadata value

        Returns:
            Self for method chaining
        """
        self.metadata[key] = value
        return self

    def then(self, task: Any, when: Optional[ConditionFunction] = None, fallback: Any = None) -> 'Flow':
        """
        Add a task to execute sequentially.

        Args:
            task: The task to execute
            when: Optional condition function. If provided and returns False,
                  the task is skipped and data passes through unchanged.
            fallback: Optional fallback task. If the primary task raises an
                      exception, the fallback task runs instead.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If task is None
        """
        self._validate_registration_state()
        task = self._coerce_task(task)
        self._validate_task(task)
        if when is not None:
            self._validate_condition(when)

        if fallback is not None:
            fallback = self._coerce_task(fallback)
            self._validate_task(fallback)

        node: ExecutionNode = {"type": NodeType.SEQUENTIAL.value, "task": task}
        if when is not None:
            node["when"] = when
        if fallback is not None:
            node["fallback"] = fallback
        self._tasks.append(node)
        return self

    def map(self, task: Any, over: str) -> 'Flow':
        """
        Execute a task once per item in a list field, in parallel.

        Args:
            task: The task to execute for each item
            over: Key in the input data containing the list to iterate over.
                  Each task invocation receives the full data dict with that
                  key replaced by the individual item.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If task is None or over is empty
        """
        self._validate_registration_state()
        task = self._coerce_task(task)
        self._validate_task(task)
        if not over:
            raise ValueError("Map 'over' key cannot be empty")

        node: ExecutionNode = {
            "type": NodeType.MAP.value,
            "task": task,
            "over": over,
        }
        self._tasks.append(node)
        return self

    def dag(self, tasks: List[Any], dependencies: Dict[str, List[str]] = None) -> 'Flow':
        """
        Add a DAG (directed acyclic graph) of tasks with automatic parallelization.

        Tasks with no dependencies run in parallel. As tasks complete, their
        dependents are unlocked and executed.

        Args:
            tasks: List of tasks to execute
            dependencies: Dict mapping task_id -> list of task_ids it depends on.
                          Tasks not listed have no dependencies.

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If task list is empty
        """
        self._validate_registration_state()
        if not tasks:
            raise ValueError("DAG task list cannot be empty")

        coerced = [self._coerce_task(t) for t in tasks]
        for task in coerced:
            self._validate_task(task)

        node: ExecutionNode = {
            "type": NodeType.DAG.value,
            "tasks": list(coerced),
            "dependencies": dependencies or {},
        }
        self._tasks.append(node)
        return self

    def parallel(self, tasks: List[Any]) -> 'Flow':
        """
        Add tasks to execute in parallel.

        Args:
            tasks: List of tasks or Flows to execute concurrently

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If task list is empty or contains None values
        """
        self._validate_registration_state()
        if not tasks:
            raise ValueError("Parallel task list cannot be empty")

        coerced = [self._coerce_task(t) for t in tasks]
        for task in coerced:
            self._validate_task(task)

        node: ExecutionNode = {
            "type": NodeType.PARALLEL.value,
            "tasks": list(coerced)
        }
        self._tasks.append(node)
        return self

    def branch(self, branches: List[Tuple[ConditionFunction, Any]]) -> 'Flow':
        """
        Add conditional branching logic.

        Executes the first task whose condition returns True.
        If no conditions match, data passes through unchanged.

        Args:
            branches: List of (condition_function, task_or_flow) tuples

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If branch list is empty, task is None, or condition is async
        """
        self._validate_registration_state()
        if not branches:
            raise ValueError("Branch list cannot be empty")

        coerced_branches = [(cond, self._coerce_task(task)) for cond, task in branches]
        for condition, task in coerced_branches:
            self._validate_task(task)
            self._validate_condition(condition)

        node: ExecutionNode = {
            "type": NodeType.BRANCH.value,
            "branches": [{"condition": cond, "task": task} for cond, task in coerced_branches]
        }
        self._tasks.append(node)
        return self

    def loop(
        self,
        condition: ConditionFunction,
        task: Any,
        max_iterations: int = 100
    ) -> 'Flow':
        """
        Execute a task repeatedly while a condition is true.

        Args:
            condition: Function that returns True to continue looping
            task: Task to execute on each iteration
            max_iterations: Maximum number of iterations to prevent infinite loops

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If task is None or condition is async
        """
        self._validate_registration_state()
        task = self._coerce_task(task)
        self._validate_task(task)
        self._validate_loop_condition(condition)

        node: ExecutionNode = {
            "type": NodeType.LOOP.value,
            "condition": condition,
            "task": task,
            "max_iterations": max_iterations
        }
        self._tasks.append(node)
        return self

    def agentic_loop(
        self,
        provider,
        tools=None,
        system_prompt: str = "",
        prompt_template: str = "",
        max_iterations: int = 10,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop_tool: bool = False,
        on_step=None,
        on_tool_call=None,
        stop_condition=None,
        observation_formatter=None,
    ) -> 'Flow':
        """Add a model-controlled agentic loop (ReAct pattern).

        The loop follows Think-Act-Observe-Repeat: the LLM reasons (Think),
        calls tools (Act), receives results (Observe), and repeats until it
        responds without tool calls or calls __done__.

        Args:
            provider: LLM provider instance for making completions.
            tools: List of Tool objects or a Toolkit instance.
            system_prompt: System prompt for the agent.
            prompt_template: Template string with {variable} placeholders for input data.
            max_iterations: Safety limit on iterations (default 10).
            temperature: LLM temperature setting.
            max_tokens: Max tokens for LLM response.
            stop_tool: If True, inject a __done__ tool for explicit completion signaling.
            on_step: Callback(iteration, step_dict) called after each Think-Act-Observe cycle.
                step_dict has keys: think (str), act (list of tool calls), observe (list of results).
            on_tool_call: Callback(tool_name, tool_args) called before each tool execution.
                Return False to reject, a dict to modify args, or None/True to proceed.
            stop_condition: Callback(steps, tool_history) returning True to stop the loop early.
            observation_formatter: Callback(tool_name, tool_args, tool_result) returning a string
                to customize how tool results are formatted before feeding back to the LLM.

        Returns:
            Self for method chaining.
        """
        self._validate_registration_state()

        if max_iterations <= 0:
            raise ValueError(f"Flow: max_iterations must be > 0, got {max_iterations}")

        # Handle stop tool injection
        actual_tools = tools
        if stop_tool:
            from water.agents.tools import Tool, Toolkit
            done_tool = Tool(
                name="__done__",
                description="Call this tool when you have completed the task and want to provide your final answer.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "final_answer": {"type": "string", "description": "Your final answer to the user's request"},
                        "metadata": {"type": "object", "description": "Optional metadata about the result", "default": {}},
                    },
                    "required": ["final_answer"],
                },
                execute=lambda final_answer, metadata=None: {"final_answer": final_answer, "metadata": metadata or {}},
            )
            if actual_tools is None:
                actual_tools = [done_tool]
            elif isinstance(actual_tools, list):
                actual_tools = actual_tools + [done_tool]
            elif isinstance(actual_tools, Toolkit):
                actual_tools = list(actual_tools) + [done_tool]

        node = {
            "type": "agentic_loop",
            "provider": provider,
            "tools": actual_tools,
            "system_prompt": system_prompt,
            "max_iterations": max_iterations,
            "config": {
                "prompt_template": prompt_template,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "on_step": on_step,
                "on_tool_call": on_tool_call,
                "stop_condition": stop_condition,
                "observation_formatter": observation_formatter,
            },
        }
        self._tasks.append(node)
        return self

    def try_catch(self, try_tasks, catch_handler=None, finally_handler=None) -> 'Flow':
        """
        Add try-catch-finally error handling to the flow.

        Args:
            try_tasks: List of tasks or single task to execute in try block
            catch_handler: Optional task or callable(error, context) -> Dict to handle errors
            finally_handler: Optional task to always execute after try/catch

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If try_tasks is empty or None
        """
        self._validate_registration_state()

        # Normalize to list
        if not isinstance(try_tasks, list):
            try_tasks = [try_tasks]

        if not try_tasks:
            raise ValueError("try_tasks cannot be empty")

        coerced_tasks = [self._coerce_task(t) for t in try_tasks]
        for task in coerced_tasks:
            self._validate_task(task)

        # Coerce catch/finally handlers if they are Tasks/Flows
        if catch_handler is not None:
            if hasattr(catch_handler, "execute"):
                catch_handler = self._coerce_task(catch_handler)
            elif not callable(catch_handler):
                raise ValueError("catch_handler must be a task or callable")

        if finally_handler is not None:
            if hasattr(finally_handler, "execute"):
                finally_handler = self._coerce_task(finally_handler)
            elif not callable(finally_handler):
                raise ValueError("finally_handler must be a task or callable")

        node: ExecutionNode = {
            "type": NodeType.TRY_CATCH.value,
            "tasks": coerced_tasks,
            "task": coerced_tasks[0] if len(coerced_tasks) == 1 else None,
            "config": {
                "catch_handler": catch_handler,
                "finally_handler": finally_handler,
            },
        }
        self._tasks.append(node)
        return self

    def on_error(self, handler) -> 'Flow':
        """
        Set a global error handler for the flow.

        Wraps all currently added tasks in a try-catch block with the given
        handler as the catch_handler. Must be called after adding tasks but
        before register().

        Args:
            handler: A task or callable(error, context) -> Dict to handle any error

        Returns:
            Self for method chaining

        Raises:
            RuntimeError: If flow is already registered
            ValueError: If handler is None or no tasks exist to wrap
        """
        self._validate_registration_state()
        if handler is None:
            raise ValueError("Error handler cannot be None")
        if not self._tasks:
            raise ValueError("No tasks to wrap with error handler")

        if hasattr(handler, "execute"):
            handler = self._coerce_task(handler)

        # Collect all existing task nodes as try_tasks for a single TRY_CATCH wrapper.
        # We rebuild the flow graph as a single try-catch node wrapping the originals.
        existing_tasks = list(self._tasks)
        self._tasks.clear()

        # Build an inner execution list — we store the original nodes and
        # the engine will run them sequentially inside the try block.
        # To keep it simple, we store the original nodes in config and
        # re-use the engine's existing node dispatch.
        node: ExecutionNode = {
            "type": NodeType.TRY_CATCH.value,
            "tasks": [],
            "task": None,
            "config": {
                "catch_handler": handler,
                "finally_handler": None,
                "_wrapped_nodes": existing_tasks,
            },
        }
        self._tasks.append(node)
        return self

    def as_task(
        self,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """
        Convert this flow into a Task that can be used inside another flow.

        Args:
            input_schema: Pydantic model for input (defaults to a generic dict model)
            output_schema: Pydantic model for output (defaults to a generic dict model)

        Returns:
            A Task instance that executes this sub-flow
        """
        from water.core.task import Task

        if not self._registered:
            raise RuntimeError("Sub-flow must be registered before converting to task")

        # Default generic schemas
        if not input_schema:
            input_schema = type(
                f"{self.id}_Input",
                (BaseModel,),
                {"__annotations__": {"data": Dict[str, Any]}},
            )
        if not output_schema:
            output_schema = type(
                f"{self.id}_Output",
                (BaseModel,),
                {"__annotations__": {"data": Dict[str, Any]}},
            )

        sub_flow = self

        async def execute_sub_flow(params, context):
            input_data = params["input_data"]
            return await sub_flow.run(input_data)

        return Task(
            id=f"subflow_{self.id}",
            description=f"Sub-flow: {self.description}",
            input_schema=input_schema,
            output_schema=output_schema,
            execute=execute_sub_flow,
        )

    @staticmethod
    def _get_model_field_names(schema: Any) -> Optional[set]:
        """Extract field names from a Pydantic model class.

        Works with both Pydantic v1 (__fields__) and v2 (model_fields).
        Returns None if the schema is not a Pydantic model.
        """
        if schema is None:
            return None
        if hasattr(schema, "model_fields"):
            return set(schema.model_fields.keys())
        if hasattr(schema, "__fields__"):
            return set(schema.__fields__.keys())
        return None

    def validate_contracts(self) -> List[Dict[str, Any]]:
        """Validate data contracts between sequential tasks.

        Walks through the task chain and checks that for sequential tasks,
        the output_schema fields of task N overlap with the input_schema
        fields of task N+1. Only checks sequential (.then()) tasks.

        Returns:
            A list of violations. Each violation is a dict with keys:
            from_task, to_task, missing_fields, message.
            An empty list means all contracts are satisfied.
        """
        violations: List[Dict[str, Any]] = []

        sequential_tasks = []
        for node in self._tasks:
            if node.get("type") == NodeType.SEQUENTIAL.value:
                task = node.get("task")
                if task is not None:
                    sequential_tasks.append(task)

        for i in range(len(sequential_tasks) - 1):
            task_a = sequential_tasks[i]
            task_b = sequential_tasks[i + 1]

            output_fields = self._get_model_field_names(
                getattr(task_a, "output_schema", None)
            )
            input_fields = self._get_model_field_names(
                getattr(task_b, "input_schema", None)
            )

            if output_fields is None or input_fields is None:
                continue

            missing = sorted(input_fields - output_fields)
            if missing:
                violations.append({
                    "from_task": task_a.id,
                    "to_task": task_b.id,
                    "missing_fields": missing,
                    "message": (
                        f"Task '{task_a.id}' output_schema is missing fields "
                        f"required by task '{task_b.id}' input_schema: {missing}"
                    ),
                })

        return violations

    def register(self) -> 'Flow':
        """
        Register the flow for execution.

        Must be called before running the flow.
        Once registered, no more tasks can be added.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If flow has no tasks, or if strict_contracts is True
                        and contract violations are found.
        """
        if not self._tasks:
            raise ValueError("Flow must have at least one task")
        self._registered = True

        violations = self.validate_contracts()
        if violations:
            if self.strict_contracts:
                messages = [v["message"] for v in violations]
                raise ValueError(
                    "Data contract violations found:\n" + "\n".join(messages)
                )
            else:
                for v in violations:
                    logger.warning("Data contract violation: %s", v["message"])

        return self

    async def run(self, input_data: InputData) -> OutputData:
        """
        Execute the flow with the provided input data.

        Args:
            input_data: Input data dictionary to process

        Returns:
            The final output data after all tasks complete

        Raises:
            RuntimeError: If flow is not registered
            FlowPausedError: If the flow was paused during execution
            FlowStoppedError: If the flow was stopped during execution
        """
        if not self._registered:
            raise RuntimeError("Flow must be registered before running")

        if self.version:
            self.metadata["_flow_version"] = self.version

        if self.secrets is not None:
            self._services["secrets"] = self.secrets

        await self.hooks.emit("on_flow_start", flow_id=self.id, input_data=input_data)

        if self.events:
            from water.middleware.events import FlowEvent
            await self.events.emit(FlowEvent("flow_start", self.id, data={"input": input_data}))

        try:
            result = await ExecutionEngine.run(
                self._tasks,
                input_data,
                flow_id=self.id,
                flow_metadata=self.metadata,
                storage=self.storage,
                hooks=self.hooks,
                event_emitter=self.events,
                telemetry=self.telemetry,
                checkpoint=self.checkpoint,
                middleware=self.middleware if self.middleware else None,
                dlq=self.dlq,
                services=self._services if self._services else None,
                max_concurrency=self.max_concurrency,
            )
            await self.hooks.emit("on_flow_complete", flow_id=self.id, output_data=result)
            if self.events:
                await self.events.emit(FlowEvent("flow_complete", self.id, data={"output": result}))
                await self.events.close()
            return result
        except (FlowPausedError, FlowStoppedError):
            raise
        except Exception as e:
            await self.hooks.emit("on_flow_error", flow_id=self.id, error=e)
            if self.events:
                await self.events.emit(FlowEvent("flow_error", self.id, data={"error": str(e)}))
                await self.events.close()
            raise

    async def run_batch(
        self,
        inputs: List[InputData],
        max_concurrency: int = 10,
        return_exceptions: bool = False,
    ) -> List[Any]:
        """
        Execute the flow against multiple inputs with concurrency control.

        Args:
            inputs: List of input data dictionaries to process
            max_concurrency: Maximum number of concurrent flow executions
            return_exceptions: If True, exceptions are returned in the results list
                               instead of being raised

        Returns:
            List of results in the same order as inputs

        Raises:
            RuntimeError: If flow is not registered
            Exception: If return_exceptions is False and any execution fails
        """
        if not self._registered:
            raise RuntimeError("Flow must be registered before running")

        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_one(input_data):
            async with semaphore:
                return await self.run(input_data)

        tasks = [run_one(inp) for inp in inputs]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def pause(self, execution_id: str) -> None:
        """
        Request a running flow to pause at the next node boundary.

        The flow will save its state and can be resumed later with resume().

        Args:
            execution_id: The execution ID of the running flow

        Raises:
            RuntimeError: If no storage backend is configured
            ValueError: If session not found or not in a pausable state
        """
        if not self.storage:
            raise RuntimeError("Storage backend required for pause/resume")

        from water.storage import FlowStatus
        session = await self.storage.get_session(execution_id)
        if not session:
            raise ValueError(f"No session found for execution: {execution_id}")
        if session.status != FlowStatus.RUNNING:
            raise ValueError(
                f"Cannot pause flow in '{session.status.value}' state "
                f"(must be 'running')"
            )

        session.status = FlowStatus.PAUSED
        await self.storage.save_session(session)

    async def stop(self, execution_id: str) -> None:
        """
        Request a running flow to stop at the next node boundary.

        The flow will save its state. Stopped flows cannot be resumed.

        Args:
            execution_id: The execution ID of the running flow

        Raises:
            RuntimeError: If no storage backend is configured
            ValueError: If session not found or not in a stoppable state
        """
        if not self.storage:
            raise RuntimeError("Storage backend required for stop")

        from water.storage import FlowStatus
        session = await self.storage.get_session(execution_id)
        if not session:
            raise ValueError(f"No session found for execution: {execution_id}")
        if session.status not in (FlowStatus.RUNNING, FlowStatus.PAUSED):
            raise ValueError(
                f"Cannot stop flow in '{session.status.value}' state "
                f"(must be 'running' or 'paused')"
            )

        session.status = FlowStatus.STOPPED
        await self.storage.save_session(session)

    async def resume(self, execution_id: str) -> OutputData:
        """
        Resume a paused flow from where it left off.

        Args:
            execution_id: The execution ID of the paused flow

        Returns:
            The final output data after all remaining tasks complete

        Raises:
            RuntimeError: If no storage backend is configured or flow not registered
            ValueError: If session not found or not in 'paused' state
        """
        if not self.storage:
            raise RuntimeError("Storage backend required for resume")
        if not self._registered:
            raise RuntimeError("Flow must be registered before resuming")

        from water.storage import FlowStatus
        session = await self.storage.get_session(execution_id)
        if not session:
            raise ValueError(f"No session found for execution: {execution_id}")
        if session.status != FlowStatus.PAUSED:
            raise ValueError(
                f"Cannot resume flow in '{session.status.value}' state "
                f"(must be 'paused')"
            )

        # Check for version mismatch between paused session and current flow
        if self.version:
            import logging
            _logger = logging.getLogger(__name__)
            session_version = session.context_state.get("flow_version")
            if session_version and session_version != self.version:
                _logger.warning(
                    f"Flow version mismatch on resume: session was paused with "
                    f"v{session_version} but current flow is v{self.version}. "
                    f"Execution may produce unexpected results."
                )

        resume_from = {
            "execution_id": session.execution_id,
            "node_index": session.current_node_index,
            "data": session.current_data,
            "context_state": session.context_state,
        }

        return await ExecutionEngine.run(
            self._tasks,
            session.input_data,
            flow_id=self.id,
            flow_metadata=self.metadata,
            storage=self.storage,
            resume_from=resume_from,
            hooks=self.hooks,
        )

    async def get_session(self, execution_id: str):
        """
        Get the session for a given execution.

        Args:
            execution_id: The execution ID to look up

        Returns:
            FlowSession if found, None otherwise

        Raises:
            RuntimeError: If no storage backend is configured
        """
        if not self.storage:
            raise RuntimeError("Storage backend required")
        return await self.storage.get_session(execution_id)

    async def get_task_runs(self, execution_id: str):
        """
        Get all task runs for a given execution.

        Args:
            execution_id: The execution ID to look up

        Returns:
            List of TaskRun records

        Raises:
            RuntimeError: If no storage backend is configured
        """
        if not self.storage:
            raise RuntimeError("Storage backend required")
        return await self.storage.get_task_runs(execution_id)

    async def dry_run(self, input_data: InputData) -> Dict[str, Any]:
        """
        Validate flow structure and data shape without executing tasks.

        Walks through the execution graph, validates input data against each
        task's input_schema using Pydantic, and returns a detailed report.

        Args:
            input_data: Input data dictionary to validate

        Returns:
            A report dict with flow_id, valid flag, per-node info, and errors.

        Raises:
            RuntimeError: If flow is not registered
        """
        if not self._registered:
            raise RuntimeError("Flow must be registered before running")

        nodes_report: List[Dict[str, Any]] = []
        top_errors: List[str] = []
        all_valid = True

        for idx, node in enumerate(self._tasks):
            node_type = NodeType(node["type"])
            node_info: Dict[str, Any] = {"index": idx, "type": node_type.value}

            if node_type == NodeType.SEQUENTIAL:
                task = node["task"]
                node_info["task_id"] = task.id
                valid, errors = self._validate_input_schema(task, input_data)
                node_info["input_valid"] = valid
                node_info["errors"] = errors
                if not valid:
                    all_valid = False
                # Report whether 'when' condition would fire
                when = node.get("when")
                if when is not None:
                    try:
                        node_info["condition_matches"] = bool(when(input_data))
                    except Exception as e:
                        logger.warning("Condition evaluation error during dry run: %s", e, exc_info=True)
                        node_info["condition_matches"] = None
                        node_info["errors"].append(f"Condition evaluation error: {e}")
                        all_valid = False

            elif node_type == NodeType.PARALLEL:
                tasks = node["tasks"]
                task_ids = [t.id for t in tasks]
                node_info["task_ids"] = task_ids
                node_errors: List[str] = []
                node_valid = True
                for t in tasks:
                    valid, errors = self._validate_input_schema(t, input_data)
                    if not valid:
                        node_valid = False
                        node_errors.extend(errors)
                node_info["input_valid"] = node_valid
                node_info["errors"] = node_errors
                if not node_valid:
                    all_valid = False

            elif node_type == NodeType.BRANCH:
                branches = node["branches"]
                branch_reports = []
                node_errors = []
                node_valid = True
                for b_idx, branch in enumerate(branches):
                    task = branch["task"]
                    condition = branch["condition"]
                    try:
                        matches = bool(condition(input_data))
                    except Exception as e:
                        matches = None
                        node_errors.append(f"Branch {b_idx} condition error: {e}")
                        all_valid = False
                        node_valid = False
                    valid, errors = self._validate_input_schema(task, input_data)
                    if not valid:
                        node_valid = False
                        node_errors.extend(errors)
                    branch_reports.append({
                        "task_id": task.id,
                        "condition_matches": matches,
                    })
                node_info["branches"] = branch_reports
                node_info["input_valid"] = node_valid
                node_info["errors"] = node_errors
                if not node_valid:
                    all_valid = False

            elif node_type == NodeType.LOOP:
                task = node["task"]
                node_info["task_id"] = task.id
                valid, errors = self._validate_input_schema(task, input_data)
                node_info["input_valid"] = valid
                node_info["errors"] = errors
                if not valid:
                    all_valid = False

            elif node_type == NodeType.MAP:
                task = node["task"]
                node_info["task_id"] = task.id
                node_info["over"] = node["over"]
                valid, errors = self._validate_input_schema(task, input_data)
                node_info["input_valid"] = valid
                node_info["errors"] = errors
                if not valid:
                    all_valid = False

            elif node_type == NodeType.DAG:
                tasks = node["tasks"]
                task_ids = [t.id for t in tasks]
                dependencies = node.get("dependencies", {})
                node_info["task_ids"] = task_ids
                node_errors = []
                node_valid = True

                # Validate input schemas
                for t in tasks:
                    valid, errors = self._validate_input_schema(t, input_data)
                    if not valid:
                        node_valid = False
                        node_errors.extend(errors)

                # Validate dependency graph: unknown deps
                all_task_ids = set(task_ids)
                for task_id, deps in dependencies.items():
                    if task_id not in all_task_ids:
                        node_errors.append(f"Dependency references unknown task: {task_id}")
                        node_valid = False
                    for dep in deps:
                        if dep not in all_task_ids:
                            node_errors.append(f"Task '{task_id}' depends on unknown task: {dep}")
                            node_valid = False

                # Detect cycles via topological sort
                in_degree = {tid: 0 for tid in all_task_ids}
                for task_id, deps in dependencies.items():
                    if task_id in in_degree:
                        in_degree[task_id] = len(deps)
                ready = [tid for tid, deg in in_degree.items() if deg == 0]
                visited = 0
                queue = list(ready)
                dependents: Dict[str, List[str]] = {tid: [] for tid in all_task_ids}
                for task_id, deps in dependencies.items():
                    for dep in deps:
                        if dep in dependents:
                            dependents[dep].append(task_id)
                while queue:
                    current = queue.pop(0)
                    visited += 1
                    for dep_id in dependents.get(current, []):
                        in_degree[dep_id] -= 1
                        if in_degree[dep_id] == 0:
                            queue.append(dep_id)
                if visited < len(all_task_ids):
                    node_errors.append("DAG has circular dependencies")
                    node_valid = False

                node_info["input_valid"] = node_valid
                node_info["errors"] = node_errors
                if not node_valid:
                    all_valid = False

            nodes_report.append(node_info)

        return {
            "flow_id": self.id,
            "valid": all_valid,
            "nodes": nodes_report,
            "errors": top_errors,
        }

    def visualize(self, format: str = "mermaid") -> str:
        """
        Generate a visual diagram of this flow's execution graph.

        Args:
            format: Output format. Currently only "mermaid" is supported.

        Returns:
            A string containing the diagram in the requested format.

        Raises:
            RuntimeError: If flow is not registered.
            ValueError: If an unsupported format is requested.
        """
        if not self._registered:
            raise RuntimeError("Flow must be registered before visualizing")

        if format != "mermaid":
            raise ValueError(f"Unsupported visualization format: '{format}'. Supported formats: mermaid")

        lines: List[str] = ["graph TD"]
        node_counter = 0

        def _next_id() -> str:
            nonlocal node_counter
            nid = f"N{node_counter}"
            node_counter += 1
            return nid

        prev_id: Optional[str] = None

        for node in self._tasks:
            node_type = NodeType(node["type"])

            if node_type == NodeType.SEQUENTIAL:
                task = node["task"]
                nid = _next_id()
                lines.append(f"    {nid}[{task.id}]")
                if prev_id is not None:
                    lines.append(f"    {prev_id} --> {nid}")
                prev_id = nid

            elif node_type == NodeType.PARALLEL:
                tasks = node["tasks"]
                fork_id = _next_id()
                lines.append(f"    {fork_id}{{{{fork}}}}")
                if prev_id is not None:
                    lines.append(f"    {prev_id} --> {fork_id}")
                join_id = _next_id()
                lines.append(f"    {join_id}{{{{join}}}}")
                for t in tasks:
                    tid = _next_id()
                    lines.append(f"    {tid}[{t.id}]")
                    lines.append(f"    {fork_id} --> {tid}")
                    lines.append(f"    {tid} --> {join_id}")
                prev_id = join_id

            elif node_type == NodeType.BRANCH:
                branches = node["branches"]
                decision_id = _next_id()
                lines.append(f"    {decision_id}{{{decision_id}}}")
                if prev_id is not None:
                    lines.append(f"    {prev_id} --> {decision_id}")
                end_id = _next_id()
                lines.append(f"    {end_id}[end_branch]")
                for b_idx, branch in enumerate(branches):
                    task = branch["task"]
                    cond = branch["condition"]
                    label = getattr(cond, "__name__", f"condition_{b_idx}")
                    tid = _next_id()
                    lines.append(f"    {tid}[{task.id}]")
                    lines.append(f"    {decision_id} -->|{label}| {tid}")
                    lines.append(f"    {tid} --> {end_id}")
                prev_id = end_id

            elif node_type == NodeType.LOOP:
                task = node["task"]
                condition = node["condition"]
                label = getattr(condition, "__name__", "condition")
                nid = _next_id()
                lines.append(f"    {nid}[{task.id}]")
                if prev_id is not None:
                    lines.append(f"    {prev_id} --> {nid}")
                lines.append(f"    {nid} -->|{label}| {nid}")
                prev_id = nid

            elif node_type == NodeType.MAP:
                task = node["task"]
                over = node["over"]
                nid = _next_id()
                lines.append(f"    {nid}[\"{task.id} (map over {over})\"]")
                if prev_id is not None:
                    lines.append(f"    {prev_id} --> {nid}")
                prev_id = nid

            elif node_type == NodeType.DAG:
                tasks = node["tasks"]
                dependencies = node.get("dependencies", {})
                task_id_to_nid: Dict[str, str] = {}
                for t in tasks:
                    nid = _next_id()
                    task_id_to_nid[t.id] = nid
                    lines.append(f"    {nid}[{t.id}]")
                if prev_id is not None:
                    for t in tasks:
                        if t.id not in dependencies or not dependencies[t.id]:
                            lines.append(f"    {prev_id} --> {task_id_to_nid[t.id]}")
                for task_id, deps in dependencies.items():
                    for dep in deps:
                        if dep in task_id_to_nid and task_id in task_id_to_nid:
                            lines.append(f"    {task_id_to_nid[dep]} --> {task_id_to_nid[task_id]}")
                if tasks:
                    all_deps_of_others: set = set()
                    for deps in dependencies.values():
                        all_deps_of_others.update(deps)
                    leaves = [t for t in tasks if t.id not in all_deps_of_others]
                    if len(leaves) == 1:
                        prev_id = task_id_to_nid[leaves[0].id]
                    else:
                        prev_id = task_id_to_nid[tasks[-1].id]

        return "\n".join(lines)

    @staticmethod
    def _validate_input_schema(task: Any, data: InputData) -> Tuple[bool, List[str]]:
        """Validate input_data against a task's input_schema using Pydantic."""
        if not hasattr(task, "input_schema") or task.input_schema is None:
            return True, []
        try:
            task.input_schema(**data)
            return True, []
        except Exception as e:
            logger.warning("Input validation failed for task '%s': %s", task.id, e, exc_info=True)
            return False, [f"Task '{task.id}' input validation failed: {e}"]

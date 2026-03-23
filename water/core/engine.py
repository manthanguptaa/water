import inspect
import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from water.core.types import (
    ExecutionGraph,
    ExecutionNode,
    InputData,
    OutputData,
    SequentialNode,
    ParallelNode,
    BranchNode,
    LoopNode
)
from water.core.context import ExecutionContext

logger = logging.getLogger(__name__)

__all__ = [
    "NodeType",
    "FlowPausedError",
    "FlowStoppedError",
    "ExecutionEngine",
]

class NodeType(Enum):
    """Enumeration of supported execution node types."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BRANCH = "branch"
    LOOP = "loop"
    MAP = "map"
    DAG = "dag"
    TRY_CATCH = "try_catch"
    AGENTIC_LOOP = "agentic_loop"


class FlowPausedError(Exception):
    """Raised when a flow execution is paused."""
    pass


class FlowStoppedError(Exception):
    """Raised when a flow execution is stopped."""
    pass


class ExecutionEngine:
    """
    Core execution engine for Water flows.

    Orchestrates the execution of different node types including sequential tasks,
    parallel execution, conditional branching, and loops. Supports pause/stop/resume
    via an optional storage backend.
    """

    # Lock to prevent race conditions when checking and modifying flow state
    # during pause/stop operations.
    _flow_state_lock = asyncio.Lock()

    @staticmethod
    async def run(
        execution_graph: ExecutionGraph,
        input_data: InputData,
        flow_id: str,
        flow_metadata: Dict[str, Any] = None,
        storage: Optional[Any] = None,
        resume_from: Optional[Dict[str, Any]] = None,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        checkpoint: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        services: Optional[Dict[str, Any]] = None,
        max_concurrency: int = 10,
    ) -> OutputData:
        """
        Execute a complete flow execution graph.

        Args:
            execution_graph: List of execution nodes to process
            input_data: Initial input data
            flow_id: Unique identifier for the flow execution
            flow_metadata: Optional metadata for the flow
            storage: Optional storage backend for persistence and pause/resume
            resume_from: Optional dict with resume state (execution_id, node_index, data, context_state)
            dlq: Optional dead letter queue for capturing failed task executions

        Returns:
            Final output data after all nodes are executed

        Raises:
            FlowPausedError: If the flow was paused during execution
            FlowStoppedError: If the flow was stopped during execution
        """
        concurrency_semaphore = asyncio.Semaphore(max_concurrency)

        if resume_from:
            context = ExecutionContext(
                flow_id=flow_id,
                execution_id=resume_from["execution_id"],
                flow_metadata=flow_metadata or {},
                input_data=input_data,
            )
            # Restore context state
            ctx_state = resume_from.get("context_state", {})
            context._task_outputs = ctx_state.get("task_outputs", {})
            context._step_history = ctx_state.get("step_history", [])
            context.step_number = ctx_state.get("step_number", 0)

            data: OutputData = resume_from["data"]
            start_index = resume_from["node_index"]

            # Register injected services on resumed context
            if services:
                for name, service in services.items():
                    context.register_service(name, service)
        else:
            context = ExecutionContext(
                flow_id=flow_id,
                flow_metadata=flow_metadata or {},
                input_data=input_data
            )
            data = input_data
            start_index = 0

            # Register injected services on the context
            if services:
                for name, service in services.items():
                    context.register_service(name, service)

            # Try to recover from a checkpoint (crash recovery)
            if checkpoint is not None:
                saved = await checkpoint.load(flow_id, context.execution_id)
                if saved is not None:
                    start_index = saved["node_index"]
                    data = saved["data"]

        # Save initial session state if storage is provided
        if storage:
            from water.storage import FlowSession, FlowStatus
            session = await storage.get_session(context.execution_id)
            if not session:
                session = FlowSession(
                    flow_id=flow_id,
                    input_data=input_data,
                    execution_id=context.execution_id,
                    status=FlowStatus.RUNNING,
                )
            else:
                session.status = FlowStatus.RUNNING
            await storage.save_session(session)

        try:
            for node_index in range(start_index, len(execution_graph)):
                # Check for pause/stop signals before each node
                if storage:
                    async with ExecutionEngine._flow_state_lock:
                        session = await storage.get_session(context.execution_id)
                        if session and session.status == FlowStatus.PAUSED:
                            # Save current state for resume
                            session.current_node_index = node_index
                            session.current_data = data
                            session.context_state = {
                                "task_outputs": context._task_outputs,
                                "step_history": context._step_history,
                                "step_number": context.step_number,
                                "flow_version": context.flow_metadata.get("_flow_version"),
                            }
                            await storage.save_session(session)
                            raise FlowPausedError(
                                f"Flow {flow_id} paused at node {node_index} "
                                f"(execution: {context.execution_id})"
                            )
                        elif session and session.status == FlowStatus.STOPPED:
                            session.current_node_index = node_index
                            session.current_data = data
                            session.context_state = {
                                "task_outputs": context._task_outputs,
                                "step_history": context._step_history,
                                "step_number": context.step_number,
                                "flow_version": context.flow_metadata.get("_flow_version"),
                            }
                            await storage.save_session(session)
                            raise FlowStoppedError(
                                f"Flow {flow_id} stopped at node {node_index} "
                                f"(execution: {context.execution_id})"
                            )

                node = execution_graph[node_index]
                data = await ExecutionEngine._execute_node(
                    node, data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq,
                    concurrency_semaphore=concurrency_semaphore,
                )

                # Save checkpoint after each successful node
                if checkpoint is not None:
                    await checkpoint.save(
                        flow_id, context.execution_id, node_index + 1, data
                    )

            # Mark as completed
            if storage:
                session = await storage.get_session(context.execution_id)
                if session:
                    session.status = FlowStatus.COMPLETED
                    session.result = data
                    session.current_data = data
                    session.current_node_index = len(execution_graph)
                    await storage.save_session(session)

            # Clear checkpoint on successful completion
            if checkpoint is not None:
                try:
                    await checkpoint.clear(flow_id, context.execution_id)
                except Exception as clear_err:
                    logger.error(
                        f"Failed to clear checkpoint for flow {flow_id} "
                        f"(execution: {context.execution_id}): {clear_err}"
                    )

        except (FlowPausedError, FlowStoppedError):
            raise
        except Exception as e:
            if storage:
                session = await storage.get_session(context.execution_id)
                if session:
                    session.status = FlowStatus.FAILED
                    session.error = str(e)
                    await storage.save_session(session)
            raise

        return data

    @staticmethod
    async def _execute_node(
        node: ExecutionNode,
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        """
        Route execution to the appropriate node type handler.
        """
        try:
            node_type = NodeType(node["type"])
        except ValueError:
            raise ValueError(f"ExecutionEngine: unknown node type {node['type']}")

        handlers = {
            NodeType.SEQUENTIAL: ExecutionEngine._execute_sequential,
            NodeType.PARALLEL: ExecutionEngine._execute_parallel,
            NodeType.BRANCH: ExecutionEngine._execute_branch,
            NodeType.LOOP: ExecutionEngine._execute_loop,
            NodeType.MAP: ExecutionEngine._execute_map,
            NodeType.DAG: ExecutionEngine._execute_dag,
            NodeType.TRY_CATCH: ExecutionEngine._execute_try_catch,
            NodeType.AGENTIC_LOOP: ExecutionEngine._execute_agentic_loop,
        }

        handler = handlers.get(node_type)
        if not handler:
            raise ValueError(f"ExecutionEngine: unhandled node type {node_type}")

        return await handler(node, data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq, concurrency_semaphore=concurrency_semaphore)

    @staticmethod
    async def _execute_task(
        task: Any,
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
    ) -> OutputData:
        """
        Execute a single task, handling both sync and async functions.
        Supports retry with backoff, per-task timeouts, hooks, events, storage recording, and telemetry.
        """
        from water.storage import TaskRun, FlowStatus
        from water.resilience.cache import cache_key as _cache_key

        params: Dict[str, InputData] = {"input_data": data}

        # --- Cache lookup (before any execution work) ---
        task_cache = getattr(task, "cache", None)
        if task_cache is not None:
            ck = _cache_key(task.id, data)
            cached = task_cache.get(ck)
            if cached is not None:
                context.task_id = task.id
                context.step_number += 1
                context.add_task_output(task.id, cached)
                return cached

        # --- Circuit breaker check (after cache, before execution) ---
        cb = getattr(task, "circuit_breaker", None)
        if cb is not None:
            if not cb.can_execute():
                from water.resilience.circuit_breaker import CircuitBreakerOpen
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open for task '{task.id}'"
                )

        # Update context with current task info
        context.task_id = task.id
        context.step_start_time = datetime.now(timezone.utc)
        context.step_number += 1

        retry_count = getattr(task, "retry_count", 0)
        retry_delay = getattr(task, "retry_delay", 0.0)
        retry_backoff = getattr(task, "retry_backoff", 1.0)
        task_timeout = getattr(task, "timeout", None)
        max_attempts = retry_count + 1
        last_error = None

        # Emit task start hook and event
        if hooks:
            await hooks.emit(
                "on_task_start",
                task_id=task.id,
                input_data=data,
                context=context,
            )
        if event_emitter:
            from water.middleware.events import FlowEvent
            await event_emitter.emit(FlowEvent(
                "task_start", context.flow_id,
                task_id=task.id, execution_id=context.execution_id,
                data={"input": data},
            ))

        # Telemetry span tracking
        _telem_span = None
        if telemetry and telemetry.is_active:
            _telem_span_ctx = telemetry.task_span(task.id, context.flow_id)
            _telem_span = _telem_span_ctx.__enter__()

        for attempt in range(1, max_attempts + 1):
            context.attempt_number = attempt

            # Create task run record
            task_run = None
            if storage:
                task_run = TaskRun(
                    execution_id=context.execution_id,
                    task_id=task.id,
                    node_index=node_index,
                    status=FlowStatus.RUNNING,
                    input_data=data,
                    started_at=datetime.now(timezone.utc),
                )
                await storage.save_task_run(task_run)

            try:
                # Rate limiting
                task_rate_limit = getattr(task, "rate_limit", None)
                if task_rate_limit:
                    from water.resilience.rate_limiter import get_rate_limiter
                    await get_rate_limiter().acquire(task.id, task_rate_limit)

                # Validate input against schema if enabled
                if getattr(task, "validate_schema", False) and hasattr(task, "input_schema"):
                    try:
                        task.input_schema(**data)
                    except Exception as ve:
                        raise ValueError(
                            f"ExecutionEngine: task '{task.id}' input validation failed - {ve}"
                        ) from ve

                # --- Middleware before_task ---
                if middleware:
                    for mw in sorted(middleware, key=lambda m: getattr(m, 'order', 0)):
                        data = await mw.before_task(task.id, data, context)
                    params = {"input_data": data}

                # Execute the task (with optional timeout)
                if inspect.iscoroutinefunction(task.execute):
                    coro = task.execute(params, context)
                    if task_timeout:
                        result = await asyncio.wait_for(coro, timeout=task_timeout)
                    else:
                        result = await coro
                else:
                    if task_timeout:
                        loop = asyncio.get_running_loop()
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, task.execute, params, context),
                            timeout=task_timeout,
                        )
                    else:
                        result = task.execute(params, context)

                # --- Middleware after_task ---
                if middleware:
                    for mw in sorted(middleware, key=lambda m: getattr(m, 'order', 0), reverse=True):
                        result = await mw.after_task(task.id, data, result, context)

                # Validate output against schema if enabled
                if getattr(task, "validate_schema", False) and hasattr(task, "output_schema"):
                    try:
                        task.output_schema(**result)
                    except Exception as ve:
                        raise ValueError(
                            f"ExecutionEngine: task '{task.id}' output validation failed - {ve}"
                        ) from ve

                # Store result in cache if enabled
                if task_cache is not None:
                    task_cache.set(ck, result)

                # Store the task result in context for future tasks to access
                context.add_task_output(task.id, result)

                # Update task run record
                if storage and task_run:
                    task_run.status = FlowStatus.COMPLETED
                    task_run.output_data = result
                    task_run.completed_at = datetime.now(timezone.utc)
                    await storage.save_task_run(task_run)

                # Emit task complete hook and event
                if hooks:
                    await hooks.emit(
                        "on_task_complete",
                        task_id=task.id,
                        input_data=data,
                        output_data=result,
                        context=context,
                    )
                if event_emitter:
                    from water.middleware.events import FlowEvent
                    await event_emitter.emit(FlowEvent(
                        "task_complete", context.flow_id,
                        task_id=task.id, execution_id=context.execution_id,
                        data={"output": result},
                    ))

                # Record circuit breaker success
                if cb is not None:
                    cb.record_success()

                if telemetry and _telem_span is not None:
                    telemetry.set_success(_telem_span)
                    _telem_span_ctx.__exit__(None, None, None)

                return result

            except Exception as e:
                last_error = e
                if storage and task_run:
                    task_run.status = FlowStatus.FAILED
                    task_run.error = str(e)
                    task_run.completed_at = datetime.now(timezone.utc)
                    await storage.save_task_run(task_run)

                if attempt < max_attempts:
                    delay = retry_delay * (retry_backoff ** (attempt - 1))
                    if delay > 0:
                        await asyncio.sleep(delay)
                    logger.info(
                        f"Retrying task {task.id} (attempt {attempt + 1}/{max_attempts}) "
                        f"after error: {e}"
                    )
                else:
                    # Record circuit breaker failure
                    if cb is not None:
                        cb.record_failure()
                    # Emit task error hook and event
                    if hooks:
                        await hooks.emit(
                            "on_task_error",
                            task_id=task.id,
                            input_data=data,
                            error=last_error,
                            context=context,
                        )
                    if event_emitter:
                        from water.middleware.events import FlowEvent
                        await event_emitter.emit(FlowEvent(
                            "task_error", context.flow_id,
                            task_id=task.id, execution_id=context.execution_id,
                            data={"error": str(last_error)},
                        ))
                    if telemetry and _telem_span is not None:
                        telemetry.record_error(_telem_span, last_error)
                        _telem_span_ctx.__exit__(type(last_error), last_error, last_error.__traceback__)
                    # Push to dead letter queue if configured
                    if dlq is not None:
                        from water.resilience.dlq import DeadLetter
                        letter = DeadLetter(
                            task_id=task.id,
                            flow_id=context.flow_id,
                            execution_id=context.execution_id,
                            input_data=data,
                            error=str(last_error),
                            error_type=type(last_error).__name__,
                            attempts=max_attempts,
                        )
                        await dlq.push(letter)
                    raise last_error

    @staticmethod
    async def _execute_sequential(
        node: SequentialNode,
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        # Support conditional skip via 'when' key
        when = node.get("when")
        if when is not None and not when(data):
            return data  # Skip task, pass data through

        task = node["task"]
        fallback = node.get("fallback")

        try:
            return await ExecutionEngine._execute_task(task, data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq)
        except Exception:
            if fallback is not None:
                logger.info(
                    f"Primary task '{task.id}' failed, running fallback task '{fallback.id}'"
                )
                return await ExecutionEngine._execute_task(fallback, data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq)
            raise

    @staticmethod
    async def _execute_parallel(
        node: ParallelNode,
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        tasks = node["tasks"]

        async def execute_single_task(task):
            if concurrency_semaphore is not None:
                async with concurrency_semaphore:
                    return await ExecutionEngine._execute_task(task, data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq)
            return await ExecutionEngine._execute_task(task, data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq)

        coroutines = [execute_single_task(task) for task in tasks]
        results: List[OutputData] = await asyncio.gather(*coroutines)

        parallel_results = {task.id: result for task, result in zip(tasks, results)}
        context.add_task_output("_parallel_results", parallel_results)

        return parallel_results

    @staticmethod
    async def _execute_branch(
        node: BranchNode,
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        branches = node["branches"]

        for branch in branches:
            condition = branch["condition"]

            if condition(data):
                task = branch["task"]
                return await ExecutionEngine._execute_task(task, data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq)

        return data

    @staticmethod
    async def _execute_loop(
        node: LoopNode,
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        condition = node["condition"]
        task = node["task"]
        max_iterations: int = node.get("max_iterations", 100)

        if max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be greater than 0, got {max_iterations}"
            )

        iteration_count: int = 0
        current_data: OutputData = data

        while iteration_count < max_iterations:
            if not condition(current_data):
                break

            # Check for pause/stop signals during loops
            if storage:
                from water.storage import FlowStatus
                async with ExecutionEngine._flow_state_lock:
                    session = await storage.get_session(context.execution_id)
                    if session and session.status == FlowStatus.PAUSED:
                        session.current_node_index = node_index
                        session.current_data = current_data
                        session.context_state = {
                            "task_outputs": context._task_outputs,
                            "step_history": context._step_history,
                            "step_number": context.step_number,
                        }
                        await storage.save_session(session)
                        raise FlowPausedError(
                            f"Flow paused during loop at node {node_index}, "
                            f"iteration {iteration_count}"
                        )
                    elif session and session.status == FlowStatus.STOPPED:
                        session.current_node_index = node_index
                        session.current_data = current_data
                        session.context_state = {
                            "task_outputs": context._task_outputs,
                            "step_history": context._step_history,
                            "step_number": context.step_number,
                        }
                        await storage.save_session(session)
                        raise FlowStoppedError(
                            f"Flow stopped during loop at node {node_index}, "
                            f"iteration {iteration_count}"
                        )

            current_data = await ExecutionEngine._execute_task(
                task, current_data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq
            )
            iteration_count += 1

        if iteration_count >= max_iterations:
            logger.warning(
                f"Loop reached maximum iterations ({max_iterations}) "
                f"for flow {context.flow_id}"
            )

        return current_data

    @staticmethod
    async def _execute_map(
        node: Dict[str, Any],
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        """
        Execute a task once per item in a list field, in parallel.

        The node must have 'task' and 'over' keys. 'over' is the key in
        the input data containing the list to iterate over.
        """
        task = node["task"]
        over_key = node["over"]

        items = data.get(over_key, [])
        if not isinstance(items, list):
            raise ValueError(f"ExecutionEngine: map 'over' key '{over_key}' must reference a list, got {type(items).__name__}")

        async def execute_for_item(item):
            item_data = {**data, over_key: item}
            if concurrency_semaphore is not None:
                async with concurrency_semaphore:
                    return await ExecutionEngine._execute_task(
                        task, item_data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq
                    )
            return await ExecutionEngine._execute_task(
                task, item_data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq
            )

        results = await asyncio.gather(*[execute_for_item(item) for item in items])
        return {"results": list(results)}

    @staticmethod
    async def _execute_dag(
        node: Dict[str, Any],
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        """
        Execute tasks as a DAG with automatic parallelization.

        The node must have a 'tasks' key (list of tasks) and a 'dependencies'
        key (dict mapping task_id -> list of dependency task_ids).
        Tasks with no dependencies run first, in parallel. As dependencies
        complete, downstream tasks are unlocked and run.

        Each task receives the original input data merged with all upstream
        task outputs available via context.get_task_output(task_id).

        Returns a dict of {task_id: result} for all tasks in the DAG.
        """
        tasks = {task.id: task for task in node["tasks"]}
        dependencies: Dict[str, List[str]] = node.get("dependencies", {})

        # Validate dependencies reference known tasks
        all_task_ids = set(tasks.keys())
        for task_id, deps in dependencies.items():
            if task_id not in all_task_ids:
                raise ValueError(f"ExecutionEngine: DAG dependency references unknown task {task_id}")
            for dep in deps:
                if dep not in all_task_ids:
                    raise ValueError(f"ExecutionEngine: task '{task_id}' depends on unknown task {dep}")

        # --- Cycle detection via DFS-based topological sort ---
        # Build adjacency list (dependency -> dependents)
        adjacency: Dict[str, List[str]] = {tid: [] for tid in all_task_ids}
        for task_id, deps in dependencies.items():
            for dep in deps:
                adjacency[dep].append(task_id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in all_task_ids}
        cycle_path: List[str] = []

        def _dfs_visit(node_id: str) -> bool:
            """Return True if a cycle is detected."""
            color[node_id] = GRAY
            cycle_path.append(node_id)
            for neighbor in adjacency[node_id]:
                if color[neighbor] == GRAY:
                    # Found a back-edge -- cycle detected
                    cycle_start = cycle_path.index(neighbor)
                    cycle_nodes = cycle_path[cycle_start:]
                    raise ValueError(
                        f"DAG contains a cycle: {' -> '.join(cycle_nodes + [neighbor])}"
                    )
                if color[neighbor] == WHITE:
                    _dfs_visit(neighbor)
            cycle_path.pop()
            color[node_id] = BLACK
            return False

        for tid in all_task_ids:
            if color[tid] == WHITE:
                _dfs_visit(tid)

        # Compute in-degrees for scheduling
        in_degree = {tid: 0 for tid in all_task_ids}
        dependents: Dict[str, List[str]] = {tid: [] for tid in all_task_ids}
        for task_id, deps in dependencies.items():
            in_degree[task_id] = len(deps)
            for dep in deps:
                dependents[dep].append(task_id)

        # Tasks with no dependencies start first
        ready = [tid for tid, deg in in_degree.items() if deg == 0]
        if not ready and all_task_ids:
            raise ValueError("ExecutionEngine: DAG has circular dependencies")

        results: Dict[str, Any] = {}
        completed = set()
        pending_futures: Dict[str, asyncio.Task] = {}

        async def run_task(task_id: str):
            task = tasks[task_id]
            # Merge upstream results into data
            task_data = dict(data)
            for dep_id in dependencies.get(task_id, []):
                if dep_id in results:
                    task_data[f"_{dep_id}_output"] = results[dep_id]
            if concurrency_semaphore is not None:
                async with concurrency_semaphore:
                    return await ExecutionEngine._execute_task(
                        task, task_data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq
                    )
            return await ExecutionEngine._execute_task(
                task, task_data, context, storage, node_index, hooks, event_emitter, telemetry, middleware, dlq
            )

        # Process DAG level by level
        while ready or pending_futures:
            # Launch all ready tasks
            for tid in ready:
                pending_futures[tid] = asyncio.create_task(run_task(tid))
            ready = []

            if not pending_futures:
                break

            # Wait for at least one task to complete
            done, _ = await asyncio.wait(
                pending_futures.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process completed tasks
            for finished in done:
                # Find which task_id this corresponds to
                finished_id = None
                for tid, fut in pending_futures.items():
                    if fut is finished:
                        finished_id = tid
                        break

                result = await finished
                results[finished_id] = result
                completed.add(finished_id)
                del pending_futures[finished_id]

                # Unlock dependents
                for dependent_id in dependents.get(finished_id, []):
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        ready.append(dependent_id)

        if len(completed) != len(all_task_ids):
            raise ValueError("ExecutionEngine: DAG has circular dependencies - not all tasks completed")

        return results

    @staticmethod
    async def _execute_try_catch(
        node: Dict[str, Any],
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        """
        Execute a try-catch-finally block.

        The node must have a 'tasks' key (list of tasks for the try block),
        and optionally 'catch_handler' and 'finally_handler' in the 'config' dict.

        - Executes all try tasks sequentially.
        - If any try task raises, the catch_handler is invoked (if provided)
          with the error info injected into data as '_error' and '_error_type'.
        - The finally_handler always runs regardless of success or failure.
        - Returns the try result on success, or the catch result on failure.
        """
        try_tasks = node.get("tasks") or []
        if not try_tasks:
            single = node.get("task")
            if single is not None:
                try_tasks = [single]

        config = node.get("config", {})
        catch_handler = config.get("catch_handler")
        finally_handler = config.get("finally_handler")
        wrapped_nodes = config.get("_wrapped_nodes")

        result = data
        error_occurred = False
        caught_error = None

        # --- Try block ---
        try:
            current_data = data
            if wrapped_nodes:
                # on_error mode: replay the original execution nodes
                for wrapped_node in wrapped_nodes:
                    current_data = await ExecutionEngine._execute_node(
                        wrapped_node, current_data, context, storage, node_index,
                        hooks, event_emitter, telemetry, middleware, dlq
                    )
            else:
                for task in try_tasks:
                    current_data = await ExecutionEngine._execute_task(
                        task, current_data, context, storage, node_index,
                        hooks, event_emitter, telemetry, middleware, dlq
                    )
            result = current_data

        except Exception as e:
            error_occurred = True
            caught_error = e

            # --- Catch block ---
            if catch_handler is not None:
                error_data = {
                    **data,
                    "_error": str(e),
                    "_error_type": type(e).__name__,
                    "_error_obj": e,
                }
                if callable(catch_handler) and not hasattr(catch_handler, "execute"):
                    # Plain callable: catch_handler(error, context) -> Dict
                    catch_result = catch_handler(e, context)
                    if inspect.isawaitable(catch_result):
                        catch_result = await catch_result
                    result = catch_result if isinstance(catch_result, dict) else error_data
                else:
                    # Task-based catch handler
                    result = await ExecutionEngine._execute_task(
                        catch_handler, error_data, context, storage, node_index,
                        hooks, event_emitter, telemetry, middleware, dlq
                    )
            else:
                # No catch handler — re-raise after finally
                pass

        # --- Finally block ---
        finally:
            if finally_handler is not None:
                finally_data = {**result} if isinstance(result, dict) else dict(data)
                finally_data["_try_success"] = not error_occurred
                if caught_error is not None:
                    finally_data["_error"] = str(caught_error)
                    finally_data["_error_type"] = type(caught_error).__name__

                if callable(finally_handler) and not hasattr(finally_handler, "execute"):
                    finally_result = finally_handler(finally_data, context)
                    if inspect.isawaitable(finally_result):
                        await finally_result
                else:
                    await ExecutionEngine._execute_task(
                        finally_handler, finally_data, context, storage, node_index,
                        hooks, event_emitter, telemetry, middleware, dlq
                    )

        # If error occurred and no catch handler, re-raise
        if error_occurred and catch_handler is None:
            raise caught_error

        return result

    @staticmethod
    async def _execute_agentic_loop(
        node,
        data: InputData,
        context: ExecutionContext,
        storage: Optional[Any] = None,
        node_index: int = 0,
        hooks: Optional[Any] = None,
        event_emitter: Optional[Any] = None,
        telemetry: Optional[Any] = None,
        middleware: Optional[List[Any]] = None,
        dlq: Optional[Any] = None,
        concurrency_semaphore: Optional[asyncio.Semaphore] = None,
    ) -> OutputData:
        """Execute a model-controlled agentic loop (ReAct pattern).

        The LLM decides when to stop by either:
        1. Returning a response with no tool calls
        2. Calling the special __done__ tool
        """
        from water.agents.tools import Toolkit, Tool

        provider = node.get("provider")
        tools = node.get("tools")
        system_prompt = node.get("system_prompt", "")
        max_iterations = node.get("max_iterations", 10)
        config = node.get("config", {})

        if max_iterations <= 0:
            raise ValueError(f"ExecutionEngine: max_iterations must be > 0, got {max_iterations}")

        # Build toolkit
        toolkit = None
        tools_schema = None
        if tools:
            if isinstance(tools, Toolkit):
                toolkit = tools
            elif isinstance(tools, list):
                toolkit = Toolkit(name="agentic_loop_tools", tools=tools)
            tools_schema = toolkit.to_openai_tools()

        # Build initial messages
        prompt_template = config.get("prompt_template", "")
        if prompt_template:
            try:
                user_message = prompt_template.format(**data) if isinstance(data, dict) else prompt_template.format(input=data)
            except (KeyError, IndexError):
                user_message = str(data)
        else:
            user_message = data.get("prompt", str(data)) if isinstance(data, dict) else str(data)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        tool_history = []
        steps = []
        last_response = None
        on_step = config.get("on_step")
        on_tool_call = config.get("on_tool_call")
        stop_condition = config.get("stop_condition")
        observation_formatter = config.get("observation_formatter")

        for iteration in range(max_iterations):
            # Check for pause/stop signals during loops
            if storage:
                from water.storage import FlowStatus
                session = await storage.get_session(context.execution_id)
                if session and session.status == FlowStatus.PAUSED:
                    session.current_node_index = node_index
                    session.current_data = data
                    session.context_state = {
                        "task_outputs": context._task_outputs,
                        "step_history": context._step_history,
                        "step_number": context.step_number,
                    }
                    await storage.save_session(session)
                    raise FlowPausedError(
                        f"Flow paused during agentic loop at node {node_index}, "
                        f"iteration {iteration}"
                    )
                elif session and session.status == FlowStatus.STOPPED:
                    session.current_node_index = node_index
                    session.current_data = data
                    session.context_state = {
                        "task_outputs": context._task_outputs,
                        "step_history": context._step_history,
                        "step_number": context.step_number,
                    }
                    await storage.save_session(session)
                    raise FlowStoppedError(
                        f"Flow stopped during agentic loop at node {node_index}, "
                        f"iteration {iteration}"
                    )

            # Call LLM
            call_kwargs = {"messages": messages}
            if tools_schema:
                call_kwargs["tools"] = tools_schema
            if config.get("temperature") is not None:
                call_kwargs["temperature"] = config["temperature"]
            if config.get("max_tokens") is not None:
                call_kwargs["max_tokens"] = config["max_tokens"]

            response = await provider.complete(**call_kwargs)
            last_response = response

            # THINK: capture the model's reasoning
            thought = response.get("content", "")

            # Check for tool calls
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # LLM is done - no more tool calls
                step = {"think": thought, "act": None, "observe": None}
                steps.append(step)
                if on_step:
                    await on_step(iteration + 1, step) if asyncio.iscoroutinefunction(on_step) else on_step(iteration + 1, step)
                return {
                    "response": thought,
                    "tool_history": tool_history,
                    "steps": steps,
                    "iterations": iteration + 1,
                }

            # ACT: process tool calls
            assistant_message = {"role": "assistant", "content": thought, "tool_calls": tool_calls}
            messages.append(assistant_message)

            step_actions = []
            step_observations = []

            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                tool_call_id = tool_call.get("id", "")

                # Check for __done__ signal
                if tool_name == "__done__":
                    step = {"think": thought, "act": [{"tool": "__done__", "arguments": tool_args}], "observe": None}
                    steps.append(step)
                    if on_step:
                        await on_step(iteration + 1, step) if asyncio.iscoroutinefunction(on_step) else on_step(iteration + 1, step)
                    return {
                        "response": tool_args.get("final_answer", ""),
                        "tool_history": tool_history,
                        "steps": steps,
                        "iterations": iteration + 1,
                        "metadata": tool_args.get("metadata", {}),
                    }

                # on_tool_call hook: can approve/reject/modify
                if on_tool_call:
                    decision = await on_tool_call(tool_name, tool_args) if asyncio.iscoroutinefunction(on_tool_call) else on_tool_call(tool_name, tool_args)
                    if decision is False:
                        tool_result = {"success": False, "error": f"Tool call '{tool_name}' rejected by on_tool_call hook"}
                        step_actions.append({"tool": tool_name, "arguments": tool_args, "rejected": True})
                        step_observations.append({"tool": tool_name, "result": tool_result})
                        tool_history.append({"iteration": iteration + 1, "tool": tool_name, "arguments": tool_args, "result": tool_result})
                        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": tool_result["error"]})
                        continue
                    elif isinstance(decision, dict):
                        tool_args = decision

                # Execute tool
                if toolkit:
                    tool = toolkit.get(tool_name)
                    if tool:
                        if isinstance(tool_args, str):
                            import json
                            tool_args = json.loads(tool_args)
                        tool_run_result = await tool.run(tool_args)
                        if tool_run_result.success:
                            tool_result = {"success": True, "result": tool_run_result.output}
                        else:
                            tool_result = {"success": False, "error": tool_run_result.error}
                    else:
                        tool_result = {"success": False, "error": f"Tool '{tool_name}' not found"}
                else:
                    tool_result = {"success": False, "error": "No tools available"}

                step_actions.append({"tool": tool_name, "arguments": tool_args})
                step_observations.append({"tool": tool_name, "result": tool_result})

                tool_history.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                })

                # OBSERVE: format and add tool result to messages
                raw_content = str(tool_result.get("result", tool_result.get("error", "")))
                if observation_formatter:
                    obs_content = observation_formatter(tool_name, tool_args, tool_result)
                else:
                    obs_content = raw_content

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": obs_content,
                })

            # Record full Think-Act-Observe step
            step = {"think": thought, "act": step_actions, "observe": step_observations}
            steps.append(step)

            if on_step:
                await on_step(iteration + 1, step) if asyncio.iscoroutinefunction(on_step) else on_step(iteration + 1, step)

            # Custom stop condition
            if stop_condition:
                should_stop = await stop_condition(steps, tool_history) if asyncio.iscoroutinefunction(stop_condition) else stop_condition(steps, tool_history)
                if should_stop:
                    logger.info(f"ExecutionEngine: agentic loop stopped by stop_condition at iteration {iteration + 1}")
                    return {
                        "response": thought,
                        "tool_history": tool_history,
                        "steps": steps,
                        "iterations": iteration + 1,
                    }

        # Max iterations reached
        logger.warning(f"ExecutionEngine: agentic loop reached max_iterations ({max_iterations})")
        return {
            "response": last_response.get("content", "") if last_response else "",
            "tool_history": tool_history,
            "steps": steps,
            "iterations": max_iterations,
        }

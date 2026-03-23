"""
Sandboxed Code Execution module for Water.

Provides safe, isolated environments for running user-provided code
with configurable resource limits, timeout enforcement, and multiple
backend options (in-memory, subprocess, Docker).
"""

import ast
import asyncio
import io
import logging
import os
import sys
import tempfile
import time
import traceback

logger = logging.getLogger(__name__)
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from water.core.task import Task

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution environments."""
    timeout: float = 30.0  # seconds
    max_memory_mb: int = 256
    max_output_size: int = 10000  # chars
    allowed_imports: Optional[List[str]] = None  # None = all allowed
    blocked_imports: Optional[List[str]] = None  # Block specific modules
    working_dir: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""
    stdout: str
    stderr: str
    return_value: Any
    exit_code: int
    execution_time: float
    timed_out: bool


class SandboxBackend(ABC):
    """Abstract base class for sandbox execution backends."""

    @abstractmethod
    async def execute(self, code: str, config: SandboxConfig) -> SandboxResult:
        """Execute code in the sandbox and return the result."""
        ...


_SAFE_BUILTINS: Dict[str, Any] = {
    # Type constructors
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "frozenset": frozenset, "bytes": bytes, "bytearray": bytearray,
    # Collection helpers
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "sorted": sorted, "reversed": reversed,
    "min": min, "max": max, "sum": sum, "all": all, "any": any,
    # Type checking
    "isinstance": isinstance, "issubclass": issubclass, "type": type,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
    # Math
    "abs": abs, "round": round, "pow": pow, "divmod": divmod,
    # String
    "repr": repr, "chr": chr, "ord": ord, "format": format, "print": print,
    # Other safe
    "__build_class__": __build_class__,
    "iter": iter, "next": next, "id": id, "hash": hash, "callable": callable,
    "staticmethod": staticmethod, "classmethod": classmethod,
    "property": property, "super": super, "object": object,
    # Exceptions
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "RuntimeError": RuntimeError,
    "StopIteration": StopIteration, "ZeroDivisionError": ZeroDivisionError,
    "NotImplementedError": NotImplementedError, "OverflowError": OverflowError,
    "ArithmeticError": ArithmeticError, "LookupError": LookupError,
}


class InMemorySandbox(SandboxBackend):
    """
    In-memory sandbox that executes code via exec() with captured output.

    Suitable for testing and trusted code. Uses a restricted namespace
    and captures stdout via io.StringIO. Supports a special __result__
    variable for return values.

    WARNING: InMemorySandbox provides basic isolation but is NOT a security
    boundary. For untrusted code, use SubprocessSandbox or DockerSandbox.
    """

    async def execute(self, code: str, config: SandboxConfig) -> SandboxResult:
        loop = asyncio.get_running_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._run_code, code, config),
                timeout=config.timeout,
            )
            return result
        except asyncio.TimeoutError:
            return SandboxResult(
                stdout="",
                stderr="Execution timed out",
                return_value=None,
                exit_code=1,
                execution_time=config.timeout,
                timed_out=True,
            )

    @staticmethod
    def _check_imports(code: str) -> None:
        """Parse code AST and reject any import statements."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("Import statements are not allowed in sandbox")

    def _run_code(self, code: str, config: SandboxConfig) -> SandboxResult:
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        namespace: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "__name__": "__sandbox__",
        }

        start_time = time.monotonic()
        exit_code = 0

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            self._check_imports(code)
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            exec(code, namespace)
        except Exception as e:
            exit_code = 1
            stderr_capture.write(traceback.format_exc())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        execution_time = time.monotonic() - start_time

        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()

        if config.max_output_size:
            stdout_text = stdout_text[:config.max_output_size]
            stderr_text = stderr_text[:config.max_output_size]

        return_value = namespace.get("__result__", None)

        return SandboxResult(
            stdout=stdout_text,
            stderr=stderr_text,
            return_value=return_value,
            exit_code=exit_code,
            execution_time=execution_time,
            timed_out=False,
        )


class SubprocessSandbox(SandboxBackend):
    """
    Subprocess-based sandbox that runs code in a separate Python process.

    Creates a temporary file with the code and executes it via
    asyncio.create_subprocess_exec with timeout enforcement.
    """

    async def execute(self, code: str, config: SandboxConfig) -> SandboxResult:
        tmp_file = None
        try:
            # Write code to a temp file
            tmp_file = tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
                dir=config.working_dir,
            )
            # Wrap code to capture __result__ and print it as a tagged line
            wrapper = (
                "import json, sys\n"
                "__result__ = None\n"
                + code
                + "\n"
                "if __result__ is not None:\n"
                "    print('__SANDBOX_RESULT__:' + json.dumps(__result__), file=sys.stderr)\n"
            )
            tmp_file.write(wrapper)
            tmp_file.flush()
            tmp_file.close()

            env = os.environ.copy()
            if config.env_vars:
                env.update(config.env_vars)

            start_time = time.monotonic()
            timed_out = False

            process = await asyncio.create_subprocess_exec(
                sys.executable, tmp_file.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=config.working_dir,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                timed_out = True
                stdout_bytes = b""
                stderr_bytes = b""

            execution_time = time.monotonic() - start_time

            stdout_text = stdout_bytes.decode("utf-8", errors="replace")
            stderr_text = stderr_bytes.decode("utf-8", errors="replace")

            # Extract __result__ from stderr
            return_value = None
            filtered_stderr_lines = []
            for line in stderr_text.splitlines():
                if line.startswith("__SANDBOX_RESULT__:"):
                    import json
                    try:
                        return_value = json.loads(line[len("__SANDBOX_RESULT__:"):])
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode subprocess sandbox result JSON: %s", line)
                        pass
                else:
                    filtered_stderr_lines.append(line)
            stderr_text = "\n".join(filtered_stderr_lines)

            if config.max_output_size:
                stdout_text = stdout_text[:config.max_output_size]
                stderr_text = stderr_text[:config.max_output_size]

            exit_code = process.returncode if not timed_out else 1

            if timed_out:
                stderr_text = "Execution timed out"

            return SandboxResult(
                stdout=stdout_text,
                stderr=stderr_text,
                return_value=return_value,
                exit_code=exit_code,
                execution_time=execution_time,
                timed_out=timed_out,
            )

        finally:
            if tmp_file and os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)


class DockerSandbox(SandboxBackend):
    """
    Docker-based sandbox that runs code in an ephemeral container.

    Uses the docker Python SDK (lazy imported). Provides strong isolation
    with configurable resource limits.
    """

    def __init__(self, image: str = "python:3.11-slim"):
        self.image = image
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import docker
            except ImportError:
                raise ImportError(
                    "Docker SDK not installed. Install it with: pip install docker"
                )
            self._client = docker.from_env()
        return self._client

    def close(self) -> None:
        """Close the Docker client and release resources."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.warning("Failed to close Docker client", exc_info=True)
            finally:
                self._client = None

    async def execute(self, code: str, config: SandboxConfig) -> SandboxResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._run_in_container, code, config
        )

    def _run_in_container(self, code: str, config: SandboxConfig) -> SandboxResult:
        client = self._get_client()

        # Write code to a temp file on the host
        tmp_dir = tempfile.mkdtemp()
        code_path = os.path.join(tmp_dir, "code.py")

        wrapper = (
            "import json, sys\n"
            "__result__ = None\n"
            + code
            + "\n"
            "if __result__ is not None:\n"
            "    print('__SANDBOX_RESULT__:' + json.dumps(__result__), file=sys.stderr)\n"
        )

        with open(code_path, "w") as f:
            f.write(wrapper)

        mem_limit = f"{config.max_memory_mb}m"
        environment = config.env_vars or {}

        start_time = time.monotonic()
        timed_out = False

        container = None
        try:
            container = client.containers.run(
                self.image,
                command=["python", "/sandbox/code.py"],
                volumes={tmp_dir: {"bind": "/sandbox", "mode": "ro"}},
                mem_limit=mem_limit,
                network_disabled=True,
                detach=True,
                environment=environment,
            )

            try:
                result = container.wait(timeout=config.timeout)
                exit_code = result.get("StatusCode", 1)
            except Exception:
                logger.warning("Docker container wait failed or timed out, killing container", exc_info=True)
                container.kill()
                timed_out = True
                exit_code = 1

            stdout_text = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr_text = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )

        except Exception as e:
            logger.exception("Docker sandbox execution failed")
            stdout_text = ""
            stderr_text = str(e)
            exit_code = 1
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    logger.warning(
                        "Failed to remove Docker container %s",
                        getattr(container, "id", "unknown"),
                        exc_info=True,
                    )
            # Clean up temp dir
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        execution_time = time.monotonic() - start_time

        # Extract __result__
        return_value = None
        filtered_stderr_lines = []
        for line in stderr_text.splitlines():
            if line.startswith("__SANDBOX_RESULT__:"):
                import json
                try:
                    return_value = json.loads(line[len("__SANDBOX_RESULT__:"):])
                except json.JSONDecodeError:
                    logger.warning("Failed to decode Docker sandbox result JSON: %s", line)
                    pass
            else:
                filtered_stderr_lines.append(line)
        stderr_text = "\n".join(filtered_stderr_lines)

        if config.max_output_size:
            stdout_text = stdout_text[:config.max_output_size]
            stderr_text = stderr_text[:config.max_output_size]

        if timed_out:
            stderr_text = "Execution timed out"

        return SandboxResult(
            stdout=stdout_text,
            stderr=stderr_text,
            return_value=return_value,
            exit_code=exit_code,
            execution_time=execution_time,
            timed_out=timed_out,
        )


# Default schemas for sandboxed tasks

class SandboxInput(BaseModel):
    """Default input schema for sandboxed tasks."""
    code: str


class SandboxOutput(BaseModel):
    """Default output schema for sandboxed tasks."""
    stdout: str
    stderr: str
    return_value: Any = None
    exit_code: int
    execution_time: float


def create_sandboxed_task(
    id: Optional[str] = None,
    description: Optional[str] = None,
    sandbox: Optional[SandboxBackend] = None,
    config: Optional[SandboxConfig] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    code_key: str = "code",
) -> Task:
    """
    Factory function to create a Task that runs code in a sandbox.

    Args:
        id: Unique identifier for the task. Auto-generated if not provided.
        description: Human-readable description.
        sandbox: Sandbox backend to use. Defaults to SubprocessSandbox.
        config: Sandbox configuration. Defaults to SandboxConfig().
        input_schema: Pydantic model for input. Defaults to SandboxInput.
        output_schema: Pydantic model for output. Defaults to SandboxOutput.
        code_key: Key in input_data containing the code to execute.

    Returns:
        A Task instance that executes code in the given sandbox.
    """
    if sandbox is None:
        sandbox = SubprocessSandbox()
    if config is None:
        config = SandboxConfig()
    if input_schema is None:
        input_schema = SandboxInput
    if output_schema is None:
        output_schema = SandboxOutput

    async def execute(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        input_data = params.get("input_data", params)
        code = input_data[code_key]

        result = await sandbox.execute(code, config)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_value": result.return_value,
            "exit_code": result.exit_code,
            "execution_time": result.execution_time,
        }

    return Task(
        id=id or f"sandbox_{uuid.uuid4().hex[:8]}",
        description=description or "Sandboxed code execution task",
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
    )

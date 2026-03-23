"""Tests for the sandboxed code execution module."""

import pytest
import asyncio
from pydantic import BaseModel

from water.agents.sandbox import (
    InMemorySandbox,
    SubprocessSandbox,
    SandboxConfig,
    SandboxResult,
    create_sandboxed_task,
)
from water.core import Flow


# --- InMemorySandbox Tests ---


@pytest.mark.asyncio
async def test_in_memory_sandbox_basic():
    """InMemorySandbox can run simple code without errors."""
    sandbox = InMemorySandbox()
    config = SandboxConfig()
    result = await sandbox.execute("x = 1 + 2", config)
    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_in_memory_sandbox_captures_stdout():
    """InMemorySandbox captures print output to stdout."""
    sandbox = InMemorySandbox()
    config = SandboxConfig()
    result = await sandbox.execute("print('hello world')", config)
    assert result.exit_code == 0
    assert "hello world" in result.stdout


@pytest.mark.asyncio
async def test_in_memory_sandbox_return_value():
    """InMemorySandbox captures return value via __result__ variable."""
    sandbox = InMemorySandbox()
    config = SandboxConfig()
    code = "__result__ = {'answer': 42}"
    result = await sandbox.execute(code, config)
    assert result.exit_code == 0
    assert result.return_value == {"answer": 42}


@pytest.mark.asyncio
async def test_in_memory_sandbox_timeout():
    """InMemorySandbox times out on long-running code.

    Uses sorted() on a large list repeatedly, which releases the GIL
    periodically and allows the asyncio timeout to fire cleanly.
    """
    sandbox = InMemorySandbox()
    config = SandboxConfig(timeout=0.5)
    code = """
data = list(range(500000))
for _ in range(1000):
    sorted(data, reverse=True)
"""
    result = await sandbox.execute(code, config)
    assert result.timed_out is True
    assert result.exit_code == 1
    assert "timed out" in result.stderr.lower()


@pytest.mark.asyncio
async def test_in_memory_sandbox_error():
    """InMemorySandbox captures exceptions in stderr."""
    sandbox = InMemorySandbox()
    config = SandboxConfig()
    code = "raise ValueError('test error')"
    result = await sandbox.execute(code, config)
    assert result.exit_code == 1
    assert "ValueError" in result.stderr
    assert "test error" in result.stderr


# --- SubprocessSandbox Tests ---


@pytest.mark.asyncio
async def test_subprocess_sandbox_basic():
    """SubprocessSandbox runs code in a separate process."""
    sandbox = SubprocessSandbox()
    config = SandboxConfig()
    code = "print('subprocess hello')"
    result = await sandbox.execute(code, config)
    assert result.exit_code == 0
    assert "subprocess hello" in result.stdout
    assert result.timed_out is False


@pytest.mark.asyncio
async def test_subprocess_sandbox_timeout():
    """SubprocessSandbox times out on long-running code."""
    sandbox = SubprocessSandbox()
    config = SandboxConfig(timeout=1.0)
    code = """
import time
time.sleep(30)
"""
    result = await sandbox.execute(code, config)
    assert result.timed_out is True
    assert result.exit_code == 1


# --- Sandboxed Task Tests ---


@pytest.mark.asyncio
async def test_sandboxed_task_in_flow():
    """Sandboxed task works correctly within a Water flow."""
    task = create_sandboxed_task(
        id="sandbox_task",
        description="Run code in sandbox",
        sandbox=InMemorySandbox(),
        config=SandboxConfig(timeout=5.0),
    )

    flow = Flow(id="sandbox_flow", description="Flow with sandboxed task")
    flow.then(task).register()

    result = await flow.run({"code": "print('from flow')"})
    assert result["stdout"].strip() == "from flow"
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_sandbox_config_defaults():
    """SandboxConfig has sensible default values."""
    config = SandboxConfig()
    assert config.timeout == 30.0
    assert config.max_memory_mb == 256
    assert config.max_output_size == 10000
    assert config.allowed_imports is None
    assert config.blocked_imports is None
    assert config.working_dir is None
    assert config.env_vars is None


@pytest.mark.asyncio
async def test_sandboxed_task_captures_output():
    """Sandboxed task returns all SandboxResult fields."""
    task = create_sandboxed_task(
        id="output_task",
        description="Capture all output",
        sandbox=InMemorySandbox(),
        config=SandboxConfig(timeout=5.0),
    )

    flow = Flow(id="output_flow", description="Flow capturing output")
    flow.then(task).register()

    code = """
print('hello stdout')
__result__ = 99
"""
    result = await flow.run({"code": code})
    assert "hello stdout" in result["stdout"]
    assert result["return_value"] == 99
    assert result["exit_code"] == 0
    assert isinstance(result["execution_time"], float)
    assert result["execution_time"] >= 0

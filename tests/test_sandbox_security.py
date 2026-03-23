"""Tests for InMemorySandbox security restrictions."""

import asyncio

import pytest

from water.agents.sandbox import InMemorySandbox, SandboxConfig


@pytest.fixture
def sandbox():
    return InMemorySandbox()


@pytest.fixture
def config():
    return SandboxConfig()


def _run(sandbox, code, config):
    return asyncio.get_event_loop().run_until_complete(sandbox.execute(code, config))


class TestImportBlocking:
    """AST-level import blocking."""

    def test_dunder_import_blocked(self, sandbox, config):
        result = _run(sandbox, "__import__('os')", config)
        assert result.exit_code == 1
        assert "Import statements are not allowed" in result.stderr or "NameError" in result.stderr

    def test_import_statement_blocked(self, sandbox, config):
        result = _run(sandbox, "import os", config)
        assert result.exit_code == 1
        assert "Import statements are not allowed" in result.stderr

    def test_from_import_blocked(self, sandbox, config):
        result = _run(sandbox, "from os import system", config)
        assert result.exit_code == 1
        assert "Import statements are not allowed" in result.stderr


class TestBuiltinRestrictions:
    """Dangerous builtins are excluded."""

    def test_open_not_available(self, sandbox, config):
        result = _run(sandbox, "open('/etc/passwd')", config)
        assert result.exit_code == 1
        assert "NameError" in result.stderr

    def test_eval_not_available(self, sandbox, config):
        result = _run(sandbox, "eval('1+1')", config)
        assert result.exit_code == 1
        assert "NameError" in result.stderr

    def test_exec_not_available(self, sandbox, config):
        result = _run(sandbox, "exec('x=1')", config)
        assert result.exit_code == 1
        assert "NameError" in result.stderr


class TestSafeCodeWorks:
    """Basic safe operations still function correctly."""

    def test_arithmetic(self, sandbox, config):
        result = _run(sandbox, "__result__ = 2 + 3 * 4", config)
        assert result.exit_code == 0
        assert result.return_value == 14

    def test_string_operations(self, sandbox, config):
        result = _run(sandbox, "__result__ = 'hello'.upper() + ' ' + 'world'", config)
        assert result.exit_code == 0
        assert result.return_value == "HELLO world"

    def test_list_comprehension(self, sandbox, config):
        result = _run(sandbox, "__result__ = [x**2 for x in range(5)]", config)
        assert result.exit_code == 0
        assert result.return_value == [0, 1, 4, 9, 16]

    def test_print_works(self, sandbox, config):
        result = _run(sandbox, "print('hello sandbox')", config)
        assert result.exit_code == 0
        assert "hello sandbox" in result.stdout

    def test_define_function(self, sandbox, config):
        code = "def add(a, b):\n    return a + b\n__result__ = add(3, 4)"
        result = _run(sandbox, code, config)
        assert result.exit_code == 0
        assert result.return_value == 7

    def test_define_class(self, sandbox, config):
        code = (
            "class Counter:\n"
            "    def __init__(self):\n"
            "        self.n = 0\n"
            "    def inc(self):\n"
            "        self.n += 1\n"
            "        return self.n\n"
            "c = Counter()\n"
            "c.inc()\n"
            "__result__ = c.inc()\n"
        )
        result = _run(sandbox, code, config)
        assert result.exit_code == 0
        assert result.return_value == 2

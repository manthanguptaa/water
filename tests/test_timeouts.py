"""Tests for configurable timeouts on LLM providers and storage backends."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from water.agents.llm import AnthropicProvider, OpenAIProvider
from water.storage.base import SQLiteStorage
from water.storage.postgres import PostgresStorage
from water.storage.redis import RedisStorage


# ---------------------------------------------------------------------------
# LLM Provider timeout parameters
# ---------------------------------------------------------------------------


class TestOpenAIProviderTimeout:
    def test_default_timeout(self):
        provider = OpenAIProvider(api_key="fake-key")
        assert provider.timeout == 60.0

    def test_custom_timeout(self):
        provider = OpenAIProvider(api_key="fake-key", timeout=10.0)
        assert provider.timeout == 10.0

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        fake_openai = MagicMock()

        async def slow_create(**kwargs):
            await asyncio.sleep(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create = slow_create
        fake_openai.AsyncOpenAI.return_value = mock_client

        provider = OpenAIProvider(api_key="fake-key", timeout=0.01)

        with patch.dict("sys.modules", {"openai": fake_openai}):
            with pytest.raises(TimeoutError, match="OpenAI API call timed out"):
                await provider.complete([{"role": "user", "content": "hi"}])


class TestAnthropicProviderTimeout:
    def test_default_timeout(self):
        provider = AnthropicProvider(api_key="fake-key")
        assert provider.timeout == 60.0

    def test_custom_timeout(self):
        provider = AnthropicProvider(api_key="fake-key", timeout=15.0)
        assert provider.timeout == 15.0

    @pytest.mark.asyncio
    async def test_timeout_raises(self):
        # Create a fake anthropic module so the lazy import succeeds
        fake_anthropic = MagicMock()

        async def slow_create(**kwargs):
            await asyncio.sleep(5)

        mock_client = MagicMock()
        mock_client.messages.create = slow_create
        fake_anthropic.AsyncAnthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="fake-key", timeout=0.01)

        with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
            with pytest.raises(TimeoutError, match="Anthropic API call timed out"):
                await provider.complete([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# Storage backend timeout parameters
# ---------------------------------------------------------------------------


class TestSQLiteStorageTimeout:
    def test_default_timeout(self, tmp_path):
        db = str(tmp_path / "test.db")
        storage = SQLiteStorage(db_path=db)
        assert storage.timeout == 30.0

    def test_custom_timeout(self, tmp_path):
        db = str(tmp_path / "test.db")
        storage = SQLiteStorage(db_path=db, timeout=10.0)
        assert storage.timeout == 10.0

    def test_timeout_passed_to_connect(self, tmp_path):
        db = str(tmp_path / "test.db")
        with patch("sqlite3.connect", wraps=sqlite3.connect) as mock_connect:
            SQLiteStorage(db_path=db, timeout=15.0)
            # _init_db calls connect once, check timeout was passed
            mock_connect.assert_called_with(db, timeout=15.0)


class TestPostgresStorageTimeout:
    def test_default_timeout(self):
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            storage = PostgresStorage(dsn="postgresql://localhost/test")
            assert storage.command_timeout == 30.0

    def test_custom_timeout(self):
        with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
            storage = PostgresStorage(dsn="postgresql://localhost/test", command_timeout=5.0)
            assert storage.command_timeout == 5.0


class TestRedisStorageTimeout:
    def test_default_timeout(self):
        with patch.dict("sys.modules", {"redis": MagicMock(), "redis.asyncio": MagicMock()}):
            storage = RedisStorage()
            assert storage.socket_timeout == 30.0

    def test_custom_timeout(self):
        with patch.dict("sys.modules", {"redis": MagicMock(), "redis.asyncio": MagicMock()}):
            storage = RedisStorage(socket_timeout=5.0)
            assert storage.socket_timeout == 5.0

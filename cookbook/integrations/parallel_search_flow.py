"""Run anonymous Parallel search and fetch as Water tasks (Python 3.10+).

From the repository root, in a virtual environment:
    pip install . 'mcp>=1.30,<2'
    python -m cookbook.integrations.parallel_search_flow

No Parallel API key or model provider is used. Anonymous access has lower rate
limits and is intended for exploration and light use. See:
https://docs.parallel.ai/integrations/mcp/search-mcp

The SDK handles Streamable HTTP initialization and JSON/SSE responses. The
session adapter below reuses Water's MCPClient.as_task without changing the
built-in client's transports. Keep all flow runs inside the session context.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from water import Flow
from water.integrations import MCPClient


SERVER_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "water-ai (Parallel Search MCP cookbook)"


class SearchInput(BaseModel):
    objective: str
    search_queries: List[str]
    session_id: str


class FetchInput(BaseModel):
    urls: List[str]
    session_id: str


class MCPOutput(BaseModel):
    # Preserve every content block and the optional structured result, rather
    # than assuming the first text block contains a JSON object.
    content: List[Dict[str, Any]]
    structuredContent: Optional[Dict[str, Any]] = None
    isError: bool = False


class SessionClient(MCPClient):
    """Adapt a live SDK session to Water's existing MCP task factory."""

    def __init__(self, session: ClientSession) -> None:
        super().__init__()
        self.session = session

    async def list_tools(self) -> List[dict]:
        result = await self.session.list_tools()
        return [tool.model_dump(mode="json", exclude_none=True) for tool in result.tools]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        result = await self.session.call_tool(name, arguments)
        if result.isError:
            raise RuntimeError(f"{name} failed: {result.model_dump_json()}")
        return result.model_dump(mode="json", exclude_none=True)


@asynccontextmanager
async def parallel_client():
    # Explicit client headers apply to discovery, search, fetch and cleanup.
    # No auth provider, saved credentials or environment API keys are loaded.
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=60.0,
    ) as http:
        async with streamable_http_client(SERVER_URL, http_client=http) as (read, write, _):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=60),
            ) as session:
                await session.initialize()
                yield SessionClient(session)


async def main():
    # Reuse one identifier across related search/fetch calls. There is no LLM
    # in this example, so it intentionally does not send a model_name.
    session_id = uuid4().hex
    async with parallel_client() as client:
        tools = await client.list_tools()
        print("Available tools:", ", ".join(tool["name"] for tool in tools))

        search = client.as_task("web_search", SearchInput, MCPOutput)
        search_flow = Flow(id="parallel_search", description="Search Python documentation")
        search_flow.then(search).register()
        result = await search_flow.run({
            "objective": "Find Python's official documentation for asyncio TaskGroup.",
            "search_queries": ["Python asyncio TaskGroup documentation"],
            "session_id": session_id,
        })
        print("Search:", json.dumps(result, indent=2))

        fetch = client.as_task("web_fetch", FetchInput, MCPOutput)
        fetch_flow = Flow(id="parallel_fetch", description="Read Python documentation")
        fetch_flow.then(fetch).register()
        result = await fetch_flow.run({
            "urls": ["https://docs.python.org/3/library/asyncio-task.html"],
            "session_id": session_id,
        })
        print("Fetch:", json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

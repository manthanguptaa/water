"""
MCP (Model Context Protocol) integration for Water.

Provides MCPServer to expose Water flows as MCP-compatible tools,
and MCPClient to consume external MCP servers as Water tasks.
"""

import asyncio
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from water.core.flow import Flow
from water.core.task import Task, create_task


class MCPServer:
    """
    Exposes registered Water flows as MCP-compatible tools.

    Each flow becomes a tool whose name is the flow's id, description is
    the flow's description, and inputSchema is derived from the first
    task's input_schema via Pydantic's model_json_schema().

    Handles JSON-RPC 2.0 requests for the MCP protocol methods:
    initialize, ping, tools/list, tools/call.
    """

    def __init__(
        self,
        flows: List[Flow],
        name: str = "water",
        version: str = "1.0.0",
    ) -> None:
        self.flows = flows
        self.name = name
        self.version = version
        self._flow_map: Dict[str, Flow] = {f.id: f for f in flows}

    def get_tool_definitions(self) -> List[dict]:
        """Return MCP tool definitions for all registered flows."""
        tools = []
        for flow in self.flows:
            input_schema: dict = {"type": "object", "properties": {}}
            # Derive input schema from the first task in the flow
            if flow._tasks:
                first_node = flow._tasks[0]
                task = first_node.get("task")
                if task is not None and hasattr(task, "input_schema") and task.input_schema is not None:
                    try:
                        input_schema = task.input_schema.model_json_schema()
                    except Exception:
                        pass

            tools.append({
                "name": flow.id,
                "description": flow.description,
                "inputSchema": input_schema,
            })
        return tools

    def handle_request(self, request: dict) -> dict:
        """
        Process an incoming MCP JSON-RPC 2.0 request synchronously.

        For tools/call, the flow is executed using asyncio.
        """
        jsonrpc = request.get("jsonrpc", "2.0")
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version,
                    },
                },
            }

        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.get_tool_definitions(),
                },
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name not in self._flow_map:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}",
                    },
                }

            flow = self._flow_map[tool_name]
            try:
                # Run the flow; handle both running inside and outside an event loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = loop.run_in_executor(
                            pool, lambda: asyncio.run(flow.run(arguments))
                        )
                        # We can't easily await here in a sync method.
                        # Fall back to creating a new loop in a thread.
                        result = asyncio.run(flow.run(arguments))
                else:
                    result = asyncio.run(flow.run(arguments))

                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(result)},
                        ],
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }

        # Unknown method
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    async def handle_request_async(self, request: dict) -> dict:
        """
        Process an incoming MCP JSON-RPC 2.0 request asynchronously.

        Preferred over handle_request when running inside an async context.
        """
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version,
                    },
                },
            }

        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.get_tool_definitions(),
                },
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            if tool_name not in self._flow_map:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}",
                    },
                }

            flow = self._flow_map[tool_name]
            try:
                result = await flow.run(arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(result)},
                        ],
                    },
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32000,
                        "message": str(e),
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}",
            },
        }

    async def serve_stdio(self) -> None:
        """
        Serve MCP over stdin/stdout (standard MCP transport).

        Reads newline-delimited JSON-RPC messages from stdin
        and writes responses to stdout.
        """
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_running_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )

        w_transport, w_protocol = await asyncio.get_running_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(w_transport, w_protocol, reader, asyncio.get_running_loop())

        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                request = json.loads(line.decode())
                response = await self.handle_request_async(request)
                writer.write((json.dumps(response) + "\n").encode())
                await writer.drain()
            except json.JSONDecodeError:
                error_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
                writer.write((json.dumps(error_resp) + "\n").encode())
                await writer.drain()

    async def serve_sse(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """
        Serve MCP over HTTP Server-Sent Events.

        Starts an HTTP server that accepts POST requests with JSON-RPC
        payloads and streams responses as SSE events.
        """
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for SSE transport. "
                "Install it with: pip install aiohttp"
            )

        async def handle_post(request: web.Request) -> web.StreamResponse:
            body = await request.json()
            response = await self.handle_request_async(body)

            sse_response = web.StreamResponse(
                status=200,
                reason="OK",
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
            await sse_response.prepare(request)
            event_data = f"data: {json.dumps(response)}\n\n"
            await sse_response.write(event_data.encode())
            await sse_response.write_eof()
            return sse_response

        app = web.Application()
        app.router.add_post("/mcp", handle_post)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()


class MCPClient:
    """
    Client for connecting to external MCP servers.

    Makes remote MCP tools available as Water tasks. Supports both
    URL-based and stdio subprocess transports. For testing, a
    mock transport can be injected via _tool_handlers.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        command: Optional[List[str]] = None,
    ) -> None:
        self.server_url = server_url
        self.command = command
        # Internal store for mock / injected tool handlers.
        # Maps tool name -> callable(arguments) -> result dict
        self._tool_handlers: Dict[str, Callable] = {}
        # Cached tool definitions for mock mode
        self._tool_definitions: List[dict] = []

    def register_mock_tool(
        self,
        name: str,
        handler: Callable[[dict], dict],
        description: str = "",
        input_schema: Optional[dict] = None,
    ) -> None:
        """
        Register a mock tool handler for testing without a real server.

        Args:
            name: Tool name.
            handler: Callable that takes arguments dict and returns result dict.
            description: Tool description.
            input_schema: JSON Schema for the tool's input.
        """
        self._tool_handlers[name] = handler
        self._tool_definitions.append({
            "name": name,
            "description": description,
            "inputSchema": input_schema or {"type": "object", "properties": {}},
        })

    async def list_tools(self) -> List[dict]:
        """Discover available tools on the connected MCP server."""
        if self._tool_handlers:
            return list(self._tool_definitions)

        if self.server_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {},
                    }
                    async with session.post(
                        self.server_url, json=payload
                    ) as resp:
                        data = await resp.json()
                        return data.get("result", {}).get("tools", [])
            except ImportError:
                raise ImportError(
                    "aiohttp is required for URL-based MCP client. "
                    "Install it with: pip install aiohttp"
                )

        return []

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Call a tool on the connected MCP server.

        Args:
            name: The tool name.
            arguments: Arguments to pass to the tool.

        Returns:
            The tool's result as a dict.
        """
        if name in self._tool_handlers:
            handler = self._tool_handlers[name]
            result = handler(arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        if self.server_url:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments},
                    }
                    async with session.post(
                        self.server_url, json=payload
                    ) as resp:
                        data = await resp.json()
                        if "error" in data:
                            raise RuntimeError(data["error"]["message"])
                        content = data.get("result", {}).get("content", [])
                        if content and content[0].get("type") == "text":
                            return json.loads(content[0]["text"])
                        return data.get("result", {})
            except ImportError:
                raise ImportError(
                    "aiohttp is required for URL-based MCP client. "
                    "Install it with: pip install aiohttp"
                )

        raise RuntimeError(f"No handler available for tool: {name}")

    def as_task(
        self,
        tool_name: str,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
    ) -> Task:
        """
        Convert an MCP tool into a Water Task.

        Args:
            tool_name: Name of the MCP tool to wrap.
            input_schema: Pydantic model for task input.
            output_schema: Pydantic model for task output.

        Returns:
            A Task instance that calls the MCP tool when executed.
        """
        return create_mcp_task(
            tool_name=tool_name,
            mcp_client=self,
            input_schema=input_schema,
            output_schema=output_schema,
        )


def create_mcp_task(
    tool_name: str,
    mcp_client: MCPClient,
    input_schema: Type[BaseModel],
    output_schema: Type[BaseModel],
) -> Task:
    """
    Factory function to create a Water Task backed by an MCP tool call.

    Args:
        tool_name: Name of the MCP tool.
        mcp_client: MCPClient instance connected to the server.
        input_schema: Pydantic model for task input.
        output_schema: Pydantic model for task output.

    Returns:
        A Task that forwards execution to the MCP tool.
    """
    client = mcp_client

    async def execute(params: dict, context: Any) -> dict:
        input_data = params.get("input_data", {})
        return await client.call_tool(tool_name, input_data)

    return create_task(
        id=f"mcp_{tool_name}",
        description=f"MCP tool: {tool_name}",
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
    )

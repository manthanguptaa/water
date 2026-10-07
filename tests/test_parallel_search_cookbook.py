"""Optional SDK transport coverage; core Water installs need no MCP SDK."""
import json

import pytest

pytest.importorskip("mcp", minversion="1.30")
import httpx

from cookbook.integrations import parallel_search_flow as example


@pytest.mark.asyncio
async def test_streamable_http_flows(monkeypatch, capsys):
    requests = []

    async def respond(request):
        payload = json.loads(request.content)
        requests.append((request, payload))
        method = payload["method"]
        if "id" not in payload:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26", "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [
                {"name": name, "inputSchema": {"type": "object"}}
                for name in ("web_search", "web_fetch")
            ]}
        else:
            assert method == "tools/call"
            result = {
                "content": [
                    {"type": "text", "text": "An excerpt, not JSON."},
                    {"type": "text", "text": "Another source."},
                ],
                "structuredContent": {"results": [{"url": "https://docs.python.org/"}]},
                "isError": False,
            }
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(example.httpx, "AsyncClient", lambda **kwargs: real_client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))
    await example.main()
    output = capsys.readouterr().out
    assert output.count("An excerpt, not JSON.") == 2
    assert output.count("Another source.") == 2
    assert output.count('"structuredContent"') == 2
    calls = [p["params"] for _, p in requests if p["method"] == "tools/call"]
    assert [c["name"] for c in calls] == ["web_search", "web_fetch"]
    assert calls[0]["arguments"]["session_id"] == calls[1]["arguments"]["session_id"]
    assert len(calls[0]["arguments"]["session_id"]) == 32
    for request, _ in requests:
        assert str(request.url) == example.SERVER_URL
        assert request.headers["user-agent"] == example.USER_AGENT
        assert "authorization" not in request.headers


@pytest.mark.asyncio
async def test_tool_error_is_not_success():
    from mcp.types import CallToolResult, TextContent
    from water import Flow

    class FailedSession:
        async def call_tool(self, name, arguments):
            return CallToolResult(
                content=[TextContent(type="text", text="Search unavailable")], isError=True,
            )

    client = example.SessionClient(FailedSession())
    flow = Flow(id="failed_parallel_search", description="Error propagation")
    flow.then(client.as_task("web_search", example.SearchInput, example.MCPOutput)).register()
    with pytest.raises(Exception, match="Search unavailable"):
        await flow.run({"objective": "Test", "search_queries": ["test"], "session_id": "x" * 32})

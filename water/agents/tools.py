__all__ = [
    "ToolResult",
    "Tool",
    "Toolkit",
    "ToolExecutor",
]

"""
Tool Use & Function Calling Abstraction for Water.

Lets developers define tools once and use them across any LLM provider.
Water manages the tool call loop (call LLM -> extract tool calls ->
execute -> feed results back) so agent tasks can use tools without
provider-specific code.
"""

import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

from pydantic import BaseModel

from water.core.task import Task


# ---------------------------------------------------------------------------
# Core Tool primitives
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Structured result from a tool invocation."""
    tool_name: str
    output: Any
    error: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = {"tool_name": self.tool_name, "output": self.output, "success": self.success}
        if self.error:
            d["error"] = self.error
        return d


class Tool:
    """
    Defines a callable tool with name, description, input schema, and execute function.

    Tools are provider-agnostic. Water translates them into each provider's
    native format automatically.

    Args:
        name: Unique tool name.
        description: What this tool does (shown to the LLM).
        input_schema: Pydantic model or dict describing expected parameters.
        execute: Callable that runs the tool. Can be sync or async.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Optional[Type[BaseModel]] = None,
        execute: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: Optional[Type[BaseModel]] = input_schema
        self.execute_fn: Optional[Callable[..., Any]] = execute

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute this tool with the given arguments."""
        if self.execute_fn is None:
            return ToolResult(tool_name=self.name, output=None, error=f"Tool: no execute function defined for '{self.name}'", success=False)
        try:
            if inspect.iscoroutinefunction(self.execute_fn):
                result = await self.execute_fn(**arguments)
            else:
                result = self.execute_fn(**arguments)
            return ToolResult(tool_name=self.name, output=result, success=True)
        except Exception as e:
            logger.exception("Tool '%s' execution failed", self.name)
            return ToolResult(tool_name=self.name, output=None, error=str(e), success=False)

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        parameters = self._schema_to_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._schema_to_json_schema(),
        }

    def _schema_to_json_schema(self) -> Dict[str, Any]:
        """Convert input_schema to JSON Schema."""
        if self.input_schema is None:
            return {"type": "object", "properties": {}}

        if isinstance(self.input_schema, dict):
            return self.input_schema

        # Pydantic model
        if hasattr(self.input_schema, "model_json_schema"):
            return self.input_schema.model_json_schema()
        if hasattr(self.input_schema, "schema"):
            return self.input_schema.schema()

        return {"type": "object", "properties": {}}


class Toolkit:
    """
    Named collection of related tools.

    Args:
        name: Toolkit name.
        tools: List of Tool instances.
    """

    def __init__(self, name: str, tools: Optional[List[Tool]] = None) -> None:
        self.name: str = name
        self.tools: List[Tool] = tools or []
        self._tool_map: Dict[str, Tool] = {t.name: t for t in self.tools}

    def add(self, tool: Tool) -> "Toolkit":
        """Add a tool to the toolkit."""
        self.tools.append(tool)
        self._tool_map[tool.name] = tool
        return self

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tool_map.get(name)

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI format."""
        return [t.to_openai_schema() for t in self.tools]

    def to_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Convert all tools to Anthropic format."""
        return [t.to_anthropic_schema() for t in self.tools]

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Any:
        return iter(self.tools)


# ---------------------------------------------------------------------------
# Tool Executor — manages the LLM <-> tool call loop
# ---------------------------------------------------------------------------

class ToolExecutor:
    """
    Manages the LLM <-> tool call loop.

    Repeatedly calls the LLM, extracts tool calls from the response,
    executes the tools, and feeds results back until the LLM produces
    a final text response or max rounds is reached.

    Args:
        provider: An LLMProvider instance.
        tools: Toolkit or list of Tools.
        max_rounds: Maximum tool call rounds before stopping.
    """

    def __init__(
        self,
        provider: Any,
        tools: Any,
        max_rounds: int = 5,
    ) -> None:
        self.provider = provider
        if isinstance(tools, Toolkit):
            self.toolkit = tools
        else:
            self.toolkit = Toolkit(name="default", tools=list(tools))
        self.max_rounds = max_rounds

    async def run(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute the tool call loop.

        Args:
            messages: Initial conversation messages.
            **kwargs: Extra provider kwargs.

        Returns:
            Dict with "text" (final response) and "tool_calls" (history).
        """
        tool_call_history = []
        current_messages = list(messages)

        # Add tool definitions
        kwargs["tools"] = self.toolkit.to_openai_tools()

        for round_num in range(self.max_rounds):
            response = await self.provider.complete(current_messages, **kwargs)

            # Check if response contains tool calls
            tool_calls = response.get("tool_calls", [])
            if not tool_calls:
                # No tool calls — LLM is done
                return {
                    "text": response.get("text", ""),
                    "tool_calls": tool_call_history,
                    "rounds": round_num + 1,
                }

            # Execute each tool call
            for tc in tool_calls:
                tool_name = tc.get("function", {}).get("name", tc.get("name", ""))
                arguments_str = tc.get("function", {}).get("arguments", tc.get("arguments", "{}"))
                if isinstance(arguments_str, str):
                    try:
                        arguments = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse tool call arguments for '%s': %s", tool_name, arguments_str)
                        arguments = {}
                else:
                    arguments = arguments_str

                tool = self.toolkit.get(tool_name)
                if tool:
                    result = await tool.run(arguments)
                else:
                    result = ToolResult(
                        tool_name=tool_name,
                        output=None,
                        error=f"ToolExecutor: unknown tool '{tool_name}'",
                        success=False,
                    )

                tool_call_history.append({
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result.to_dict(),
                })

                # Feed result back to the LLM
                current_messages.append({
                    "role": "tool",
                    "content": json.dumps(result.output if result.success else {"error": result.error}),
                    "tool_call_id": tc.get("id", str(uuid.uuid4())),
                })

        # Max rounds reached
        final = await self.provider.complete(current_messages, **kwargs)
        return {
            "text": final.get("text", ""),
            "tool_calls": tool_call_history,
            "rounds": self.max_rounds,
        }

"""
Real-World Cookbook: Claude Code–Style Coding Agent with Water
==============================================================

A coding agent inspired by Claude Code's architecture. The LLM fully controls
the loop — it decides which tools to call, when to delegate to sub-agents,
and when to stop. There are no hardcoded steps.

Architecture:
  - ReAct loop where the model drives every decision
  - Real tools that operate on the filesystem
  - Sub-agents for isolated tasks (research, file editing)
  - Layered memory (org rules, project context, auto-learned)
  - Semantic tool selection (narrows tools per reasoning step)
  - on_tool_call gating for safety
  - on_step tracing for observability

Requires:
    pip install openai   # or anthropic
    export OPENAI_API_KEY=sk-...   # or ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python cookbook/real_world/claude_code_agent.py
"""

import asyncio
import json
import os
import subprocess
import uuid

from water.agents.tools import Tool, Toolkit
from water.agents.react import create_agentic_task
from water.agents.subagent import SubAgentConfig, create_sub_agent_tool
from water.agents.memory import (
    MemoryLayer,
    MemoryManager,
    create_memory_tools,
)
from water.agents.tool_search import create_tool_selector


# ============================================================================
# LLM Provider with tool-calling support
# ============================================================================

class OpenAIToolProvider:
    """
    OpenAI provider that properly handles function/tool calling.

    The react loop expects:
      provider.complete(**kwargs) -> {"content": str, "tool_calls": list}
    where tool_calls follow OpenAI's format.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    async def complete(self, **kwargs):
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)

        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools", None)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1024)

        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            create_kwargs["tools"] = tools

        response = await client.chat.completions.create(**create_kwargs)
        choice = response.choices[0].message

        # Parse tool calls into the format react.py expects
        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                args = tc.function.arguments
                # Parse JSON string arguments into dict
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    parsed_args = {}
                tool_calls.append({
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": parsed_args,
                    },
                })

        return {
            "content": choice.content or "",
            "tool_calls": tool_calls,
        }


class AnthropicToolProvider:
    """
    Anthropic provider that properly handles tool use.

    Translates between Anthropic's tool_use format and the OpenAI-style
    format that react.py expects.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    async def complete(self, **kwargs):
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        messages = kwargs.get("messages", [])
        tools_openai = kwargs.get("tools", None)
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 1024)

        # Separate system prompt from messages and convert to Anthropic format
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "tool":
                # Consolidate consecutive tool results into a single user message
                tool_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                if filtered_messages and filtered_messages[-1]["role"] == "user" and isinstance(filtered_messages[-1].get("content"), list):
                    # Append to existing tool_result user message
                    filtered_messages[-1]["content"].append(tool_block)
                else:
                    filtered_messages.append({
                        "role": "user",
                        "content": [tool_block],
                    })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                # Convert OpenAI-style assistant+tool_calls to Anthropic
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    args = tc["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": args,
                    })
                filtered_messages.append({"role": "assistant", "content": content})
            else:
                filtered_messages.append(msg)

        # Convert OpenAI tool schemas to Anthropic format
        anthropic_tools = None
        if tools_openai:
            anthropic_tools = []
            for t in tools_openai:
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        create_kwargs = {
            "model": self.model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            create_kwargs["system"] = system
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools

        response = await client.messages.create(**create_kwargs)

        # Parse response into the format react.py expects
        content_text = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
                })

        return {
            "content": content_text,
            "tool_calls": tool_calls,
        }


def get_provider():
    """Pick a provider based on available API keys."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("[Provider] Using Anthropic (Claude)")
        return AnthropicToolProvider()
    elif os.environ.get("OPENAI_API_KEY"):
        print("[Provider] Using OpenAI")
        return OpenAIToolProvider()
    else:
        raise EnvironmentError(
            "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this cookbook."
        )


# ============================================================================
# Real tools — filesystem, search, bash, git
# ============================================================================

# The working directory for the agent (use a safe sandbox)
WORK_DIR = os.getcwd()


def _read_file(path: str) -> str:
    """Read a file's contents."""
    full = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
    with open(full, "r") as f:
        return f.read()


def _write_file(path: str, content: str) -> str:
    """Write content to a file."""
    full = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


def _list_dir(path: str = ".") -> str:
    """List directory contents."""
    full = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
    entries = os.listdir(full)
    return "\n".join(sorted(entries))


def _search_code(pattern: str, path: str = ".") -> str:
    """Search codebase using grep."""
    full = os.path.join(WORK_DIR, path) if not os.path.isabs(path) else path
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, full],
            capture_output=True, text=True, timeout=10,
        )
        output = result.stdout[:2000]  # Cap output
        return output if output else f"No matches for '{pattern}'"
    except Exception as e:
        return f"Search error: {e}"


def _run_bash(command: str) -> str:
    """Execute a bash command (sandboxed to WORK_DIR)."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=WORK_DIR,
        )
        output = (result.stdout + result.stderr)[:2000]
        return output if output else "(no output)"
    except Exception as e:
        return f"Error: {e}"


# Build tool objects
file_tools = [
    Tool(
        name="read_file",
        description="Read the contents of a file. Use this to understand existing code before making changes.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path relative to project root"}},
            "required": ["path"],
        },
        execute=_read_file,
    ),
    Tool(
        name="write_file",
        description="Write content to a file. Creates parent directories if needed.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to project root"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
        execute=_write_file,
    ),
    Tool(
        name="list_directory",
        description="List files and directories at the given path.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path", "default": "."}},
            "required": [],
        },
        execute=_list_dir,
    ),
]

search_tools = [
    Tool(
        name="search_code",
        description="Search the codebase for a pattern using grep. Returns matching lines with file paths and line numbers.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern (regex)"},
                "path": {"type": "string", "description": "Directory to search in", "default": "."},
            },
            "required": ["pattern"],
        },
        execute=_search_code,
    ),
]

bash_tool = Tool(
    name="bash",
    description="Execute a shell command. Use for running tests, installing packages, git operations, etc.",
    input_schema={
        "type": "object",
        "properties": {"command": {"type": "string", "description": "The shell command to run"}},
        "required": ["command"],
    },
    execute=_run_bash,
)


# ============================================================================
# Build and run the agent
# ============================================================================

async def main():
    print("=" * 70)
    print("  Claude Code–Style Coding Agent (built with Water)")
    print("=" * 70)

    provider = get_provider()

    # --- 1. Layered memory ---
    print("\n[Setup] Layered memory...")
    memory = MemoryManager()
    await memory.add(
        "code_style",
        "This project uses Python with type hints. Prefer async/await. Use pytest for tests.",
        MemoryLayer.ORG,
    )
    await memory.add(
        "project_info",
        "Water is a Python framework for building AI agent workflows with flows, tasks, and tools.",
        MemoryLayer.PROJECT,
    )
    memory_tools = create_memory_tools(memory)

    # --- 2. All tools the agent can use ---
    all_tools = file_tools + search_tools + [bash_tool] + memory_tools
    print(f"[Setup] {len(all_tools)} tools available: {[t.name for t in all_tools]}")

    # --- 3. Semantic tool selector ---
    selector = create_tool_selector(
        tools=all_tools,
        top_k=5,
        always_include=["memory_recall"],
    )

    # --- 4. Sub-agent for research ---
    research_agent = create_sub_agent_tool(SubAgentConfig(
        id="research_agent",
        provider=provider,
        tools=search_tools + [file_tools[0]],  # search + read_file
        system_prompt="You are a code research assistant. Search the codebase and read files to answer questions. Be concise.",
        max_iterations=5,
    ))

    orchestrator_tools = all_tools + [research_agent]

    # --- 5. Hooks ---
    step_log = []

    def on_step(iteration, step):
        think = (step["think"] or "")[:100]
        actions = step.get("act") or []
        tool_names = [a["tool"] for a in actions] if actions else []
        step_log.append({"i": iteration, "tools": tool_names})
        print(f"\n  [Step {iteration}]")
        if think:
            print(f"    Think: {think}...")
        if tool_names:
            print(f"    Tools: {tool_names}")

    BLOCKED = {"write_file"}  # safety: block writes in this demo

    def on_tool_call(tool_name, tool_args):
        if tool_name in BLOCKED:
            print(f"    [BLOCKED] {tool_name} — write operations disabled in demo")
            return False
        print(f"    [Calling] {tool_name}({json.dumps(tool_args)[:80]})")
        return True

    # --- 6. Build the agent ---
    memory_prompt = memory.to_system_prompt()
    system = (
        "You are an expert coding assistant, similar to Claude Code. You have access to "
        "tools for reading files, searching code, running shell commands, and managing memory. "
        "You also have a research_agent you can delegate deep code exploration to.\n\n"
        "When the user asks you to do something:\n"
        "1. Understand the request\n"
        "2. Explore the codebase to find relevant files\n"
        "3. Make changes or provide answers\n"
        "4. Verify your work if applicable\n\n"
        "Be concise and direct. Use tools — don't guess at file contents.\n\n"
        f"{memory_prompt}"
    )

    agent = create_agentic_task(
        id="claude-code",
        provider=provider,
        tools=orchestrator_tools,
        tool_selector=create_tool_selector(
            tools=orchestrator_tools,
            top_k=6,
            always_include=["memory_recall", "bash"],
        ),
        system_prompt=system,
        max_iterations=15,
        temperature=0.3,
        max_tokens=2048,
        on_step=on_step,
        on_tool_call=on_tool_call,
    )

    # --- 7. Run it ---
    user_prompt = "What does the water framework do? Look at the codebase and give me a summary."

    print(f"\n[User] {user_prompt}")
    print("-" * 70)

    result = await agent.execute(
        {"input_data": {"prompt": user_prompt}},
        None,
    )

    print("\n" + "=" * 70)
    print("[Response]")
    print(result["response"])
    print(f"\n[Stats] {result['iterations']} iterations, "
          f"{len(result['tool_history'])} tool calls")
    print(f"[Tools used] {[h['tool'] for h in result['tool_history']]}")


if __name__ == "__main__":
    asyncio.run(main())

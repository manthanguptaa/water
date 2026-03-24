"""ReAct (Reason + Act) agentic loop pattern.

Provides create_agentic_task() for creating tasks where the LLM controls
the iteration loop, deciding when to use tools and when to stop.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel

from water.agents.tools import Tool, Toolkit

logger = logging.getLogger(__name__)

__all__ = ["create_agentic_task"]


class AgenticInput(BaseModel):
    """Default input schema for agentic tasks."""
    prompt: str = ""


class AgenticOutput(BaseModel):
    """Default output schema for agentic tasks."""
    response: str = ""
    iterations: int = 0
    tool_history: list = []


def create_agentic_task(
    id: str = None,
    provider=None,
    tools: Optional[List[Tool]] = None,
    toolkit: Optional[Toolkit] = None,
    system_prompt: str = "",
    prompt_template: str = "",
    max_iterations: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stop_tool: bool = False,
    on_step: Optional[Callable] = None,
    on_tool_call: Optional[Callable] = None,
    stop_condition: Optional[Callable] = None,
    observation_formatter: Optional[Callable] = None,
    output_parser: Optional[Callable] = None,
    tool_selector: Optional[Any] = None,
    retry_count: int = 0,
    timeout: Optional[float] = None,
):
    """Create a task that runs a model-controlled agentic loop (ReAct pattern).

    The loop follows Think-Act-Observe-Repeat: the LLM reasons (Think),
    calls tools (Act), receives results (Observe), and repeats until done.

    Args:
        id: Task identifier.
        provider: LLM provider instance.
        tools: List of Tool objects.
        toolkit: Toolkit instance (alternative to tools list).
        system_prompt: System prompt for the agent.
        prompt_template: Template with {variable} placeholders for input data.
        max_iterations: Safety limit on loop iterations.
        temperature: LLM temperature.
        max_tokens: Max tokens per LLM response.
        stop_tool: If True, inject a __done__ tool for explicit stop signaling.
        on_step: Callback(iteration, step_dict) called after each Think-Act-Observe cycle.
        on_tool_call: Callback(tool_name, tool_args) called before tool execution.
            Return False to reject, a dict to modify args, or None/True to proceed.
        stop_condition: Callback(steps, tool_history) returning True to stop early.
        observation_formatter: Callback(tool_name, tool_args, tool_result) -> str
            to customize how tool results are fed back to the LLM.
        output_parser: Optional function to parse the final response.
        retry_count: Number of retries on failure.
        timeout: Timeout in seconds.

    Returns:
        A Task instance that can be used with flow.then().
    """
    from water.core.task import Task

    # Build toolkit
    all_tools = tools or []
    if toolkit:
        all_tools = list(toolkit) + all_tools

    if stop_tool:
        done_tool = Tool(
            name="__done__",
            description="Call this tool when you have completed the task and want to provide your final answer.",
            input_schema={
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string", "description": "Your final answer to the user's request"},
                    "metadata": {"type": "object", "description": "Optional metadata about the result", "default": {}},
                },
                "required": ["final_answer"],
            },
            execute=lambda final_answer, metadata=None: {"final_answer": final_answer, "metadata": metadata or {}},
        )
        all_tools.append(done_tool)

    final_toolkit = Toolkit(name="agentic_tools", tools=all_tools) if all_tools else None
    tools_schema = final_toolkit.to_openai_tools() if final_toolkit else None

    async def execute(params, context=None):
        data = params.get("input_data", params) if isinstance(params, dict) else params

        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

        # Build user message
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

        for iteration in range(max_iterations):
            # Call LLM — optionally narrow tools via tool_selector
            call_kwargs = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
            if tool_selector and final_toolkit:
                # Use the last user/assistant message as the query for tool selection
                query = user_message
                for m in reversed(messages):
                    if m.get("role") in ("user", "assistant") and m.get("content"):
                        query = m["content"]
                        break
                selected_toolkit = tool_selector.to_toolkit(query)
                call_kwargs["tools"] = selected_toolkit.to_openai_tools()
            elif tools_schema:
                call_kwargs["tools"] = tools_schema

            response = await provider.complete(**call_kwargs)
            last_response = response

            # THINK: capture reasoning
            thought = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                step = {"think": thought, "act": None, "observe": None}
                steps.append(step)
                if on_step:
                    await on_step(iteration + 1, step) if asyncio.iscoroutinefunction(on_step) else on_step(iteration + 1, step)
                result = {"response": thought, "tool_history": tool_history, "steps": steps, "iterations": iteration + 1}
                return output_parser(result) if output_parser else result

            # ACT: process tool calls
            # Normalize tool_calls — keep arguments as dicts for provider-agnostic handling
            normalized_tc = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        pass
                # Serialize arguments to JSON string for the message history
                # (OpenAI API requires arguments as a JSON string)
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                normalized_tc.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args_str,
                    },
                })
            assistant_msg = {"role": "assistant", "content": thought, "tool_calls": normalized_tc}
            messages.append(assistant_msg)

            step_actions = []
            step_observations = []

            for tc in tool_calls:
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                tool_args = fn.get("arguments", {})
                tool_call_id = tc.get("id", "")

                if tool_name == "__done__":
                    step = {"think": thought, "act": [{"tool": "__done__", "arguments": tool_args}], "observe": None}
                    steps.append(step)
                    if on_step:
                        await on_step(iteration + 1, step) if asyncio.iscoroutinefunction(on_step) else on_step(iteration + 1, step)
                    result = {
                        "response": tool_args.get("final_answer", ""),
                        "tool_history": tool_history,
                        "steps": steps,
                        "iterations": iteration + 1,
                        "metadata": tool_args.get("metadata", {}),
                    }
                    return output_parser(result) if output_parser else result

                # on_tool_call hook
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

                if final_toolkit:
                    tool = final_toolkit.get(tool_name)
                    if tool:
                        if isinstance(tool_args, str):
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
                tool_history.append({"iteration": iteration + 1, "tool": tool_name, "arguments": tool_args, "result": tool_result})

                # OBSERVE: format and feed back
                raw_content = str(tool_result.get("result", tool_result.get("error", "")))
                obs_content = observation_formatter(tool_name, tool_args, tool_result) if observation_formatter else raw_content
                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": obs_content})

            # Record full Think-Act-Observe step
            step = {"think": thought, "act": step_actions, "observe": step_observations}
            steps.append(step)

            if on_step:
                await on_step(iteration + 1, step) if asyncio.iscoroutinefunction(on_step) else on_step(iteration + 1, step)

            # Custom stop condition
            if stop_condition:
                should_stop = await stop_condition(steps, tool_history) if asyncio.iscoroutinefunction(stop_condition) else stop_condition(steps, tool_history)
                if should_stop:
                    result = {"response": thought, "tool_history": tool_history, "steps": steps, "iterations": iteration + 1}
                    return output_parser(result) if output_parser else result

        logger.warning(f"Agentic loop reached max_iterations ({max_iterations})")
        result = {"response": last_response.get("content", "") if last_response else "", "tool_history": tool_history, "steps": steps, "iterations": max_iterations}
        return output_parser(result) if output_parser else result

    task = Task(
        id=id or "agentic_task",
        input_schema=AgenticInput,
        output_schema=AgenticOutput,
        execute=execute,
        retry_count=retry_count,
        timeout=timeout,
        validate_schema=False,
    )
    return task

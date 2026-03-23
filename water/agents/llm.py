


__all__ = [
    "AgentInput",
    "AgentOutput",
    "LLMProvider",
    "MockProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "CustomProvider",
    "create_agent_task",
]


"""
LLM-Native Agent Tasks for Water.

Provides tasks that wrap LLM calls so flows can orchestrate AI agents
alongside regular Python tasks. Includes provider abstractions for
OpenAI, Anthropic, custom callables, and a mock provider for testing.
"""

import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel

from water.core.task import Task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_DEFAULT_CHARS_PER_TOKEN = 4


def estimate_token_count(text: str, chars_per_token: float = _DEFAULT_CHARS_PER_TOKEN) -> int:
    """
    Estimate the number of tokens in *text* using a simple heuristic.

    Uses ``len(text) / chars_per_token`` (default 4 chars per token).
    This is a rough approximation; for exact counts use a proper
    tokeniser such as ``tiktoken``.

    Args:
        text: The input text to estimate.
        chars_per_token: Average characters per token.

    Returns:
        Estimated token count (at least 1 for non-empty text, 0 for empty).
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


# ---------------------------------------------------------------------------
# Default schemas
# ---------------------------------------------------------------------------

class AgentInput(BaseModel):
    """Default input schema for agent tasks."""
    prompt: str = ""


class AgentOutput(BaseModel):
    """Default output schema for agent tasks."""
    response: str = ""


# ---------------------------------------------------------------------------
# LLM Provider hierarchy
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        """
        Send messages to the LLM and return a response dict.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Dict with at least a ``"text"`` key containing the response string.
        """
        ...


class MockProvider(LLMProvider):
    """
    Returns predefined responses. Useful for testing without real API keys.

    If *responses* is provided, they are returned in order (cycling if
    exhausted). Otherwise every call returns *default_response*.
    """

    def __init__(
        self,
        default_response: str = "mock response",
        responses: Optional[List[str]] = None,
    ) -> None:
        self.default_response = default_response
        self.responses = list(responses) if responses else []
        self._index = 0
        self.call_history: List[List[Dict[str, str]]] = []

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        self.call_history.append(messages)
        if self.responses:
            text = self.responses[self._index % len(self.responses)]
            self._index += 1
        else:
            text = self.default_response
        return {"text": text}


class OpenAIProvider(LLMProvider):
    """Wraps the OpenAI chat completions API (lazy import)."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        try:
            import openai  # lazy import
        except ImportError:
            raise ImportError(
                f"OpenAIProvider: the 'openai' package is required. "
                f"Install it with: pip install openai"
            )

        if not self.api_key:
            raise ValueError(
                f"OpenAIProvider: API key not provided. Pass api_key or set the "
                f"OPENAI_API_KEY environment variable."
            )

        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        text = response.choices[0].message.content or ""
        return {"text": text, "usage": response.usage}


class AnthropicProvider(LLMProvider):
    """Wraps the Anthropic messages API (lazy import)."""

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        try:
            import anthropic  # lazy import

        except ImportError:
            raise ImportError(
                f"AnthropicProvider: the 'anthropic' package is required. "
                f"Install it with: pip install anthropic"
            )

        if not self.api_key:
            raise ValueError(
                f"AnthropicProvider: API key not provided. Pass api_key or set the "
                f"ANTHROPIC_API_KEY environment variable."
            )

        # Anthropic separates system prompt from messages
        system = None
        filtered = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered.append(msg)

        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        create_kwargs: Dict[str, Any] = dict(
            model=kwargs.get("model", self.model),
            messages=filtered,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        if system:
            create_kwargs["system"] = system

        response = await client.messages.create(**create_kwargs)
        text = response.content[0].text if response.content else ""
        return {"text": text, "usage": response.usage}


class CustomProvider(LLMProvider):
    """
    Wraps any async callable with signature:

        async def my_llm(messages: list, **kwargs) -> str
    """

    def __init__(self, fn: Callable) -> None:
        if not callable(fn):
            raise ValueError(f"CustomProvider: requires a callable")
        self._fn = fn

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        result = await self._fn(messages, **kwargs)
        if isinstance(result, dict):
            return result
        return {"text": str(result)}


# ---------------------------------------------------------------------------
# Usage normalisation
# ---------------------------------------------------------------------------

def _normalize_usage(raw_usage: Any) -> Optional[Dict[str, int]]:
    """Normalise provider-specific token usage into ``{input_tokens, output_tokens}``.

    Handles OpenAI (``prompt_tokens``/``completion_tokens``), Anthropic
    (``input_tokens``/``output_tokens``), plain dicts, and SDK model objects.
    Returns ``None`` when *raw_usage* is ``None`` or empty.
    """
    if raw_usage is None:
        return None
        
    if isinstance(raw_usage, dict):
        in_t = raw_usage.get("input_tokens")
        out_t = raw_usage.get("output_tokens")
        return {
            "input_tokens": in_t if in_t is not None else raw_usage.get("prompt_tokens", 0),
            "output_tokens": out_t if out_t is not None else raw_usage.get("completion_tokens", 0),
        }
        
    # SDK objects (e.g. openai.types.CompletionUsage, anthropic.types.Usage)
    input_t = getattr(raw_usage, "input_tokens", None)
    if input_t is None:
        input_t = getattr(raw_usage, "prompt_tokens", 0)
        
    output_t = getattr(raw_usage, "output_tokens", None)
    if output_t is None:
        output_t = getattr(raw_usage, "completion_tokens", 0)
        
    return {"input_tokens": input_t, "output_tokens": output_t}


# ---------------------------------------------------------------------------
# Provider factory helper
# ---------------------------------------------------------------------------

def _resolve_provider(
    provider: str,
    model: str,
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    custom_fn: Optional[Callable] = None,
    provider_instance: Optional[LLMProvider] = None,
) -> LLMProvider:
    """Return an LLMProvider instance based on the *provider* string."""
    if provider_instance is not None:
        return provider_instance

    if provider == "openai":
        return OpenAIProvider(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )
    elif provider == "custom":
        if custom_fn is None:
            raise ValueError(
                f"CustomProvider: requires a callable passed via the "
                f"'custom_fn' parameter of create_agent_task."
            )
        return CustomProvider(fn=custom_fn)
    elif provider == "mock":
        return MockProvider()
    else:
        raise ValueError(f"LLMProvider: unknown provider {provider!r}")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_agent_task(
    id: Optional[str] = None,
    description: Optional[str] = None,
    prompt_template: str = "",
    model: str = "default",
    provider: str = "openai",
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    tools: Optional[List[Dict]] = None,
    output_parser: Optional[Callable] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    retry_count: int = 0,
    timeout: Optional[float] = None,
    custom_fn: Optional[Callable] = None,
    provider_instance: Optional[LLMProvider] = None,
) -> Task:
    """
    Create a Task that wraps an LLM call.

    The returned object is a regular :class:`water.task.Task` so it
    integrates seamlessly with Flow, retries, caching, etc.

    Args:
        id: Task identifier. Auto-generated if not provided.
        description: Human-readable description.
        prompt_template: A string with ``{variable}`` placeholders that
            will be formatted with the task's input data.
        model: Model identifier passed to the provider.
        provider: One of ``"openai"``, ``"anthropic"``, ``"custom"``,
            ``"mock"``.
        api_key: API key for the provider (or use env vars).
        system_prompt: Optional system message prepended to the
            conversation.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        tools: Tool definitions the LLM can use (provider-specific).
        output_parser: ``Callable[[str], dict]`` that transforms the raw
            LLM text into a result dict.
        input_schema: Pydantic model for task input validation.
        output_schema: Pydantic model for task output validation.
        retry_count: Number of retries on failure.
        timeout: Timeout in seconds for the task execution.
        custom_fn: Async callable for the ``"custom"`` provider.
        provider_instance: Pre-built :class:`LLMProvider` instance
            (takes precedence over *provider*).

    Returns:
        A :class:`Task` instance ready to be added to a Flow.
    """

    task_id = id or f"agent_{uuid.uuid4().hex[:8]}"

    if not input_schema:
        input_schema = type(
            f"{task_id}_Input",
            (BaseModel,),
            {"__annotations__": {"prompt": str}},
        )
    if not output_schema:
        output_schema = type(
            f"{task_id}_Output",
            (BaseModel,),
            {"__annotations__": {"response": str}},
        )

    llm_provider = _resolve_provider(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        custom_fn=custom_fn,
        provider_instance=provider_instance,
    )

    async def execute(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        input_data = params.get("input_data", params)

        # 1. Format prompt template with input variables
        if prompt_template:
            try:
                user_content = prompt_template.format(**input_data)
            except KeyError as exc:
                raise ValueError(
                    f"AgentTask: prompt template variable {exc} not found in input data. "
                    f"Available keys: {list(input_data.keys())}"
                )
        else:
            # Fall back to a 'prompt' key in the input data
            user_content = str(input_data.get("prompt", ""))

        # 2. Build messages
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # 3. Estimate tokens and log before calling the provider
        total_input_text = "".join(m["content"] for m in messages)
        estimated_tokens = estimate_token_count(total_input_text)
        logger.info(
            "LLM call [%s]: estimated input tokens ~%d (from %d chars)",
            task_id, estimated_tokens, len(total_input_text),
        )

        # 4. Call the provider
        extra_kwargs: Dict[str, Any] = {}
        if tools:
            extra_kwargs["tools"] = tools
        response = await llm_provider.complete(messages, **extra_kwargs)
        response_text = response.get("text", "")

        # Build cost-tracking metadata so CostTracker middleware can record usage
        cost_meta: Dict[str, Any] = {}
        usage = _normalize_usage(response.get("usage"))
        if usage:
            cost_meta["usage"] = usage
            cost_meta["model"] = getattr(llm_provider, "model", "unknown")

        # 5. Parse response
        if output_parser:
            parsed = output_parser(response_text)
            if isinstance(parsed, dict):
                return {**parsed, **cost_meta}
            return {"response": parsed, **cost_meta}

        # 6. Default: merge response with input data
        return {"response": response_text, **input_data, **cost_meta}

    return Task(
        id=task_id,
        description=description or f"Agent task: {task_id}",
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
        retry_count=retry_count,
        timeout=timeout,
    )

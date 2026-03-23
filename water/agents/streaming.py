"""
Async Streaming LLM Responses for Water.

Extends the agent task system with streaming support so flows can
consume LLM output token-by-token (or word-by-word) while still
producing a final Dict result compatible with the rest of the framework.

Includes:
  - StreamChunk dataclass for incremental deltas
  - StreamingResponse collector that accumulates chunks into full text
  - StreamingProvider base class with both complete() and stream() methods
  - MockStreamProvider for testing (yields words one at a time)
  - OpenAIStreamProvider / AnthropicStreamProvider for production use
  - create_streaming_agent_task() factory mirroring create_agent_task()
"""

import os
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
)

from pydantic import BaseModel

from water.agents.llm import AgentInput, AgentOutput, LLMProvider
from water.core.task import Task


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """A single incremental piece of a streaming LLM response."""

    delta: str = ""
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingResponse:
    """
    Accumulates StreamChunk objects and exposes the assembled text.

    Usage::

        sr = StreamingResponse()
        async for chunk in provider.stream(messages):
            sr.add(chunk)
        print(sr.text)
    """

    def __init__(self) -> None:
        self.chunks: List[StreamChunk] = []

    def add(self, chunk: StreamChunk) -> None:
        """Append a chunk to the response."""
        self.chunks.append(chunk)

    @property
    def text(self) -> str:
        """Return the concatenated text from all collected chunks."""
        return "".join(c.delta for c in self.chunks)

    @property
    def finish_reason(self) -> Optional[str]:
        """Return the finish_reason from the last chunk that carries one."""
        for chunk in reversed(self.chunks):
            if chunk.finish_reason is not None:
                return chunk.finish_reason
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        """Merge metadata dicts from all chunks (later chunks win)."""
        merged: Dict[str, Any] = {}
        for chunk in self.chunks:
            merged.update(chunk.metadata)
        return merged


# ---------------------------------------------------------------------------
# Streaming provider hierarchy
# ---------------------------------------------------------------------------

class StreamingProvider(LLMProvider):
    """
    Extended LLM provider that supports token-level streaming.

    Subclasses must implement :meth:`stream`.  The default
    :meth:`complete` implementation collects the full stream, so
    subclasses only need to override it if they want a more efficient
    non-streaming path.
    """

    async def complete(self, messages: List[Dict[str, str]], **kwargs) -> dict:
        """Fallback: consume the stream and return the assembled text."""
        sr = StreamingResponse()
        async for chunk in self.stream(messages, **kwargs):
            sr.add(chunk)
        return {"text": sr.text}

    async def stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        Yield StreamChunk objects as the LLM generates tokens.

        Must be overridden by concrete subclasses.
        """
        raise NotImplementedError("Subclasses must implement stream()")
        # Make the function an async generator even though it raises.
        yield  # pragma: no cover


class MockStreamProvider(StreamingProvider):
    """
    Yields words from a predefined response one at a time.

    Useful for testing streaming pipelines without real API keys.
    If *responses* is provided they are cycled in order, otherwise
    *default_response* is used every time.
    """

    def __init__(
        self,
        default_response: str = "mock streaming response",
        responses: Optional[List[str]] = None,
    ) -> None:
        self.default_response = default_response
        self.responses = list(responses) if responses else []
        self._index = 0
        self.call_history: List[List[Dict[str, str]]] = []

    def _next_response(self) -> str:
        if self.responses:
            text = self.responses[self._index % len(self.responses)]
            self._index += 1
            return text
        return self.default_response

    async def stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[StreamChunk]:
        self.call_history.append(messages)
        text = self._next_response()
        words = text.split(" ")
        for i, word in enumerate(words):
            # Re-insert the space that split removed (except for the first word)
            delta = word if i == 0 else " " + word
            is_last = i == len(words) - 1
            yield StreamChunk(
                delta=delta,
                finish_reason="stop" if is_last else None,
                metadata={"index": i},
            )


class OpenAIStreamProvider(StreamingProvider):
    """Wraps the OpenAI chat completions streaming API (lazy import)."""

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

    async def stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[StreamChunk]:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required to use OpenAIStreamProvider. "
                "Install it with: pip install openai"
            )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Pass api_key or set the "
                "OPENAI_API_KEY environment variable."
            )

        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        async for event in response:
            choice = event.choices[0] if event.choices else None
            if choice is None:
                continue
            delta_content = choice.delta.content or ""
            finish = choice.finish_reason
            yield StreamChunk(
                delta=delta_content,
                finish_reason=finish,
                metadata={},
            )


class AnthropicStreamProvider(StreamingProvider):
    """Wraps the Anthropic messages streaming API (lazy import)."""

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

    async def stream(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> AsyncIterator[StreamChunk]:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required to use AnthropicStreamProvider. "
                "Install it with: pip install anthropic"
            )

        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Pass api_key or set the "
                "ANTHROPIC_API_KEY environment variable."
            )

        # Separate system prompt from messages (Anthropic API requirement)
        system = None
        filtered: List[Dict[str, str]] = []
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

        async with client.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text, finish_reason=None, metadata={})
            yield StreamChunk(delta="", finish_reason="end_turn", metadata={})


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_streaming_agent_task(
    id: Optional[str] = None,
    description: Optional[str] = None,
    prompt_template: str = "",
    provider_instance: Optional[StreamingProvider] = None,
    system_prompt: Optional[str] = None,
    on_chunk: Optional[Callable[[StreamChunk], Any]] = None,
    input_schema: Optional[Type[BaseModel]] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    retry_count: int = 0,
    timeout: Optional[float] = None,
) -> Task:
    """
    Create a Task that wraps a streaming LLM call.

    The returned Task is fully compatible with :class:`water.core.task.Task`
    and collects all streamed chunks into a single Dict result.  If an
    *on_chunk* callback is supplied it is called for every
    :class:`StreamChunk` as it arrives, enabling real-time UI updates.

    Args:
        id: Task identifier (auto-generated if omitted).
        description: Human-readable description.
        prompt_template: String with ``{variable}`` placeholders filled
            from the task's input data.
        provider_instance: A :class:`StreamingProvider` to use.
        system_prompt: Optional system message prepended to the
            conversation.
        on_chunk: Optional callback invoked with each StreamChunk.
        input_schema: Pydantic model for task input validation.
        output_schema: Pydantic model for task output validation.
        retry_count: Number of retries on failure.
        timeout: Timeout in seconds for the task execution.

    Returns:
        A :class:`Task` instance ready to be added to a Flow.
    """

    task_id = id or f"streaming_agent_{uuid.uuid4().hex[:8]}"

    if provider_instance is None:
        provider_instance = MockStreamProvider()

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

    async def execute(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        input_data = params.get("input_data", params)

        # 1. Format the prompt
        if prompt_template:
            try:
                user_content = prompt_template.format(**input_data)
            except KeyError as exc:
                raise ValueError(
                    f"Prompt template variable {exc} not found in input data. "
                    f"Available keys: {list(input_data.keys())}"
                )
        else:
            user_content = str(input_data.get("prompt", ""))

        # 2. Build messages
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        # 3. Stream from the provider
        sr = StreamingResponse()
        async for chunk in provider_instance.stream(messages):
            sr.add(chunk)
            if on_chunk is not None:
                on_chunk(chunk)

        # 4. Return collected result
        return {"response": sr.text, **input_data}

    return Task(
        id=task_id,
        description=description or f"Streaming agent task: {task_id}",
        input_schema=input_schema,
        output_schema=output_schema,
        execute=execute,
        retry_count=retry_count,
        timeout=timeout,
    )

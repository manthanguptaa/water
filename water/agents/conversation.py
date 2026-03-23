"""
Multi-Turn Conversation Management for Water.

Provides structured conversation state tracking, turn-based execution,
and a factory function that creates a Task for single-turn processing
within a flow.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from water.agents.context import ContextManager
from water.agents.llm import LLMProvider
from water.core.task import Task

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationState:
    """Persistent state for a single conversation."""

    conversation_id: str = ""
    history: List[Turn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    slots: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.conversation_id:
            self.conversation_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ConversationManager
# ---------------------------------------------------------------------------


class ConversationManager:
    """
    Orchestrates multi-turn dialogue with an LLM provider.

    Maintains conversation history within a :class:`ConversationState`,
    optionally applying token windowing via a :class:`ContextManager`.

    Args:
        provider: The LLM provider to use for completions.
        system_prompt: System message prepended to every call.
        max_history: Maximum number of turns retained in state.
        context_manager: Optional context manager for token windowing.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "",
        max_history: int = 50,
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.context_manager = context_manager

    async def send(self, message: str, state: ConversationState) -> Turn:
        """
        Send a user message and get an assistant response.

        Appends the user turn and assistant turn to *state.history*,
        truncating to *max_history* if needed.

        Args:
            message: The user's message text.
            state: The conversation state to read/update.

        Returns:
            The assistant's :class:`Turn`.
        """
        user_turn = Turn(role="user", content=message)
        state.history.append(user_turn)

        # Build messages for the provider
        messages = self.get_history(state)

        # Apply token windowing if a context manager is configured
        if self.context_manager is not None:
            messages = await self.context_manager.prepare_messages(messages)

        response = await self.provider.complete(messages)
        response_text = response.get("text", "")

        assistant_turn = Turn(role="assistant", content=response_text)
        state.history.append(assistant_turn)

        # Enforce max_history (count user+assistant pairs, keep last N turns)
        if len(state.history) > self.max_history:
            state.history = state.history[-self.max_history:]

        return assistant_turn

    def get_history(
        self, state: ConversationState, max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Return conversation history formatted for the LLM provider.

        Args:
            state: The conversation state.
            max_turns: If set, only include the last *max_turns* turns.

        Returns:
            List of message dicts with ``role`` and ``content`` keys.
        """
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        turns = state.history
        if max_turns is not None:
            turns = turns[-max_turns:]

        for turn in turns:
            messages.append({"role": turn.role, "content": turn.content})

        return messages

    def clear(self, state: ConversationState) -> None:
        """Reset conversation history and slots."""
        state.history.clear()
        state.slots.clear()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_conversation_task(
    id: Optional[str] = None,
    provider: Optional[LLMProvider] = None,
    system_prompt: str = "",
    max_history: int = 50,
    context_manager: Optional[ContextManager] = None,
) -> Task:
    """
    Create a :class:`Task` that processes a single conversation turn.

    The task manages :class:`ConversationState` objects internally,
    keyed by ``conversation_id``.

    Input schema::

        {"message": str, "conversation_id": str}

    Output schema::

        {"response": str, "conversation_id": str, "turn_count": int}

    Args:
        id: Task identifier (auto-generated if omitted).
        provider: LLM provider instance.
        system_prompt: System prompt for the conversation.
        max_history: Maximum turns kept in state.
        context_manager: Optional context manager for token windowing.

    Returns:
        A :class:`Task` that can be added to a Flow.
    """
    task_id = id or f"conversation_{uuid.uuid4().hex[:8]}"

    if provider is None:
        raise ValueError("create_conversation_task requires a provider")

    manager = ConversationManager(
        provider=provider,
        system_prompt=system_prompt,
        max_history=max_history,
        context_manager=context_manager,
    )

    # Internal store of conversation states keyed by conversation_id
    states: Dict[str, ConversationState] = {}

    async def execute(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
        input_data = params.get("input_data", params)
        message = str(input_data.get("message", ""))
        conversation_id = str(input_data.get("conversation_id", "default"))

        if conversation_id not in states:
            states[conversation_id] = ConversationState(
                conversation_id=conversation_id
            )

        state = states[conversation_id]
        assistant_turn = await manager.send(message, state)

        return {
            "response": assistant_turn.content,
            "conversation_id": conversation_id,
            "turn_count": len(state.history),
        }

    class _ConvInput(BaseModel):
        message: str = ""
        conversation_id: str = "default"

    class _ConvOutput(BaseModel):
        response: str = ""
        conversation_id: str = ""
        turn_count: int = 0

    return Task(
        id=task_id,
        description=f"Conversation task: {task_id}",
        input_schema=_ConvInput,
        output_schema=_ConvOutput,
        execute=execute,
        validate_schema=False,
    )

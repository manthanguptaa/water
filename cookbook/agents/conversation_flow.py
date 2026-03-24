"""
Cookbook: Multi-Turn Conversation Flow
======================================

Demonstrates how to use ConversationManager with a real OpenAI LLM
to build a support bot that preserves dialogue state across turns.
"""

import asyncio

from water.agents.conversation import ConversationManager, ConversationState
from water.agents.llm import OpenAIProvider


async def main():
    # Set up a real OpenAI provider
    provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.7)

    # Create a conversation manager with a system prompt
    manager = ConversationManager(
        provider=provider,
        system_prompt="You are a friendly customer support agent. Keep responses brief (1-2 sentences).",
        max_history=50,
    )

    # Create conversation state
    state = ConversationState(conversation_id="support-session-1")

    # Simulate a multi-turn dialogue
    user_messages = [
        "Hi, I need help.",
        "I can't log into my account.",
        "Yes, please reset my password.",
        "Thank you so much!",
        "No, that's all. Goodbye!",
    ]

    print("=== Support Bot Conversation ===\n")

    for message in user_messages:
        print(f"User: {message}")
        turn = await manager.send(message, state)
        print(f"Bot:  {turn.content}")
        print()

    # Show final state
    print(f"--- Conversation ID: {state.conversation_id} ---")
    print(f"--- Total turns: {len(state.history)} ---")
    print(f"--- Slots: {state.slots} ---")


if __name__ == "__main__":
    asyncio.run(main())

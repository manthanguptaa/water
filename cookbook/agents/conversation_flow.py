"""
Cookbook: Multi-Turn Conversation Flow
======================================

Demonstrates how to use ConversationManager with a MockProvider
to build a support bot that preserves dialogue state across turns.
"""

import asyncio

from water.agents.conversation import ConversationManager, ConversationState
from water.agents.llm import MockProvider


async def main():
    # Set up a mock provider that returns canned support-bot responses
    provider = MockProvider(
        responses=[
            "Hello! I'm your support bot. How can I help you today?",
            "I understand you're having trouble logging in. Let me look into that.",
            "I've reset your password. You should receive an email shortly.",
            "You're welcome! Is there anything else I can help with?",
            "Glad I could help. Have a great day!",
        ]
    )

    # Create a conversation manager with a system prompt
    manager = ConversationManager(
        provider=provider,
        system_prompt="You are a friendly customer support agent.",
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


# This example shows how new_messages is different from all_messages:
# - all_messages() is used to maintain the conversation context between interactions
# - new_messages_json() is used to log only the latest interaction
# - Each interaction is logged separately while the full conversation history is maintained independently

# The key is that they serve different purposes:
# - all_messages: conversation continuity
# - new_messages: interaction tracking/logging

"""Example demonstrating new_messages in PydanticAI.

Shows how to log individual interactions while maintaining separate chat history.
"""

from datetime import datetime
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

support_agent = Agent(
    "openai:gpt-4o",
    system_prompt="You are a customer service agent."
)

def log_interaction(ticket_id: str, interaction_number: int, messages_json: bytes) -> None:
    """Log just this interaction to a separate file."""
    Path(f"ticket_{ticket_id}_interaction_{interaction_number}.json").write_bytes(messages_json)

async def main():
    # Initialize conversation
    history = []
    ticket_id = "T123"
    
    # First interaction
    result = await support_agent.run(
        "I need help with my account",
        message_history=history
    )
    # Log just this interaction
    log_interaction(ticket_id, 1, result.new_messages_json())
    # Update full history for context
    history = result.all_messages()
    
    # Second interaction
    result = await support_agent.run(
        "I can't see my latest transaction",
        message_history=history
    )
    # Log just the new interaction
    log_interaction(ticket_id, 2, result.new_messages_json())
    
    # Demonstrate what's in the logs
    for i in (1, 2):
        log_file = f"ticket_{ticket_id}_interaction_{i}.json"
        interaction = ModelMessagesTypeAdapter.validate_json(
            Path(log_file).read_bytes()
        )
        print(f"\nInteraction {i}:")
        for msg in interaction:
            print(f"- {msg.parts[0].content[:50]}...")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

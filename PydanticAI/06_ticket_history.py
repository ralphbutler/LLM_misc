
# This example demonstrates message serialization by:
# 
# - Using ModelMessagesTypeAdapter for proper serialization/deserialization
# - Showing how to save messages to a file and load them back
# - Demonstrating how to use loaded history in new conversations
# - Keeping conversation context across multiple interactions
#
# The example is focused but shows a real use case where message serialization is
#    necessary (support ticket system with persistent history).

"""Example demonstrating message serialization in PydanticAI.

Shows how to save and load conversation history for a support ticket system.
"""

from pathlib import Path
import json
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

support_agent = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You are a support agent. Maintain context from previous messages "
        "and reference ticket history when appropriate."
    )
)

def save_ticket_history(ticket_id: str, messages: list) -> None:
    """Save conversation history to a JSON file."""
    # Serialize messages using the ModelMessagesTypeAdapter
    messages_json = ModelMessagesTypeAdapter.dump_json(messages)
    
    # Save to file
    Path(f"ticket_{ticket_id}.json").write_bytes(messages_json)
    
def load_ticket_history(ticket_id: str) -> list:
    """Load conversation history from a JSON file."""
    try:
        # Read the file
        messages_json = Path(f"ticket_{ticket_id}.json").read_bytes()
        
        # Deserialize messages using the ModelMessagesTypeAdapter
        return ModelMessagesTypeAdapter.validate_json(messages_json)
    except FileNotFoundError:
        return []

# Example usage
async def main():
    # Simulate a ticket conversation
    ticket_id = "12345"
    
    # Start a new conversation
    result = await support_agent.run(
        "My printer isn't working",
        message_history=[]  # New conversation
    )
    print("First response:", result.data)
    
    # Save the conversation
    save_ticket_history(ticket_id, result.all_messages())
    
    # Later... load the history and continue the conversation
    history = load_ticket_history(ticket_id)
    result = await support_agent.run(
        "It's still not working after trying those steps",
        message_history=history
    )
    print("\nFollow-up response:", result.data)
    
    # Save updated history
    save_ticket_history(ticket_id, result.all_messages())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

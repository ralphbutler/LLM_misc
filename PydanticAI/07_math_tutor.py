
# This example demonstrates chat history reuse by:
# 
# - Using message_history parameter to maintain context across interactions
# - Using result.all_messages() to get the updated history after each interaction
# - Showing how the agent can reference previous explanations in follow-up responses
# - Building a continuous conversation that maintains context

# all_messages and new_messages serve different purposes, and it would be good to do
#    a separate example (08) for new_messages. Here's the key difference:
# all_messages(): Returns the complete conversation history (previous history + new interactions)
# - Used when you want to maintain conversation context
# - Perfect for chatbots, tutoring, and continuous conversations
# - What we showed in the history reuse example


new_messages() and new_messages_json(): Returns only the messages from the current interaction

"""Example demonstrating chat history reuse in PydanticAI.

Shows how to maintain context through a math tutoring session.
"""

from typing import List
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

math_tutor = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You are a math tutor helping a student solve problems. "
        "Reference previous explanations when building on concepts."
    )
)

async def continue_lesson(messages: List[ModelMessage], new_question: str) -> None:
    """Continue a tutoring session using previous chat history."""
    result = await math_tutor.run(
        new_question,
        message_history=messages
    )
    print(f"\nStudent: {new_question}")
    print(f"Tutor: {result.data}")
    return result.all_messages()

async def main():
    # Start with no history
    history = []
    
    # First question about basic concept
    result = await math_tutor.run(
        "Can you explain what a prime number is?",
        message_history=history
    )
    print(f"Student: Can you explain what a prime number is?")
    print(f"Tutor: {result.data}")
    
    # Use history from first interaction
    history = result.all_messages()
    
    # Follow-up questions building on previous explanation
    history = await continue_lesson(
        history,
        "Is 15 a prime number? Please explain why or why not."
    )
    
    # Another follow-up using accumulated context
    history = await continue_lesson(
        history,
        "What's the next prime number after 15?"
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

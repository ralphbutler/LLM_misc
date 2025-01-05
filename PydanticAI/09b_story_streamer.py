
# This second version adds:
# - Word counting during streaming
# - Progress tracking with \r carriage return
# - Demonstrates debounce_by for smoother output
# - Shows how to access streaming metadata (timestamps)

"""Example demonstrating streaming text responses in PydanticAI.

Shows how to stream a generated story with progress tracking.
"""

from pydantic_ai import Agent

story_agent = Agent(
    "openai:gpt-4o",
    system_prompt=(
        "You are a storyteller. Create a very short story (2-3 sentences) "
        "based on the given theme. Keep it simple and family-friendly."
    )
)

async def main():
    print("Generating story...\n")
    words_seen = 0
    
    async with story_agent.run_stream("Tell me a story about a magic library") as result:
        print("Story: ", end="", flush=True)
        async for chunk in result.stream(debounce_by=0.1):  # Smooth out the streaming
            words_seen += len(chunk.split())
            print(chunk, end="", flush=True)
            print(f"\rWords generated: {words_seen}", end="", flush=True)
    
    print("\n\nStory complete!")
    
    # Can also get timestamps
    print(f"Generation started at: {result.timestamp()}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

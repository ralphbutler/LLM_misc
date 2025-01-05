
# This example demonstrates streaming responses by:
# - Using run_stream() to create a streaming context
# - Using async for to iterate over chunks of the response
# - Showing how to handle the streamed output in real-time
# - Demonstrating the async context manager pattern

# Key features:
# - Uses async with to manage the streaming context
# - Shows real-time output with flush=True
# - Demonstrates how to access the final complete result
# - Keeps the example focused while showing practical usage

"""Example demonstrating streaming text responses in PydanticAI.

Shows how to stream a generated story word by word with progress markers.
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
    
    async with story_agent.run_stream("Tell me a story about a magic library") as result:
        print("Story: ", end="", flush=True)
        async for chunk in result.stream():
            # Print each chunk without newline and flush to show streaming
            print(chunk, end="", flush=True)
    
    # Print final newlines after story is complete
    print("\n\nStory generation complete!")
    
    # You can also get the final result
    print("\nFinal story length:", len(result.data))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

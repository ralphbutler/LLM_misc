
# This example simplifies the structured streaming concept while keeping the key elements:
# - Simple TypedDict with just 3 fields:
#     - name (basic string)
#     - cookTime (integer with min/max validation)
#     - difficulty (literal with fixed options)
# - Shows structured streaming with:
#     - stream_structured() for getting updates
#     - validate_structured_result() with partial validation
#     - Real-time display updates as fields are populated
# - Keeps the power of validation while being more approachable

"""Example demonstrating structured streaming responses with validation in PydanticAI.

Shows how to stream and validate a recipe as it's being generated.
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import Field
from pydantic_ai import Agent
from rich.live import Live
from rich.panel import Panel

class Recipe(TypedDict):
    name: str
    cookTime: Annotated[int, Field(description='Cooking time in minutes', ge=5, le=120)]
    difficulty: Literal['easy', 'medium', 'hard']

recipe_agent = Agent(
    "openai:gpt-4o",
    result_type=Recipe,
    system_prompt=(
        "You are a recipe generator. Create simple recipes with a name, "
        "cooking time (in minutes), and difficulty level (easy/medium/hard)."
    )
)

async def main():
    print("Generating recipe...\n")
    
    with Live("", refresh_per_second=4) as live:
        async with recipe_agent.run_stream("Create a pasta recipe") as result:
            async for message, is_final in result.stream_structured():
                try:
                    recipe = await result.validate_structured_result(
                        message, 
                        allow_partial=not is_final
                    )
                    # Create display panel with available information
                    content = []
                    if 'name' in recipe:
                        content.append(f"Recipe: {recipe['name']}")
                    if 'cookTime' in recipe:
                        content.append(f"Cooking Time: {recipe['cookTime']} minutes")
                    if 'difficulty' in recipe:
                        content.append(f"Difficulty: {recipe['difficulty']}")
                    
                    live.update(Panel("\n".join(content), title="Recipe Details"))
                    
                except Exception as e:
                    live.update(f"Validation error: {str(e)}")

    print("\nFinal recipe generated!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

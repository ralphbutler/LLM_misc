
# This example demonstrates dynamic system prompts because:

# - The prompt changes based on the current time
# - It's impossible to achieve this with a static prompt since the time and appropriate mood change
# - The @greeter.system_prompt decorator specifies how to generate the prompt for each interaction
# - It shows how to access context and generate different prompts based on conditions

# The example is intentionally short but shows a real use case where dynamic prompts are necessary
#    rather than just convenient.

from datetime import datetime, timezone
from pydantic_ai import Agent, RunContext

greeter = Agent(
    "openai:gpt-4o",
    system_prompt="Base prompt - will be replaced dynamically"
)

@greeter.system_prompt
async def time_aware_prompt(ctx: RunContext[None]) -> str:
    """Generate a system prompt based on the current time."""
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    if hour < 12:
        time_of_day = "morning"
        mood = "energetic and ready to start the day"
    elif hour < 17:
        time_of_day = "afternoon"
        mood = "productive and focused"
    else:
        time_of_day = "evening"
        mood = "relaxed and winding down"
        
    return (
        f"You are a helpful assistant during the {time_of_day}. "
        f"Keep responses {mood}. "
        f"Current time: {now.strftime('%H:%M %Z')}"
    )

# Example usage
result = greeter.run_sync("Hello! How are you?")
print(f"AI: {result.data}")

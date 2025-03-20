import os
import json
import sys
import asyncio
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, stdio_client

# Claude API key (should be set as environment variable)
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not CLAUDE_API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)

# Path to the MCP server script
SERVER_SCRIPT_PATH = "arc_server.py"
# Specific puzzle to solve (can be changed)
TARGET_PUZZLE = "d037b0a7.json"

# Initialize Claude client
claude = Anthropic(api_key=CLAUDE_API_KEY)

async def main():
    print(f"Starting ARC puzzle client to solve puzzle: {TARGET_PUZZLE}")
    
    # Start the MCP server process and connect to it
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_SCRIPT_PATH],
    )
    
    try:
        # Connect to the MCP server
        print("Connecting to ARC puzzle server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # Step 1: Use the tool to list available puzzles
                print("Listing available puzzles...")
                available_puzzles = await session.call_tool("list_available_puzzles")
                available_puzzles = [puzzle.text for puzzle in available_puzzles.content]
                
                # Check if our target puzzle is available
                if TARGET_PUZZLE not in available_puzzles:
                    print(f"Error: Puzzle {TARGET_PUZZLE} not found in available puzzles")
                    print("Available puzzles:", ", ".join(available_puzzles[:10]) + 
                          ("..." if len(available_puzzles) > 10 else ""))
                    return
                
                print(f"Found target puzzle {TARGET_PUZZLE} in available puzzles")
                
                # Step 2: Use the resource to get the puzzle data
                print(f"Retrieving puzzle data for {TARGET_PUZZLE}...")
                resource_response = await session.read_resource(f"puzzle://{TARGET_PUZZLE}")
                
                # Extract the text content from the resource response based on the actual structure
                puzzle_data = None
                # Check if the response matches the format we see in the output
                if hasattr(resource_response, 'contents') and len(resource_response.contents) > 0:
                    # Try the first content item
                    puzzle_data = resource_response.contents[0].text
                    
                if not puzzle_data:
                    print("Error: Could not extract puzzle data from resource response")
                    print("Resource response:", resource_response)
                    return
                
                # Check if we got valid puzzle data
                try:
                    puzzle_json = json.loads(puzzle_data)
                    print(f"Successfully retrieved puzzle data ({len(puzzle_data)} bytes)")
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse puzzle data as JSON: {str(e)}")
                    print("Raw data:", puzzle_data[:200] + "..." if len(puzzle_data) > 200 else puzzle_data)
                    return
                
                # Step 3: Get the prompt template
                print("Retrieving prompt template...")
                prompt_template = await session.get_prompt("arc_puzzle_prompt", {"puzzle_data": puzzle_data})
                prompt_text = prompt_template.messages[0].content.text

                # Step 4: Use the prompt and puzzle data to query Claude
                print("Sending puzzle to the LLM for solving...")
                response = claude.messages.create(
                    # model="claude-3-7-sonnet-20250219",
                    model="claude-3-7-sonnet-latest",
                    max_tokens=4000,
                    messages=[ {"role": "user", "content": prompt_text} ]
                )

                # Step 5: Print Claude's response
                print("\n" + "=" * 80)
                print("CLAUDE'S SOLUTION:")
                print("=" * 80)
                print(response.content[0].text)
                print("=" * 80)
                
                print("\nPuzzle solving complete!")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

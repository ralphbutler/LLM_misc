import os
import sys
import asyncio
import json
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, stdio_client

# Claude API key (should be set as environment variable)
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not CLAUDE_API_KEY:
    print("Error: ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)

# Path to your MCP server script
SERVER_SCRIPT_PATH = "rand_server.py"

# Initialize Claude client
claude = Anthropic(api_key=CLAUDE_API_KEY)

async def main():
    print("Starting Sonnet MCP client to demonstrate tool usage")
    
    # Start the MCP server process and connect to it
    server_params = StdioServerParameters(
        command="python",
        args=[SERVER_SCRIPT_PATH],
    )
    
    try:
        # Connect to the MCP server
        print("Connecting to random number server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # List available tools to verify what we're working with
                tools = await session.list_tools()
                # Inspect the structure of the tools response to handle it properly
                print(f"Tools response: {tools}")
                
                # Try to extract tool names safely
                tool_names = []
                if isinstance(tools, list):
                    for tool in tools:
                        if hasattr(tool, 'name'):
                            tool_names.append(tool.name)
                        elif isinstance(tool, dict) and 'name' in tool:
                            tool_names.append(tool['name'])
                        elif isinstance(tool, tuple) and len(tool) > 0:
                            # If it's a tuple, try to get the first element as the name
                            tool_names.append(str(tool[0]))
                
                print(f"Available tools: {', '.join(tool_names) if tool_names else 'None found'}")
                
                # Step 1: Call the tool directly first to verify it works
                print("\nTesting randtool directly...")
                test_result = await session.call_tool("randtool", {"input_string": "Test result:"})
                
                # Inspect and extract the actual result text safely
                print(f"Raw tool response structure: {type(test_result)}")
                test_result_text = "Unable to extract result"
                
                # Try different ways to extract the text based on possible response structures
                if hasattr(test_result, 'text'):
                    test_result_text = test_result.text
                elif hasattr(test_result, 'content'):
                    if isinstance(test_result.content, list):
                        for item in test_result.content:
                            if hasattr(item, 'text'):
                                test_result_text = item.text
                                break
                    elif hasattr(test_result.content, 'text'):
                        test_result_text = test_result.content.text
                elif isinstance(test_result, str):
                    test_result_text = test_result
                
                print(f"Direct tool call result: {test_result_text}")
                
                # Step 2: Prepare the system message that instructs Claude about the tool
                system_message = {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": """You have access to the following tools:

randtool: A tool that takes a string input and appends a random number between 1-100 to it.
Input: input_string (string) - The text to prepend to the random number
Output: The input string followed by a space and a random integer between 1 and 100

When you need to use a tool, use the following format:
<tool>
{
  "name": "tool_name",
  "input": {
    "parameter_name": "parameter_value"
  }
}
</tool>

The user will process your tool request and provide the result."""
                        }
                    ]
                }
                
                # Step 3: Create a user message that will trigger Claude to use the tool
                user_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please call the randtool with the input \"Today's lucky number is\" and then tell me what I could use this random number for."
                        }
                    ]
                }
                
                # Step 4: Send the request to Claude
                print("\nSending request to Claude...")
                response = claude.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1000,
                    system=system_message["content"][0]["text"],
                    messages=[user_message]
                )
                
                # Step 5: Process Claude's response to extract tool call if present
                claude_response = response.content[0].text
                print("\n" + "=" * 80)
                print("CLAUDE'S INITIAL RESPONSE:")
                print("=" * 80)
                print(claude_response)
                print("=" * 80)
                
                # Step 6: Parse tool calls from Claude's response using a basic approach
                tool_calls = []
                if "<tool>" in claude_response and "</tool>" in claude_response:
                    # Extract content between tool tags
                    tool_parts = claude_response.split("<tool>")
                    for part in tool_parts[1:]:  # Skip the first part before any tool
                        if "</tool>" in part:
                            tool_json = part.split("</tool>")[0].strip()
                            try:
                                tool_call = json.loads(tool_json)
                                tool_calls.append(tool_call)
                            except json.JSONDecodeError:
                                print(f"Error parsing tool call: {tool_json}")
                
                # Step 7: Execute any tool calls and send results back to Claude
                follow_up_messages = [user_message]
                
                for tool_call in tool_calls:
                    print(f"\nExecuting tool: {tool_call['name']}")
                    # Call the actual tool via MCP
                    tool_result = await session.call_tool(tool_call['name'], tool_call['input'])
                    
                    # Extract the result text safely
                    tool_result_text = "Unable to extract result"
                    
                    # Try different ways to extract the text based on possible response structures
                    if hasattr(tool_result, 'text'):
                        tool_result_text = tool_result.text
                    elif hasattr(tool_result, 'content'):
                        if isinstance(tool_result.content, list):
                            for item in tool_result.content:
                                if hasattr(item, 'text'):
                                    tool_result_text = item.text
                                    break
                        elif hasattr(tool_result.content, 'text'):
                            tool_result_text = tool_result.content.text
                    elif isinstance(tool_result, str):
                        tool_result_text = tool_result
                    
                    print(f"Tool result: {tool_result_text}")
                    
                    # Add the tool result as a message
                    follow_up_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Tool result: {tool_result_text}"
                            }
                        ]
                    })
                
                # Only make a follow-up call if we executed tools
                if tool_calls:
                    # Step 8: Send the follow-up request to Claude with tool results
                    print("\nSending follow-up request to Claude with tool results...")
                    follow_up_response = claude.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=1000,
                        system=system_message["content"][0]["text"],
                        messages=follow_up_messages
                    )
                    
                    # Print the final response
                    final_response = follow_up_response.content[0].text
                    print("\n" + "=" * 80)
                    print("CLAUDE'S FINAL RESPONSE:")
                    print("=" * 80)
                    print(final_response)
                    print("=" * 80)
                
                print("\nClient demo complete!")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

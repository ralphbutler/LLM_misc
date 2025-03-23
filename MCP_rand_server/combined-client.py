import os
import sys
import asyncio
import json
from openai import OpenAI
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, stdio_client


# Path to MCP server script
SERVER_SCRIPT_PATH = "/Users/rbutler/MCPSERVERS/rand_server/rand_server.py"

# Initialize API clients
openai_client = OpenAI()
claude_client = Anthropic()

async def test_tool_directly(session):
    """Test the randtool directly to verify it works"""
    print("\nTesting randtool directly...")
    test_result = await session.call_tool("randtool", {"input_string": "Test result:"})
    
    # Extract the result text safely
    test_result_text = "Unable to extract result"
    
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
    return test_result_text

async def run_openai_demo(session):
    """Run the GPT-4o demonstration"""
    if not openai_client:
        print("\nSkipping OpenAI GPT-4o demo: API key not available")
        return
    
    print("\n" + "=" * 80)
    print("RUNNING OPENAI GPT-4o DEMO")
    print("=" * 80)
    
    # Create tools description for OpenAI API
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "randtool",
                "description": "A tool that takes a string input and appends a random number between 1-100 to it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_string": {
                            "type": "string",
                            "description": "The text to prepend to the random number"
                        }
                    },
                    "required": ["input_string"]
                }
            }
        }
    ]
    
    # Send request to GPT-4o
    print("Sending request to GPT-4o...")
    messages = [
        {
            "role": "system", 
            "content": "You have access to a tool called 'randtool' that generates random numbers. When appropriate, use the tool to complete tasks."
        },
        {
            "role": "user", 
            "content": "Please call the randtool with the input \"Today's lucky number is\" and then tell me what I could use this random number for."
        }
    ]
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
        max_tokens=1000
    )
    
    print("\n" + "-" * 70)
    print("GPT-4o INITIAL RESPONSE:")
    print("-" * 70)
    assistant_message = response.choices[0].message
    print(f"Content: {assistant_message.content}")
    print(f"Tool calls: {assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else 'None'}")
    
    # Handle tool calls if present
    if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
        # Create a new messages array with previous messages and assistant's response
        updated_messages = messages + [assistant_message.model_dump()]
        
        # Process each tool call
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"\nExecuting tool: {function_name}")
            print(f"With arguments: {function_args}")
            
            # Execute the tool via MCP
            tool_result = await session.call_tool(function_name, function_args)
            
            # Extract the result text safely
            tool_result_text = "Unable to extract result"
            
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
            
            # Add the tool result to the messages
            updated_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": tool_result_text
            })
        
        # Send the follow-up request to GPT-4o with tool results
        print("\nSending follow-up request to GPT-4o with tool results...")
        follow_up_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=updated_messages,
            max_tokens=1000
        )
        
        # Print the final response
        final_response = follow_up_response.choices[0].message.content
        print("\n" + "-" * 70)
        print("GPT-4o FINAL RESPONSE:")
        print("-" * 70)
        print(final_response)
    
    print("OpenAI demo complete!")

async def run_claude_demo(session):
    """Run the Claude Sonnet demonstration"""
    print("\n" + "=" * 80)
    print("RUNNING CLAUDE SONNET DEMO")
    print("=" * 80)
    
    # Prepare the system message that instructs Claude about the tool
    system_message = """You have access to the following tools:

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
    
    # Create a user message that will trigger Claude to use the tool
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Please call the randtool with the input \"Today's lucky number is\" and then tell me what I could use this random number for."
            }
        ]
    }
    
    # Send the request to Claude
    print("Sending request to Claude...")
    response = claude_client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        system=system_message,
        messages=[user_message]
    )
    
    # Process Claude's response
    claude_response = response.content[0].text
    print("\n" + "-" * 70)
    print("CLAUDE'S INITIAL RESPONSE:")
    print("-" * 70)
    print(claude_response)
    
    # Parse tool calls from Claude's response
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
    
    # Execute any tool calls and send results back to Claude
    follow_up_messages = [user_message]
    
    for tool_call in tool_calls:
        print(f"\nExecuting tool: {tool_call['name']}")
        # Call the actual tool via MCP
        tool_result = await session.call_tool(tool_call['name'], tool_call['input'])
        
        # Extract the result text safely
        tool_result_text = "Unable to extract result"
        
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
        # Send the follow-up request to Claude with tool results
        print("\nSending follow-up request to Claude with tool results...")
        follow_up_response = claude_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1000,
            system=system_message,
            messages=follow_up_messages
        )
        
        # Print the final response
        final_response = follow_up_response.content[0].text
        print("\n" + "-" * 70)
        print("CLAUDE'S FINAL RESPONSE:")
        print("-" * 70)
        print(final_response)
    
    print("Claude demo complete!")

async def main():
    print("Starting Combined LLM MCP Client Demo")
    
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
                
                # List available tools
                tools = await session.list_tools()
                tool_names = []
                if isinstance(tools, list):
                    for tool in tools:
                        if hasattr(tool, 'name'):
                            tool_names.append(tool.name)
                        elif isinstance(tool, dict) and 'name' in tool:
                            tool_names.append(tool['name'])
                        elif isinstance(tool, tuple) and len(tool) > 0:
                            tool_names.append(str(tool[0]))
                
                print(f"Available tools: {', '.join(tool_names) if tool_names else 'None found'}")
                
                # Test the tool directly
                await test_tool_directly(session)
                
                # Run OpenAI demo if API key is available
                if openai_client:
                    await run_openai_demo(session)
                
                # Run Claude demo
                await run_claude_demo(session)
                
                print("\nCombined demo complete!")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

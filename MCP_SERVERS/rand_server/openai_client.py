import os
import sys
import asyncio
import json
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters, stdio_client

# OpenAI API key (should be set as environment variable)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Path to your MCP server script
SERVER_SCRIPT_PATH = "rand_server.py"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

async def main():
    print("Starting OpenAI GPT-4o MCP client to demonstrate tool usage")
    
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
                
                # Step 2: Create tools description for OpenAI API in their expected format
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
                
                # Step 3: Send request to GPT-4o
                print("\nSending request to GPT-4o...")
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
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",  # let the model decide
                    max_tokens=1000
                )
                
                print("\n" + "=" * 80)
                print("GPT-4o INITIAL RESPONSE:")
                print("=" * 80)
                assistant_message = response.choices[0].message
                print(f"Content: {assistant_message.content}")
                print(f"Tool calls: {assistant_message.tool_calls if hasattr(assistant_message, 'tool_calls') else 'None'}")
                print("=" * 80)
                
                # Step 4: Handle tool calls if present
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
                        
                        # Add the tool result to the messages
                        updated_messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": tool_result_text
                        })
                    
                    # Step 5: Send the follow-up request to GPT-4o with tool results
                    print("\nSending follow-up request to GPT-4o with tool results...")
                    follow_up_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=updated_messages,
                        max_tokens=1000
                    )
                    
                    # Print the final response
                    final_response = follow_up_response.choices[0].message.content
                    print("\n" + "=" * 80)
                    print("GPT-4o FINAL RESPONSE:")
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

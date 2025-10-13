# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

"""
01_agent_basics.py - Claude Agent SDK Core Functionality Demo

This program demonstrates:
- ClaudeSDKClient with stateful conversation
- Creating custom tools with @tool decorator
- Using built-in tools (Read, Write, Bash)
- Streaming responses with receive_response()
- Basic error handling
- ClaudeAgentOptions configuration

Run with: uv run 01_agent_basics.py
"""

import anyio
from pathlib import Path
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    CLINotFoundError,
    ProcessError,
)


# ============================================================================
# CUSTOM TOOL DEFINITION
# ============================================================================

@tool(
    name="calculate",
    description="Perform mathematical calculations. Supports basic arithmetic operations. Provide a mathematical expression like '2 + 2' or '10 * 5'.",
    input_schema={"expression": str}
)
async def calculate(args):
    """
    Custom calculator tool that evaluates mathematical expressions.

    This demonstrates how to create a custom tool using the @tool decorator.
    Tools must return a dict with 'content' key containing a list of content blocks.
    """
    try:
        expression = args.get("expression", "")
        # Note: eval() is used here for demo purposes only
        # In production, use a proper math expression parser
        result = eval(expression, {"__builtins__": {}}, {})

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Calculation: {expression} = {result}"
                }
            ]
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error calculating '{expression}': {str(e)}"
                }
            ]
        }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    """
    Main demonstration function showing core Claude Agent SDK features.
    """

    print("=" * 80)
    print("Claude Agent SDK - Core Functionality Demo")
    print("=" * 80)
    print()

    # Create a temporary directory for file operations
    work_dir = Path.cwd() / "demo_01_output"
    work_dir.mkdir(exist_ok=True)
    print(f"📁 Working directory: {work_dir}")
    print()

    # ========================================================================
    # STEP 1: Create an SDK MCP server for our custom tools
    # ========================================================================
    print("🔧 Step 1: Setting up custom tools...")

    # Create an in-process MCP server with our custom calculator tool
    # This makes the tool available to Claude
    mcp_server = create_sdk_mcp_server(
        name="math-tools",
        version="1.0.0",
        tools=[calculate]
    )
    print("   ✓ Created 'math-tools' MCP server with calculate tool")
    print()

    # ========================================================================
    # STEP 2: Configure ClaudeAgentOptions
    # ========================================================================
    print("⚙️  Step 2: Configuring agent options...")

    options = ClaudeAgentOptions(
        # System prompt defines the agent's behavior and personality
        system_prompt="You are a helpful coding assistant with access to file operations and calculation tools.",

        # Allowed tools - include both built-in and custom tools
        # Custom tools use the format: mcp__<server-name>__<tool-name>
        allowed_tools=[
            "Write",                          # Built-in: create/overwrite files
            "Read",                           # Built-in: read files
            "Bash",                           # Built-in: execute shell commands
            "mcp__math-tools__calculate"      # Custom: our calculator tool
        ],

        # MCP servers dictionary - maps server name to server instance
        mcp_servers={
            "math-tools": mcp_server
        },

        # Working directory for file operations
        cwd=str(work_dir),

        # Permission mode - 'acceptEdits' auto-accepts file modifications
        # Other options: 'default', 'plan', 'bypassPermissions'
        permission_mode="acceptEdits"
    )
    print("   ✓ Configured with custom tools and built-in file operations")
    print("   ✓ Permission mode: acceptEdits (auto-accept file changes)")
    print()

    # ========================================================================
    # STEP 3: Create a stateful conversation with ClaudeSDKClient
    # ========================================================================
    print("🤖 Step 3: Starting stateful conversation with Claude...")
    print("-" * 80)
    print()

    try:
        # Use async context manager for automatic connection/disconnection
        async with ClaudeSDKClient(options=options) as client:

            # ================================================================
            # CONVERSATION TURN 1: Use custom calculator tool
            # ================================================================
            print("💬 User: Calculate the result of (15 * 7) + 25")
            print()

            await client.query("Calculate the result of (15 * 7) + 25")

            # Stream the response using receive_response()
            # This returns an async iterator of messages
            print("🤖 Claude:")
            async for message in client.receive_response():
                # Messages can be of various types
                # For simplicity, we'll just print them
                print(f"   {message}")

            print()
            print("-" * 80)
            print()

            # ================================================================
            # CONVERSATION TURN 2: Write calculation results to a file
            # ================================================================
            print("💬 User: Write that result to a file called 'result.txt'")
            print()

            await client.query(
                "Write the calculation result to a file called 'result.txt' "
                "with a nice formatted message."
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()
            print("-" * 80)
            print()

            # ================================================================
            # CONVERSATION TURN 3: Read the file back
            # ================================================================
            print("💬 User: Read the file back to verify the content")
            print()

            await client.query("Read the file 'result.txt' and confirm its contents.")

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()
            print("-" * 80)
            print()

            # ================================================================
            # CONVERSATION TURN 4: Execute a bash command
            # ================================================================
            print("💬 User: List all files in the working directory")
            print()

            await client.query("Use the bash tool to list all files in the current directory.")

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()
            print("-" * 80)
            print()

    except CLINotFoundError:
        print("❌ Error: Claude Code CLI not found.")
        print("   Please ensure Claude Code is installed and available in your PATH.")
        print("   Visit: https://docs.claude.com/en/api/agent-sdk/overview")
        return

    except ProcessError as e:
        print(f"❌ Process Error: {e}")
        print(f"   Exit code: {e.exit_code}")
        return

    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  1. ✓ Created custom tool with @tool decorator")
    print("  2. ✓ Set up MCP server with create_sdk_mcp_server()")
    print("  3. ✓ Configured ClaudeAgentOptions with tools and permissions")
    print("  4. ✓ Used ClaudeSDKClient for stateful conversation")
    print("  5. ✓ Streamed responses with receive_response() async iterator")
    print("  6. ✓ Used built-in tools: Write, Read, Bash")
    print("  7. ✓ Maintained conversation context across multiple turns")
    print()
    print(f"📁 Output files created in: {work_dir}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the async main function
    anyio.run(main)

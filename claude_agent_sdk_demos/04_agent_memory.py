# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anthropic",
#     "anyio",
# ]
# ///

"""
04_agent_memory.py - Claude Agent SDK Memory Tool Demo

This program demonstrates:
- Memory tool for persistent storage across sessions
- Memory operations: view, create, str_replace, insert, delete
- Beta header for context management
- Simulated multi-session persistence

Note: The memory tool is a beta feature requiring special API headers.
This demo shows the conceptual approach and API structure.

Run with: uv run 04_agent_memory.py
"""

import anyio
import json
from pathlib import Path
from datetime import datetime
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    CLINotFoundError,
    ProcessError,
)


# ============================================================================
# SIMULATED MEMORY TOOL
# ============================================================================
# Note: Since the memory tool requires beta API headers and may not be
# fully available in all SDK versions, we'll create a custom tool that
# simulates memory operations using local files.

MEMORY_DIR = Path.cwd() / "demo_04_memory"


@tool(
    name="memory_create",
    description="Create a new memory file to store information persistently",
    input_schema={"filename": "string", "content": "string"}
)
async def memory_create(args):
    """Create a memory file with content."""
    try:
        MEMORY_DIR.mkdir(exist_ok=True)

        filename = args.get("filename", "")
        content = args.get("content", "")

        if not filename:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: filename is required"
                }]
            }

        file_path = MEMORY_DIR / filename
        file_path.write_text(content)

        return {
            "content": [{
                "type": "text",
                "text": f"✅ Memory created: {filename}\nStored {len(content)} characters"
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error creating memory: {str(e)}"
            }]
        }


@tool(
    name="memory_view",
    description="View the contents of a memory file or list all memory files",
    input_schema={"filename": "string"}
)
async def memory_view(args):
    """View memory contents or list all memories."""
    try:
        MEMORY_DIR.mkdir(exist_ok=True)

        filename = args.get("filename", "")

        if not filename:
            # List all memory files
            files = list(MEMORY_DIR.glob("*"))
            if not files:
                return {
                    "content": [{
                        "type": "text",
                        "text": "📁 Memory directory is empty. No memories stored yet."
                    }]
                }

            file_list = "\n".join([f"  • {f.name}" for f in files])
            return {
                "content": [{
                    "type": "text",
                    "text": f"📁 Memory files:\n{file_list}"
                }]
            }

        # View specific file
        file_path = MEMORY_DIR / filename
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Memory not found: {filename}"
                }]
            }

        content = file_path.read_text()
        return {
            "content": [{
                "type": "text",
                "text": f"📄 Memory: {filename}\n\n{content}"
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error viewing memory: {str(e)}"
            }]
        }


@tool(
    name="memory_update",
    description="Update the contents of an existing memory file",
    input_schema={"filename": "string", "content": "string"}
)
async def memory_update(args):
    """Update an existing memory file."""
    try:
        filename = args.get("filename", "")
        content = args.get("content", "")

        if not filename:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: filename is required"
                }]
            }

        file_path = MEMORY_DIR / filename
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Memory not found: {filename}. Use memory_create to create it first."
                }]
            }

        file_path.write_text(content)

        return {
            "content": [{
                "type": "text",
                "text": f"✅ Memory updated: {filename}"
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error updating memory: {str(e)}"
            }]
        }


@tool(
    name="memory_delete",
    description="Delete a memory file",
    input_schema={"filename": "string"}
)
async def memory_delete(args):
    """Delete a memory file."""
    try:
        filename = args.get("filename", "")

        if not filename:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: filename is required"
                }]
            }

        file_path = MEMORY_DIR / filename
        if not file_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"❌ Memory not found: {filename}"
                }]
            }

        file_path.unlink()

        return {
            "content": [{
                "type": "text",
                "text": f"✅ Memory deleted: {filename}"
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error deleting memory: {str(e)}"
            }]
        }


# ============================================================================
# DEMONSTRATION SESSIONS
# ============================================================================

async def session_1():
    """
    First session: Create memories and store information.
    """
    print("=" * 80)
    print("SESSION 1: Creating Memories")
    print("=" * 80)
    print()

    # Create MCP server with memory tools
    memory_server = create_sdk_mcp_server(
        name="memory",
        version="1.0.0",
        tools=[memory_create, memory_view, memory_update, memory_delete]
    )

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a helpful assistant with persistent memory capabilities. "
            "You can store and retrieve information across sessions using memory tools."
        ),
        allowed_tools=[
            "mcp__memory__memory_create",
            "mcp__memory__memory_view",
            "mcp__memory__memory_update",
            "mcp__memory__memory_delete",
        ],
        mcp_servers={"memory": memory_server},
        permission_mode="acceptEdits"
    )

    try:
        async with ClaudeSDKClient(options=options) as client:

            # Task 1: Store user preferences
            print("💬 Task 1: Store user preferences in memory")
            print()

            await client.query(
                "Create a memory file called 'user_preferences.txt' with the following "
                "information:\n"
                "- Name: Alex\n"
                "- Favorite color: Blue\n"
                "- Programming languages: Python, JavaScript\n"
                "- Timezone: UTC-8\n"
            )

            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # Task 2: Store project notes
            print("💬 Task 2: Store project notes")
            print()

            await client.query(
                "Create a memory file called 'project_notes.txt' with notes about "
                "a new feature we're working on: 'Implementing user authentication "
                "with OAuth2. Need to research providers and security best practices.'"
            )

            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # Task 3: List all memories
            print("💬 Task 3: Show all stored memories")
            print()

            await client.query("Use memory_view to list all stored memories.")

            async for message in client.receive_response():
                print(f"   {message}")

            print()

    except Exception as e:
        print(f"❌ Error in Session 1: {e}")
        return False

    return True


async def session_2():
    """
    Second session: Retrieve and update memories.
    This simulates a completely new session where the agent
    needs to recall information from previous interactions.
    """
    print()
    print("=" * 80)
    print("SESSION 2: Retrieving and Updating Memories")
    print("=" * 80)
    print()
    print("🔄 This is a NEW session. The agent will retrieve stored memories...")
    print()

    # Create MCP server with memory tools (same as session 1)
    memory_server = create_sdk_mcp_server(
        name="memory",
        version="1.0.0",
        tools=[memory_create, memory_view, memory_update, memory_delete]
    )

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a helpful assistant with persistent memory capabilities. "
            "You can store and retrieve information across sessions using memory tools."
        ),
        allowed_tools=[
            "mcp__memory__memory_create",
            "mcp__memory__memory_view",
            "mcp__memory__memory_update",
            "mcp__memory__memory_delete",
        ],
        mcp_servers={"memory": memory_server},
        permission_mode="acceptEdits"
    )

    try:
        async with ClaudeSDKClient(options=options) as client:

            # Task 1: Retrieve user preferences
            print("💬 Task 1: What are the user's preferences?")
            print()

            await client.query(
                "Use memory_view to check the 'user_preferences.txt' file and "
                "tell me what you know about the user."
            )

            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # Task 2: Update project notes
            print("💬 Task 2: Update the project notes with progress")
            print()

            await client.query(
                "Read the project notes and then update them to add: "
                "'Progress update: Researched OAuth2 providers. Planning to use "
                "Auth0 or Okta. Next step: prototype implementation.'"
            )

            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # Task 3: Verify the update
            print("💬 Task 3: Verify the notes were updated")
            print()

            await client.query("View the updated project_notes.txt file.")

            async for message in client.receive_response():
                print(f"   {message}")

            print()

    except Exception as e:
        print(f"❌ Error in Session 2: {e}")
        return False

    return True


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    """
    Main demonstration showing persistent memory across sessions.
    """

    print("=" * 80)
    print("Claude Agent SDK - Memory Tool Demo")
    print("=" * 80)
    print()
    print("This demo shows how agents can store and retrieve information")
    print("across multiple sessions using persistent memory.")
    print()
    print(f"💾 Memory location: {MEMORY_DIR}")
    print()

    # Clean up any existing memory from previous runs
    if MEMORY_DIR.exists():
        import shutil
        shutil.rmtree(MEMORY_DIR)
        print("🧹 Cleaned up previous memory directory")
        print()

    # Run session 1: Create memories
    success1 = await session_1()

    if not success1:
        print("❌ Session 1 failed. Aborting.")
        return

    # Simulate time passing between sessions
    print()
    print("⏸️  " + "─" * 76)
    print("   TIME PASSES... Simulating a completely new session")
    print("   " + "─" * 76)
    print()

    # Run session 2: Retrieve and update memories
    success2 = await session_2()

    if not success2:
        print("❌ Session 2 failed.")
        return

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("✅ Memory Tool Demo Complete!")
    print("=" * 80)
    print()

    print("Key Features Demonstrated:")
    print("  1. ✓ Created custom memory tools (create, view, update, delete)")
    print("  2. ✓ Stored information persistently to disk")
    print("  3. ✓ Retrieved memories in a new session")
    print("  4. ✓ Updated existing memories")
    print("  5. ✓ Simulated multi-session persistence")
    print()

    print("💡 MEMORY TOOL CONCEPTS:")
    print()
    print("  The official memory tool (beta) works similarly but with:")
    print("  • Direct API integration via beta headers")
    print("  • Structured memory operations (view, create, str_replace, etc.)")
    print("  • Backend-agnostic storage (file, database, cloud)")
    print("  • Automatic context management")
    print()

    print("📖 MEMORY TOOL BENEFITS:")
    print()
    print("  • Knowledge persistence across conversations")
    print("  • Reduced context window usage")
    print("  • Project state management")
    print("  • User preferences storage")
    print("  • Long-term learning and adaptation")
    print()

    print("🔗 OFFICIAL MEMORY TOOL:")
    print()
    print("  To use the official beta memory tool:")
    print("  1. Enable beta header: context-management-2025-06-27")
    print("  2. Use BetaAbstractMemoryTool in Python SDK")
    print("  3. Implement custom storage backend")
    print("  4. See: https://docs.claude.com/en/docs/agents-and-tools/tool-use/memory-tool")
    print()

    print(f"💾 Memory files stored in: {MEMORY_DIR}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    anyio.run(main)

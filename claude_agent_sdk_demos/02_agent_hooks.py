# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

"""
02_agent_hooks.py - Claude Agent SDK Hooks Demo

This program demonstrates:
- PreToolUse hooks for validation and blocking
- PostToolUse hooks for logging and monitoring
- HookMatcher for tool-specific hooks
- Permission decisions (allow/deny)
- Hook context and tool inspection

Run with: uv run 02_agent_hooks.py
"""

import anyio
from datetime import datetime
from pathlib import Path
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    CLINotFoundError,
    ProcessError,
)


# ============================================================================
# HOOK DEFINITIONS
# ============================================================================

async def pre_bash_validator(input_data, tool_use_id, context):
    """
    PreToolUse hook that validates and blocks dangerous bash commands.

    This hook runs BEFORE the Bash tool executes, allowing us to:
    - Inspect the command
    - Block dangerous operations
    - Log security warnings
    - Provide custom error messages

    Returns a dict with permission decision to allow or deny execution.
    """
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Only process Bash tool
    if tool_name != "Bash":
        return {}

    command = tool_input.get("command", "")

    print(f"   🔍 [PreToolUse] Validating bash command: {command}")

    # Define dangerous command patterns
    dangerous_patterns = [
        "rm -rf",
        "sudo rm",
        "mkfs",
        "> /dev/",
        "dd if=",
        ":(){ :|:& };:",  # Fork bomb
    ]

    # Check for dangerous patterns
    for pattern in dangerous_patterns:
        if pattern in command:
            print(f"   🚫 [PreToolUse] BLOCKED: Command contains dangerous pattern '{pattern}'")
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Command blocked for safety: contains dangerous pattern '{pattern}'. "
                        f"This command could cause system damage or data loss."
                    ),
                }
            }

    print(f"   ✅ [PreToolUse] Command approved")
    return {}


async def post_tool_logger(output_data, tool_use_id, context):
    """
    PostToolUse hook that logs all tool executions.

    This hook runs AFTER a tool completes, allowing us to:
    - Log tool execution details
    - Monitor performance
    - Track tool usage patterns
    - Send notifications or alerts

    Returns an empty dict (no modifications needed).
    """
    tool_name = output_data.get("tool_name", "")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"   📝 [PostToolUse] Tool '{tool_name}' completed at {timestamp}")

    # You could add more sophisticated logging here:
    # - Write to log files
    # - Send metrics to monitoring systems
    # - Trigger notifications
    # - Analyze tool usage patterns

    return {}


async def post_file_operation_logger(output_data, tool_use_id, context):
    """
    PostToolUse hook specifically for file operations.

    This demonstrates using HookMatcher to target specific tools.
    Only runs for Write, Edit, and similar file modification tools.
    """
    tool_name = output_data.get("tool_name", "")
    tool_input = output_data.get("tool_input", {})

    print(f"   💾 [PostToolUse] File operation logged: {tool_name}")

    # Extract file path if available
    file_path = tool_input.get("file_path", "unknown")
    if file_path != "unknown":
        print(f"      → File: {file_path}")

    return {}


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    """
    Main demonstration showing PreToolUse and PostToolUse hooks.
    """

    print("=" * 80)
    print("Claude Agent SDK - Hooks Demo")
    print("=" * 80)
    print()

    # Create working directory
    work_dir = Path.cwd() / "demo_02_hooks"
    work_dir.mkdir(exist_ok=True)
    print(f"📁 Working directory: {work_dir}")
    print()

    # ========================================================================
    # Configure hooks with HookMatcher
    # ========================================================================
    print("🔧 Setting up hooks...")
    print()

    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant with access to bash and file operations.",

        allowed_tools=[
            "Bash",
            "Write",
            "Read",
        ],

        # Hooks configuration - maps hook types to lists of hook matchers
        hooks={
            # PreToolUse hooks run BEFORE tool execution
            "PreToolUse": [
                HookMatcher(
                    matcher="Bash",  # Only match Bash tool
                    hooks=[pre_bash_validator]
                ),
            ],

            # PostToolUse hooks run AFTER tool execution
            "PostToolUse": [
                # Generic logger for all tools
                HookMatcher(
                    matcher=".*",  # Match all tools with regex
                    hooks=[post_tool_logger]
                ),

                # Specific logger for file operations
                HookMatcher(
                    matcher="Write|Edit",  # Match Write OR Edit tools
                    hooks=[post_file_operation_logger]
                ),
            ],
        },

        cwd=str(work_dir),
        permission_mode="acceptEdits"
    )

    print("   ✓ Configured PreToolUse hook for Bash validation")
    print("   ✓ Configured PostToolUse hook for general logging")
    print("   ✓ Configured PostToolUse hook for file operation logging")
    print()

    # ========================================================================
    # Test the hooks with various commands
    # ========================================================================

    try:
        async with ClaudeSDKClient(options=options) as client:

            # ================================================================
            # TEST 1: Safe bash command (should succeed)
            # ================================================================
            print("=" * 80)
            print("TEST 1: Safe Bash Command")
            print("=" * 80)
            print()
            print("💬 User: Use bash to list the current directory contents")
            print()

            await client.query("Use the bash tool to run 'ls -la' to list the current directory.")

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TEST 2: Dangerous bash command (should be blocked)
            # ================================================================
            print("=" * 80)
            print("TEST 2: Dangerous Bash Command (Should Be Blocked)")
            print("=" * 80)
            print()
            print("💬 User: Try to run a dangerous rm -rf command")
            print()

            await client.query(
                "Use bash to run 'rm -rf /tmp/test' to remove a test directory."
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TEST 3: File write operation (triggers file operation logger)
            # ================================================================
            print("=" * 80)
            print("TEST 3: File Write Operation")
            print("=" * 80)
            print()
            print("💬 User: Create a test file")
            print()

            await client.query(
                "Create a file called 'test.txt' with the content 'Hello from hooks demo!'"
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TEST 4: Read operation (only triggers general logger)
            # ================================================================
            print("=" * 80)
            print("TEST 4: File Read Operation")
            print("=" * 80)
            print()
            print("💬 User: Read the test file")
            print()

            await client.query("Read the file 'test.txt' and show me its contents.")

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

    except CLINotFoundError:
        print("❌ Error: Claude Code CLI not found.")
        print("   Please ensure Claude Code is installed.")
        return

    except ProcessError as e:
        print(f"❌ Process Error: {e}")
        return

    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("✅ Hooks Demo Complete!")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  1. ✓ PreToolUse hook for command validation")
    print("  2. ✓ Blocked dangerous bash commands")
    print("  3. ✓ PostToolUse hook for logging all tools")
    print("  4. ✓ PostToolUse hook for specific file operations")
    print("  5. ✓ HookMatcher with regex patterns")
    print("  6. ✓ Permission decisions (allow/deny)")
    print("  7. ✓ Custom error messages for blocked operations")
    print()
    print("Hook Benefits:")
    print("  • Security: Block dangerous operations before execution")
    print("  • Monitoring: Track all tool usage in real-time")
    print("  • Auditing: Log file operations for compliance")
    print("  • Control: Intercept and modify tool behavior")
    print()
    print(f"📁 Output files created in: {work_dir}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    anyio.run(main)

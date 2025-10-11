# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

"""
03_agent_permissions.py - Claude Agent SDK Permission Modes Demo

This program demonstrates:
- default mode: Standard permission prompts
- acceptEdits mode: Auto-accept file modifications
- plan mode: Read-only analysis without execution
- bypassPermissions mode: Skip all permission checks

Each mode is tested with the same task to show behavioral differences.

Run with: uv run 03_agent_permissions.py
"""

import anyio
from pathlib import Path
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    CLINotFoundError,
    ProcessError,
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def test_permission_mode(mode_name: str, permission_mode: str, work_dir: Path):
    """
    Test a specific permission mode with a standard task.

    Args:
        mode_name: Human-readable name for display
        permission_mode: The actual permission mode value
        work_dir: Working directory for file operations
    """

    print("=" * 80)
    print(f"Testing Permission Mode: {mode_name}")
    print("=" * 80)
    print()
    print(f"📋 Mode: {permission_mode}")
    print()

    # Configure options for this specific mode
    options = ClaudeAgentOptions(
        system_prompt="You are a helpful assistant that can read and write files.",
        allowed_tools=["Write", "Read", "Bash"],
        permission_mode=permission_mode,
        cwd=str(work_dir)
    )

    # Describe the mode
    mode_descriptions = {
        "default": (
            "In DEFAULT mode, Claude will prompt for permission on first use of each tool.\n"
            "This is the safest mode for interactive use."
        ),
        "acceptEdits": (
            "In ACCEPT_EDITS mode, file modifications are automatically accepted.\n"
            "This is useful for automation where you trust the agent's actions."
        ),
        "plan": (
            "In PLAN mode, Claude can analyze but not modify files or execute commands.\n"
            "This is ideal for read-only analysis and planning."
        ),
        "bypassPermissions": (
            "In BYPASS_PERMISSIONS mode, all permission checks are skipped.\n"
            "⚠️  Use with extreme caution - only in controlled environments!"
        ),
    }

    print(f"ℹ️  {mode_descriptions.get(permission_mode, 'Unknown mode')}")
    print()

    try:
        async with ClaudeSDKClient(options=options) as client:

            # ================================================================
            # Standard task: Create a file with system information
            # ================================================================
            print("💬 Task: Create a file with system information")
            print()

            task = (
                "Create a file called 'system_info.txt' that contains:\n"
                "1. The current date and time\n"
                "2. A list of files in the current directory\n"
                "3. A friendly message\n\n"
                "Use bash to get the date and list files, then write to the file."
            )

            await client.query(task)

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # Verification: Try to read the file back
            # ================================================================
            print("💬 Verification: Read the file back")
            print()

            await client.query("Read the 'system_info.txt' file and show me its contents.")

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

    except CLINotFoundError:
        print("❌ Error: Claude Code CLI not found.")
        return False

    except ProcessError as e:
        print(f"❌ Process Error: {e}")
        print(f"   This might be expected in '{permission_mode}' mode.")
        print()
        return False

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        print()
        return False

    return True


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    """
    Main demonstration comparing all four permission modes.
    """

    print("=" * 80)
    print("Claude Agent SDK - Permission Modes Comparison")
    print("=" * 80)
    print()
    print("This demo shows how different permission modes affect agent behavior.")
    print("Each mode will be tested with the same task: creating a system info file.")
    print()

    # Create separate working directories for each mode
    base_dir = Path.cwd() / "demo_03_permissions"
    base_dir.mkdir(exist_ok=True)

    # ========================================================================
    # MODE 1: default
    # ========================================================================
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MODE 1: DEFAULT".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    mode1_dir = base_dir / "mode_default"
    mode1_dir.mkdir(exist_ok=True)

    await test_permission_mode("Default", "default", mode1_dir)

    # ========================================================================
    # MODE 2: acceptEdits
    # ========================================================================
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MODE 2: ACCEPT EDITS".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    mode2_dir = base_dir / "mode_acceptEdits"
    mode2_dir.mkdir(exist_ok=True)

    await test_permission_mode("Accept Edits", "acceptEdits", mode2_dir)

    # ========================================================================
    # MODE 3: plan
    # ========================================================================
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MODE 3: PLAN (Read-Only)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    mode3_dir = base_dir / "mode_plan"
    mode3_dir.mkdir(exist_ok=True)

    await test_permission_mode("Plan", "plan", mode3_dir)

    # ========================================================================
    # MODE 4: bypassPermissions
    # ========================================================================
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "MODE 4: BYPASS PERMISSIONS ⚠️".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    mode4_dir = base_dir / "mode_bypass"
    mode4_dir.mkdir(exist_ok=True)

    await test_permission_mode("Bypass Permissions", "bypassPermissions", mode4_dir)

    # ========================================================================
    # SUMMARY AND COMPARISON
    # ========================================================================
    print()
    print("=" * 80)
    print("✅ Permission Modes Comparison Complete!")
    print("=" * 80)
    print()

    print("📊 SUMMARY OF PERMISSION MODES:")
    print()

    print("┌─────────────────────┬──────────────────────────────────────────────────────┐")
    print("│ Mode                │ Behavior                                              │")
    print("├─────────────────────┼──────────────────────────────────────────────────────┤")
    print("│ default             │ Prompts for permission on first tool use             │")
    print("│                     │ Best for: Interactive development                    │")
    print("├─────────────────────┼──────────────────────────────────────────────────────┤")
    print("│ acceptEdits         │ Auto-accepts file modifications                      │")
    print("│                     │ Best for: Trusted automation scripts                 │")
    print("├─────────────────────┼──────────────────────────────────────────────────────┤")
    print("│ plan                │ Read-only analysis, no execution                     │")
    print("│                     │ Best for: Planning and analysis tasks                │")
    print("├─────────────────────┼──────────────────────────────────────────────────────┤")
    print("│ bypassPermissions   │ Skips ALL permission checks                          │")
    print("│                     │ Best for: Controlled/sandboxed environments only ⚠️   │")
    print("└─────────────────────┴──────────────────────────────────────────────────────┘")
    print()

    print("💡 KEY TAKEAWAYS:")
    print()
    print("  1. Use 'default' for interactive sessions where you want control")
    print("  2. Use 'acceptEdits' for automation when you trust the agent")
    print("  3. Use 'plan' when you only want analysis without changes")
    print("  4. Use 'bypassPermissions' only in sandboxed/test environments")
    print()

    print("⚠️  SECURITY CONSIDERATIONS:")
    print()
    print("  • Always use the most restrictive mode that meets your needs")
    print("  • 'bypassPermissions' should never be used in production")
    print("  • Consider using hooks (see 02_agent_hooks.py) for additional safety")
    print("  • Audit tool usage regularly in automated systems")
    print()

    print(f"📁 Test outputs created in: {base_dir}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    anyio.run(main)

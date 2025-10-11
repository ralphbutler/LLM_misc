# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

"""
07_agent_websearch.py - Claude Agent SDK WebSearch & WebFetch Demo

This program demonstrates:
- WebSearch built-in tool for internet searches
- WebFetch for retrieving specific URLs
- Research synthesis and report generation
- File output with Write tool

Simple workflow:
1. Search the web for information
2. Fetch detailed content from a specific URL
3. Synthesize findings into a concise report
4. Save report to file

Run with: uv run 07_agent_websearch.py
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
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    """
    Simple web research demonstration with WebSearch and WebFetch.
    """

    print("=" * 80)
    print("Claude Agent SDK - WebSearch & WebFetch Demo")
    print("=" * 80)
    print()

    print("This demo shows how to:")
    print("  • Use WebSearch to find information on the internet")
    print("  • Use WebFetch to retrieve detailed content from URLs")
    print("  • Synthesize research findings into a report")
    print("  • Save results to a markdown file")
    print()

    # Create output directory
    work_dir = Path.cwd() / "demo_07_research"
    work_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {work_dir}")
    print()

    # ========================================================================
    # Configure research agent
    # ========================================================================
    print("🔧 Configuring research agent...")
    print()

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a concise research assistant. Your goal is to:\n"
            "1. Search for information using WebSearch\n"
            "2. Fetch detailed content from relevant URLs using WebFetch\n"
            "3. Create a brief, well-structured markdown report\n"
            "4. Save the report using the Write tool\n\n"
            "Keep reports concise (2-3 pages maximum) and well-cited."
        ),

        # Enable web tools and file operations
        allowed_tools=[
            "WebSearch",  # Search the internet
            "WebFetch",   # Fetch specific URLs
            "Write",      # Save research findings
        ],

        # Use acceptEdits to auto-approve file saves
        permission_mode="acceptEdits",

        cwd=str(work_dir),
    )

    print("   ✓ WebSearch and WebFetch tools enabled")
    print("   ✓ File operations configured")
    print()

    # ========================================================================
    # Research Task: Simple single-query research
    # ========================================================================

    try:
        async with ClaudeSDKClient(options=options) as client:

            print("=" * 80)
            print("RESEARCH TASK: Claude Agent SDK Overview")
            print("=" * 80)
            print()

            print("💬 User: Research and create report on Claude Agent SDK")
            print()

            await client.query(
                "Please research the Claude Agent SDK by Anthropic.\n\n"
                "STEPS:\n"
                "1. Use WebSearch to find information about the Claude Agent SDK\n"
                "2. Use WebFetch to get detailed information from the official Anthropic "
                "documentation or blog post (if you find a good URL)\n"
                "3. Create a concise markdown report (2-3 pages) covering:\n"
                "   - What is the Claude Agent SDK?\n"
                "   - Key features and capabilities\n"
                "   - Main use cases\n"
                "   - How it differs from the Anthropic API\n"
                "   - Sources cited\n\n"
                "4. CRITICAL: Use the Write tool to save the report.\n"
                "   File name: 'claude_agent_sdk_report.md'\n"
                "   DO NOT use /tmp/ - save it in the current working directory.\n"
                "   Just use the filename without any path prefix.\n\n"
                "Keep it concise and well-structured. Make sure to create the file in the current directory!"
            )

            print("🤖 Research Agent:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

    except CLINotFoundError:
        print("❌ Error: Claude Code CLI not found.")
        print("   Please ensure Claude Code is installed.")
        return

    except ProcessError as e:
        print(f"❌ Process Error: {e}")
        print(f"   Exit code: {e.exit_code}")
        return

    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("✅ WebSearch & WebFetch Demo Complete!")
    print("=" * 80)
    print()

    print("Key Features Demonstrated:")
    print("  1. ✓ WebSearch - Search the internet for information")
    print("  2. ✓ WebFetch - Retrieve detailed content from specific URLs")
    print("  3. ✓ Research synthesis - Analyze and summarize findings")
    print("  4. ✓ File output - Save reports with Write tool")
    print()

    print("🌐 WEB TOOLS:")
    print()
    print("  WebSearch:")
    print("  • Search the internet for current information")
    print("  • Get snippets and links from search results")
    print("  • Find relevant sources quickly")
    print()
    print("  WebFetch:")
    print("  • Retrieve full content from specific URLs")
    print("  • Parse and analyze web pages")
    print("  • Extract detailed information")
    print("  • Follow up on search results for deeper research")
    print()

    print("💡 USE CASES:")
    print()
    print("  • Quick research on technical topics")
    print("  • Fact-checking and verification")
    print("  • Gathering information for documentation")
    print("  • Market and competitive research")
    print("  • News monitoring and summarization")
    print()

    print("⚠️  NOTES:")
    print()
    print("  • Requires internet connectivity")
    print("  • Some URLs may be inaccessible or rate-limited")
    print("  • Always verify critical information")
    print("  • Typical cost: ~$0.20-0.40 per run")
    print("  • Typical duration: 2-3 minutes")
    print()

    print(f"📁 Research output saved in: {work_dir}")
    print()

    # Show any generated files
    files = list(work_dir.glob("*.md"))
    if files:
        print("📄 Generated files:")
        for file in files:
            print(f"   • {file.name}")
        print()
    else:
        print("⚠️  No files generated - agent may not have used Write tool")
        print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    anyio.run(main)

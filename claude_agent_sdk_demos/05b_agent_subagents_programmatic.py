# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

"""
05b_agent_subagents_programmatic.py - Claude Agent SDK Programmatic Subagents Demo

This program demonstrates:
- Programmatic subagent definition using AgentDefinition
- No filesystem dependencies - pure Python configuration
- Runtime agent creation and composition
- Same capabilities as filesystem approach but more flexible

This is the PROGRAMMATIC alternative to 05_agent_subagents.py

Key differences from 05_agent_subagents.py:
- Uses AgentDefinition objects instead of .claude/agents/*.md files
- No filesystem I/O for agent definitions
- Better for dynamic/runtime agent generation
- Easier to test and version control

Run with: uv run 05b_agent_subagents_programmatic.py
"""

import anyio
from pathlib import Path
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    CLINotFoundError,
    ProcessError,
)


# ============================================================================
# SAMPLE CODE FOR DEMONSTRATION
# ============================================================================

def create_sample_code():
    """Create sample code files for the subagents to work with."""

    samples_dir = Path.cwd() / "demo_05b_sample_code"
    samples_dir.mkdir(exist_ok=True)

    # Sample Python function
    sample_py = samples_dir / "calculator.py"
    sample_py.write_text("""def calculate(a, b, operation):
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        return a / b
    else:
        return None
""")

    return samples_dir


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    """
    Main demonstration showing programmatic subagent creation.
    """

    print("=" * 80)
    print("Claude Agent SDK - Programmatic Subagents Demo")
    print("=" * 80)
    print()

    # ========================================================================
    # SETUP: Create sample code
    # ========================================================================
    print("📝 Setup: Creating sample code files...")
    print()

    samples_dir = create_sample_code()
    print(f"   ✓ Created sample code in: {samples_dir}")
    print()

    # ========================================================================
    # DEFINE SUBAGENTS PROGRAMMATICALLY
    # ========================================================================
    print("🔧 Setup: Defining subagents programmatically with AgentDefinition...")
    print()

    # Subagent 1: Code Reviewer
    code_reviewer = AgentDefinition(
        description="Expert code review specialist for quality, security, and best practices analysis",
        prompt="""You are a meticulous code reviewer with expertise in:
- Code quality and maintainability
- Security vulnerabilities and common pitfalls
- Best practices and design patterns
- Performance optimization opportunities
- Documentation completeness

When reviewing code:
1. Start by understanding the overall structure
2. Identify potential issues and improvements
3. Provide specific, actionable feedback
4. Suggest concrete improvements with examples
5. Highlight both strengths and areas for improvement

Focus on being constructive and educational in your feedback.""",
        tools=["Read", "Grep", "Glob"],
        model="sonnet"
    )

    # Subagent 2: Documentation Writer
    docs_writer = AgentDefinition(
        description="Technical documentation specialist for creating clear, comprehensive documentation",
        prompt="""You are a technical documentation expert who excels at:
- Writing clear, concise documentation
- Creating helpful code examples
- Explaining complex concepts simply
- Structuring information logically
- Using appropriate formatting and style

When writing documentation:
1. Understand the purpose and audience
2. Structure content with clear headings
3. Provide relevant code examples
4. Use consistent formatting and terminology
5. Include practical usage instructions

Write in a friendly, accessible style that balances technical accuracy with readability.""",
        tools=["Read", "Write", "Glob"],
        model="sonnet"
    )

    # Subagent 3: Test Engineer
    test_engineer = AgentDefinition(
        description="Testing specialist for creating comprehensive test suites and test plans",
        prompt="""You are a test engineering expert focused on:
- Test strategy and planning
- Writing comprehensive test cases
- Unit, integration, and end-to-end testing
- Test automation best practices
- Coverage analysis

When creating tests:
1. Understand the functionality to test
2. Identify edge cases and error conditions
3. Write clear, maintainable test code
4. Follow testing frameworks best practices
5. Ensure good coverage of critical paths

Prioritize test clarity, maintainability, and comprehensive coverage.""",
        tools=["Read", "Write", "Bash"],
        model="sonnet"
    )

    print("   ✓ Created subagent: code-reviewer")
    print("     Type: AgentDefinition (programmatic)")
    print("   ✓ Created subagent: docs-writer")
    print("     Type: AgentDefinition (programmatic)")
    print("   ✓ Created subagent: test-engineer")
    print("     Type: AgentDefinition (programmatic)")
    print()

    # ========================================================================
    # CONFIGURE MAIN AGENT WITH PROGRAMMATIC SUBAGENTS
    # ========================================================================
    print("⚙️  Configuring main orchestrator agent with programmatic subagents...")
    print()

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are an orchestrator agent that coordinates specialized subagents. "
            "You have access to:\n"
            "- code-reviewer: For code quality and security review\n"
            "- docs-writer: For creating documentation\n"
            "- test-engineer: For creating tests\n\n"
            "Delegate tasks to the appropriate subagent based on the request. "
            "When delegating, explicitly mention which subagent should handle the task."
        ),

        # Main agent has access to basic tools and can spawn subagents
        allowed_tools=["Read", "Write", "Glob"],

        # Programmatic subagent definitions using AgentDefinition
        agents={
            "code-reviewer": code_reviewer,
            "docs-writer": docs_writer,
            "test-engineer": test_engineer
        },

        permission_mode="acceptEdits",
        cwd=str(Path.cwd())
    )

    print("   ✓ Main agent configured with 3 programmatic subagents")
    print("   ✓ No filesystem dependencies required")
    print()

    # ========================================================================
    # DEMONSTRATION: Task delegation to subagents
    # ========================================================================

    try:
        async with ClaudeSDKClient(options=options) as client:

            # ================================================================
            # TASK 1: Code Review
            # ================================================================
            print("=" * 80)
            print("TASK 1: Code Review (Delegating to code-reviewer subagent)")
            print("=" * 80)
            print()

            print("💬 User: Please review the calculator.py file for quality and security")
            print()

            await client.query(
                "Please review the file 'demo_05b_sample_code/calculator.py' for code quality, "
                "security issues, and best practices. Have the code-reviewer subagent "
                "handle this task."
            )

            print("🤖 Main Agent → code-reviewer subagent:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TASK 2: Documentation
            # ================================================================
            print("=" * 80)
            print("TASK 2: Documentation (Delegating to docs-writer subagent)")
            print("=" * 80)
            print()

            print("💬 User: Create comprehensive documentation for calculator.py")
            print()

            await client.query(
                "Create comprehensive documentation for the calculator.py file in demo_05b_sample_code/. "
                "Have the docs-writer subagent create a README.md file in the "
                "demo_05b_sample_code directory with usage examples and API documentation."
            )

            print("🤖 Main Agent → docs-writer subagent:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TASK 3: Test Creation
            # ================================================================
            print("=" * 80)
            print("TASK 3: Test Suite (Delegating to test-engineer subagent)")
            print("=" * 80)
            print()

            print("💬 User: Create comprehensive tests for calculator.py")
            print()

            await client.query(
                "Create a comprehensive test suite for calculator.py in demo_05b_sample_code/. "
                "Have the test-engineer subagent create test_calculator.py with "
                "unit tests covering all operations and edge cases."
            )

            print("🤖 Main Agent → test-engineer subagent:")
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
        import traceback
        traceback.print_exc()
        return

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 80)
    print("✅ Programmatic Subagents Demo Complete!")
    print("=" * 80)
    print()

    print("Key Features Demonstrated:")
    print("  1. ✓ Created subagents with AgentDefinition objects")
    print("  2. ✓ No filesystem dependencies - pure Python configuration")
    print("  3. ✓ Defined specialized subagents:")
    print("      • code-reviewer: Quality and security analysis")
    print("      • docs-writer: Documentation creation")
    print("      • test-engineer: Test suite development")
    print("  4. ✓ Delegated tasks from main agent to subagents")
    print("  5. ✓ Demonstrated isolated contexts for each subagent")
    print()

    print("💡 PROGRAMMATIC vs FILESYSTEM APPROACH:")
    print()
    print("  PROGRAMMATIC (AgentDefinition - this demo):")
    print("  ✓ No filesystem I/O required")
    print("  ✓ Better for dynamic/runtime agent generation")
    print("  ✓ Easier to test and version control")
    print("  ✓ Can be constructed from configs, APIs, databases")
    print("  ✓ Fully type-safe with IDE support")
    print()
    print("  FILESYSTEM (setting_sources - see 05_agent_subagents.py):")
    print("  ✓ Easy to edit without code changes")
    print("  ✓ Shareable across team via .claude/ directory")
    print("  ✓ Integrates with Claude Code CLI conventions")
    print("  ✓ Good for stable, project-level agents")
    print("  ✓ Human-readable YAML + Markdown format")
    print()

    print("🔀 HYBRID APPROACH:")
    print()
    print("  You can use BOTH approaches together:")
    print()
    print("  options = ClaudeAgentOptions(")
    print("      agents={")
    print("          'dynamic-agent': AgentDefinition(...)  # Runtime agent")
    print("      },")
    print("      setting_sources=['project']  # Load from .claude/agents/")
    print("  )")
    print()

    print("📊 WHEN TO USE EACH:")
    print()
    print("  Use PROGRAMMATIC when:")
    print("  • Generating agents dynamically based on user input")
    print("  • Building agents from external configs/APIs")
    print("  • Writing automated tests")
    print("  • Need runtime flexibility")
    print()
    print("  Use FILESYSTEM when:")
    print("  • Defining stable, project-level agents")
    print("  • Sharing agents across team")
    print("  • Non-developers need to modify agents")
    print("  • Integrating with Claude Code conventions")
    print()

    print("🔧 AgentDefinition Parameters:")
    print()
    print("  AgentDefinition(")
    print("      description='What the agent does',")
    print("      prompt='System prompt defining behavior',")
    print("      tools=['Read', 'Write', ...],  # Optional")
    print("      model='sonnet'  # or 'opus', 'haiku', 'inherit'")
    print("  )")
    print()

    print("📖 BEST PRACTICES:")
    print()
    print("  1. Keep subagent prompts focused and specific")
    print("  2. Limit tools to only what each subagent needs")
    print("  3. Use clear, descriptive names and descriptions")
    print("  4. Consider subagent granularity (not too broad, not too narrow)")
    print("  5. Test subagents individually before integration")
    print("  6. Choose programmatic vs filesystem based on use case")
    print()

    print(f"📁 Output files created in: {samples_dir}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    anyio.run(main)

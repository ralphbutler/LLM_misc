# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

"""
05_agent_subagents.py - Claude Agent SDK Subagents Demo

This program demonstrates:
- Creating subagent definitions in .claude/agents/
- YAML frontmatter structure for subagent configuration
- Automatic task delegation to specialized subagents
- Isolated contexts for different tasks
- Programmatic subagent definition

Subagents enable:
- Parallelization: Multiple subagents working simultaneously
- Context isolation: Each subagent has its own context window
- Specialization: Task-specific agents with focused tools

Run with: uv run 05_agent_subagents.py
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
# SUBAGENT SETUP
# ============================================================================

def setup_subagents():
    """
    Create .claude/agents/ directory structure and subagent definition files.

    Subagents are defined as Markdown files with YAML frontmatter containing:
    - name: Unique identifier for the subagent
    - description: What the subagent does (used for automatic delegation)
    - tools: List of allowed tools for this subagent
    - System prompt: Instructions defining the subagent's behavior
    """

    agents_dir = Path.cwd() / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # SUBAGENT 1: Code Reviewer
    # ========================================================================
    code_reviewer_content = """---
name: code-reviewer
description: Expert code review specialist for quality, security, and best practices analysis
tools: Read, Grep, Glob
---

You are a meticulous code reviewer with expertise in:
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

Focus on being constructive and educational in your feedback.
"""

    code_reviewer_path = agents_dir / "code-reviewer.md"
    code_reviewer_path.write_text(code_reviewer_content)

    # ========================================================================
    # SUBAGENT 2: Documentation Writer
    # ========================================================================
    docs_writer_content = """---
name: docs-writer
description: Technical documentation specialist for creating clear, comprehensive documentation
tools: Read, Write, Glob
---

You are a technical documentation expert who excels at:
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

Write in a friendly, accessible style that balances technical accuracy with readability.
"""

    docs_writer_path = agents_dir / "docs-writer.md"
    docs_writer_path.write_text(docs_writer_content)

    # ========================================================================
    # SUBAGENT 3: Test Engineer
    # ========================================================================
    test_engineer_content = """---
name: test-engineer
description: Testing specialist for creating comprehensive test suites and test plans
tools: Read, Write, Bash
---

You are a test engineering expert focused on:
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

Prioritize test clarity, maintainability, and comprehensive coverage.
"""

    test_engineer_path = agents_dir / "test-engineer.md"
    test_engineer_path.write_text(test_engineer_content)

    return {
        "code-reviewer": code_reviewer_path,
        "docs-writer": docs_writer_path,
        "test-engineer": test_engineer_path,
    }


# ============================================================================
# SAMPLE CODE FOR DEMONSTRATION
# ============================================================================

def create_sample_code():
    """Create sample code files for the subagents to work with."""

    samples_dir = Path.cwd() / "demo_05_sample_code"
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
    Main demonstration showing subagent creation and delegation.
    """

    print("=" * 80)
    print("Claude Agent SDK - Subagents Demo")
    print("=" * 80)
    print()

    # ========================================================================
    # SETUP: Create subagent definitions and sample code
    # ========================================================================
    print("🔧 Setup: Creating subagent definitions...")
    print()

    subagents = setup_subagents()

    for name, path in subagents.items():
        print(f"   ✓ Created subagent: {name}")
        print(f"     Location: {path}")

    print()

    print("📝 Setup: Creating sample code files...")
    print()

    samples_dir = create_sample_code()
    print(f"   ✓ Created sample code in: {samples_dir}")
    print()

    # ========================================================================
    # CONFIGURE MAIN AGENT
    # ========================================================================
    print("⚙️  Configuring main orchestrator agent...")
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

        # Load subagent definitions from filesystem
        # Note: setting_sources enables loading from .claude directory
        setting_sources=["project"],

        permission_mode="acceptEdits",
        cwd=str(Path.cwd())
    )

    print("   ✓ Main agent configured with subagent access")
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
                "Please review the file 'sample_code/calculator.py' for code quality, "
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
                "Create comprehensive documentation for the calculator.py file. "
                "Have the docs-writer subagent create a README.md file in the "
                "sample_code directory with usage examples and API documentation."
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
                "Create a comprehensive test suite for calculator.py. "
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
    print("✅ Subagents Demo Complete!")
    print("=" * 80)
    print()

    print("Key Features Demonstrated:")
    print("  1. ✓ Created subagent definitions with YAML frontmatter")
    print("  2. ✓ Set up .claude/agents/ directory structure")
    print("  3. ✓ Defined specialized subagents:")
    print("      • code-reviewer: Quality and security analysis")
    print("      • docs-writer: Documentation creation")
    print("      • test-engineer: Test suite development")
    print("  4. ✓ Delegated tasks from main agent to subagents")
    print("  5. ✓ Demonstrated isolated contexts for each subagent")
    print()

    print("💡 SUBAGENT BENEFITS:")
    print()
    print("  • Specialization: Each subagent focuses on specific tasks")
    print("  • Context Isolation: Subagents have separate context windows")
    print("  • Parallelization: Multiple subagents can work simultaneously")
    print("  • Maintainability: Easy to update individual subagent behavior")
    print("  • Scalability: Add new subagents without modifying main agent")
    print()

    print("📁 SUBAGENT FILES:")
    print()
    print("  Filesystem-based subagents:")
    for name, path in subagents.items():
        print(f"    • {name}: {path}")
    print()

    print("🔄 PROGRAMMATIC SUBAGENTS:")
    print()
    print("  You can also define subagents programmatically using the 'agents'")
    print("  parameter in ClaudeAgentOptions:")
    print()
    print("  agents = {")
    print("      'sql-expert': {")
    print("          'description': 'SQL query optimization specialist',")
    print("          'prompt': 'You are an expert in SQL...',")
    print("          'tools': ['Read', 'Bash']")
    print("      }")
    print("  }")
    print()

    print("📖 BEST PRACTICES:")
    print()
    print("  1. Keep subagent prompts focused and specific")
    print("  2. Limit tools to only what each subagent needs")
    print("  3. Use clear, descriptive names and descriptions")
    print("  4. Consider subagent granularity (not too broad, not too narrow)")
    print("  5. Test subagents individually before integration")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    anyio.run(main)

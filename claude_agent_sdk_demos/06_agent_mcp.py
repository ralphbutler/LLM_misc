# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "claude-agent-sdk",
#     "anyio",
# ]
# ///

"""
06_agent_mcp.py - Claude Agent SDK MCP Servers Demo

This program demonstrates:
- Creating in-process MCP servers with create_sdk_mcp_server()
- Registering multiple custom tools in a single server
- Tool naming convention: mcp__<server-name>__<tool-name>
- Combining multiple MCP servers
- Benefits of in-process vs external MCP servers

MCP (Model Context Protocol) allows:
- Standardized tool interfaces
- Reusable tool definitions
- Integration with external services
- Modular tool organization

Run with: uv run 06_agent_mcp.py
"""

import anyio
import json
import csv
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
# MCP SERVER 1: DATA TOOLS
# ============================================================================

@tool(
    name="json_format",
    description="Format and pretty-print JSON data. Provide a JSON string to format.",
    input_schema={"json_string": str}
)
async def json_format(args):
    """Format JSON data for better readability."""
    try:
        json_string = args.get("json_string", "")
        data = json.loads(json_string)
        formatted = json.dumps(data, indent=2, sort_keys=True)

        return {
            "content": [{
                "type": "text",
                "text": f"Formatted JSON:\n```json\n{formatted}\n```"
            }]
        }
    except json.JSONDecodeError as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: Invalid JSON - {str(e)}"
            }]
        }


@tool(
    name="json_validate",
    description="Validate JSON syntax and structure. Provide a JSON string to validate.",
    input_schema={"json_string": str}
)
async def json_validate(args):
    """Validate JSON data."""
    try:
        json_string = args.get("json_string", "")
        data = json.loads(json_string)

        info = {
            "valid": True,
            "type": type(data).__name__,
            "length": len(data) if isinstance(data, (list, dict)) else None
        }

        return {
            "content": [{
                "type": "text",
                "text": f"✅ Valid JSON\nType: {info['type']}\n" +
                       (f"Length: {info['length']}" if info['length'] is not None else "")
            }]
        }
    except json.JSONDecodeError as e:
        return {
            "content": [{
                "type": "text",
                "text": f"❌ Invalid JSON\nError: {str(e)}"
            }]
        }


@tool(
    name="csv_analyze",
    description="Analyze CSV data and provide statistics. Provide CSV content to analyze.",
    input_schema={"csv_content": str}
)
async def csv_analyze(args):
    """Analyze CSV data."""
    try:
        csv_content = args.get("csv_content", "")
        lines = csv_content.strip().split('\n')

        if not lines:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: Empty CSV data"
                }]
            }

        # Parse CSV
        reader = csv.reader(lines)
        rows = list(reader)

        if not rows:
            return {
                "content": [{
                    "type": "text",
                    "text": "Error: No data rows"
                }]
            }

        headers = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []

        analysis = f"""CSV Analysis:
📊 Rows: {len(data_rows)} (excluding header)
📋 Columns: {len(headers)}
📝 Headers: {', '.join(headers)}
"""

        return {
            "content": [{
                "type": "text",
                "text": analysis
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error analyzing CSV: {str(e)}"
            }]
        }


# ============================================================================
# MCP SERVER 2: STRING UTILITIES
# ============================================================================

@tool(
    name="string_reverse",
    description="Reverse a string. Provide text to reverse.",
    input_schema={"text": str}
)
async def string_reverse(args):
    """Reverse a string."""
    text = args.get("text", "")
    reversed_text = text[::-1]

    return {
        "content": [{
            "type": "text",
            "text": f"Original: {text}\nReversed: {reversed_text}"
        }]
    }


@tool(
    name="string_stats",
    description="Get statistics about a string. Provide text to analyze.",
    input_schema={"text": str}
)
async def string_stats(args):
    """Analyze string statistics."""
    text = args.get("text", "")

    stats = f"""String Statistics:
📏 Length: {len(text)}
🔤 Letters: {sum(c.isalpha() for c in text)}
🔢 Digits: {sum(c.isdigit() for c in text)}
⎵ Spaces: {sum(c.isspace() for c in text)}
📝 Words: {len(text.split())}
"""

    return {
        "content": [{
            "type": "text",
            "text": stats
        }]
    }


@tool(
    name="string_case_convert",
    description="Convert string case. Provide text and case_type (upper, lower, title, snake, or camel).",
    input_schema={"text": str, "case_type": str}
)
async def string_case_convert(args):
    """Convert string case."""
    text = args.get("text", "")
    case_type = args.get("case_type", "upper").lower()

    conversions = {
        "upper": text.upper(),
        "lower": text.lower(),
        "title": text.title(),
        "snake": text.lower().replace(" ", "_"),
        "camel": "".join(word.capitalize() for word in text.split())
    }

    result = conversions.get(case_type, text)

    return {
        "content": [{
            "type": "text",
            "text": f"Original: {text}\n{case_type.title()} case: {result}"
        }]
    }


# ============================================================================
# MCP SERVER 3: TIME UTILITIES
# ============================================================================

@tool(
    name="current_timestamp",
    description="Get the current timestamp in various formats. Provide format type (iso, unix, readable, date, or time).",
    input_schema={"format": str}
)
async def current_timestamp(args):
    """Get current timestamp."""
    format_type = args.get("format", "iso").lower()
    now = datetime.now()

    formats = {
        "iso": now.isoformat(),
        "unix": str(int(now.timestamp())),
        "readable": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    }

    result = formats.get(format_type, now.isoformat())

    return {
        "content": [{
            "type": "text",
            "text": f"Current timestamp ({format_type}): {result}"
        }]
    }


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    """
    Main demonstration showing multiple MCP servers with various tools.
    """

    print("=" * 80)
    print("Claude Agent SDK - MCP Servers Demo")
    print("=" * 80)
    print()

    # Create a temporary directory for file operations
    work_dir = Path.cwd() / "demo_06_mcp"
    work_dir.mkdir(exist_ok=True)
    print(f"📁 Working directory: {work_dir}")
    print()

    # ========================================================================
    # STEP 1: Create multiple MCP servers
    # ========================================================================
    print("🔧 Step 1: Creating MCP servers with custom tools...")
    print()

    # Data tools server
    data_server = create_sdk_mcp_server(
        name="data-tools",
        version="1.0.0",
        tools=[json_format, json_validate, csv_analyze]
    )
    print("   ✓ Created 'data-tools' MCP server")
    print("     Tools: json_format, json_validate, csv_analyze")

    # String utilities server
    string_server = create_sdk_mcp_server(
        name="string-utils",
        version="1.0.0",
        tools=[string_reverse, string_stats, string_case_convert]
    )
    print("   ✓ Created 'string-utils' MCP server")
    print("     Tools: string_reverse, string_stats, string_case_convert")

    # Time utilities server
    time_server = create_sdk_mcp_server(
        name="time-utils",
        version="1.0.0",
        tools=[current_timestamp]
    )
    print("   ✓ Created 'time-utils' MCP server")
    print("     Tools: current_timestamp")

    print()

    # ========================================================================
    # STEP 2: Configure agent with all MCP servers
    # ========================================================================
    print("⚙️  Step 2: Configuring agent with all MCP servers...")
    print()

    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a helpful assistant with access to data processing, "
            "string manipulation, and time utility tools."
        ),

        # Register all MCP servers
        mcp_servers={
            "data-tools": data_server,
            "string-utils": string_server,
            "time-utils": time_server,
        },

        # Allow all tools from all servers
        # Format: mcp__<server-name>__<tool-name>
        allowed_tools=[
            # Data tools
            "mcp__data-tools__json_format",
            "mcp__data-tools__json_validate",
            "mcp__data-tools__csv_analyze",

            # String utils
            "mcp__string-utils__string_reverse",
            "mcp__string-utils__string_stats",
            "mcp__string-utils__string_case_convert",

            # Time utils
            "mcp__time-utils__current_timestamp",

            # Built-in tools
            "Write",
            "Read",
        ],

        # Working directory for file operations
        cwd=str(work_dir),

        permission_mode="acceptEdits"
    )

    print("   ✓ Configured agent with 3 MCP servers and 7 custom tools")
    print()

    # ========================================================================
    # STEP 3: Demonstrate tool usage
    # ========================================================================

    try:
        async with ClaudeSDKClient(options=options) as client:

            # ================================================================
            # TEST 1: JSON Tools
            # ================================================================
            print("=" * 80)
            print("TEST 1: JSON Tools")
            print("=" * 80)
            print()

            print("💬 User: Format and validate this JSON: {'name':'Alice','age':30}")
            print()

            await client.query(
                "Use the json tools to format and validate this JSON: "
                '{"name":"Alice","age":30,"city":"New York"}'
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TEST 2: String Tools
            # ================================================================
            print("=" * 80)
            print("TEST 2: String Utilities")
            print("=" * 80)
            print()

            print("💬 User: Analyze and manipulate the string 'Hello World'")
            print()

            await client.query(
                "Use the string utilities to:\n"
                "1. Get statistics about 'Hello World'\n"
                "2. Reverse it\n"
                "3. Convert it to snake_case"
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TEST 3: CSV Analysis
            # ================================================================
            print("=" * 80)
            print("TEST 3: CSV Analysis")
            print("=" * 80)
            print()

            csv_data = """name,age,city
Alice,30,New York
Bob,25,London
Charlie,35,Tokyo"""

            print("💬 User: Analyze this CSV data")
            print()

            await client.query(
                f"Analyze this CSV data:\n{csv_data}"
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TEST 4: Time Utilities
            # ================================================================
            print("=" * 80)
            print("TEST 4: Time Utilities")
            print("=" * 80)
            print()

            print("💬 User: Get current timestamp in different formats")
            print()

            await client.query(
                "Get the current timestamp in ISO, Unix, and readable formats."
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

            # ================================================================
            # TEST 5: Combined workflow
            # ================================================================
            print("=" * 80)
            print("TEST 5: Combined Workflow (Multiple MCP Servers)")
            print("=" * 80)
            print()

            print("💬 User: Create a JSON report with timestamp and string analysis")
            print()

            await client.query(
                "Create a JSON file called 'report.json' that contains:\n"
                "1. A timestamp (use time utils)\n"
                "2. String analysis of 'Claude Agent SDK' (use string utils)\n"
                "3. Format the JSON nicely (use json tools)\n"
                "Write the result to a file."
            )

            print("🤖 Claude:")
            async for message in client.receive_response():
                print(f"   {message}")

            print()

    except CLINotFoundError:
        print("❌ Error: Claude Code CLI not found.")
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
    print("✅ MCP Servers Demo Complete!")
    print("=" * 80)
    print()

    print("Key Features Demonstrated:")
    print("  1. ✓ Created 3 separate MCP servers with create_sdk_mcp_server()")
    print("  2. ✓ Registered 7 custom tools across servers:")
    print("      • data-tools: JSON & CSV processing")
    print("      • string-utils: String manipulation")
    print("      • time-utils: Timestamp generation")
    print("  3. ✓ Used tool naming convention: mcp__<server>__<tool>")
    print("  4. ✓ Combined multiple MCP servers in one agent")
    print("  5. ✓ Demonstrated cross-server workflow")
    print()

    print("💡 MCP SERVER BENEFITS:")
    print()
    print("  In-Process MCP Servers (SDK):")
    print("  • No subprocess management - runs in same process")
    print("  • Better performance - no IPC overhead")
    print("  • Simpler deployment - single Python process")
    print("  • Easy debugging - standard Python debugging")
    print()

    print("  External MCP Servers:")
    print("  • Language-agnostic - any language")
    print("  • Separate lifecycle - independent updates")
    print("  • Reusable - shared across applications")
    print("  • Sandboxed - isolated execution")
    print()

    print("🔀 HYBRID APPROACH:")
    print()
    print("  You can combine both SDK and external MCP servers:")
    print()
    print("  mcp_servers = {")
    print("      'data-tools': data_server,        # In-process SDK server")
    print("      'external-api': external_config   # External MCP server")
    print("  }")
    print()

    print("📦 MCP TOOL ORGANIZATION:")
    print()
    print("  Best practices for organizing MCP tools:")
    print("  • Group related tools in the same server")
    print("  • Keep servers focused (data, strings, time, etc.)")
    print("  • Use clear, descriptive server and tool names")
    print("  • Version your servers for compatibility")
    print("  • Document tool inputs and outputs")
    print()

    print("🔗 LEARN MORE:")
    print()
    print("  Model Context Protocol: https://modelcontextprotocol.io")
    print("  Claude Agent SDK: https://docs.claude.com/en/api/agent-sdk/python")
    print()
    print(f"📁 Output files created in: {work_dir}")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    anyio.run(main)

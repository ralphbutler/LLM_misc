# Claude Agent SDK - Comprehensive Demo Suite

A complete collection of demonstration programs showcasing all major features of the Claude Agent SDK for Python.

## 📋 Overview

The Claude Agent SDK is a high-level framework for building production-ready AI agents with advanced capabilities like automatic context management, built-in tools, custom tool creation, hooks, and more. This demo suite provides practical, runnable examples of every major feature.

## 📦 What's Included

| Program | Features Demonstrated |
|---------|----------------------|
| **01_agent_basics.py** | Core functionality: ClaudeSDKClient, custom tools, built-in tools, streaming |
| **02_agent_hooks.py** | Hooks system: PreToolUse/PostToolUse for validation and logging |
| **03_agent_permissions.py** | All 4 permission modes: default, acceptEdits, plan, bypassPermissions |
| **04_agent_memory.py** | Persistent storage: Memory tool for cross-session data persistence |
| **05_agent_subagents.py** | Subagents: Isolated contexts, task delegation, specialization |
| **06_agent_mcp.py** | MCP servers: In-process custom tool servers with multiple tools |
| **07_agent_websearch.py** | Web research: WebSearch, WebFetch, automatic context management |

## ✅ Prerequisites

- Python 3.10 or higher
- `uv` package manager ([installation guide](https://github.com/astral-sh/uv))
- Claude Code CLI installed and configured
- `ANTHROPIC_API_KEY` environment variable set

### Installing uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setting up API Key

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## 🚀 Quick Start

Each program is self-contained and can be run independently using `uv run`:

```bash
# Run any program
uv run 01_agent_basics.py
uv run 02_agent_hooks.py
# ... etc
```

The first run will automatically install dependencies. Subsequent runs are instant.

## 🎯 Program Details

### ⚡ 01_agent_basics.py - Core Functionality

**What it demonstrates:**
- Creating a stateful conversation with `ClaudeSDKClient`
- Defining custom tools with `@tool` decorator
- Creating MCP servers with `create_sdk_mcp_server()`
- Using built-in tools (Write, Read, Bash)
- Streaming responses with `receive_response()`
- Configuring `ClaudeAgentOptions`

**Key concepts:**
- Custom calculator tool implementation
- Tool registration and naming conventions
- Async conversation patterns
- Error handling

**What to watch for:**
- The calculator tool performs a calculation and writes result to file
- Files are created in `demo_01_output/` directory
- Note: MCP appears here because it's required for registering any custom tool (see FAQ below)

```bash
uv run 01_agent_basics.py
```

---

### 🎣 02_agent_hooks.py - Hooks System

**What it demonstrates:**
- `PreToolUse` hooks for command validation and blocking
- `PostToolUse` hooks for logging and monitoring
- `HookMatcher` for tool-specific hooks
- Permission decisions (allow/deny)
- Security through validation

**Key concepts:**
- Blocking dangerous bash commands
- Logging all tool usage
- Targeted hooks for specific tools
- Real-time monitoring

**What to watch for:**
- The hook will actually **block** the dangerous `rm -rf` command
- You'll see logging output for every tool execution
- Different hooks fire for file operations vs. general tools

```bash
uv run 02_agent_hooks.py
```

---

### 🔐 03_agent_permissions.py - Permission Modes

**What it demonstrates:**
- `default` - Standard permission prompts
- `acceptEdits` - Auto-accept file modifications
- `plan` - Read-only analysis mode
- `bypassPermissions` - Skip all checks (use with caution)

**Key concepts:**
- Security vs. automation tradeoffs
- Use cases for each mode
- Safety considerations
- Mode comparison

**What to watch for:**
- **"plan" mode is particularly interesting** - it will analyze what needs to be done but won't actually execute or create files
- Each mode creates files in separate subdirectories under `demo_03_permissions/`
- Compare the behavior differences for the same task across modes

```bash
uv run 03_agent_permissions.py
```

---

### 💾 04_agent_memory.py - Persistent Storage

**What it demonstrates:**
- Custom memory tools for persistence
- Creating, viewing, updating, and deleting memories
- Multi-session persistence simulation
- File-based storage backend

**Key concepts:**
- Knowledge persistence across sessions
- Reduced context window usage
- Project state management
- Memory tool API structure

**What to watch for:**
- The program runs **two separate "sessions"** to demonstrate persistence
- Session 2 successfully retrieves memories created in Session 1
- Memory files are stored in `demo_04_memory/` directory and persist between runs

```bash
uv run 04_agent_memory.py
```

---

### 👥 05_agent_subagents.py - Specialized Agents

**What it demonstrates:**
- Creating subagent definitions with YAML frontmatter
- Setting up `.claude/agents/` directory structure
- Task delegation to specialized subagents
- Isolated contexts for different tasks
- Three example subagents:
  - `code-reviewer` - Code quality analysis
  - `docs-writer` - Documentation creation
  - `test-engineer` - Test suite development

**Key concepts:**
- Specialization and modularity
- Context isolation
- Parallel execution
- Automatic delegation

**What to watch for:**
- The program **creates `.claude/agents/` directory** with three subagent definition files
- Sample code is created in `demo_05_sample_code/` for the subagents to analyze
- Each subagent has its own specialized system prompt and limited tool access

```bash
uv run 05_agent_subagents.py
```

---

### 🏗️ 06_agent_mcp.py - MCP Servers

**What it demonstrates:**
- Creating multiple in-process MCP servers
- Registering custom tools with servers
- Tool naming: `mcp__<server-name>__<tool-name>`
- Three example servers:
  - `data-tools` - JSON/CSV processing
  - `string-utils` - String manipulation
  - `time-utils` - Timestamp generation

**Key concepts:**
- Tool organization and modularity
- In-process vs external MCP servers
- Hybrid approaches
- Benefits of MCP architecture

**What to watch for:**
- Three separate MCP servers are created with 7 total tools
- The final test demonstrates using tools from multiple servers in one workflow
- Compare this architectural approach with the basic single-tool approach in 01

```bash
uv run 06_agent_mcp.py
```

---

### 🌐 07_agent_websearch.py - Web Research

**What it demonstrates:**
- `WebSearch` tool for internet searches
- `WebFetch` tool for retrieving detailed URL content
- Simple focused research task
- Research synthesis and report generation

**Key concepts:**
- Web research workflow
- Information gathering and synthesis
- File output with Write tool
- Practical use of web tools

**What to watch for:**
- This program performs **actual web searches** - results will vary based on current web content
- Single focused research query about Claude Agent SDK
- Should create `claude_agent_sdk_report.md` in `demo_07_research/`
- **Typical runtime**: 2-3 minutes
- **Typical cost**: $0.20-0.40 (similar to other demos)
- Much simpler than complex multi-phase designs - focused on demonstrating the tools clearly

```bash
uv run 07_agent_websearch.py
```

## ⭐ Key Features of Claude Agent SDK

### Automatic Context Management
- **Context Compaction**: Summarizes old messages when approaching limits
- **Context Editing**: Removes stale tool calls and results
- **Performance**: 84% token reduction, 39% performance improvement

### Built-in Tools
- **Bash**: Execute shell commands
- **Read/Write/Edit**: File operations
- **Grep/Glob**: Search and find files
- **WebSearch/WebFetch**: Internet access

### Custom Tools
- Define with `@tool` decorator
- Create in-process MCP servers
- Structured input schemas
- Type-safe parameters

### Hooks System
- **PreToolUse**: Validate/block before execution
- **PostToolUse**: Log/monitor after completion
- **HookMatcher**: Target specific tools
- **Permission decisions**: Control tool access

### Permission Modes
- **default**: Interactive with prompts
- **acceptEdits**: Automated file changes
- **plan**: Read-only analysis
- **bypassPermissions**: Unrestricted (caution!)

### Subagents
- Isolated context windows
- Task specialization
- Parallel execution
- Automatic delegation

## 🏗️ Architecture

```
Claude Agent SDK
├── Core
│   ├── ClaudeSDKClient (stateful conversations)
│   ├── query() (stateless queries)
│   └── ClaudeAgentOptions (configuration)
├── Tools
│   ├── Built-in (Bash, Read, Write, etc.)
│   ├── Custom (@tool decorator)
│   └── MCP Servers (in-process or external)
├── Features
│   ├── Hooks (PreToolUse, PostToolUse)
│   ├── Permissions (4 modes)
│   ├── Memory (persistent storage)
│   ├── Subagents (specialized agents)
│   └── Context Management (automatic)
└── Integration
    ├── Streaming (async iterators)
    ├── Error Handling (typed exceptions)
    └── Session Management (connect/disconnect)
```

## 📝 Common Patterns

### Basic Agent Setup

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    system_prompt="You are a helpful assistant",
    allowed_tools=["Read", "Write"],
    permission_mode="acceptEdits"
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Your task here")
    async for message in client.receive_response():
        print(message)
```

### Custom Tool Creation

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool(
    name="my_tool",
    description="What the tool does",
    input_schema={"param": "string"}
)
async def my_tool(args):
    result = process(args["param"])
    return {
        "content": [{
            "type": "text",
            "text": f"Result: {result}"
        }]
    }

server = create_sdk_mcp_server(
    name="my-server",
    version="1.0.0",
    tools=[my_tool]
)
```

### Hooks Implementation

```python
async def my_hook(input_data, tool_use_id, context):
    # Validation logic
    if should_block:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "Reason here"
            }
        }
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[my_hook])
        ]
    }
)
```

## 🔧 Troubleshooting

### "Claude Code CLI not found"
- Ensure Claude Code is installed: [Installation Guide](https://docs.claude.com/en/api/agent-sdk/overview)
- Check that it's in your PATH: `which claude`

### "ANTHROPIC_API_KEY not set"
- Set the environment variable: `export ANTHROPIC_API_KEY='your-key'`
- Or add to your shell profile (`.bashrc`, `.zshrc`)

### Permission errors
- Make sure you're using appropriate permission modes
- Check file/directory permissions in your working directory
- Use hooks to validate operations before execution

### Subagents not found
- Verify `.claude/agents/` directory exists
- Check YAML frontmatter syntax
- Ensure `setting_sources=["project"]` is set

## 📚 Additional Resources

- **Official Docs**: https://docs.claude.com/en/api/agent-sdk/python
- **GitHub Repository**: https://github.com/anthropics/claude-agent-sdk-python
- **Building Agents Guide**: https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk
- **Model Context Protocol**: https://modelcontextprotocol.io

## 📁 Project Structure

```
WORK/
├── README.md                    # This file
├── PLAN01.md                    # Detailed planning document
├── 01_agent_basics.py          # Core functionality
├── 02_agent_hooks.py           # Hooks system
├── 03_agent_permissions.py     # Permission modes
├── 04_agent_memory.py          # Memory tool
├── 05_agent_subagents.py       # Subagents
├── 06_agent_mcp.py             # MCP servers
├── 07_agent_websearch.py       # Web research
├── demo_01_output/             # Output from 01
├── demo_02_hooks/              # Output from 02
├── demo_03_permissions/        # Output from 03
├── demo_04_memory/             # Output from 04
├── demo_05_sample_code/        # Output from 05
├── demo_07_research/           # Output from 07
└── .claude/
    └── agents/                 # Subagent definitions
        ├── code-reviewer.md
        ├── docs-writer.md
        └── test-engineer.md
```

## 🎓 Learning Path

Recommended order for exploring the demos:

1. **01_agent_basics.py** - Start here to understand core concepts
2. **06_agent_mcp.py** - Learn about tool organization
3. **02_agent_hooks.py** - Add validation and monitoring
4. **03_agent_permissions.py** - Understand security modes
5. **05_agent_subagents.py** - Explore specialization
6. **04_agent_memory.py** - Add persistence
7. **07_agent_websearch.py** - Build a complete research agent

## ❓ Frequently Asked Questions

### Why does MCP appear in both 01_agent_basics.py and 06_agent_mcp.py?

**Short answer:** MCP is the required infrastructure for registering any custom tool in the SDK.

**Detailed explanation:**
- **Program 01** focuses on "how to create a custom tool" - MCP is the mechanism you must use, but it's not the main point
- **Program 06** focuses on "MCP architecture and organization" - demonstrating how to structure multiple tools across multiple servers

Think of it like:
- **01** = "How to send an email" (you need SMTP, but that's not the focus)
- **06** = "Understanding SMTP architecture" (SMTP itself is the topic)

You cannot create a custom tool without wrapping it in an MCP server via `create_sdk_mcp_server()`. Even a single custom tool requires this pattern.

### What's the difference between the Claude Agent SDK and the Anthropic API?

The Anthropic API is low-level and requires you to manually manage conversation state, implement tool calling, and handle context. The Claude Agent SDK is a high-level framework that provides:
- Automatic context management (compaction, editing)
- Built-in tools (file ops, bash, web search)
- Hooks system for validation and monitoring
- Agent-oriented architecture for autonomous workflows
- Production-ready error handling and session management

Use the API when you need fine-grained control. Use the SDK when building agents.

### Do I need Claude Code CLI installed?

Yes, the Claude Agent SDK requires the Claude Code CLI to be installed and available in your PATH. The SDK communicates with Claude through this CLI.

### Can I use these programs in production?

These are demonstration programs designed for learning. For production use:
- Review and modify permission modes appropriately
- Add comprehensive error handling
- Implement proper logging and monitoring
- Consider security implications of tool access
- Test thoroughly in your specific environment

## 🤝 Contributing

This is a demonstration project. Feel free to:
- Modify the examples for your use cases
- Extend with additional features
- Create your own custom tools and subagents
- Share improvements and insights

## 📄 License

These demonstrations are provided as educational examples for the Claude Agent SDK.

## 💬 Support

For SDK issues:
- GitHub Issues: https://github.com/anthropics/claude-agent-sdk-python/issues
- Documentation: https://docs.claude.com/en/api/agent-sdk/python

---

**Happy building with Claude Agent SDK!** 🤖✨

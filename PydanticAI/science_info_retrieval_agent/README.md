# Science Information Retrieval Agent

A small agent built with PydanticAI that retrieves structured scientific information from LLMs.

## Overview

This project demonstrates how to use PydanticAI to build a specialized agent for retrieving scientific information. The agent takes natural language queries about scientific topics and returns structured, validated responses containing:

- Main scientific concept details
- Detailed explanation
- Related concepts
- Key references
- Confidence level
- Limitations of the provided information

The agent showcases several key features of the PydanticAI framework:

1. **Structured outputs** via Pydantic models
2. **Type safety** throughout the application
3. **Dependency injection** for configurable behavior
4. **Tool functions** to augment LLM capabilities
5. **Model-agnostic design** allowing easy switching between different LLM providers

## Installation

### Prerequisites

- Python 3.9+
- pip or uv (as mentioned in your preferences)

### Setup with pip

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install pydantic-ai rich
```

### Setup with uv

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies
uv pip install pydantic-ai rich
```

### API Keys

Before running the code, make sure you have the appropriate API keys for the LLM provider you want to use. Set these as environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY=your_api_key_here

# For other providers (if needed)
export ANTHROPIC_API_KEY=your_api_key_here
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

### Basic Usage

```python
from science_agent import get_science_info_sync

# Get information about a scientific topic
result = get_science_info_sync("Explain quantum entanglement")

# Access the structured data
print(f"Concept: {result.main_concept.name}")
print(f"Description: {result.main_concept.short_description}")
print(f"Explanation: {result.explanation}")
```

### Advanced Usage with Configuration

```python
import asyncio
from science_agent import get_science_info, ScienceAgentConfig
from datetime import date

# Create a custom configuration
config = ScienceAgentConfig(
    detail_level=8,  # Higher detail (1-10 scale)
    max_references=5,  # Include up to 5 references
    include_limitations=True,  # Include limitations section
    current_date=date.today()  # Provide current date context
)

# Run asynchronously with the configuration
async def main():
    result = await get_science_info(
        "Explain the current state of fusion energy research", 
        config=config
    )
    print(result)

asyncio.run(main())
```

### Running the Example Script

To run the example script that demonstrates multiple use cases:

```bash
python example_usage.py
```

## How It Works

### Architecture

The agent uses PydanticAI's framework to:

1. **Define structured output models** using Pydantic
2. **Create an agent** with a specific LLM backend
3. **Configure system prompts** both statically and dynamically
4. **Implement tools** to enhance the agent's capabilities
5. **Validate responses** to ensure they match the expected schema

### Key Components

#### Output Models

The agent defines a hierarchical structure of Pydantic models:

- `ScienceResponse`: The top-level response
  - `ScienceConcept`: Information about the main concept
    - `ScienceReference`: References to scientific sources

This structure ensures that the LLM's responses are consistently formatted and contain all required fields.

#### Type-safe Agent Implementation

The agent uses the dependency injection pattern with explicitly typed parameters:

```python
science_agent = Agent(
    'openai:gpt-4o',
    deps_type=ScienceAgentConfig,
    system_prompt="You are a scientific research assistant..."
)
```

This approach provides type checking and IDE support through the explicit `deps_type` parameter.

#### Agent Configuration

The `ScienceAgentConfig` dataclass allows customizing:

- Level of detail in responses
- Maximum number of references to include
- Whether to include limitations
- Current date for context

This demonstrates PydanticAI's dependency injection capability.

#### Tools

The agent includes a `lookup_scientific_terminology` tool that simulates accessing a scientific database. In a real application, this could connect to actual scientific APIs or databases.

## Extending the Agent

### Using Different LLM Providers

The agent is model-agnostic. To use a different LLM provider:

```python
from science_agent import science_agent

# Switch to Anthropic
result = science_agent.run_sync(
    "Explain CRISPR gene editing", 
    model="anthropic:claude-3-5-sonnet-latest"
)

# Or use Google's Gemini
result = science_agent.run_sync(
    "Explain CRISPR gene editing", 
    model="google-gla:gemini-1.5-pro"
)
```

### Adding New Tools

To add new capabilities, define additional tools using the `@science_agent.tool` decorator:

```python
@science_agent.tool
async def search_scientific_papers(
    ctx: RunContext[ScienceAgentConfig],
    query: str,
    max_results: int = 5
) -> List[Dict[str, str]]:
    """
    Search for scientific papers related to the query.
    In a real implementation, this would connect to academic APIs.
    """
    # Implementation details...
    return papers
```

## Limitations

- The current implementation simulates access to scientific databases
- A production version would need to connect to real scientific APIs
- Some LLM providers may have different capabilities regarding structured outputs

## License

MIT

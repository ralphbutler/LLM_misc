# PydanticAI Examples

A collection of examples demonstrating key features and patterns for building AI-powered applications with PydanticAI.

## Examples Overview

### Basic Features

- **01_structured_result.py** - Demonstrates structured outputs using Pydantic models with a book analysis system
- **02_result_validation_retry.py** - Shows robust validation and retry mechanisms through a restaurant reservation system
- **03_currency_converter.py** - Illustrates dependency injection for managing external data (exchange rates)

### Tools and Context

- **04a_plain_tool.py** - Shows simple tool usage with stock price lookups
- **04a_validate_password.py** - Demonstrates plain tools for password validation
- **04b_tool_runcontext.py** - Examples of stateful tools using RunContext for expense tracking

### Dynamic Systems

- **05_time_greeter.py** - Shows dynamic system prompts that adapt based on current time
- **06_ticket_history.py** - Demonstrates message serialization for persistent support tickets
- **07_math_tutor.py** - Illustrates chat history reuse in a tutoring context
- **08_interaction_logger.py** - Shows the difference between all_messages and new_messages for logging

### Streaming Responses

- **09a_story_streamer.py** - Basic text streaming example with a story generator
- **09b_story_streamer.py** - Enhanced streaming with progress tracking and metadata
- **10_recipe_streamer.py** - Demonstrates structured streaming with validation using TypedDict

## Key Concepts

- Structured outputs with validation
- Dependency injection
- Tool patterns (plain vs context)
- Message history and serialization
- Streaming responses (text and structured)
- Dynamic system prompts

## Getting Started

Each example is self-contained and includes detailed comments explaining the concepts and implementation. The examples progress from basic to more advanced features, making them suitable for both learning and reference.

## Requirements

- Python 3.8+
- PydanticAI
- Additional requirements vary by example (see individual files)

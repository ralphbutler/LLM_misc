
# This program demonstrates how to use PydanticAI to create a structured AI-powered book analysis system.
# It defines a BookAnalysis data model that specifies exactly what information should be extracted from
# a book description, including:
# - Title and author
# - Genre (limited to specific categories like Fiction, Non-Fiction, etc.)
# - Themes (1-5 main themes)
# - Rating (1-10 scale)
# - A short recommendation (max 200 characters)

# This program creates an AI agent using this model and includes an example that analyzes George Orwell's "1984".
# When run, it outputs a structured analysis of the book along with API usage statistics.
# The key benefit is that it enforces a consistent format for book analyses through Pydantic's data validation.


"""Example demonstrating how to use PydanticAI with structured result types.

This example shows how to define and use structured result types with PydanticAI agents.
It creates an agent that analyzes book information and returns structured data.
"""

# import os
# os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"
import logfire
logfire.configure(send_to_logfire='if-token-present')

import asyncio
from typing import Annotated, List
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class BookAnalysis(BaseModel):
    """Structured result type for book analysis."""
    title: str = Field(description="The book's title")
    author: str = Field(description="The book's author")
    genre: Annotated[str, Field(
        description="Primary genre of the book",
        pattern="^(Fiction|Non-Fiction|Mystery|Science Fiction|Fantasy|Biography|History)$"
    )]
    themes: List[str] = Field(
        description="Main themes explored in the book",
        min_items=1,
        max_items=5
    )
    rating: Annotated[int, Field(
        description="Rating out of 10",
        ge=1,
        le=10
    )]
    recommendation: Annotated[str, Field(
        description="Short recommendation for who would enjoy this book",
        max_length=200
    )]

# Create an agent that returns BookAnalysis objects
agent = Agent(
    "openai:gpt-4o",
    result_type=BookAnalysis,
    system_prompt="""You are a literary analyst. Given a book description, 
    provide a structured analysis including genre, themes, and recommendations.
    Be specific and concise in your analysis."""
)

async def main():
    # Example book description
    book_description = """
    '1984' by George Orwell depicts a dystopian society where the government 
    maintains power through surveillance, manipulation of language, and control 
    of information. The story follows Winston Smith as he rebels against this 
    totalitarian system.
    """
    
    # Run the analysis
    result = await agent.run(book_description)
    
    # Print the structured result
    print("\nStructured Book Analysis:")
    print(f"Title: {result.data.title}")
    print(f"Author: {result.data.author}")
    print(f"Genre: {result.data.genre}")
    print(f"Themes: {', '.join(result.data.themes)}")
    print(f"Rating: {result.data.rating}/10")
    print(f"Recommendation: {result.data.recommendation}")
    
    # Print usage statistics
    print("\nAPI Usage:")
    print(result.usage())

if __name__ == '__main__':
    asyncio.run(main())

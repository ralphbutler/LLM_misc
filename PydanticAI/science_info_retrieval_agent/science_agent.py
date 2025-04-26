"""
Science Information Retrieval Agent

This agent uses PydanticAI to retrieve scientific information from an LLM
and returns structured data about scientific topics.
"""

import asyncio
from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


# Define output schema
class ScienceReference(BaseModel):
    """A reference to a scientific source."""
    title: str = Field(description="Title of the scientific source")
    authors: List[str] = Field(description="Authors of the scientific source")
    year: int = Field(description="Publication year", ge=1600, le=2025)
    doi: Optional[str] = Field(description="DOI identifier if available", default=None)


class ScienceConcept(BaseModel):
    """Information about a specific scientific concept."""
    name: str = Field(description="Name of the scientific concept")
    short_description: str = Field(description="Brief explanation of the concept (1-2 sentences)")
    field: str = Field(description="Field of science this concept belongs to")
    related_concepts: List[str] = Field(description="Other related scientific concepts")
    key_references: List[ScienceReference] = Field(
        description="Key references for further reading",
        max_items=3
    )


class ScienceResponse(BaseModel):
    """Structured response for scientific information retrieval."""
    main_concept: ScienceConcept = Field(description="The primary scientific concept being explained")
    explanation: str = Field(description="Detailed explanation of the concept (3-5 paragraphs)")
    confidence: int = Field(
        description="Confidence level in the information provided (1-10)",
        ge=1, le=10
    )
    limitations: str = Field(description="Limitations or caveats about the provided information")


# Define dependencies to customize agent behavior
@dataclass
class ScienceAgentConfig:
    """Configuration for the science agent."""
    detail_level: int = 5  # 1-10 scale for how detailed responses should be
    max_references: int = 3
    include_limitations: bool = True
    current_date: date = date.today()


# Initialize the agent
science_agent = Agent(
    # Using OpenAI's model as default, but this can be changed when running the agent
    'openai:gpt-4o',
    deps_type=ScienceAgentConfig,
    output_type=ScienceResponse,
    system_prompt=(
        "You are a scientific research assistant specialized in retrieving accurate "
        "scientific information. Provide well-structured, factual responses about scientific "
        "topics with appropriate detail, references, and limitations."
    ),
)


# Add dynamic system prompt to customize based on configuration
@science_agent.system_prompt
async def add_detail_instruction(ctx: RunContext[ScienceAgentConfig]) -> str:
    """Add instruction about detail level based on configuration."""
    detail_level = ctx.deps.detail_level
    return (
        f"Provide responses with a detail level of {detail_level}/10. "
        f"Include up to {ctx.deps.max_references} key references per concept. "
        f"Today's date is {ctx.deps.current_date}."
    )


# Add tool for terminology lookup
@science_agent.tool
async def lookup_scientific_terminology(
    ctx: RunContext[ScienceAgentConfig], 
    term: str
) -> str:
    """
    Look up technical scientific terminology to provide standardized definitions.
    This simulates accessing a scientific terminology database.
    
    Args:
        term: The scientific term to look up
        
    Returns:
        A standardized definition of the term
    """
    # In a real implementation, this would query an actual database or API
    # Here we're simulating the behavior with a few example terms
    terminology_database = {
        "quantum entanglement": "A quantum mechanical phenomenon where the quantum states of two or more particles become correlated, even when separated by large distances.",
        "photosynthesis": "The process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water, generating oxygen as a byproduct.",
        "natural selection": "The process whereby organisms better adapted to their environment tend to survive and produce more offspring.",
        "general relativity": "Einstein's theory of gravitation in which gravity is described as a consequence of the curvature of spacetime.",
        "crispr": "A family of DNA sequences found in the genomes of prokaryotic organisms such as bacteria, containing snippets of DNA from viruses that have attacked the bacterium."
    }
    
    # Case-insensitive lookup with fallback
    term_lower = term.lower()
    for key, definition in terminology_database.items():
        if term_lower in key.lower():
            return f"Definition of '{key}': {definition}"
    
    # If term not found, return a message indicating this
    return f"Term '{term}' not found in the terminology database."


# Main function to run the agent
async def get_science_info(query: str, config: Optional[ScienceAgentConfig] = None) -> ScienceResponse:
    """
    Run the science agent to get information about a scientific topic.
    
    Args:
        query: The scientific query to process
        config: Optional configuration for the agent
        
    Returns:
        A structured ScienceResponse with information about the topic
    """
    if config is None:
        config = ScienceAgentConfig()
        
    result = await science_agent.run(query, deps=config)
    return result.data


# Synchronous version for easier use
def get_science_info_sync(query: str, config: Optional[ScienceAgentConfig] = None) -> ScienceResponse:
    """Synchronous version of get_science_info"""
    if config is None:
        config = ScienceAgentConfig()
        
    result = science_agent.run_sync(query, deps=config)
    return result.data


# Example usage as script
if __name__ == "__main__":
    # Set up sample query and configuration
    sample_query = "Explain quantum computing and its potential applications in machine learning"
    config = ScienceAgentConfig(
        detail_level=7,
        max_references=3,
        include_limitations=True
    )
    
    # Run the agent and print the result
    result = asyncio.run(get_science_info(sample_query, config))
    
    # Print the results in a structured way
    print(f"CONCEPT: {result.main_concept.name}")
    print(f"FIELD: {result.main_concept.field}")
    print("\nDESCRIPTION:")
    print(result.main_concept.short_description)
    print("\nDETAILED EXPLANATION:")
    print(result.explanation)
    print("\nRELATED CONCEPTS:")
    for concept in result.main_concept.related_concepts:
        print(f"- {concept}")
    print("\nKEY REFERENCES:")
    for ref in result.main_concept.key_references:
        authors_text = ", ".join(ref.authors)
        doi_text = f" DOI: {ref.doi}" if ref.doi else ""
        print(f"- {ref.title} ({ref.year}) by {authors_text}{doi_text}")
    print(f"\nCONFIDENCE: {result.confidence}/10")
    print("\nLIMITATIONS:")
    print(result.limitations)

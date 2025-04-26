"""
Example Usage of Science Information Retrieval Agent

This script demonstrates how to use the Science Information Retrieval Agent
for several different types of scientific queries.
"""

import asyncio
from datetime import date
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from science_agent import ScienceAgentConfig, get_science_info, get_science_info_sync

console = Console()


async def run_examples():
    """Run multiple examples of the science agent with different configurations"""
    
    # Example 1: Basic usage with default configuration
    console.print(Panel("[bold cyan]Example 1: Basic Usage[/bold cyan]"))
    console.print("Query: Explain CRISPR gene editing technology")
    
    result1 = await get_science_info("Explain CRISPR gene editing technology")
    
    console.print(Panel(f"[bold green]{result1.main_concept.name}[/bold green]"))
    console.print(f"Field: {result1.main_concept.field}")
    console.print(Markdown(result1.main_concept.short_description))
    console.print(Markdown(result1.explanation))
    console.print(f"Confidence: {result1.confidence}/10")
    console.print()
    
    # Example 2: High detail configuration
    console.print(Panel("[bold cyan]Example 2: High Detail Configuration[/bold cyan]"))
    console.print("Query: Explain how quantum computing could impact cryptography")
    
    detailed_config = ScienceAgentConfig(
        detail_level=8,
        max_references=3,
        include_limitations=True
    )
    
    result2 = await get_science_info(
        "Explain how quantum computing could impact cryptography", 
        config=detailed_config
    )
    
    console.print(Panel(f"[bold green]{result2.main_concept.name}[/bold green]"))
    console.print(f"Field: {result2.main_concept.field}")
    console.print(Markdown(result2.main_concept.short_description))
    console.print(Markdown(result2.explanation))
    
    console.print("[bold]Related Concepts:[/bold]")
    for concept in result2.main_concept.related_concepts:
        console.print(f"- {concept}")
    
    console.print("\n[bold]Key References:[/bold]")
    for ref in result2.main_concept.key_references:
        authors = ", ".join(ref.authors)
        doi = f" DOI: {ref.doi}" if ref.doi else ""
        console.print(f"- {ref.title} ({ref.year}) by {authors}{doi}")
    
    console.print(f"\nConfidence: {result2.confidence}/10")
    console.print("\n[bold]Limitations:[/bold]")
    console.print(Markdown(result2.limitations))
    console.print()
    
    # Example 3: Using with different LLM model
    console.print(Panel("[bold cyan]Example 3: Using a Different LLM Model[/bold cyan]"))
    console.print("Query: Explain the current state of fusion energy research")
    
    # In a real scenario, you'd import the science_agent module and modify the model
    # For brevity, we'll just use the same agent but note the change
    console.print("[yellow]Note: In a real implementation, we would change the underlying LLM model here[/yellow]")
    
    result3 = await get_science_info(
        "Explain the current state of fusion energy research",
        config=ScienceAgentConfig(detail_level=6)
    )
    
    console.print(Panel(f"[bold green]{result3.main_concept.name}[/bold green]"))
    console.print(Markdown(result3.explanation))
    console.print(f"Confidence: {result3.confidence}/10")
    console.print()
    
    # Example 4: Compare different configurations
    console.print(Panel("[bold cyan]Example 4: Comparing Detail Levels[/bold cyan]"))
    console.print("Query: Explain machine learning interpretability")
    
    # Low detail configuration
    low_detail = ScienceAgentConfig(detail_level=3, max_references=1)
    result_low = await get_science_info(
        "Explain machine learning interpretability", 
        config=low_detail
    )
    
    # High detail configuration
    high_detail = ScienceAgentConfig(detail_level=9, max_references=3)
    result_high = await get_science_info(
        "Explain machine learning interpretability", 
        config=high_detail
    )
    
    console.print(Panel("[bold yellow]Low Detail (Level 3)[/bold yellow]"))
    console.print(Markdown(result_low.explanation))
    console.print(f"Word count: {len(result_low.explanation.split())}")
    
    console.print(Panel("[bold yellow]High Detail (Level 9)[/bold yellow]"))
    console.print(Markdown(result_high.explanation))
    console.print(f"Word count: {len(result_high.explanation.split())}")
    
    # Final summary
    console.print(Panel("[bold magenta]Summary of Agent Capabilities[/bold magenta]"))
    console.print(
        "The Science Information Retrieval Agent demonstrates how PydanticAI can be used to:\n"
        "1. Create structured outputs from LLM responses\n"
        "2. Implement configurable behavior through dependency injection\n"
        "3. Use tools to augment the LLM's capabilities\n"
        "4. Ensure type safety and validation of responses\n"
    )


if __name__ == "__main__":
    asyncio.run(run_examples())

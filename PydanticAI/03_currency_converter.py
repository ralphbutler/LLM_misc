
# This demo showcases several key aspects of pydanticAI's dependency system while remaining focused:
# 
# It demonstrates a clear use case for dependencies (exchange rates that need to be accessed by the agent)
# - Shows how to create a typed dependency class using @dataclass
# - Illustrates how to use RunContext to access dependencies in tools
# - Provides a practical example that's not too trivial but also not overly complex
# - Shows how to pass dependencies when running the agent

# This example is self-contained and demonstrates a real-world scenario (currency conversion)
#   that naturally requires external data (exchange rates) to be passed to the agent.

"""Example demonstrating how to use PydanticAI agent dependencies (deps_type).

This example shows how to create an agent that converts currencies using
external exchange rate data passed through dependencies.
"""

from dataclasses import dataclass
from typing import Dict

from pydantic_ai import Agent, RunContext


@dataclass
class CurrencyDeps:
    """Dependencies for the currency converter agent.
    
    In a real application, these rates would likely come from an API,
    but we're using static data for demonstration purposes.
    """
    exchange_rates: Dict[str, float]  # Rates relative to USD


# Create the agent with proper typing for dependencies
currency_agent = Agent(
    'openai:gpt-4o',
    deps_type=CurrencyDeps,
    system_prompt=(
        'You are a helpful currency conversion assistant. '
        'Use the provided exchange rates to convert between currencies. '
        'Format ALL responses in plain text like this exact example:\n'
        '100 USD = 100 × 0.92 = 92 EUR\n'
        'Never use special characters, LaTeX notation, or mathematical symbols like \\text or \\times. '
        'Use a plain "x" for multiplication. Keep responses to a single line with no explanations.'
    ),
)


@currency_agent.tool
async def get_exchange_rate(ctx: RunContext[CurrencyDeps], currency: str) -> float:
    """Get the exchange rate for a currency relative to USD.
    
    Args:
        ctx: The context containing exchange rates.
        currency: The three-letter currency code (e.g., 'EUR', 'GBP').
    
    Returns:
        float: The exchange rate relative to USD.
    """
    currency = currency.upper()
    if currency == 'USD':
        return 1.0
    if currency not in ctx.deps.exchange_rates:
        raise ValueError(f"Exchange rate not found for {currency}")
    return ctx.deps.exchange_rates[currency]


async def main():
    # Set up example exchange rates (relative to USD)
    rates = {
        'EUR': 0.92,
        'GBP': 0.79,
        'JPY': 151.56,
        'CAD': 1.35,
    }
    
    # Create dependencies instance
    deps = CurrencyDeps(exchange_rates=rates)
    
    # Example conversions
    questions = [
        "Convert 100 USD to EUR",
        "How much is 50 GBP in JPY?",
        "Convert 1000 JPY to CAD",
    ]
    
    for question in questions:
        result = await currency_agent.run(question, deps=deps)
        print(f"\nQ: {question}")
        print(f"A: {result.data}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())


# This example is clean for cases where we don't need any context or dependencies.
# It is a good example of when to use tool_plain - simple functions that just take inputs
# and return outputs without needing any context or state.

from pydantic_ai import Agent
from pydantic import BaseModel
import yfinance as yf

class StockPriceResult(BaseModel):
    symbol: str
    price: float
    currency: str = "USD"
    message: str

stock_agent = Agent(
    "openai:gpt-4o",
    result_type=StockPriceResult,
    system_prompt="You are a helpful financial assistant that uses a tool to retrieve stock prices."
)

@stock_agent.tool_plain
def get_stock_price(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    price = ticker.fast_info.last_price
    return {
        "price": round(price, 2),
        "currency": "USD"
    }
    
result = stock_agent.run_sync("What is Apple's current stock price?")

print(f"Stock Price for {result.data.symbol}:  ${result.data.price:.2f} {result.data.currency}")
print(f"Message: {result.data.message}")

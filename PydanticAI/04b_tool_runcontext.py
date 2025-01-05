
# This example demonstrates why RunContext is necessary because:
# 
#   - The add_expense tool needs to maintain state across multiple calls (the running total and
#         category totals)
#   - It accesses and modifies shared data through ctx.deps
#   - The state persists between different agent invocations since we pass the same state object
# 
# This couldn't be done with a plain tool because:
# 
#   - A plain tool has no way to access or update the running totals
#   - Each call would be isolated with no shared state
#   - Category totals would be lost between invocations


from dataclasses import dataclass
from typing import Dict
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

@dataclass
class ExpenseState:
    running_total: float
    categories: Dict[str, float]

class ExpenseResult(BaseModel):
    amount: float
    category: str
    message: str
    new_total: float

expense_agent = Agent(
    "openai:gpt-4o",
    deps_type=ExpenseState,
    result_type=ExpenseResult,
    system_prompt="You are an expense tracking assistant. Use the tools to track expenses and maintain category totals."
)

@expense_agent.tool
async def add_expense(ctx: RunContext[ExpenseState], amount: float, category: str) -> dict:
    """Add an expense and update running totals.
    
    Args:
        ctx: Contains the running total and category totals
        amount: Amount to add
        category: Expense category
    """
    ctx.deps.running_total += amount
    ctx.deps.categories[category] = ctx.deps.categories.get(category, 0) + amount
    return {
        "total": round(ctx.deps.running_total, 2),
        "category_total": round(ctx.deps.categories[category], 2)
    }

# Example usage
state = ExpenseState(running_total=0.0, categories={})
result = expense_agent.run_sync(
    "Add a coffee expense of $4.50", 
    deps=state
)
print(f"{result.data.message}")
print(f"New total: ${result.data.new_total:.2f}")

result = expense_agent.run_sync(
    "Add lunch for $12.75",
    deps=state
)
print(f"{result.data.message}")
print(f"New total: ${result.data.new_total:.2f}")

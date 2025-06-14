import os
import json
from datetime import timedelta
from typing import List, Dict, Any

from pydantic import BaseModel, Field

# LiteLLM setup
os.environ['LITELLM_LOG'] = 'DEBUG'
import litellm
litellm.enable_json_schema_validation = True
from litellm import completion, exceptions

# Rich components used in core functions
from rich.table import Table
from rich.text import Text
from rich.style import Style

# Define the Pydantic model for structured output
class PuzzleResult(BaseModel):
    """Structured result type for ARC challenge grid puzzle result."""
    code: str = Field(description="Python code that transforms input grid to output grid")
    solution: List[List[int]] = Field(description="list of rows, each of which is a list of integers")
    reasoning: str = Field(description="explanation of how the code transforms input to output")

# Base prompt template
arc_prompt = """
You are an expert at solving ARC challenge puzzles.

<task>
Write a function named transform_grid(grid) that transforms input grids to output grids.
- Input/Output: Takes a 2D list of integers, returns a 2D list of integers
- Must work for all examples and generalize to the test case
- Use only Python standard library functions
- Include comments explaining key steps
- Write concise, readable code without markdown annotations
</task>

<grid_info>
- Grids are 2D arrays of integers (0 represents empty space)
- Grid sizes vary - solution must be size-independent
- Same integer values maintain consistent meaning across grids
- All necessary information is in the input grid
- Positions are zero-indexed from top-left corner
</grid_info>

Here is the puzzle:
PUZZLE
"""

def call_llm(model_name: str, api_base: str, max_tokens: int, prompt: str, Pydantic_Model: type) -> tuple[Any, float]:
    """
    Call an LLM with the given prompt and extract structured output using litellm.
    Returns the response model and the cost of the call.
    """
    cost = 0.0
    try:
        response = completion(
            model=model_name,
            api_base=api_base,
            # max_tokens=max_tokens,   # not for o4
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            # temperature=0.0,   # not for o4
            # budget_tokens=1024,  # gemini flash did not give an error for this
            # thinking_budget=1024,  # gemini flash did not give an error for this
            request_timeout=60,
            response_format=Pydantic_Model,  # not response_model
        )

        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
            response_model = Pydantic_Model.model_validate(json.loads(content))

            # Track cost if available
            if hasattr(response, '_hidden_params') and 'response_cost' in response._hidden_params:
                cost = response._hidden_params['response_cost']

            return response_model, cost
        else:
            raise Exception("Received empty response from model")

    except Exception as e:
        print(f"*** call_llm failed calling {model_name}")
        print(f"*** type error msg: {type(e)}")
        print(f"*** error msg: {e}")
        if isinstance(e, litellm.exceptions.RateLimitError):
            print("**** EXITING DUE TO RATELIMIT ERROR ****")
            # Re-raise the exception so the main loop can catch it and exit
            raise e
        # Return a default failed result and 0 cost on other errors
        return PuzzleResult(code="def transform_grid(example): return",
                          solution=[[0,0,0],[0,0,0],[0,0,0]],
                          reasoning="FAILED TO CREATE USABLE CODE DUE TO LLM ERROR"), 0.0


def create_augmented_prompt(base_prompt: str, puzzle_data: Dict, previous_attempts: List[Dict] = None) -> str:
    """Create a prompt with the puzzle and optionally add previous attempt information"""

    # Create a simplified puzzle object that only includes train data and test input
    puzzle_for_llm = {
        'train': puzzle_data['train'],
        'test': {'input': puzzle_data['test'][0]['input']}
    }

    # Insert the puzzle into the prompt
    prompt = base_prompt.replace("PUZZLE", str(puzzle_for_llm))

    # Add ASCII representation of grids for better visualization
    prompt += "\n\n<ascii_grids>\n"
    prompt += "Train Examples:\n"
    for i, example in enumerate(puzzle_data['train']):
        prompt += f"\nExample {i+1} Input:\n"
        prompt += format_grid_for_comparison(example['input']) + "\n"
        prompt += f"Example {i+1} Output:\n"
        prompt += format_grid_for_comparison(example['output']) + "\n"

    prompt += "\nTest Input:\n"
    prompt += format_grid_for_comparison(puzzle_data['test'][0]['input'])
    prompt += "\n</ascii_grids>\n"

    # Add previous attempts information if available
    if previous_attempts and len(previous_attempts) > 0:
        prompt += "\n<previous_attempts>\n"
        prompt += "Your following attempted solutions failed to correctly solve the puzzle.\n"
        prompt += "Propose a new strategy that is different from these previous approaches.\n\n"

        for i, attempt in enumerate(previous_attempts, 1):
            prompt += f"Attempt {i} Reasoning:\n{attempt['reasoning']}\n"
            prompt += f"Result: {attempt['result']}\n\n"

        prompt += "Your solution must use a new approach that differs from the failed attempts above."
        prompt += "\n</previous_attempts>\n"

    return prompt

def format_grid_for_comparison(grid: List[List[Any]]) -> str:
    """Format a grid as a string for display"""
    return "\n".join([" ".join(map(str, row)) for row in grid])

def create_comparison_table(correct_answer: List[List[int]], solution: List[List[int]]) -> Table:
    """Create a rich table comparing expected and actual solutions"""
    table = Table(title="Grid Comparison")
    table.add_column("Correct Answer", justify="left")
    table.add_column("Solution", justify="left")

    # Get the maximum dimensions from both grids
    max_rows = max(len(correct_answer), len(solution))

    for i in range(max_rows):
        correct_row = correct_answer[i] if i < len(correct_answer) else []
        solution_row = solution[i] if i < len(solution) else []

        # Format the correct row
        correct_str = " ".join(map(str, correct_row))

        # Format the solution row with differences highlighted
        solution_text = Text()
        max_cols = max(len(correct_row), len(solution_row))
        for j in range(max_cols):
            val = solution_row[j] if j < len(solution_row) else None
            correct_val = correct_row[j] if j < len(correct_row) else None

            if val is not None and correct_val is not None and val == correct_val:
                 solution_text.append(f"{val} ")  # Matching values in default color
            elif val is not None:
                 solution_text.append(f"{val} ", style="red")  # Mismatched or extra values in red
            elif correct_val is not None:
                 # This case handles missing values in the solution grid
                 solution_text.append("  ", style="red") # Represent missing with space, highlighted red
            else:
                 solution_text.append("  ") # Should not happen if max_cols is correct

        table.add_row(correct_str, solution_text)

    return table

def format_time(seconds):
    """Format time in a human-readable way"""
    return str(timedelta(seconds=round(seconds)))


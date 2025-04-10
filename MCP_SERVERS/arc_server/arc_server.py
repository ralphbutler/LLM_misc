import json
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("ARC Puzzle Server")

# Define the directory where ARC puzzles are stored
# PUZZLE_DIR = "/Users/rbutler/Desktop/work_ARC/ARC-AGI-official/data/training"
PUZZLE_DIR = "/Users/rbutler/MCPSERVERS/arc_server/SAMPLE_PUZZLES"

@mcp.resource("puzzle://{puzzle_name}")
def get_puzzle(puzzle_name: str) -> str:
    """
    Get an ARC puzzle by name.
    
    Args:
        puzzle_name: The name of the puzzle file (e.g., 'd037b0a7.json')
    
    Returns:
        The puzzle data as a JSON string
    """
    try:
        # Ensure the puzzle name has a .json extension
        if not puzzle_name.endswith('.json'):
            puzzle_name = f"{puzzle_name}.json"
        
        # Build the full path to the puzzle file
        puzzle_path = os.path.join(PUZZLE_DIR, puzzle_name)
        
        # Ensure the path exists
        if not os.path.exists(puzzle_path):
            return json.dumps({
                "error": f"Puzzle {puzzle_name} not found",
                "puzzle_dir": PUZZLE_DIR
            })
        
        # Read and return the puzzle data
        with open(puzzle_path, 'r') as f:
            puzzle_data = f.read()
            
        # Validate that it's proper JSON
        json.loads(puzzle_data)  # This will raise an exception if not valid JSON
        return puzzle_data
        
    except Exception as e:
        return json.dumps({"error": f"Failed to retrieve puzzle: {str(e)}"})

@mcp.tool()
def list_available_puzzles() -> list:
    """
    List all available ARC puzzles in the puzzle directory.
    
    Returns:
        A list of available puzzle names
    """
    try:
        puzzle_dir = Path(PUZZLE_DIR)
        if not puzzle_dir.exists():
            return [f"Puzzle directory not found: {PUZZLE_DIR}"]
            
        puzzles = [f.name for f in puzzle_dir.glob("*.json")]
        return puzzles
    except Exception as e:
        return [f"Error listing puzzles: {str(e)}"]

@mcp.prompt()
def arc_puzzle_prompt(puzzle_data: str) -> str:
    """
    A basic prompt for presenting an ARC puzzle to the model.
    
    Args:
        puzzle_data: The JSON string containing the puzzle data
    
    Returns:
        A formatted prompt string
    """
    return f"""
Please solve this ARC puzzle:

{puzzle_data}

Analyze the training examples to understand the pattern, then apply the transformation to the test input.
Explain your reasoning and provide the expected output.
"""

# Simple test to make sure the directory exists
if not os.path.exists(PUZZLE_DIR):
    print(f"Warning: Puzzle directory does not exist: {PUZZLE_DIR}")
    print(f"Please update PUZZLE_DIR in the script.")

if __name__ == "__main__":
    mcp.run()  # debug=True)

import random
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Random Number Generator")

@mcp.tool()
def randtool(input_string: str) -> str:
    """
    Generate a random integer between 1 and 100 inclusive and append it to the input string.
    
    Args:
        input_string: A string to be concatenated with the random number
    
    Returns:
        The input string followed by a space and a random integer between 1 and 100
    """
    random_number = random.randint(1, 100)
    return f"{input_string} {random_number}"

if __name__ == "__main__":
    mcp.run()

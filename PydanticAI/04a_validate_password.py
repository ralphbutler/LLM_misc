
# This example is clean for cases where we don't need any context or dependencies.
# It is a good example of when to use tool_plain - simple functions that just take inputs
# and return outputs without needing any context or state.

"""Example demonstrating how to use PydanticAI plain tools.

This example shows how to create a simple password validation agent
that uses tools to check different password requirements.
"""

from pydantic_ai import Agent


# Create the agent
password_agent = Agent(
    'openai:gpt-4o',
    system_prompt=(
        'You are a password validation assistant. '
        'Use the provided tools to check if passwords meet security requirements. '
        'Respond with a clear yes/no and list any failed requirements.'
    ),
)


@password_agent.tool_plain
def check_length(password: str) -> bool:
    """Check if password meets minimum length requirement (8 characters).
    
    Args:
        password: The password to check.
    """
    return len(password) >= 8


@password_agent.tool_plain
def check_complexity(password: str) -> dict[str, bool]:
    """Check if password contains required character types.
    
    Args:
        password: The password to check.
        
    Returns:
        Dictionary with results of each complexity check.
    """
    return {
        'has_uppercase': any(c.isupper() for c in password),
        'has_lowercase': any(c.islower() for c in password),
        'has_number': any(c.isdigit() for c in password),
        'has_special': any(not c.isalnum() for c in password)
    }


async def main():
    # Test some passwords
    passwords = [
        "short",
        "NoSpecialChars123",
        "Secure@Password123",
    ]
    
    for password in passwords:
        result = await password_agent.run(
            f'Is this a valid password? "{password}"'
        )
        print(f"\nPassword: {password}")
        print(f"Result: {result.data}")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())

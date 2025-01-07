
# This program demonstrates how to build a restaurant reservation system using PydanticAI with robust validation logic.
# It shows how to handle both successful and failed reservations through two data models:
#    ValidReservation and InvalidReservation.
# The program implements sophisticated validation rules that:
# - Prevents reservations for past dates
# - Limits bookings to 1 year in advance
# - Restricts weekend party sizes to 15 people
# - Checks for date availability (simulated by marking the 15th of each month as fully booked)

# When validation fails, the system uses a retry mechanism that gives the AI model feedback about
# why the reservation failed and allows it to adjust its response. The example includes test cases
# for various scenarios like valid bookings, oversized weekend parties, and attempts to book on
# fully booked dates.


"""Example demonstrating how to use result validation with PydanticAI.

This example shows how to implement custom validation logic for agent responses,
including both data validation and business logic validation.
"""

import asyncio
from typing import Union, Annotated
from datetime import date, datetime

import logfire
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypeAlias

from pydantic_ai import Agent, ModelRetry, RunContext

# Configure logfire
logfire.configure(send_to_logfire='if-token-present')

class ValidReservation(BaseModel):
    """Response when a reservation can be made."""
    guest_name: str = Field(min_length=2)
    date: date
    party_size: Annotated[int, Field(ge=1, le=20)]
    special_requests: str = ""

class InvalidReservation(BaseModel):
    """Response when a reservation cannot be made."""
    error_message: str

# Define response type alias
Response: TypeAlias = Union[ValidReservation, InvalidReservation]

# Create the agent
agent = Agent(
    'openai:gpt-4',
    result_type=Response,  # type: ignore
    system_prompt="""You are a restaurant reservation system.
    Process reservation requests and validate them against business rules.
    Return either a valid reservation or an error message."""
)

@agent.result_validator
async def validate_result(ctx: RunContext[None], result: Response) -> Response:
    """Custom validation logic for reservations.
    
    When ModelRetry is raised, PydanticAI will retry the query with the error message,
    allowing the model to adjust its response. This continues until either:
    - A valid response is achieved
    - The retry limit (3) is reached
    - An InvalidReservation is returned
    """
    if isinstance(result, InvalidReservation):
        return result
    
    # Get current date for comparisons
    today = datetime.now().date()
    
    try:
        # Validate reservation date isn't in the past
        if result.date < today:
            raise ModelRetry(
                "Please provide a future date - reservations cannot be made for past dates. "
                f"Today is {today}."
            )
        
        # Validate reservation isn't too far in the future
        max_future_date = today.replace(year=today.year + 1)
        if result.date > max_future_date:
            raise ModelRetry(
                f"Please provide a date before {max_future_date} - "
                "reservations cannot be made more than 1 year in advance."
            )
        
        # Validate weekend party size limits
        if result.date.weekday() >= 5 and result.party_size > 15:  # Weekend
            suggested_size = min(result.party_size, 15)
            raise ModelRetry(
                f"Weekend reservations are limited to 15 people. "
                f"Please adjust the party size or choose a weekday. "
                f"Current party size: {result.party_size}"
            )
        
        # Hypothetical database check for availability
        if is_fully_booked(result.date):
            return InvalidReservation(
                error_message=(
                    f"Sorry, we are fully booked on {result.date}. "
                    "Please try a different date."
                )
            )
        
        return result
        
    except ValueError as e:
        # Handle any Pydantic validation errors clearly
        raise ModelRetry(f"Invalid reservation details: {str(e)}")

def is_fully_booked(check_date: date) -> bool:
    """Simulate checking if a date is fully booked.
    
    In a real system, this would query a database.
    For demo purposes, we'll say the 15th of any month is fully booked.
    """
    return check_date.day == 15

async def process_reservation(request: str):
    """Process a reservation request and print the result."""
    try:
        result = await agent.run(request)
        
        if isinstance(result.data, ValidReservation):
            print("\nReservation Confirmed:")
            print(f"Guest: {result.data.guest_name}")
            print(f"Date: {result.data.date}")
            print(f"Party Size: {result.data.party_size}")
            if result.data.special_requests:
                print(f"Special Requests: {result.data.special_requests}")
        else:
            print(f"\nReservation Failed: {result.data.error_message}")
            
    except Exception as e:
        if "maximum retries" in str(e).lower():
            print(
                "\nUnable to create a valid reservation after several attempts. "
                "Please check the reservation details and try again with more specific information."
            )
        else:
            print(f"\nError processing reservation: {str(e)}")

async def main():
    # Test various reservation scenarios
    test_requests = [
        # Valid reservation
        "Book a table for John Smith, party of 4, on February 10th, 2025",
        
        # Should trigger party size validation but allow retry
        "I need a table for 16 people on Saturday January 11th, 2025. We're celebrating a birthday!",
        
        # Should return InvalidReservation (fully booked date)
        "Book for Alice Johnson on January 15th, 2025, party of 2",
        
        # Valid with special request
        "Make a reservation for Bob Wilson on March 1st, 2025, party of 6, high chair needed"
    ]
    
    for request in test_requests:
        print(f"\nProcessing request: {request}")
        await process_reservation(request)

if __name__ == '__main__':
    asyncio.run(main())

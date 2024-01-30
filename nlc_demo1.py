
import sys, os, time

from NLC import NLC_Agent


def get_weather(city: str) -> str:
    """Get the current weather given a city."""
    rv = f"The weather in {city} is warm and breezy."
    return rv

from typing import List   # just to demo function with List arg
def calculate_mortgage_payment(
    loan_amount: int,
    interest_rate: float,
    loan_term: int,
    # list_example: List[int],
) -> float:
    """Get the monthly mortgage payment given an interest rate percentage."""

    # P = L[c(1 + c)^n]/[(1 + c)^n - 1]
    #   - P = monthly payment
    #   - L = loan amount
    #   - c = monthly interest rate (annual interest rate divided by 12)
    #   - n = total number of payments (loan term in years multiplied by 12)
    c = interest_rate / 12.0
    cn = (1+c)**loan_term
    monthly_payment = (loan_amount * c * cn) / (cn-1)
    rv = f"The monthly payment would be {monthly_payment:0.2f}."
    return rv


def main():
    nlc_agent = NLC_Agent(model="ollama/mistral")
    nlc_agent.add_tool_func("calculate_mortgage_payment", calculate_mortgage_payment)
    nlc_agent.add_tool_func("get_weather", get_weather)

    usr1 = "Please get the weather in Boston for today."
    usr2 = "Who was the first president of the USA?"
    usr3 = """
        Determine the monthly mortgage payment for a loan amount of $200,000,
        an interest rate of 4%, and a loan term of 360 months.
    """

    for usr_msg in [usr1,usr2,usr3]:
        nlc_agent.run(usr_msg)

main()


from pydantic import BaseModel, model_validator

from openai import OpenAI

class LineItem(BaseModel):
    name: str
    price: float
    quantity: int

class Recipe(BaseModel) :
    recipe_name: str
    products: list[LineItem]
    total_cost: float
    @model_validator(mode="after")
    def total_cost_adds_up(self) -> "Recipe":
        curr_cost = sum(prod.price * prod.quantity for prod in self.products)
        assert self.total_cost == curr_cost, "Total cost does not add up"
        return self

client = OpenAI()

msg = """We can make the chili-mac recipe for $4.25 by combining:
             2 boxes-macaroni for $1.50 each and 1 can-of-chili for $1.25"""
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06", # "gpt-4o-mini",
    messages=[
        {"role": "user", "content": msg },
    ],
    response_format=Recipe,
)

message = completion.choices[0].message
if message.parsed:
    print("RECIPE", message.parsed.recipe_name)
    print("PRODUCTS")
    for prod in message.parsed.products:
        print("   ",prod)
    print("TOTAL COST", message.parsed.total_cost)
else:
    print(message.refusal)

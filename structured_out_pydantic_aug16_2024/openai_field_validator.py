
from pydantic import BaseModel, field_validator

from openai import OpenAI

class User(BaseModel):
    name: str
    age: int

    @field_validator("name")
    def name_must_be_uppercase(cls, v: str):
        if v.upper() != v:
            raise ValueError("Name must be uppercase, please fix.")
        return v

client = OpenAI()

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06", # "gpt-4o-mini",
    messages=[
        {"role": "user", "content": "JIMBOB is 20 years old."},
    ],
    response_format=User,
)

message = completion.choices[0].message
if message.parsed:
    print(message.parsed.name)
    print(message.parsed.age)
else:
    print(message.refusal)

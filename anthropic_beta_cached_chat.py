import sys, os, time

from anthropic import Anthropic
client = Anthropic()

with open("./511145.12.first8") as f:
    bio_text = f.read()

import anthropic

client = anthropic.Anthropic()

query1 = "Provide a brief summary of the cached text."

stime = time.time()
response = client.beta.prompt_caching.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    system=[
      {
        "type": "text",
        "text": "You are an AI assistant tasked with answering questions about biological data."
      },
      {
        "type": "text",
        "text": bio_text,
        "cache_control": {"type": "ephemeral"}
      }
    ],
    messages=[{"role": "user", "content": "Provide a brief summary of the cached text."} ],
)
print(response)
print("USAGE:", response.usage)
print(f"TIME {time.time()-stime:0.2f}")
print("-"*50)

user_query = "What is the name of the genome with id 511145.12 and how many base pairs are in it?"

stime = time.time()
response = client.beta.prompt_caching.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "You are an AI assistant tasked with answering questions about biological data."
        },
        {
            "type": "text",
            "text": bio_text,
            "cache_control": {"type": "ephemeral"}
        }
    ],
    messages=[
        {"role": "user", "content": f"Using cached text, answer this question: {user_query}"},
    ],
)
print(response)
print("USAGE:", response.usage)
print(f"TIME {time.time()-stime:0.2f}")
print("-"*50)

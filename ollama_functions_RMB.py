
# this program demos a technique to use until Ollama officially supports function-calling

# have ollama create JSON format to suggest a function call that the user can make

import sys, os, time, json

from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

def multiply_large_integers(int1: int, int2: int):
    return int1 * int2

functions=[
    {
        "name": "multiply_large_integers",
        "description": "accept two large numbers and compute their result",
        "parameters": {
            "number1": {
                "type": "int",
                "description": "the first integer",
            },
            "number2": {
                "type": "int",
                "description": "the second integer",
            },
        },
    }
]

func_schema = {
    "function_name": "name of the function to be executed",
    "arg_list": "list of argument values for invocation, in order, comma-separated, and of the correct type based on the schema"
}
answer_schema = {
    "answer": "answer to user query"
}

system_message = f"""
    You are a helpful assistant.
    You have access to these functions:
        {functions}

    Always create JSON output.

    If you determine that the best reply to a query is to suggest a function
    call, then the proper format for the JSON output is to use this schema:
        {func_schema}

    If you are not going to suggest a function call, then the proper format
    for the JSON output is to use this different schema:
        {answer_schema}
"""

user_messages = [
    "tell the name of the first president of the USA",
    "compute the product 123456 * 654321",
]

for user_message in user_messages:
    messages = [
        { "role": "system", "content": system_message },
        { "role": "user", "content": user_message },
    ]

    response = client.chat.completions.create(
        model="mistral",
        temperature=0.0,
        messages=messages,
    )
    response = response.choices[0].message.content
    response = response.replace("\\","")
    print(response)
    json_response = json.loads(response)
    if "answer" in json_response:
        print("ANSWER:", json_response["answer"])
    elif "function_name" in json_response:
        funcname = json_response["function_name"]
        arg_list = json_response["arg_list"]
        func = globals()[funcname]
        rc = func(*arg_list)
        print("RC:",rc)
    print("-"*50)

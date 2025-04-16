
import sys, os, json

os.environ['LITELLM_LOG'] = 'DEBUG'
from litellm import completion 
import litellm
litellm.enable_json_schema_validation = True   ## RMB: nec for gemini

from pydantic import BaseModel 

class CookieRecipe(BaseModel):
    name: str
    ingredients: list[str]
    instructions: str

model2apibase = {
    # "openai/DeepSeek-R1-Distill-Qwen-32B" : "http://localhost:1234/v1",  # lmstudio
    # "openai/qwen2.5-7b-instruct-mlx"      : "http://localhost:1234/v1",  # lmstudio
    # "openai/qwen2.5-32b-instruct"         : "http://localhost:1234/v1",  # lmstudio
    # "openai/phi-4"                        : "http://localhost:1234/v1",  # lmstudio
    # "openai/gemma-3-27b-it"               : "http://localhost:1234/v1",  # lmstudio
    # "openai/qwq-32b"                      : "http://localhost:1234/v1",  # lmstudio

    # RMB NEC TO USE ollama_chat for structured output for pydantic models
    "ollama_chat/llama3.2:3b"                    : "http://localhost:11434",    # ollama

    # "gpt-4o-mini"                         : "",  # known to litellm
    # "gpt-4o"                              : "",  # known to litellm
    # "o3-mini"                             : "",  # known to litellm

    # "gemini/gemini-2.0-flash"             : "",  # known to litellm
    # "gemini/gemini-2.5-pro-exp-03-25"     : "",  # known to litellm

    # "claude-3-7-sonnet-latest"            : "",  # known to litellm
}

messages = [
    {"role": "user", "content": "provide a good cookie recipe."},
]

model_name = list(model2apibase.keys())[0]
api_base = model2apibase[model_name]
print("MODEL", model_name, "API_BASE", api_base)

response = completion(
    model=model_name,
    api_base=api_base,
    messages=messages,
    max_tokens=2048,
    temperature=0.0,
    response_format=CookieRecipe,   ## RMB: NOT response_model
)

print("-"*50)
print("response")
print(response)
print()

content = response.choices[0].message.content
print("content", content)

jcontent = json.loads(content)
print("-"*50)
print("jcontent")
print(jcontent)
print(list(jcontent.keys()))
print()

response_model = CookieRecipe(**jcontent)
print("-"*50)
print("response_model")
print("name:",response_model.name)
print("ingredients:",response_model.ingredients)
print("instructions:",response_model.instructions)

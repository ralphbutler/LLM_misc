import os
import textwrap
import polyllm
from pydantic import BaseModel, Field

IMAGE_PATH = "" # "/Users/rmbutler/Desktop/Gemini_Generated_Image.jpg"

# I heavily suggest llama3.1 for good  JSON  &  function calling / tool usage
# Llama3.1 found here: `https://huggingface.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main`
LLAMA_PYTHON_MODEL = "" # "path/to/model.gguf"

# Make sure llama_cpp is running: `python -m llama_cpp.server --model path/to/model.gguf`
# I heavily suggest llama3.1 for good  JSON  &  function calling / tool usage
# Llama3.1 found here: `https://huggingface.co/bullerwins/Meta-Llama-3.1-8B-Instruct-GGUF/tree/main`
LLAMA_PYTHON_SERVER_PORT = "" # "8000"

# Make sure Ollama is running
# I heavily suggest llama3.1 for good  JSON  &  function calling / tool usage
# Llama3.1 found here: `ollama pull llama3.1`
OLLAMA_MODEL = "" # "llama3.1"

OPENAI_MODEL = "" # "gpt-4o-2024-08-06"

# "gemini-1.5-pro" sometimes hits rate limits for free tier usage
# "gemini-1.5-pro" is required and has been hardcoded for structured object use below
GOOGLE_MODEL = "gemini-1.5-pro-002"

ANTHROPIC_MODEL = "" # "claude-3-5-sonnet-20240620"


image_exists = os.path.isfile(IMAGE_PATH)
if not image_exists:
    print("\033[38;5;197mWarning: The image file does not exist.\033[0m")
    print("\033[38;5;197mMulti-modal tests will be skipped.\033[0m")
    print("\033[38;5;197mPlease update IMAGE_PATH in this script to point to a valid image file.\033[0m")
    print()
if not LLAMA_PYTHON_MODEL:
    print("\033[38;5;197mNo llama-cpp-python model specified.\033[0m")
    print("\033[38;5;197mllama-cpp-python tests will be skipped.\033[0m")
    print("\033[38;5;197mPlease update LLAMA_PYTHON_MODEL in this script to point to a .gguf file.\033[0m")
    print()
if not LLAMA_PYTHON_SERVER_PORT:
    print("\033[38;5;197mNo llama.cpp port specified.\033[0m")
    print("\033[38;5;197mllama.cpp tests will be skipped.\033[0m")
    print("\033[38;5;197mPlease update LLAMA_PYTHON_SERVER_PORT in this script to point to a running llama_cpp_python server.\033[0m")
    print()
if not OLLAMA_MODEL:
    print("\033[38;5;197mNo Ollama model specified.\033[0m")
    print("\033[38;5;197mOllama tests will be skipped.\033[0m")
    print("\033[38;5;197mPlease update OLLAMA_MODEL in this script to specify a downloaded Ollama model.\033[0m")
    print()
if not OPENAI_MODEL:
    print("\033[38;5;197mNo OpenAI model specified.\033[0m")
    print("\033[38;5;197mOpenAI tests will be skipped.\033[0m")
    print("\033[38;5;197mPlease update OPENAI_MODEL in this script to specify an OpenAI model.\033[0m")
    print()
if not GOOGLE_MODEL:
    print("\033[38;5;197mNo Google model specified.\033[0m")
    print("\033[38;5;197mGoogle tests will be skipped.\033[0m")
    print("\033[38;5;197mPlease update GOOGLE_MODEL in this script to specify a Google model.\033[0m")
    print()
if not ANTHROPIC_MODEL:
    print("\033[38;5;197mNo Anthropic model specified.\033[0m")
    print("\033[38;5;197mAnthropic tests will be skipped.\033[0m")
    print("\033[38;5;197mPlease update ANTHROPIC_MODEL in this script to specify an Anthropic model.\033[0m")
    print()


# Example for plain text conversations.
# Tests correct handling of system, user, and assistant message roles.
text_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "generate a small image of a kitty."},
    # {"role": "assistant", "content": "Why did the scarecrow win an award?\nBecause he was outstanding in his field!"},
    # {"role": "user", "content": "Great! Tell me another joke!"},
]


# Example for multimodal conversations.
# Tests correct handling of images.
image_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": IMAGE_PATH}},
        ],
    },
]


# Example for json_object output.
# Tests enforcement of JSON responses.
json_messages = [
    {
        "role": "user",
        "content": textwrap.dedent("""
            Find the name of the first president of the USA and get the years that he served.
            Produce the result in JSON that matches this schema:
                {
                    "first_name": "first name",
                    "last_name":  "last name",
                    "years_served": "years served"
                }
            """).strip()
    }
]



class Flight(BaseModel):
    departure_time: str = Field(description="The time the flight departs")
    destination: str = Field(description="The destination of the flight")
class FlightList(BaseModel):
    flights: list[Flight] = Field(description="A list of known flight details")
flight_list_schema = polyllm.pydantic_to_schema(FlightList, indent=2)
pydantic_messages = [
    {
        "role": "user",
        "content": textwrap.dedent(f"""
            Write a list of 2 to 5 random flight details.
            Produce the result in JSON that matches this schema:
            {flight_list_schema}
            """).strip()
    }
]



# Example for function calling / tool usage.
# Tests ability to use tools to answer questions, answer questions without tools when no tool is helpful, and avoid using tools or trying to answer when neither is applicable.
def multiply_large_numbers(a: int, b: int) -> int:
    """Multiplies two large numbers."""
    return a * b
# Extracted: function name, argument names, argument types, docstring

tool_message0 = [{"role": "user", "content": "What is 123456 multiplied by 654321?"}]
tool_message1 = [{"role": "user", "content": "How old was George Washington when he became president?"}]
tool_message2 = [{"role": "user", "content": "What is the current temperature in Kalamazoo?"}]



# Examples

if LLAMA_PYTHON_MODEL:
    from llama_cpp import Llama
    llm = Llama(
        model_path=LLAMA_PYTHON_MODEL,
        n_ctx=1024,
        n_gpu_layers=-1,
        verbose=False,
    )
    print('\033[38;5;41m======== LlamaCPP Python\033[0m')
    print("\033[38;5;93m==== Testing plain text conversation: (Should tell a joke)\033[0m")
    print(polyllm.generate(llm, text_messages))
    print("\033[38;5;93m\n==== Testing JSON mode: (Should format: George, Washington, 1789-1797)\033[0m")
    print(polyllm.generate(llm, json_messages, json_object=True))
    print("\033[38;5;93m\n==== Testing Structured Output mode: (Should list 2-5 random flight times and destinations)\033[0m")
    output = polyllm.generate(llm, json_messages, json_schema=FlightList)
    print(output)
    print(polyllm.json_to_pydantic(output, FlightList))
    print("\033[38;5;93m\n==== Testing tool usage: (Should choose multiply_large_numbers, a=123456, b=654321)\033[0m")
    print(polyllm.generate_tools(llm, tool_message0, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool: (Should respond 57)\033[0m")
    print(polyllm.generate_tools(llm, tool_message1, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool or knowledge: (Should refuse to respond)\033[0m")
    print(polyllm.generate_tools(llm, tool_message2, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing streaming: (Should tell a joke)\033[0m")
    for chunk in polyllm.generate_stream(llm, text_messages):
        print(chunk, end='', flush=True)
    print()

if LLAMA_PYTHON_SERVER_PORT:
    print('\033[38;5;41m======== LlamaCPP Python Server\033[0m')
    print("\033[38;5;93m==== Testing plain text conversation: (Should tell a joke)\033[0m")
    print(polyllm.generate(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", text_messages))
    print("\033[38;5;93m\n==== Testing JSON mode: (Should format: George, Washington, 1789-1797)\033[0m")
    print(polyllm.generate(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", json_messages, json_object=True))
    print("\033[38;5;93m\n==== Testing Structured Output mode: (Should list 2-5 random flight times and destinations)\033[0m")
    output = polyllm.generate(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", json_messages, json_schema=FlightList)
    print(output)
    print(polyllm.json_to_pydantic(output, FlightList))
    print("\033[38;5;93m\n==== Testing tool usage: (Should choose multiply_large_numbers, a=123456, b=654321)\033[0m")
    print(polyllm.generate_tools(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", tool_message0, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool: (Should respond 57)\033[0m")
    print(polyllm.generate_tools(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", tool_message1, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool or knowledge: (Should refuse to respond)\033[0m")
    print(polyllm.generate_tools(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", tool_message2, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing streaming: (Should tell a joke)\033[0m")
    for chunk in polyllm.generate_stream(f"llamacpp/{LLAMA_PYTHON_SERVER_PORT}", text_messages):
        print(chunk, end='', flush=True)
    print()

if OLLAMA_MODEL:
    print('\n\033[38;5;41m======== Ollama\033[0m')
    print("\033[38;5;93m==== Testing plain text conversation: (Should tell a joke)\033[0m")
    print(polyllm.generate(f"ollama/{OLLAMA_MODEL}", text_messages))
    print("\033[38;5;93m\n==== Testing JSON mode: (Should format: George, Washington, 1789-1797)\033[0m")
    print(polyllm.generate(f"ollama/{OLLAMA_MODEL}", json_messages, json_object=True))
    print("\033[38;5;93m\n==== Testing tool usage: (Should choose multiply_large_numbers, a=123456, b=654321)\033[0m")
    print(polyllm.generate_tools(f"ollama/{OLLAMA_MODEL}", tool_message0, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool: (Should respond 57)\033[0m")
    print(polyllm.generate_tools(f"ollama/{OLLAMA_MODEL}", tool_message1, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool or knowledge: (Should refuse to respond)\033[0m")
    print(polyllm.generate_tools(f"ollama/{OLLAMA_MODEL}", tool_message2, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing streaming: (Should tell a joke)\033[0m")
    for chunk in polyllm.generate_stream(f"ollama/{OLLAMA_MODEL}", text_messages):
        print(chunk, end='', flush=True)
    print()

if OPENAI_MODEL:
    print('\n\033[38;5;41m======== OpenAI\033[0m')
    print("\033[38;5;93m==== Testing plain text conversation: (Should tell a joke)\033[0m")
    print(polyllm.generate(OPENAI_MODEL, text_messages))
    if image_exists:
        print("\033[38;5;93m\n==== Testing multi-modal image input: (Should describe your image)\033[0m")
        print(polyllm.generate(OPENAI_MODEL, image_messages))
    print("\033[38;5;93m\n==== Testing JSON mode: (Should format: George, Washington, 1789-1797)\033[0m")
    print(polyllm.generate(OPENAI_MODEL, json_messages, json_object=True))
    print("\033[38;5;93m\n==== Testing Structured Output mode: (Should list 2-5 random flight times and destinations)\033[0m")
    output = polyllm.generate(OPENAI_MODEL, json_messages, json_schema=FlightList)
    print(output)
    print(polyllm.json_to_pydantic(output, FlightList))
    print("\033[38;5;93m\n==== Testing tool usage: (Should choose multiply_large_numbers, a=123456, b=654321)\033[0m")
    print(polyllm.generate_tools(OPENAI_MODEL, tool_message0, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool: (Should respond 57)\033[0m")
    print(polyllm.generate_tools(OPENAI_MODEL, tool_message1, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool or knowledge: (Should refuse to respond)\033[0m")
    print(polyllm.generate_tools(OPENAI_MODEL, tool_message2, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing streaming: (Should tell a joke)\033[0m")
    for chunk in polyllm.generate_stream(OPENAI_MODEL, text_messages):
        print(chunk, end='', flush=True)
    print()

if GOOGLE_MODEL:
    print('\n\033[38;5;41m======== Google\033[0m')
    print("\033[38;5;93m==== Testing plain text conversation: (Should tell a joke)\033[0m")
    print(polyllm.generate(GOOGLE_MODEL, text_messages))
    if image_exists:
        print("\033[38;5;93m\n==== Testing multi-modal image input: (Should describe your image)\033[0m")
        print(polyllm.generate(GOOGLE_MODEL, image_messages))
    print("\033[38;5;93m\n==== Testing JSON mode: (Should format: George, Washington, 1789-1797)\033[0m")
    print(polyllm.generate(GOOGLE_MODEL, json_messages, json_object=True))
    print("\033[38;5;93m\n==== Testing Structured Output mode: (Should list 2-5 random flight times and destinations)\033[0m")
    output = polyllm.generate("gemini-1.5-pro", json_messages, json_schema=FlightList)
    print(output)
    print(polyllm.json_to_pydantic(output, FlightList))
    print("\033[38;5;93m\n==== Testing tool usage: (Should choose multiply_large_numbers, a=123456, b=654321)\033[0m")
    print(polyllm.generate_tools(GOOGLE_MODEL, tool_message0, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool: (Should respond 57)\033[0m")
    print(polyllm.generate_tools(GOOGLE_MODEL, tool_message1, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool or knowledge: (Should refuse to respond)\033[0m")
    print(polyllm.generate_tools(GOOGLE_MODEL, tool_message2, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing streaming: (Should tell a joke)\033[0m")
    for chunk in polyllm.generate_stream(GOOGLE_MODEL, text_messages):
        print(chunk, end='', flush=True)
    print()

if ANTHROPIC_MODEL:
    print('\n\033[38;5;41m======== Anthropic\033[0m')
    print("\033[38;5;93m==== Testing plain text conversation: (Should tell a joke)\033[0m")
    print(polyllm.generate(ANTHROPIC_MODEL, text_messages))
    if image_exists:
        print("\033[38;5;93m\n==== Testing multi-modal image input: (Should describe your image)\033[0m")
        print(polyllm.generate(ANTHROPIC_MODEL, image_messages))
    print("\033[38;5;93m\n==== Testing JSON mode: (Should format: George, Washington, 1789-1797)\033[0m")
    print(polyllm.generate(ANTHROPIC_MODEL, json_messages, json_object=True))
    print("\033[38;5;93m\n==== Testing tool usage: (Should choose multiply_large_numbers, a=123456, b=654321)\033[0m")
    print(polyllm.generate_tools(ANTHROPIC_MODEL, tool_message0, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool: (Should respond 57)\033[0m")
    print(polyllm.generate_tools(ANTHROPIC_MODEL, tool_message1, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing tool usage with no relevant tool or knowledge: (Should refuse to respond)\033[0m")
    print(polyllm.generate_tools(ANTHROPIC_MODEL, tool_message2, tools=[multiply_large_numbers]))
    print("\033[38;5;93m\n==== Testing streaming: (Should tell a joke)\033[0m")
    for chunk in polyllm.generate_stream(ANTHROPIC_MODEL, text_messages):
        print(chunk, end='', flush=True)
    print()

import backoff
import base64
import json
import os
import textwrap
import time
from typing import Callable, Generator
from pydantic import BaseModel


try:
    from llama_cpp import Llama, LlamaGrammar
    from llama_cpp.llama_grammar import json_schema_to_gbnf
    llamapython_import = True
except ImportError:
    llamapython_import = False
    Llama = None

try:
    from openai import OpenAI
    openai_import = True
except ImportError:
    openai_import = False

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
    google_import = True
except ImportError:
    google_import = False

try:
    import anthropic
    anthropic_import = True
except ImportError:
    anthropic_import = False


openai_key = None
google_key = None
anthropic_key = None
lazy_loaded = False
def lazy_load():
    global openai_client, openai_models, openai_key
    global google_models, google_key
    global anthropic_client, anthropic_models, anthropic_key
    global lazy_loaded

    if lazy_loaded:
        return

    if openai_import and os.environ.get("OPENAI_API_KEY"):
        openai_client = OpenAI()
        openai_models = [model.id for model in list(openai_client.models.list())]
        openai_key = True
    else:
        openai_models = []
        openai_key = False

    if google_import and os.environ.get("GOOGLE_API_KEY"):
        genai.configure()
        google_models = [model.name.split('/')[1] for model in genai.list_models()]
        google_key = True
    else:
        google_models = []
        google_key = False

    if anthropic_import and os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic()
        anthropic_models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
        anthropic_key = True
    else:
        anthropic_models = []
        anthropic_key = False

    lazy_loaded = True

MODEL_ERR_MSG = "Unsupported model: {model}"
if not all((llamapython_import, openai_import, google_import, anthropic_import)):
    missing = []
    if not llamapython_import:
        missing.append('llama-cpp-python')
    if not openai_import:
        missing.append('openai')
    if not google_import:
        missing.append('google-generativeai')
    if not anthropic_import:
        missing.append('anthropic')
    MODEL_ERR_MSG += ". Failed imports: pip install " + " ".join(missing) + "."
if any((openai_import and (openai_key is False), google_import and (google_key is False), anthropic_import and (anthropic_key is False))):
    missing = []
    if openai_import and (openai_key is False):
        missing.append('OPENAI_API_KEY')
    if google_import and (google_key is False):
        missing.append('GOOGLE_API_KEY')
    if anthropic_import and (anthropic_key is False):
        missing.append('ANTHROPIC_API_KEY')
    MODEL_ERR_MSG += ". Missing API keys: " + ", ".join(missing) + "."


def generate(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    json_object: bool = False,
    json_schema: BaseModel|None = None,
) -> str:
    if json_object and json_schema:
        raise ValueError("generate() cannot simultaneously support JSON mode (json_object) and Structured Object mode (json_schema)")

    if llamapython_import and isinstance(model, Llama):
        return _llamapython(model, messages, temperature, json_object, json_schema)
    elif model.startswith('llamacpp/'):
        return _llamacpp(model, messages, temperature, json_object, json_schema)
    elif model.startswith('ollama/'):
        return _ollama(model, messages, temperature, json_object, json_schema)
    else:
        lazy_load()
        if model in openai_models:
            return _openai(model, messages, temperature, json_object, json_schema)
        elif model in google_models:
            return _google(model, messages, temperature, json_object, json_schema)
        elif model in anthropic_models:
            return _anthropic(model, messages, temperature, json_object, json_schema)
        else:
            raise ValueError(MODEL_ERR_MSG.format(model=model))

def generate_stream(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    json_object: bool = False,
    json_schema: BaseModel|None = None,
) -> Generator[str, None, None]:
    if json_object and json_schema:
        raise ValueError("generate_stream() cannot simultaneously support JSON mode (json_object) and Structured Object mode (json_schema)")

    if llamapython_import and isinstance(model, Llama):
        return _llamapython_stream(model, messages, temperature, json_object, json_schema)
    elif model.startswith('llamacpp/'):
        return _llamacpp_stream(model, messages, temperature, json_object, json_schema)
    elif model.startswith('ollama/'):
        return _ollama_stream(model, messages, temperature, json_object, json_schema)
    else:
        lazy_load()
        if model in openai_models:
            return _openai_stream(model, messages, temperature, json_object, json_schema)
        elif model in google_models:
            return _google_stream(model, messages, temperature, json_object, json_schema)
        elif model in anthropic_models:
            return _anthropic_stream(model, messages, temperature, json_object, json_schema)
        else:
            raise ValueError(MODEL_ERR_MSG.format(model=model))

def generate_tools(
    model: str|Llama, # type: ignore
    messages: list,
    temperature: float = 0.0,
    tools: list[Callable] = None,
) -> list:
    if llamapython_import and isinstance(model, Llama):
        return _llamapython_tools(model, messages, temperature, tools)
    elif model.startswith('llamacpp/'):
        return _llamacpp_tools(model, messages, temperature, tools)
    elif model.startswith('ollama/'):
        return _ollama_tools(model, messages, temperature, tools)
    else:
        lazy_load()
        if model in openai_models:
            return _openai_tools(model, messages, temperature, tools)
        elif model in google_models:
            return _google_tools(model, messages, temperature, tools)
        elif model in anthropic_models:
            return _anthropic_tools(model, messages, temperature, tools)
        else:
            raise ValueError(MODEL_ERR_MSG.format(model=model))

def pydantic_to_schema(json_schema: BaseModel, indent: int|str|None = None) -> str:
    return json.dumps(json_schema.model_json_schema(), indent=indent)

def json_to_pydantic(json_response: str, pydantic_model: type[BaseModel]) -> BaseModel:
    try:
        data = json.loads(json_response)
        instance = pydantic_model(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}")
    except ValueError as e:
        raise ValueError(f"Error creating Pydantic model: {e}")

    return instance


def _llamapython(
    model: Llama, # type: ignore
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_llamacpp_messages(messages)

    kwargs = {
        "messages": transformed_messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": -1,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        schema = pydantic_to_schema(json_schema)
        grammar = LlamaGrammar.from_json_schema(schema, verbose=False)
        kwargs["grammar"] = grammar

    response = model.create_chat_completion(**kwargs)

    text = response['choices'][0]['message']['content']
    return text

def _llamapython_stream(
    model: Llama, # type: ignore
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_llamacpp_messages(messages)

    kwargs = {
        "messages": transformed_messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": -1,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        schema = pydantic_to_schema(json_schema)
        grammar = LlamaGrammar.from_json_schema(schema, verbose=False)
        kwargs["grammar"] = grammar

    response = model.create_chat_completion(**kwargs)

    next(response)
    for chunk in response:
        if chunk['choices'][0]['finish_reason'] is not None:
            break
        token = chunk['choices'][0]['delta']['content']
        if not token:
            break
        yield token

def _llamapython_tools(
    model: Llama, # type: ignore
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = _prepare_llamacpp_messages(messages)
    transformed_tools = _prepare_openai_tools(tools) if tools else None

    system_message = textwrap.dedent(f"""
        You are a helpful assistant.
        You have access to these tools:
            {transformed_tools}

        Always prefer a tool that can produce an answer if such a tool is available.

        Otherwise try to answer it on your own to the best of your ability, i.e. just provide a
        simple answer to the question, without elaborating.

        Always create JSON output.
        If the output requires a tool invocation, format the JSON in this way:
            {{
                "tool_name": "the_tool_name",
                "arguments": {{ "arg1_name": arg1, "arg2_name": arg2, ... }}
            }}
        If the output does NOT require a tool invocation, format the JSON in this way:
            {{
                "tool_name": "",  # empty string for tool name
                "result": response_to_the_query  # place the text response in a string here
            }}
    """).strip()

    transformed_messages.insert(0, {"role": "system", "content": system_message})

    kwargs = {
        "messages": transformed_messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": -1,
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "result": {"type": "string"},
                },
                "required": ["tool_name"],
            },
        },
    }

    response = model.create_chat_completion(**kwargs)

    j = json.loads(response['choices'][0]['message']['content'])

    text = ''
    tool = ''
    args = {}

    if 'tool_name' in j:
        if j['tool_name'] and 'arguments' in j:
            tool = j['tool_name']
            args = j['arguments']
        elif 'result' in j:
            text = j['result']
        else:
            text = 'Did not produce a valid response.'
    else:
        text = 'Did not produce a valid response.'

    return text, tool, args


def _llamacpp(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_llamacpp_messages(messages)

    kwargs = {
        "model": model[9:],
        "messages": transformed_messages,
        "stream": False,
        "temperature": temperature,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        schema = pydantic_to_schema(json_schema)
        gbnf = json_schema_to_gbnf(schema)
        kwargs["extra_body"] = {"grammar": gbnf}

    client = OpenAI(
        base_url=f'http://localhost:{model[9:]}/v1',
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    text = response.choices[0].message.content
    return text

def _llamacpp_stream(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_llamacpp_messages(messages)

    kwargs = {
        "model": model[9:],
        "messages": transformed_messages,
        "stream": True,
        "temperature": temperature,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        schema = pydantic_to_schema(json_schema)
        gbnf = json_schema_to_gbnf(schema)
        kwargs["extra_body"] = {"grammar": gbnf}

    client = OpenAI(
        base_url=f'http://localhost:{model[9:]}/v1',
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def _llamacpp_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = _prepare_llamacpp_messages(messages)
    transformed_tools = _prepare_openai_tools(tools) if tools else None

    system_message = textwrap.dedent(f"""
        You are a helpful assistant.
        You have access to these tools:
            {transformed_tools}

        Always prefer a tool that can produce an answer if such a tool is available.

        Otherwise try to answer it on your own to the best of your ability, i.e. just provide a
        simple answer to the question, without elaborating.

        Always create JSON output.
        If the output requires a tool invocation, format the JSON in this way:
            {{
                "tool_name": "the_tool_name",
                "arguments": {{ "arg1_name": arg1, "arg2_name": arg2, ... }}
            }}
        If the output does NOT require a tool invocation, format the JSON in this way:
            {{
                "tool_name": "",  # empty string for tool name
                "result": response_to_the_query  # place the text response in a string here
            }}
    """).strip()

    transformed_messages.insert(0, {"role": "system", "content": system_message})

    kwargs = {
        "model": model[9:],
        "messages": transformed_messages,
        "stream": False,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    client = OpenAI(
        base_url=f'http://localhost:{model[9:]}/v1',
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    j = json.loads(response.choices[0].message.content)

    text = ''
    tool = ''
    args = {}

    if 'tool_name' in j:
        if j['tool_name'] and 'arguments' in j:
            tool = j['tool_name']
            args = j['arguments']
        elif 'result' in j:
            text = j['result']
        else:
            text = 'Did not produce a valid response.'
    else:
        text = 'Did not produce a valid response.'

    return text, tool, args


def _ollama(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_ollama_messages(messages)

    kwargs = {
        "model": model[7:],
        "messages": transformed_messages,
        "stream": False,
        "temperature": temperature,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        raise NotImplementedError("Ollama does not support Structured Output")

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    text = response.choices[0].message.content
    return text

def _ollama_stream(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_ollama_messages(messages)

    kwargs = {
        "model": model[7:],
        "messages": transformed_messages,
        "stream": True,
        "temperature": temperature,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        raise NotImplementedError("Ollama does not support Structured Output")

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def _ollama_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = _prepare_ollama_messages(messages)
    transformed_tools = _prepare_openai_tools(tools) if tools else None

    system_message = textwrap.dedent(f"""
        You are a helpful assistant.
        You have access to these tools:
            {transformed_tools}

        Always prefer a tool that can produce an answer if such a tool is available.

        Otherwise try to answer it on your own to the best of your ability, i.e. just provide a
        simple answer to the question, without elaborating.

        Always create JSON output.
        If the output requires a tool invocation, format the JSON in this way:
            {{
                "tool_name": "the_tool_name",
                "arguments": {{ "arg1_name": arg1, "arg2_name": arg2, ... }}
            }}
        If the output does NOT require a tool invocation, format the JSON in this way:
            {{
                "tool_name": "",  # empty string for tool name
                "result": response_to_the_query  # place the text response in a string here
            }}
    """).strip()

    transformed_messages.insert(0, {"role": "system", "content": system_message})

    kwargs = {
        "model": model[7:],
        "messages": transformed_messages,
        "stream": False,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='-',
    )
    response = client.chat.completions.create(**kwargs)

    j = json.loads(response.choices[0].message.content)

    text = ''
    tool = ''
    args = {}

    if 'tool_name' in j:
        if j['tool_name'] and 'arguments' in j:
            tool = j['tool_name']
            args = j['arguments']
        elif 'result' in j:
            text = j['result']
        else:
            text = 'Did not produce a valid response.'
    else:
        text = 'Did not produce a valid response.'

    return text, tool, args


def _openai(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_openai_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "max_tokens": 4096,
        "temperature": temperature,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        kwargs["response_format"] = json_schema

    if json_schema:
        response = openai_client.beta.chat.completions.parse(**kwargs)
        if (response.choices[0].message.refusal):
            text = response.choices[0].message.refusal
        else:
            # Auto-generated Pydantic object here:
            #     response.choices[0].message.parsed
            text = response.choices[0].message.content
    else:
        response = openai_client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content

    return text

def _openai_stream(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    transformed_messages = _prepare_openai_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "stream": True,
        "max_tokens": 4096,
        "temperature": temperature,
    }

    if json_object:
        kwargs["response_format"] = {"type": "json_object"}
    if json_schema:
        # Structured Object mode doesn't currently support streaming
        kwargs.pop('stream')
        kwargs["response_format"] = json_schema

    if json_schema:
        response = openai_client.beta.chat.completions.parse(**kwargs)
        if (response.choices[0].message.refusal):
            text = response.choices[0].message.refusal
        else:
            # Auto-generated Pydantic object here:
            #     response.choices[0].message.parsed
            text = response.choices[0].message.content
        yield text
    else:
        response = openai_client.chat.completions.create(**kwargs)
        for chunk in response:
            text = chunk.choices[0].delta.content
            if text:
                yield text

def _openai_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    transformed_messages = _prepare_openai_messages(messages)
    transformed_tools = _prepare_openai_tools(tools) if tools else None

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "stream": False,
        "max_tokens": 4096,
        "temperature": temperature,
    }

    if transformed_tools:
        kwargs["tools"] = transformed_tools
        kwargs["tool_choice"] = "auto"

    response = openai_client.chat.completions.create(**kwargs)

    text = ''
    if response.choices[0].message.content:
        text = response.choices[0].message.content

    tool = ''
    args = {}
    if response.choices[0].message.tool_calls:
        func = response.choices[0].message.tool_calls[0].function
        tool = func.name
        args = json.loads(func.arguments)

    return text, tool, args


@backoff.on_exception(
    backoff.constant,
    GoogleResourceExhausted,
    interval=60,
    max_tries=4,
)
def _google(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    system_message = _prepare_google_system_message(messages)
    transformed_messages = _prepare_google_messages(messages)

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": 8192,
    }

    if json_object or json_schema:
        generation_config["response_mime_type"] = "application/json"
    else:
        generation_config["response_mime_type"] = "text/plain"

    if json_schema:
        generation_config["response_schema"] = json_schema

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_message,
        generation_config=generation_config,
    )

    response = gemini_model.generate_content(
        transformed_messages,
        stream=False,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    return response.text

@backoff.on_exception(
    backoff.constant,
    GoogleResourceExhausted,
    interval=60,
    max_tries=4,
)
def _google_stream(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    system_message = _prepare_google_system_message(messages)
    transformed_messages = _prepare_google_messages(messages)

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": 8192,
    }

    if json_object or json_schema:
        generation_config["response_mime_type"] = "application/json"
    else:
        generation_config["response_mime_type"] = "text/plain"

    if json_schema:
        generation_config["response_schema"] = json_schema

    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_message,
        generation_config=generation_config,
    )

    response = gemini_model.generate_content(
        transformed_messages,
        stream=True,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    for chunk in response:
        if chunk.parts:
            for part in chunk.parts:
                if part.text:
                    yield part.text
        elif chunk.text:
            yield chunk.text

def _google_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    system_message = _prepare_google_system_message(messages)
    transformed_messages = _prepare_google_messages(messages)

    if tools:
        system_message = (system_message or '') + "\nIf you do not have access to a function that can help answer the question, answer it on your own to the best of your ability."

    generation_config = {
        "temperature": temperature,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain"
    }
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_message,
        generation_config=generation_config,
        tools=tools,
    )
    try:
        response = gemini_model.generate_content(
            transformed_messages,
            stream=False,
            tool_config={"function_calling_config": "AUTO"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    except GoogleResourceExhausted:
        time.sleep(60)
        response = gemini_model.generate_content(
            transformed_messages,
            stream=False,
            tool_config={"function_calling_config": "AUTO"},
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

    text = ''
    tool = ''
    args = {}

    for part in response.candidates[0].content.parts:
        if not text and part.text:
            text = part.text

        if not tool and part.function_call:
            func = part.function_call
            tool = func.name
            args = dict(func.args)

    return text, tool, args


def _anthropic(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    system_message = _prepare_anthropic_system_message(messages)
    transformed_messages = _prepare_anthropic_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "max_tokens": 4000,
        "temperature": temperature,
    }

    if system_message:
        kwargs["system"] = system_message

    if json_object:
        transformed_messages.append(
            {
                "role": "assistant",
                "content": "Here is the JSON requested:\n{"
            }
        )
    if json_schema:
        raise NotImplementedError("Anthropic does not support Structured Output")

    response = anthropic_client.messages.create(**kwargs)

    text = response.content[0].text
    if json_object:
        text = '{' + text[:text.rfind("}") + 1]

    return text

def _anthropic_stream(
    model: str,
    messages: list,
    temperature: float,
    json_object: bool,
    json_schema: BaseModel|None,
):
    system_message = _prepare_anthropic_system_message(messages)
    transformed_messages = _prepare_anthropic_messages(messages)

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "max_tokens": 4000,
        "temperature": temperature,
    }

    if system_message:
        kwargs["system"] = system_message

    if json_object:
        transformed_messages.append(
            {
                "role": "assistant",
                "content": "Here is the JSON requested:\n{"
            }
        )
        yield '{'
    if json_schema:
        raise NotImplementedError("Anthropic does not support Structured Output")

    with anthropic_client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            yield text

def _anthropic_tools(
    model: str,
    messages: list,
    temperature: float,
    tools: list[Callable],
):
    system_message = _prepare_anthropic_system_message(messages)
    transformed_messages = _prepare_anthropic_messages(messages)
    transformed_tools = _prepare_anthropic_tools(tools) if tools else None

    kwargs = {
        "model": model,
        "messages": transformed_messages,
        "max_tokens": 4000,
        "temperature": temperature,
    }

    if system_message:
        kwargs["system"] = system_message

    if transformed_tools:
        kwargs["tools"] = transformed_tools

    response = anthropic_client.messages.create(**kwargs)

    text = response.content[0].text

    tool = ''
    args = {}
    if response.stop_reason == "tool_use":
        func = response.content[1]
        tool = func.name
        args = func.input

    return text, tool, args


def _prepare_llamacpp_messages(messages):
    messages_out = []

    for message in messages:
        if 'content' in message and isinstance(message['content'], list):
            content = []
            for item in message['content']:
                if item.get('type') == 'image_url':
                    continue
                else:
                    content.append(item)
        else:
            content = message.get('content', '')
        messages_out.append({'role': message['role'], 'content': content})

    return messages_out

def _prepare_ollama_messages(messages):
    messages_out = []

    for message in messages:
        if 'content' in message and isinstance(message['content'], list):
            content = []
            for item in message['content']:
                if item.get('type') == 'image_url':
                    continue
                else:
                    content.append(item)
        else:
            content = message.get('content', '')
        messages_out.append({'role': message['role'], 'content': content})

    return messages_out

def _prepare_openai_messages(messages):
    messages_out = []

    for message in messages:
        if 'content' in message and isinstance(message['content'], list):
            content = []
            for item in message['content']:
                if item.get('type') == 'image_url':
                    with open(item['image_url']['url'], 'rb') as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                else:
                    content.append(item)
        else:
            content = message.get('content', '')
        messages_out.append({'role': message['role'], 'content': content})

    return messages_out

def _prepare_openai_tools(tools: list[Callable]):
    openai_tools = []

    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.__name__,
                "description": tool.__doc__,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param: {"type": "number" if annotation == int else "string"}
                        for param, annotation in tool.__annotations__.items()
                        if param != 'return'
                    },
                    "required": list(tool.__annotations__.keys())[:-1]
                }
            }
        })

    return openai_tools

def _prepare_google_messages(messages):
    messages_out = []

    for message in messages:
        if message['role'] == 'system':
            continue
        elif message['role'] == 'assistant':
            role = 'model'
        else:
            role = message['role']

        if isinstance(message['content'], list):
            content = []
            for item in message['content']:
                if item['type'] == 'text':
                    content.append(item['text'])
                elif item['type'] == 'image_url':
                    content.append({'mime_type': 'image/jpeg', 'data': _load_image(item['image_url']['url'])})
        else:
            content = [message['content']]

        messages_out.append({'role': role, 'parts': content})

    return messages_out

def _prepare_google_system_message(messages):
    system_message = None

    for message in messages:
        if message['role'] == 'system':
            system_message = message['content']
            break

    return system_message

def _prepare_anthropic_messages(messages):
    messages_out = []

    for message in messages:
        if message['role'] == 'system':
            continue

        if isinstance(message['content'], list):
            content = []
            for item in message['content']:
                if item['type'] == 'text':
                    content.append({"type": "text", "text": item['text']})
                elif item['type'] == 'image_url':
                    with open(item['image_url']['url'], 'rb') as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    })
        else:
            content = [{"type": "text", "text": message['content']}]

        messages_out.append({'role': message['role'], 'content': content})

    return messages_out

def _prepare_anthropic_system_message(messages):
    system_message = None

    for message in messages:
        if message['role'] == 'system':
            system_message = message['content']
            break

    return system_message

def _prepare_anthropic_tools(tools: list[Callable]):
    anthropic_tools = []

    for tool in tools:
        anthropic_tools.append({
            "name": tool.__name__,
            "description": tool.__doc__,
            "input_schema": {
                "type": "object",
                "properties": {
                    param: {"type": "number" if annotation == int else "string"}
                    for param, annotation in tool.__annotations__.items()
                    if param != 'return'
                },
                "required": list(tool.__annotations__.keys())[:-1]
            }
        })

    return anthropic_tools


def _load_image(image_path):
    with open(image_path, "rb") as image_file:
        return image_file.read()

if __name__ == '__main__':
    lazy_load()
    print('OpenAI:')
    for model in openai_models:
        print(' ', model)
    print()

    print('Google:')
    for model in google_models:
        print(' ', model)
    print()

    print('Anthropic:')
    for model in anthropic_models:
        print(' ', model)

import sys, os, json
import asyncio
import traceback
import subprocess
import webbrowser
import polyllm

from pydantic import BaseModel, Field

import yfinance as yf
import chainlit as cl
import plotly

from duckduckgo_search import AsyncDDGS

ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
OPENAI_MODEL = "gpt-4o"

SCRATCH_PAD_DIR = "./scratchpad"

query_stock_price_def = {
    "name": "query_stock_price",
    "description": "Queries the latest stock price information for a given stock symbol.",
    "parameters": {
      "type": "object",
      "properties": {
        "symbol": {
          "type": "string",
          "description": "The stock symbol to query (e.g., 'AAPL' for Apple Inc.)"
        },
        "period": {
          "type": "string",
          "description": "The time period for which to retrieve stock data (e.g., '1d' for one day, '1mo' for one month)"
        }
      },
      "required": ["symbol", "period"]
    }
}

async def query_stock_price_handler(symbol, period):
    """
    Queries the latest stock price information for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": "No data found for the given symbol."}
        return hist.to_json()
 
    except Exception as e:
        print(f"❌ Error performing query_stock_price_handler: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

query_stock_price = (query_stock_price_def, query_stock_price_handler)

draw_plotly_chart_def = {
    "name": "draw_plotly_chart",
    "description": "Draws a Plotly chart based on the provided JSON figure and displays it with an accompanying message.",
    "parameters": {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "The message to display alongside the chart"
        },
        "plotly_json_fig": {
          "type": "string",
          "description": "A JSON string representing the Plotly figure to be drawn"
        }
      },
      "required": ["message", "plotly_json_fig"]
    }
}

async def draw_plotly_chart_handler(message: str, plotly_json_fig):
    fig = plotly.io.from_json(plotly_json_fig)
    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]

    await cl.Message(content=message, elements=elements).send()
    
draw_plotly_chart = (draw_plotly_chart_def, draw_plotly_chart_handler)


internet_search_def = {
    "name": "internet_search",
    "description": "Performs an internet search using the Tavily API.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the internet (e.g., 'What's the weather like in Madrid tomorrow?').",
            },
        },
        "required": ["query"],
    },
}


async def internet_search_handler(query):
    """
    Executes an internet search using the DuckDuckGo API and returns the result.
    """
    try:
        print(f"🕵 Performing internet search for query: '{query}'")
        results = await AsyncDDGS().atext(query, region='wt-wt', safesearch='off',
                                          timelimit='y', max_results=1)

        # Extracting the result for formatting
        if not results:
            await cl.Message(content=f"No results found for '{query}'.").send()
            return None

        # Formatting the results in a more readable way
        formatted_results = "\n".join(
            [
                f"{i+1}. [{result['title']}]({result['href']})\n{result['body'][:200]}..."
                for (i,result) in enumerate(results)
            ]
        )

        message_content = f"Search Results for '{query}':\n\n{formatted_results}"
        await cl.Message(content=message_content).send()

        print(f"📏 Search results for '{query}' retrieved successfully.")
        return results[0]
    except Exception as e:
        print(f"❌ Error performing internet search: {str(e)}")
        print(traceback.format_exc())
        await cl.Message(
            content=f"An error occurred while performing the search: {str(e)}"
        ).send()


internet_search = (internet_search_def, internet_search_handler)

create_python_file_def = {
    "name": "create_python_file",
    "description": "Creates a Python file based on a given topic or content description.",
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The name of the Python file to be created (e.g., 'script.py').",
            },
            "content_description": {
                "type": "string",
                "description": "The content description for the Python file (e.g., 'Generate a random number').",
            },
        },
        "required": ["filename", "topic"],
    },
}


class PythonFile(BaseModel):
    """
    Python file content.
    """

    filename: str = Field(
        ...,
        description="The name of the Python file with the extension .py",
    )
    program: str = Field(
        ...,
        description="The Python code to be saved in the file",
    )


async def create_python_file_handler(filename: str, content_description: str):
    """
    Creates a Python file with the provided filename based on a given topic or content description.
    """
    print("DBG INSIDE CREATE PY")
    try:
        print(f"📝 Drafting Python file that '{content_description}'")

        system_prompt = f"""
        Create a Python script for the given topic. The script should be well-commented,
        use best practices, and aim to be simple yet effective. 
        Include informative docstrings and comments where necessary.

        # Topic
        {content_description}

        # Requirements
        1. **Define Purpose**: Write a brief docstring explaining the purpose of the script.
        2. **Implement Logic**: Implement the logic related to the topic, keeping the script easy to understand.
        3. **Best Practices**: Follow Python best practices, such as using functions where appropriate and adding comments to clarify the code.

        """
        system_prompt += """
        Produce the result in JSON that matches this schema:
            {
                "filename": "file to write the python progrm to",
                "program":  "the python program",
            }
        """

        # generate the Python file content
        text_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_description},
        ]
        print("DBG CALLING GENERATE")
        # RMB
        # content = "FOOBAR"
        response = polyllm.generate(OPENAI_MODEL, text_messages, json_schema=PythonFile)
        print("DBG DONE CALLING GENERATE")

        filepath = os.path.join(SCRATCH_PAD_DIR, filename)
        os.makedirs(SCRATCH_PAD_DIR, exist_ok=True)

        # RMB
        structured_object = polyllm.json_to_pydantic(response, PythonFile)
        python_pgm = structured_object.program
        with open(filepath, "w") as f:
            f.write(python_pgm)

        print(f"💾 Python file '{filename}' created successfully at {filepath}")
        msg = f"Python file {filename} created successfully based on topic {content_description}"
        await cl.Message(content=msg).send()
        print("DBG SENT MSG")
        rv = { "filename": filename, "code": "CODE" }
        return json.dumps(rv)

    except Exception as e:
        print(f"❌ Error creating Python file: {str(e)}")
        print(traceback.format_exc())
        await cl.Message(
            content=f"An error occurred while creating the Python file: {str(e)}"
        ).send()
        return {"error": f"An error occurred while creating the Python file: {str(e)}" }


create_python_file = (create_python_file_def, create_python_file_handler)


execute_python_file_def = {
    "name": "execute_python_file",
    "description": "Executes a Python file in the scratchpad directory using the current Python environment.",
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The name of the Python file to be executed (e.g., 'script.py').",
            },
        },
        "required": ["filename"],
    },
}


async def execute_python_file_handler(filename: str):
    """Executes a Python file in the scratchpad directory using the current Python environment."""
    print("DBGFN",filename)
    try:
        filepath = os.path.join(SCRATCH_PAD_DIR, filename)

        print("DBGFP",filepath)
        if not os.path.exists(filepath):
            error_message = (
                f"Python file '{filename}' not found in scratchpad directory."
            )
            print(f"❌ {error_message}")
            await cl.Message(content=error_message).send()
            return error_message

        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"✅ Successfully executed Python file '{filename}'")
            output_message = result.stdout
            await cl.Message( content=f"Output of '{filename}':\n\n{output_message}").send()
            return output_message
        else:
            error_message = f"Error executing Python file '{filename}': {result.stderr}"
            print(f"❌ {error_message}")
            await cl.Message(content=error_message).send()
            return error_message

    except Exception as e:
        print(f"❌ Error executing Python file: {str(e)}")
        print(traceback.format_exc())
        await cl.Message(
            content=f"An error occurred while executing the Python file: {str(e)}"
        ).send()
        return f"An error occurred while executing the Python file: {str(e)}"


execute_python_file = (execute_python_file_def, execute_python_file_handler)


open_browser_def = {
    "name": "open_browser",
    "description": "Opens a browser tab with the URL specified in the user's prompt.",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The user's prompt to determine which URL to open.",
            },
        },
        "required": ["prompt"],
    },
}

class URL(BaseModel):
    """
    URL to be visited in browser
    """

    url: str = Field(
        ...,
        description="The URL to be opened in a browser",
    )

async def open_browser_handler(prompt: str):
    """
    Open a browser tab with the URL in the user's prompt.
    """
    try:
        print(f"📖 open_browser() Prompt: {prompt}")

        llm_prompt = f"""
        Extract the user's desired URL from the prompt.

        # Prompt:
        {prompt}
        """
        llm_prompt += """
        Produce the result in JSON that matches this schema:
            {
                "url":  "the URL to visit in the browser",
            }
        """

        text_messages = [
            {"role": "user", "content": llm_prompt},
        ]
        response = polyllm.generate(OPENAI_MODEL, text_messages, json_schema=URL)

        structured_object = polyllm.json_to_pydantic(response, URL)
        url = structured_object.url
        if not url.startswith("http"):
            url = "http://www." + url

        # Open the URL if it's not empty
        browser = "firefox"
        if url:
            print(f"📖 open_browser() Opening URL: {url}")
            result = subprocess.run(
                ["open", "-n", "-a", "firefox", url],
                capture_output=False,
                text=False,
            )
            ## loop = asyncio.get_running_loop()
            ## with ThreadPoolExecutor() as pool:
                ## await loop.run_in_executor(
                    ## pool, webbrowser.get(browser).open, url
                ## )
            return f"URL opened successfully in the browser: {url}"
        else:
            error_message = f"Error retrieving URL from the prompt: {prompt}"
            print(f"❌ {error_message}")
            await cl.Message(content=error_message).send()
            return error_message

    except Exception as e:
        print(f"❌ Error opening browser: {str(e)}")
        print(traceback.format_exc())
        return {"status": "Error", "message": str(e)}


open_browser = (open_browser_def, open_browser_handler)


describe_image_def = {
    "name": "describe_image",
    "description": "describe an image based on user query",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The user asks a question about an image.",
            },
        },
        "required": ["prompt"],
    },
}

async def describe_image_handler(prompt: str):
    """
    Describe an image based on user question.
    """
    try:
        print(f"📖 describe_image() Prompt: {prompt}")

        llm_prompt = f"""
        Examine the image and respond to the user's prompt about it.
        
        # Prompt
        {prompt}
        """

        image_filename = cl.user_session.get("image_filename")
        image_path = os.path.join(SCRATCH_PAD_DIR, image_filename)
        if not os.path.exists(image_path):
            error_message = ( f"image file '{image_filename}' not found in scratchpad directory.")
            print(f"❌ {error_message}")
            await cl.Message(content=error_message).send()
            return error_message

        image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": llm_prompt},
                    {"type": "image_url", "image_url": {"url": image_path}},
                ],
            },
        ]

        response = polyllm.generate(ANTHROPIC_MODEL, image_messages)
        print("DBGRESP",response)

        await cl.Message(content=response).send()
        print("DBG SENT RESP MSG")
        # rv = { "prompt": prompt, "response": response }
        # return json.dumps(rv)
        return response

    except Exception as e:
        print(f"❌ Error handling image: {str(e)}")
        print(traceback.format_exc())
        return {"status": "Error", "message": str(e)}


describe_image = (describe_image_def, describe_image_handler)


tools = [
    query_stock_price,
    draw_plotly_chart,
    internet_search,
    create_python_file,
    execute_python_file,
    open_browser,
    describe_image,
]

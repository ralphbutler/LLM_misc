
import json, inspect
from litellm import completion
from typing import get_type_hints, Callable

class NLC_Agent():
    def __init__(self, model):
        self.model = model
        self.name2func = {} # or get from globals(): func = globals()[func_name]

    def function_to_json(self,func):
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        function_info = {
            "tool-name": func.__name__,
            "description": func.__doc__,
            "parameters": {},
            "returns": type_hints.get("return", "void").__name__,
        }

        for (name,_) in signature.parameters.items():
            param_type = type_hints.get(name,type(None))
            if "typing." in str(param_type):
                param_type = str(param_type).replace("typing.","")
            else:
                param_type = str( param_type.__name__ )
            function_info["parameters"][name] = param_type

        rv = json.dumps(function_info, indent=2)  # put json into a str
        return rv

    def add_tool_func(self, tool_name: str, tool_func: Callable) -> None:
        self.name2func[tool_name] = tool_func

    def run(self, usr_msg: str) -> None:
        print("\nDOING:",usr_msg)
        # flow -> accept the user input
        # flow -> ask LLM which tool to use (if any)

        # build the sys msg to query about which tools to use
        tool_query_sys_msg = f"""
            You are an expert assistant.

            You have access to the following list of tools:
            [
        """
        for (funcname,func) in self.name2func.items():
            tool_query_sys_msg += f"{self.function_to_json(func)}\n"
        tool_query_sys_msg += f"""
            ]

            You must follow these instructions:
            If any of the tools are appropriate to helping satisfy the user query below,
            then select them based on the query.
            If a tool is selected, you must respond in the JSON format matching the following schema:
            {{
               "tools": {{
                    "tool-name":  <name of the selected tool>,
                    "tool-inputs": <parameters for the selected tool, matching the tool's JSON schema
               }}
            }}
            If multiple tools are selected, make sure a list of tools are returned in a JSON array.
            If there is no tool that match the user request, you will respond with empty json.
            Do not add any additional Notes or Explanations

            You MUST reply only with JSON as described above.

        """
        # end build sys msg to query tools

        messages = [
            { "role": "system", "content": tool_query_sys_msg},
            { "role": "user",   "content": usr_msg},
        ]
        response = completion(
            model=self.model,
            messages=messages,
        )
        text = response.choices[0].message.content
        text = text.replace("\\","")
        try:
            jresponse = json.loads(text)  # json from str
        except ValueError as e:
            jresponse = {}  # will cause tool_response to be set correctly below
        if jresponse and jresponse["tools"]:
            # print(f"{jresponse}\n","-"*50)
            nice_print_response = json.dumps(jresponse, indent=3)  # json to prettier str
            print(nice_print_response)
            for tool in jresponse["tools"]:
                tool_name = tool["tool-name"]
                tool_inputs = tool["tool-inputs"]
            func = self.name2func[tool_name]
            tool_response = func(**tool_inputs)
            print("TOOLRESP",tool_response)
        else:
            tool_response = ""
            print("No tools are useful to solve the query")

        # flow -> use the tool if one was suggested; continue at CONT_1 below
        #         we go ahead and feed its output to the LLM here
        if tool_response:
            sys_msg_with_tool_info = f"""
                You are a helpful assistant.
                Use the info from the tool if provided, do NOT embellish it in any way.
                In any case, provide a reasonable response to the user query.
                TOOL INFO:
                {tool_response}
            """
            messages = [
                { "role": "system", "content": sys_msg_with_tool_info},
                { "role": "user",   "content": usr_msg},
            ]
            response = completion(
                model="ollama/mistral",
                messages = messages,
            )
            text = response.choices[0].message.content  # str
            print(text)
        else:   # flow -> get LLM answer or fail
            # try llm query with original question
            sys_msg_plain = "You are a helpful assistant."
            messages = [
                { "role": "system", "content": sys_msg_plain},
                { "role": "user",   "content": usr_msg},
            ]
            response = completion(
                model="ollama/mistral",
                messages = messages,
            )
            text = response.choices[0].message.content  # str
            print(text)
            # flow -> if that fails: give bad response and exit
        # flow -> CONT_1 (skip because already fed tool output to LLM)

import sys, os, subprocess

import anthropic
from typing import Dict, Any, List

BASH_SYSTEM_PROMPT = "You are a helpful assistant that can execute bash commands."

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
messages = []

cost_per_million_input_tokens  =  3.0  # $ 3.00 per million input tokens
cost_per_million_output_tokens = 15.0  # $15.00 per million output tokens

def _run_bash_command(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Handle bash command execution"""
    try:
        command = tool_call.get("command")
        restart = tool_call.get("restart", False)

        if restart:
            print("Bash session restarted.")
            return {"content": "Bash session restarted."}

        if not command:
            print("No command provided to execute.")
            return {"error": "No command provided to execute."}

        # execute the command in a subprocess
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ,
            text=True,
            executable="/bin/bash",
        )

        output = result.stdout.strip()
        error_output = result.stderr.strip()

        # print outputs
        if output:
            print(f"Command output:\n\n```output for '{command[:20]}...'\n{output}\n```")
        if error_output:
            print(f"Command error output:\n\n```error for '{command}'\n{error_output}\n```")

        if result.returncode != 0:
            error_message = error_output or "Command execution failed."
            return {"error": error_message}

        return {"content": output}

    except Exception as e:
        print(f"Error in _run_bash_command: {str(e)}")
        return {"error": str(e)}

def process_tool_calls(tool_calls: List[anthropic.types.ContentBlock]) -> List[Dict[str, Any]]:
    """Process tool calls and return results"""
    results = []

    for tool_call in tool_calls:
        if tool_call.type == "tool_use" and tool_call.name == "bash":
            print(f"Bash tool call input: {tool_call.input}")

            result = _run_bash_command(tool_call.input)

            # convert result to match expected tool result format
            is_error = False

            if result.get("error"):
                is_error = True
                tool_result_content = [{"type": "text", "text": result["error"]}]
            else:
                tool_result_content = [
                    {"type": "text", "text": result.get("content", "")}
                ]

            results.append(
                {
                    "tool_call_id": tool_call.id,
                    "output": {
                        "type": "tool_result",
                        "content": tool_result_content,
                        "tool_use_id": tool_call.id,
                        "is_error": is_error,
                    },
                }
            )

    return results

def process_user_bash_request(bash_prompt: str) -> None:
    """Main method to process bash commands via the assistant"""
    try:
        # initial message with proper content structure
        api_message = {
            "role": "user",
            "content": [{"type": "text", "text": bash_prompt}],
        }
        messages = [api_message]
        total_input_tokens = 0
        total_output_tokens = 0

        print(f"User input: {api_message}")

        while True:
            response = client.beta.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                messages=messages,
                system=BASH_SYSTEM_PROMPT,
                tools=[{"type": "bash_20241022", "name": "bash"}],
                betas=["computer-use-2024-10-22"],
            )

            # extract token usage from the response
            input_tokens = getattr(response.usage, "input_tokens", 0)
            output_tokens = getattr(response.usage, "output_tokens", 0)
            print(
                f"API usage: input_tokens={input_tokens}, output_tokens={output_tokens}"
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            print(f"API response: {response.model_dump()}")

            # convert response content to message params
            response_content = []
            for block in response.content:
                if block.type == "text":
                    response_content.append({"type": "text", "text": block.text})
                else:
                    response_content.append(block.model_dump())

            # add assistant response to messages
            messages.append({"role": "assistant", "content": response_content})

            if response.stop_reason != "tool_use":
                # print the assistant's final response
                print(response.content[0].text)
                break

            tool_results = process_tool_calls(response.content)

            # add tool results as user message
            if tool_results:
                messages.append(
                    {"role": "user", "content": [tool_results[0]["output"]]}
                )

                if tool_results[0]["output"]["is_error"]:
                    print(f"Error: {tool_results[0]['output']['content']}")
                    break

        total_input_cost  = (total_input_tokens / 1_000_000)  * cost_per_million_input_tokens
        total_output_cost = (total_output_tokens / 1_000_000) * cost_per_million_output_tokens
        total_cost = total_input_cost + total_output_cost
        print(f"Total input cost: ${total_input_cost:.6f}")
        print(f"Total output cost: ${total_output_cost:.6f}")
        print(f"Total cost: ${total_cost:.6f}")

    except Exception as e:
        print(f"Error in process_user_bash_request: {str(e)}")
        exit(-1)


def main():
    process_user_bash_request(sys.argv[1])

if __name__ == "__main__":
    main()

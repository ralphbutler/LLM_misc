import sys, os, subprocess, traceback, json

import anthropic
from typing import Dict, Any, List

## EDITOR_SYSTEM_PROMPT = "You are a helpful assistant that helps users edit text files."
EDITOR_SYSTEM_PROMPT = """You are a helpful assistant that helps users edit text files.

IMPORTANT RULES:
1. ALL file paths must start with '/repo/' and include the full filename
2. For example, use '/repo/pingpong.txt' not just 'pingpong.txt'
3. Never use empty paths in tool calls
4. When using the insert command, you MUST include the actual text to insert in the new_str field

For file operations:
1. Always use complete paths starting with '/repo/'
2. View file contents before modifying
3. Include all required fields in tool calls

Example - if user mentions 'pingpong.txt', always use '/repo/pingpong.txt' in your tool calls.

For the insert command, you MUST include all of these fields:
- "command": "insert"
- "path": "/repo/filename.txt"
- "insert_line": line number (0-based)
- "new_str": the actual text to insert
"""

EDITOR_DIR = os.path.join(os.getcwd(), "editor_dir")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
messages = []

cost_per_million_input_tokens  =  3.0  # $ 3.00 per million input tokens
cost_per_million_output_tokens = 15.0  # $15.00 per million output tokens

def _get_editor_path(path: str) -> str:
    """Convert API path to local editor directory path"""
    if not path:
        raise ValueError("Path cannot be empty")
    clean_path = path.replace("/repo/", "", 1)  # Strips /repo/ prefix
    full_path = os.path.join(EDITOR_DIR, clean_path)  # Uses editor_dir
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

def validate_tool_call(tool_call: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
    """Validate tool call has required fields"""
    missing_fields = [field for field in required_fields if field not in tool_call]
    if missing_fields:
        return {"error": f"Missing required fields: {', '.join(missing_fields)}"}
    return {}

def _handle_view(path: str, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Handle view command"""
    if error := validate_tool_call(tool_call, ["path"]):
        return error
    
    editor_path = _get_editor_path(path)
    if os.path.exists(editor_path):
        try:
            with open(editor_path, "r") as f:
                return {"content": f.read()}
        except IOError as e:
            return {"error": f"Failed to read file {editor_path}: {str(e)}"}
    return {"error": f"File {editor_path} does not exist"}

def _handle_create(path: str, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Handle create command"""
    if error := validate_tool_call(tool_call, ["path", "file_text"]):
        return error
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(tool_call["file_text"])
        return {"content": f"File created at {path}"}
    except IOError as e:
        return {"error": f"Failed to create file {path}: {str(e)}"}

def _handle_str_replace(path: str, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Handle str_replace command"""
    if error := validate_tool_call(tool_call, ["path", "old_str"]):
        return error
    
    try:
        with open(path, "r") as f:
            content = f.read()
        if tool_call["old_str"] not in content:
            return {"error": f"String '{tool_call['old_str']}' not found in file"}
        new_content = content.replace(
            tool_call["old_str"], tool_call.get("new_str", "")
        )
        with open(path, "w") as f:
            f.write(new_content)
        return {"content": "File updated successfully"}
    except IOError as e:
        return {"error": f"Failed to update file {path}: {str(e)}"}

def _handle_insert(path: str, tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Handle insert command"""
    if error := validate_tool_call(tool_call, ["path", "insert_line", "new_str"]):
        return error

    try:
        # Read all lines and ensure consistent line endings
        with open(path, "r") as f:
            lines = [line.rstrip('\n') for line in f]

        insert_line = int(tool_call["insert_line"])
        if insert_line > len(lines):
            return {"error": f"Insert line {insert_line} is beyond file length {len(lines)}"}

        # Insert new line
        lines.insert(insert_line, tool_call["new_str"])

        # Write back with consistent line endings
        with open(path, "w") as f:
            for line in lines:
                f.write(line + '\n')

        return {"content": "Content inserted successfully"}
    except (IOError, ValueError) as e:
        return {"error": f"Failed to insert into file {path}: {str(e)}"}

def handle_text_editor_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Handle text editor tool calls"""
    try:
        if not isinstance(tool_call, dict):
            return {"error": "Invalid tool call format"}
            
        if error := validate_tool_call(tool_call, ["command", "path"]):
            return error

        handlers = {
            "view": _handle_view,
            "create": _handle_create,
            "str_replace": _handle_str_replace,
            "insert": _handle_insert,
        }

        command = tool_call["command"]
        handler = handlers.get(command)
        if not handler:
            return {"error": f"Unknown command: {command}"}

        path = _get_editor_path(tool_call["path"])
        return handler(path, tool_call)

    except Exception as e:
        print(f"Error in handle_text_editor_tool: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Internal error: {str(e)}"}

def process_tool_calls(tool_calls: List[anthropic.types.ContentBlock]) -> List[Dict[str, Any]]:
    """Process tool calls and return results with detailed logging"""
    results = []

    print("\n=== Processing Tool Calls ===")
    print(f"Number of content blocks: {len(tool_calls)}")

    for block in tool_calls:
        print(f"\nContent Block Type: {block.type}")
        
        if block.type == "text":
            print(f"Text Content: {block.text}")
            continue
            
        if block.type == "tool_use":
            print(f"Tool Call ID: {block.id}")
            print(f"Tool Name: {block.name}")
            print("Tool Input:")
            print(json.dumps(block.input, indent=2))

            if block.name == "str_replace_editor":
                result = handle_text_editor_tool(block.input)
                print("\nHandler Result:")
                print(json.dumps(result, indent=2))

                is_error = "error" in result
                tool_result_content = [
                    {
                        "type": "text", 
                        "text": result.get("error") if is_error else result.get("content", "")
                    }
                ]

                results.append({
                    "tool_call_id": block.id,
                    "output": {
                        "type": "tool_result",
                        "content": tool_result_content,
                        "tool_use_id": block.id,
                        "is_error": is_error,
                    }
                })

    return results

def process_user_edit_request(edit_prompt: str) -> None:
    """Main function to process editing prompts with API interaction logging"""
    try:
        api_message = {
            "role": "user",
            "content": [{"type": "text", "text": edit_prompt}],
        }
        messages = [api_message]
        
        print("\n=== Starting Edit Request ===")
        print(f"Initial prompt: {edit_prompt}")

        while True:
            print("\n=== Making API Request ===")
            print("Messages to send:")
            print(json.dumps(messages, indent=2))
            
            response = client.beta.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                system=EDITOR_SYSTEM_PROMPT,
                messages=messages,
                tools=[{"type": "text_editor_20241022", "name": "str_replace_editor"}],
                betas=["computer-use-2024-10-22"],
            )

            print("\n=== API Response ===")
            print(f"Stop reason: {response.stop_reason}")
            print("Response content:")
            for block in response.content:
                if block.type == "text":
                    print(f"Text block: {block.text}")
                elif block.type == "tool_use":
                    print(f"Tool use block: {json.dumps(block.model_dump(), indent=2)}")

            response_content = [
                block.model_dump() if block.type != "text" 
                else {"type": "text", "text": block.text}
                for block in response.content
            ]

            messages.append({"role": "assistant", "content": response_content})

            if response.stop_reason != "tool_use":
                if response.content:
                    print(response.content[0].text)
                break

            tool_results = process_tool_calls(response.content)

            if tool_results:
                messages.append(
                    {"role": "user", "content": [tool_results[0]["output"]]}
                )
                if tool_results[0]["output"]["is_error"]:
                    print(f"\nTool call failed: {tool_results[0]['output']['content'][0]['text']}")
                    break

    except Exception as e:
        print(f"Error in process_edit: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python workedit.py '<edit prompt>'")
        sys.exit(1)
    process_user_edit_request(sys.argv[1])

if __name__ == "__main__":
    main()

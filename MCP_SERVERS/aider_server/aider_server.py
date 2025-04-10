#!/usr/bin/env python3
"""
Aider FastMCP Server

A simplified MCP server using FastMCP to provide Aider AI coding functionality.
This implementation consolidates multiple files and uses FastMCP's decorator pattern.
"""

import json
import sys
import os
import subprocess
import logging
from typing import Dict, Any, List, Union, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context

# Try importing Aider dependencies - handle gracefully if not installed
try:
    from aider.models import Model, fuzzy_match_models
    from aider.coders import Coder
    from aider.io import InputOutput
    AIDER_AVAILABLE = True
except ImportError:
    AIDER_AVAILABLE = False
    print("Warning: Aider not available. Please install with: pip install aider-chat")

# Default model to use (can be overridden in environment)
DEFAULT_EDITOR_MODEL = os.environ.get("AIDER_MODEL", "gemini/gemini-2.5-pro-exp-03-25") 

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("aider-mcp")

# Type alias for response dictionary
ResponseDict = Dict[str, Union[bool, str]]

#
# Aider integration functions (consolidated from aider_ai_code.py)
#

def get_changes_diff_or_content(relative_editable_files: List[str], working_dir: str = None) -> str:
    """
    Get the git diff for the specified files, or their content if git fails.
    """
    diff = ""
    # Use current directory if no working directory specified
    working_dir = working_dir or os.getcwd()
    logger.info(f"Getting diff in working directory: {working_dir}")

    # Get the full paths of the files
    files_arg = " ".join(relative_editable_files)
    
    try:
        # Use git to get diff
        diff_cmd = f"git -C {working_dir} diff -- {files_arg}"
        logger.info(f"Running git command: {diff_cmd}")
        diff = subprocess.check_output(diff_cmd, shell=True, text=True, stderr=subprocess.PIPE)
        logger.info("Successfully obtained git diff.")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git diff failed. Falling back to reading file contents.")
        diff = "Git diff failed. Current file contents:\n\n"
        
        for file_path in relative_editable_files:
            full_path = os.path.join(working_dir, file_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r") as f:
                        content = f.read()
                        diff += f"--- {file_path} ---\n{content}\n\n"
                except Exception as read_e:
                    diff += f"--- {file_path} --- (Error reading file: {read_e})\n\n"
            else:
                diff += f"--- {file_path} --- (File not found)\n\n"
    except Exception as e:
        diff = f"Error getting git diff: {str(e)}\n\n"
        
    return diff


def check_for_meaningful_changes(relative_editable_files: List[str], working_dir: str = None) -> bool:
    """
    Check if the edited files contain meaningful content.
    """
    working_dir = working_dir or os.getcwd()
    
    for file_path in relative_editable_files:
        full_path = os.path.join(working_dir, file_path)
        
        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    content = f.read()
                    stripped_content = content.strip()
                    
                    # Check if the file has substantive content
                    if stripped_content and (
                        len(stripped_content.split("\n")) > 1
                        or any(kw in content for kw in ["def ", "class ", "import ", "from ", "async def"])
                    ):
                        logger.info(f"Meaningful content found in: {file_path}")
                        return True
            except Exception as e:
                logger.error(f"Failed reading file {full_path}: {e}")
                continue
    
    logger.info("No meaningful changes detected.")
    return False


def process_coder_results(relative_editable_files: List[str], working_dir: str = None) -> ResponseDict:
    """
    Process the results after Aider has run, checking for meaningful changes
    and retrieving the diff or content.
    """
    working_dir = working_dir or os.getcwd()
    diff_output = get_changes_diff_or_content(relative_editable_files, working_dir)
    has_meaningful_content = check_for_meaningful_changes(relative_editable_files, working_dir)

    if has_meaningful_content:
        return {"success": True, "diff": diff_output}
    else:
        return {
            "success": False,
            "diff": diff_output or "No meaningful changes detected."
        }


def code_with_aider(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: List[str] = [],
    model: str = DEFAULT_EDITOR_MODEL,
    working_dir: str = None
) -> str:
    """
    Run Aider to perform AI coding tasks based on the provided prompt and files.
    """
    if not AIDER_AVAILABLE:
        return json.dumps({
            "success": False, 
            "diff": "Aider is not installed. Please install with 'pip install aider-chat'"
        })
    
    logger.info(f"Starting Aider coding session with prompt: '{ai_coding_prompt}'")
    
    # Use current directory if not specified
    working_dir = working_dir or os.getcwd()
    logger.info(f"Working directory: {working_dir}")
    
    try:
        # Configure the model
        ai_model = Model(model)
        
        # Create full paths for files
        abs_editable_files = [os.path.join(working_dir, file) for file in relative_editable_files]
        abs_readonly_files = [os.path.join(working_dir, file) for file in relative_readonly_files]
        
        # Set up chat history file
        chat_history_file = os.path.join(working_dir, ".aider.chat.history.md")
        
        # Create coder instance
        coder = Coder.create(
            main_model=ai_model,
            io=InputOutput(yes=True, chat_history_file=chat_history_file),
            fnames=abs_editable_files,
            read_only_fnames=abs_readonly_files,
            auto_commits=False,
            suggest_shell_commands=False,
            detect_urls=False,
            use_git=True
        )
        
        # Run the coding session
        result = coder.run(ai_coding_prompt)
        logger.info(f"Aider coding session completed.")
        # f = open("RMBDBG", "w")
        # print(f"DBG: Aider coding session completed.", file=f)
        
        # Process the results
        response = process_coder_results(relative_editable_files, working_dir)
        
    except Exception as e:
        logger.exception(f"Error in code_with_aider: {str(e)}")
        response = {
            "success": False,
            "diff": f"Error during Aider execution: {str(e)}"
        }
    
    return json.dumps(response)


def list_models(substring: str = "") -> List[str]:
    """
    List available models that match the provided substring.
    """
    if not AIDER_AVAILABLE:
        return ["Aider is not installed. Please install with 'pip install aider-chat'"]
    
    return fuzzy_match_models(substring)


#
# MCP Server implementation using FastMCP
#

# Server context to store working directory
@dataclass
class AiderContext:
    working_dir: str

# Lifespan manager for server context
@asynccontextmanager
async def aider_lifespan(server: FastMCP):
    """Initialize server context with working directory."""
    # Use the current directory as the working directory
    working_dir = os.getcwd()
    logger.info(f"Initializing Aider MCP Server in: {working_dir}")
    
    try:
        yield AiderContext(working_dir=working_dir)
    finally:
        logger.info("Shutting down Aider MCP Server.")

# Create the MCP server with lifespan support
mcp = FastMCP("Aider Coding", lifespan=aider_lifespan)

@mcp.tool()
def aider_ai_code(
    ctx: Context,
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: List[str] = [],
    model: str = ""
) -> str:
    """
    Run Aider to perform AI coding tasks based on the provided prompt and files.
    
    Args:
        ai_coding_prompt: The prompt for the AI to execute
        relative_editable_files: List of relative paths to files that can be edited
        relative_readonly_files: List of relative paths to files that can be read but not edited
        model: The AI model to use (leave blank for default)
    
    Returns:
        JSON with success status and diff showing changes
    """
    working_dir = ctx.request_context.lifespan_context.working_dir
    
    # Ensure editable files is a list
    if isinstance(relative_editable_files, str):
        relative_editable_files = [relative_editable_files]
    
    # Ensure readonly files is a list
    if isinstance(relative_readonly_files, str):
        relative_readonly_files = [relative_readonly_files]
    
    # Use specified model or fall back to default
    model_to_use = model if model else DEFAULT_EDITOR_MODEL
    
    # Run Aider and return results
    result = code_with_aider(
        ai_coding_prompt=ai_coding_prompt,
        relative_editable_files=relative_editable_files,
        relative_readonly_files=relative_readonly_files,
        model=model_to_use,
        working_dir=working_dir
    )
    
    # Parse result to extract JSON fields
    try:
        result_dict = json.loads(result)
        return result_dict
    except:
        return {"success": False, "diff": "Error processing response"}

@mcp.tool()
def list_available_models(substring: str = "") -> Dict[str, List[str]]:
    """
    List available models that match the provided substring.
    
    Args:
        substring: Substring to match against available models (optional)
    
    Returns:
        Dictionary containing matching model names
    """
    models = list_models(substring)
    return {"models": models}

if __name__ == "__main__":
    if not AIDER_AVAILABLE:
        print("Warning: Aider not available. Functionality will be limited.")
        print("Please install with: pip install aider-chat")
    
    logger.info(f"Starting Aider FastMCP Server with default model: {DEFAULT_EDITOR_MODEL}")
    
    # No command-line arguments needed - uses current directory by default
    mcp.run()

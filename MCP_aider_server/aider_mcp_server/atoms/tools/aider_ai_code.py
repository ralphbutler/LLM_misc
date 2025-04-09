import json
from typing import List, Optional, Dict, Any, Union
import os
import os.path
import subprocess
from aider.models import Model
from aider.coders import Coder
from aider.io import InputOutput
from aider_mcp_server.atoms.logging import get_logger

# Configure logging for this module
logger = get_logger(__name__)

# Type alias for response dictionary
ResponseDict = Dict[str, Union[bool, str]]


def _get_changes_diff_or_content(
    relative_editable_files: List[str], working_dir: str = None
) -> str:
    """
    Get the git diff for the specified files, or their content if git fails.

    Args:
        relative_editable_files: List of files to check for changes
        working_dir: The working directory where the git repo is located
    """
    diff = ""
    # Log current directory for debugging
    current_dir = os.getcwd()
    logger.info(f"Current directory during diff: {current_dir}")
    if working_dir:
        logger.info(f"Using working directory: {working_dir}")

    # Always attempt to use git
    files_arg = " ".join(relative_editable_files)
    logger.info(f"Attempting to get git diff for: {' '.join(relative_editable_files)}")

    try:
        # Use git -C to specify the repository directory
        if working_dir:
            diff_cmd = f"git -C {working_dir} diff -- {files_arg}"
        else:
            diff_cmd = f"git diff -- {files_arg}"

        logger.info(f"Running git command: {diff_cmd}")
        diff = subprocess.check_output(
            diff_cmd, shell=True, text=True, stderr=subprocess.PIPE
        )
        logger.info("Successfully obtained git diff.")
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Git diff command failed with exit code {e.returncode}. Error: {e.stderr.strip()}"
        )
        logger.warning("Falling back to reading file contents.")
        diff = "Git diff failed. Current file contents:\n\n"
        for file_path in relative_editable_files:
            full_path = (
                os.path.join(working_dir, file_path) if working_dir else file_path
            )
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r") as f:
                        content = f.read()
                        diff += f"--- {file_path} ---\n{content}\n\n"
                        logger.info(f"Read content for {file_path}")
                except Exception as read_e:
                    logger.error(
                        f"Failed reading file {full_path} for content fallback: {read_e}"
                    )
                    diff += f"--- {file_path} --- (Error reading file)\n\n"
            else:
                logger.warning(f"File {full_path} not found during content fallback.")
                diff += f"--- {file_path} --- (File not found)\n\n"
    except Exception as e:
        logger.error(f"Unexpected error getting git diff: {str(e)}")
        diff = f"Error getting git diff: {str(e)}\n\n"  # Provide error in diff string as fallback
    return diff


def _check_for_meaningful_changes(
    relative_editable_files: List[str], working_dir: str = None
) -> bool:
    """
    Check if the edited files contain meaningful content.

    Args:
        relative_editable_files: List of files to check
        working_dir: The working directory where files are located
    """
    for file_path in relative_editable_files:
        # Use the working directory if provided
        full_path = os.path.join(working_dir, file_path) if working_dir else file_path
        logger.info(f"Checking for meaningful content in: {full_path}")

        if os.path.exists(full_path):
            try:
                with open(full_path, "r") as f:
                    content = f.read()
                    # Check if the file has more than just whitespace or a single comment line,
                    # or contains common code keywords. This is a heuristic.
                    stripped_content = content.strip()
                    if stripped_content and (
                        len(stripped_content.split("\n")) > 1
                        or any(
                            kw in content
                            for kw in [
                                "def ",
                                "class ",
                                "import ",
                                "from ",
                                "async def",
                            ]
                        )
                    ):
                        logger.info(f"Meaningful content found in: {file_path}")
                        return True
            except Exception as e:
                logger.error(
                    f"Failed reading file {full_path} during meaningful change check: {e}"
                )
                # If we can't read it, we can't confirm meaningful change from this file
                continue
        else:
            logger.info(
                f"File not found or empty, skipping meaningful check: {full_path}"
            )

    logger.info("No meaningful changes detected in any editable files.")
    return False


def _process_coder_results(
    relative_editable_files: List[str], working_dir: str = None
) -> ResponseDict:
    """
    Process the results after Aider has run, checking for meaningful changes
    and retrieving the diff or content.

    Args:
        relative_editable_files: List of files that were edited
        working_dir: The working directory where the git repo is located

    Returns:
        Dictionary with success status and diff output
    """
    diff_output = _get_changes_diff_or_content(relative_editable_files, working_dir)
    logger.info("Checking for meaningful changes in edited files...")
    has_meaningful_content = _check_for_meaningful_changes(
        relative_editable_files, working_dir
    )

    if has_meaningful_content:
        logger.info("Meaningful changes found. Processing successful.")
        return {"success": True, "diff": diff_output}
    else:
        logger.warning(
            "No meaningful changes detected. Processing marked as unsuccessful."
        )
        # Even if no meaningful content, provide the diff/content if available
        return {
            "success": False,
            "diff": diff_output
            or "No meaningful changes detected and no diff/content available.",
        }


def _format_response(response: ResponseDict) -> str:
    """
    Format the response dictionary as a JSON string.

    Args:
        response: Dictionary containing success status and diff output

    Returns:
        JSON string representation of the response
    """
    return json.dumps(response, indent=4)


def code_with_aider(
    ai_coding_prompt: str,
    relative_editable_files: List[str],
    relative_readonly_files: List[str] = [],
    model: str = "gemini/gemini-2.5-pro-exp-03-25",
    working_dir: str = None,
) -> str:
    """
    Run Aider to perform AI coding tasks based on the provided prompt and files.

    Args:
        ai_coding_prompt (str): The prompt for the AI to execute.
        relative_editable_files (List[str]): List of files that can be edited.
        relative_readonly_files (List[str], optional): List of files that can be read but not edited. Defaults to [].
        model (str, optional): The model to use. Defaults to "gemini/gemini-2.5-pro-exp-03-25".
        working_dir (str, required): The working directory where git repository is located and files are stored.

    Returns:
        Dict[str, Any]: {'success': True/False, 'diff': str with git diff output}
    """
    logger.info("Starting code_with_aider process.")
    logger.info(f"Prompt: '{ai_coding_prompt}'")

    # Working directory must be provided
    if not working_dir:
        error_msg = "Error: working_dir is required for code_with_aider"
        logger.error(error_msg)
        return json.dumps({"success": False, "diff": error_msg})

    logger.info(f"Working directory: {working_dir}")
    logger.info(f"Editable files: {relative_editable_files}")
    logger.info(f"Readonly files: {relative_readonly_files}")
    logger.info(f"Model: {model}")

    try:
        # Configure the model
        logger.info("Configuring AI model...")  # Point 1: Before init
        ai_model = Model(model)
        logger.info(f"Configured model: {model}")
        logger.info("AI model configured.")  # Point 2: After init

        # Create the coder instance
        logger.info("Creating Aider coder instance...")
        # Use working directory for chat history file if provided
        history_dir = working_dir
        abs_editable_files = [
            os.path.join(working_dir, file) for file in relative_editable_files
        ]
        abs_readonly_files = [
            os.path.join(working_dir, file) for file in relative_readonly_files
        ]
        chat_history_file = os.path.join(history_dir, ".aider.chat.history.md")
        logger.info(f"Using chat history file: {chat_history_file}")

        coder = Coder.create(
            main_model=ai_model,
            io=InputOutput(
                yes=True,
                chat_history_file=chat_history_file,
            ),
            fnames=abs_editable_files,
            read_only_fnames=abs_readonly_files,
            auto_commits=False,  # We'll handle commits separately
            suggest_shell_commands=False,
            detect_urls=False,
            use_git=True,  # Always use git
        )
        logger.info("Aider coder instance created successfully.")

        # Run the coding session
        logger.info("Starting Aider coding session...")  # Point 3: Before run
        result = coder.run(ai_coding_prompt)
        logger.info(f"Aider coding session result: {result}")
        logger.info("Aider coding session finished.")  # Point 4: After run

        # Process the results after the coder has run
        logger.info("Processing coder results...")  # Point 5: Processing results
        try:
            response = _process_coder_results(relative_editable_files, working_dir)
            logger.info("Coder results processed.")
        except Exception as e:
            logger.exception(
                f"Error processing coder results: {str(e)}"
            )  # Point 6: Error
            response = {
                "success": False,
                "diff": f"Error processing files after execution: {str(e)}",
            }

    except Exception as e:
        logger.exception(
            f"Critical Error in code_with_aider: {str(e)}"
        )  # Point 6: Error
        response = {
            "success": False,
            "diff": f"Unhandled Error during Aider execution: {str(e)}",
        }

    formatted_response = _format_response(response)
    logger.info(
        f"code_with_aider process completed. Success: {response.get('success')}"
    )
    logger.info(
        f"Formatted response: {formatted_response}"
    )  # Log complete response for debugging
    return formatted_response

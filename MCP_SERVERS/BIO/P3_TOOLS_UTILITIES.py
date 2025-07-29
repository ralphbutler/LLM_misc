#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp",
# ]
# ///

"""
P3-Tools Utilities MCP Server

This MCP server provides access to P3-Tools utility functions for file management,
data processing, and pipeline support. It implements the tools documented in
P3_TOOLS_UTILITIES.md, covering essential infrastructure for P3-Tools workflows.

The server exposes utility tools covering:
- Authentication and System Tools: p3-login, p3-logout, p3-whoami
- Workspace File Management: p3-ls, p3-cp, p3-rm, p3-cat
- Data Processing Tools: p3-extract, p3-collate, p3-count (core tools implemented)
- Output Formatting Tools: p3-format-results, p3-aggregates-to-html (stubbed)
- Additional Processing Tools: Various p3-* utilities (mix of implemented and stubbed)

Core tools are fully implemented with robust error handling.
Advanced tools are stubbed with clear "not yet implemented" messages.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("P3-Tools Utilities Server")

# Core utility functions from existing servers

def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300) -> Dict[str, Any]:
    """
    Execute P3 tool with proper error handling and output parsing
    """
    try:
        # Source the BV-BRC environment before running the command
        command_str = ' '.join(command)
        full_command = f"source /Applications/BV-BRC.app/user-env.sh && {command_str}"
        
        if input_data:
            result = subprocess.run(
                full_command,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                check=False
            )
        else:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                check=False
            )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'returncode': result.returncode,
            'command': full_command
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command timed out after {timeout} seconds',
            'returncode': -1,
            'command': ' '.join(command)
        }
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Subprocess error: {str(e)}',
            'returncode': -1,
            'command': ' '.join(command)
        }

def diagnose_p3_error(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose P3 tool errors and provide remediation suggestions
    """
    stderr = result.get('stderr', '').lower()
    
    diagnosis = {
        'error_type': 'unknown',
        'retry_recommended': False,
        'remediation': 'Check command syntax and parameters'
    }
    
    # Authentication errors
    if any(phrase in stderr for phrase in ['not logged in', 'authentication', 'unauthorized']):
        diagnosis.update({
            'error_type': 'authentication',
            'retry_recommended': False,
            'remediation': 'Run p3-login to authenticate with BV-BRC'
        })
    
    # Network/server errors
    elif any(phrase in stderr for phrase in ['connection', 'timeout', 'server error', '500', '502', '503']):
        diagnosis.update({
            'error_type': 'network',
            'retry_recommended': True,
            'remediation': 'Server or network issue - retry may succeed'
        })
    
    # Parameter errors
    elif any(phrase in stderr for phrase in ['invalid', 'unknown option', 'missing', 'required']):
        diagnosis.update({
            'error_type': 'parameter',
            'retry_recommended': False,
            'remediation': 'Check command parameters and syntax'
        })
    
    # Tool not available
    elif 'command not found' in stderr:
        diagnosis.update({
            'error_type': 'tool_missing',
            'retry_recommended': False,
            'remediation': 'P3-Tools not installed or not in PATH'
        })
    
    return diagnosis

def robust_p3_execution(command: List[str], input_data: str = None, 
                       timeout: int = 300, max_retries: int = 2) -> Dict[str, Any]:
    """
    Enhanced P3 tool execution with retry logic and error diagnosis
    """
    for attempt in range(max_retries + 1):
        result = run_p3_tool(command, input_data, timeout)
        
        if result['success']:
            return result
        
        # Diagnose the error
        diagnosis = diagnose_p3_error(result)
        result['diagnosis'] = diagnosis
        
        # Retry for transient errors
        if diagnosis.get('retry_recommended', False) and attempt < max_retries:
            time.sleep(2 ** attempt)  # Exponential backoff
            continue
        
        # No more retries or non-transient error
        return result
    
    return result

def parse_p3_tabular_output(output: str) -> List[Dict[str, str]]:
    """
    Parse standard P3-Tools tabular output into list of dictionaries
    """
    if not output.strip():
        return []
    
    lines = output.strip().split('\n')
    
    # Filter out welcome message and non-data lines
    data_lines = []
    headers = None
    found_header = False
    
    for line in lines:
        # Skip welcome messages
        if "Welcome to the BV-BRC" in line:
            continue
        
        # Skip empty lines
        if not line.strip():
            continue
            
        # Look for TSV header
        if not found_header and ('\t' in line and ('.' in line or line.count('\t') > 0)):
            headers = line.split('\t')
            found_header = True
            continue
        
        # This is actual data
        if found_header:
            data_lines.append(line)
    
    # Parse data rows
    results = []
    if headers and data_lines:
        # Clean up header names
        clean_headers = []
        for header in headers:
            clean_header = header
            for prefix in ['genome.', 'subsystem.', 'feature.', 'drug.', 'taxonomy.']:
                if header.startswith(prefix):
                    clean_header = header.replace(prefix, '')
                    break
            clean_headers.append(clean_header)
        
        for line in data_lines:
            values = line.split('\t')
            if len(values) == len(clean_headers):
                row_dict = dict(zip(clean_headers, values))
                results.append(row_dict)
    
    return results

def create_temp_input_file(content: str, suffix: str = '.tmp') -> str:
    """Create temporary input file for tool operations"""
    # Use home directory since BV-BRC can't access /tmp/
    home_dir = os.path.expanduser('~')
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, dir=home_dir) as f:
        f.write(content)
        return f.name

def cleanup_temp_file(filepath: str) -> None:
    """Safely remove temporary file"""
    try:
        if filepath and os.path.exists(filepath):
            os.unlink(filepath)
    except OSError:
        pass

# Helper function for not-yet-implemented tools
def not_yet_implemented(tool_name: str, description: str = "") -> str:
    """Return standard message for not-yet-implemented tools"""
    return json.dumps({
        'implemented': False,
        'tool_name': tool_name,
        'status': 'not_yet_implemented',
        'message': f'{tool_name} is not yet implemented in this MCP server',
        'description': description,
        'note': 'This is a stub - the tool may or may not exist in the actual P3-Tools suite'
    }, indent=2)

# Authentication and System Tools (CORE - IMPLEMENTED)

@mcp.tool()
def p3_login(username: Optional[str] = None, password: Optional[str] = None) -> str:
    """
    Authenticate with BV-BRC using p3-login
    
    Args:
        username: BV-BRC username (if None, will prompt)
        password: BV-BRC password (if None, will prompt) - NOT logged or returned
    
    Returns:
        JSON string with authentication result
    """
    
    command = ['p3-login']
    
    # Add credentials if provided (otherwise tool will prompt)
    if username:
        command.extend(['--user', username])
    if password:
        command.extend(['--password', password])
    
    result = robust_p3_execution(command, timeout=60)
    
    if result['success']:
        return json.dumps({
            'authenticated': True,
            'message': 'Successfully logged in to BV-BRC',
            'username': username if username else 'prompted',
            'output': result['stdout']
        }, indent=2)
    else:
        return json.dumps({
            'authenticated': False,
            'error': result['stderr'],
            'message': 'Authentication failed',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_logout() -> str:
    """
    End BV-BRC authentication session
    
    Returns:
        JSON string with logout result
    """
    
    result = robust_p3_execution(['p3-logout'])
    
    return json.dumps({
        'success': result['success'],
        'message': result['stdout'] if result['success'] else result['stderr'],
        'logout_completed': result['success']
    }, indent=2)

@mcp.tool()
def p3_whoami() -> str:
    """
    Check current BV-BRC authentication status
    
    Returns:
        JSON string with authentication status and username
    """
    
    result = robust_p3_execution(['p3-whoami'])
    
    if result['success']:
        return json.dumps({
            'authenticated': True,
            'username': result['stdout'].strip(),
            'message': 'User is authenticated'
        }, indent=2)
    else:
        return json.dumps({
            'authenticated': False,
            'username': None,
            'message': 'Not authenticated - run p3-login',
            'error': result['stderr']
        }, indent=2)

# Workspace File Management Tools (CORE - IMPLEMENTED)

@mcp.tool()
def p3_ls(path: str = "/rbutler@bvbrc/home", 
          long_format: bool = False,
          recursive: bool = False) -> str:
    """
    List workspace files using p3-ls
    
    Args:
        path: Workspace path to list (default: /rbutler@bvbrc/home)
        long_format: Show detailed file information
        recursive: List subdirectories recursively
    
    Returns:
        JSON string with file listing results
    """
    
    command = ['p3-ls']
    
    if long_format:
        command.append('-l')
    if recursive:
        command.append('-R')
    
    command.append(path)
    
    result = robust_p3_execution(command)
    
    if result['success']:
        files = [line.strip() for line in result['stdout'].split('\n') if line.strip()]
        
        if long_format:
            # Parse detailed listing
            detailed_files = []
            for line in files:
                if line.strip() and not line.startswith('total'):
                    parts = line.split()
                    if len(parts) >= 9:
                        detailed_files.append({
                            'permissions': parts[0],
                            'links': parts[1],
                            'owner': parts[2],
                            'group': parts[3],
                            'size': parts[4],
                            'date': ' '.join(parts[5:8]),
                            'name': ' '.join(parts[8:])
                        })
            
            return json.dumps({
                'success': True,
                'path': path,
                'file_count': len(detailed_files),
                'files': detailed_files,
                'long_format': True
            }, indent=2)
        else:
            return json.dumps({
                'success': True,
                'path': path,
                'file_count': len(files),
                'files': files,
                'long_format': False
            }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'path': path,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_cp_to_workspace(local_path: str, workspace_path: str) -> str:
    """
    Copy local file to workspace
    
    Args:
        local_path: Path to local file
        workspace_path: Destination workspace path
    
    Returns:
        JSON string with copy operation result
    """
    
    command = ['p3-cp', local_path, workspace_path]
    result = robust_p3_execution(command, timeout=120)
    
    return json.dumps({
        'success': result['success'],
        'local_path': local_path,
        'workspace_path': workspace_path,
        'message': 'File copied successfully' if result['success'] else 'Copy failed',
        'error': result['stderr'] if not result['success'] else None,
        'diagnosis': result.get('diagnosis', {}) if not result['success'] else None
    }, indent=2)

@mcp.tool()
def p3_cp_from_workspace(workspace_path: str, local_path: str) -> str:
    """
    Copy workspace file to local system
    
    Args:
        workspace_path: Source workspace path
        local_path: Destination local path
    
    Returns:
        JSON string with copy operation result
    """
    
    command = ['p3-cp', workspace_path, local_path]
    result = robust_p3_execution(command, timeout=120)
    
    return json.dumps({
        'success': result['success'],
        'workspace_path': workspace_path,
        'local_path': local_path,
        'message': 'File copied successfully' if result['success'] else 'Copy failed',
        'error': result['stderr'] if not result['success'] else None,
        'diagnosis': result.get('diagnosis', {}) if not result['success'] else None
    }, indent=2)

@mcp.tool()
def p3_cp_within_workspace(source_path: str, dest_path: str) -> str:
    """
    Copy file within workspace
    
    Args:
        source_path: Source workspace path
        dest_path: Destination workspace path
    
    Returns:
        JSON string with copy operation result
    """
    
    command = ['p3-cp', source_path, dest_path]
    result = robust_p3_execution(command)
    
    return json.dumps({
        'success': result['success'],
        'source_path': source_path,
        'dest_path': dest_path,
        'message': 'File copied successfully' if result['success'] else 'Copy failed',
        'error': result['stderr'] if not result['success'] else None,
        'diagnosis': result.get('diagnosis', {}) if not result['success'] else None
    }, indent=2)

@mcp.tool()
def p3_rm(workspace_path: str, recursive: bool = False, force: bool = False) -> str:
    """
    Remove workspace files or directories
    
    Args:
        workspace_path: Workspace path to remove
        recursive: Remove directories recursively
        force: Force removal without confirmation
    
    Returns:
        JSON string with removal operation result
    """
    
    command = ['p3-rm']
    
    if recursive:
        command.append('-r')
    if force:
        command.append('-f')
    
    command.append(workspace_path)
    
    result = robust_p3_execution(command)
    
    return json.dumps({
        'success': result['success'],
        'workspace_path': workspace_path,
        'recursive': recursive,
        'force': force,
        'message': 'File removed successfully' if result['success'] else 'Removal failed',
        'error': result['stderr'] if not result['success'] else None,
        'diagnosis': result.get('diagnosis', {}) if not result['success'] else None
    }, indent=2)

@mcp.tool()
def p3_cat(workspace_path: str, max_lines: Optional[int] = None) -> str:
    """
    Display workspace file contents
    
    Args:
        workspace_path: Workspace file path
        max_lines: Maximum number of lines to show
    
    Returns:
        JSON string with file contents
    """
    
    command = ['p3-cat', workspace_path]
    
    result = robust_p3_execution(command)
    
    if result['success']:
        output = result['stdout']
        
        # Apply line limiting if specified
        if max_lines:
            lines = output.split('\n')
            output = '\n'.join(lines[:max_lines])
            truncated = len(lines) > max_lines
        else:
            truncated = False
        
        return json.dumps({
            'success': True,
            'workspace_path': workspace_path,
            'content': output,
            'lines': len(output.split('\n')) if output else 0,
            'truncated': truncated,
            'max_lines_applied': max_lines
        }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'workspace_path': workspace_path,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# Data Processing Tools (MIXED - CORE IMPLEMENTED, OTHERS STUBBED)

@mcp.tool()
def p3_count_simple(input_data: str) -> str:
    """
    Count lines/records in input data (simple implementation)
    
    Args:
        input_data: Input data to count
    
    Returns:
        JSON string with count results
    """
    
    if not input_data.strip():
        return json.dumps({
            'success': True,
            'total_lines': 0,
            'non_empty_lines': 0,
            'input_length': 0
        }, indent=2)
    
    lines = input_data.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    return json.dumps({
        'success': True,
        'total_lines': len(lines),
        'non_empty_lines': len(non_empty_lines),
        'input_length': len(input_data),
        'note': 'This is a simple line counter - actual p3-count may have different functionality'
    }, indent=2)

@mcp.tool()
def p3_extract_columns_simple(input_data: str, columns: List[str]) -> str:
    """
    Extract specific columns from tabular data (simple implementation)
    
    Args:
        input_data: Tabular input data
        columns: List of column names to extract
    
    Returns:
        JSON string with extracted data
    """
    
    try:
        parsed_data = parse_p3_tabular_output(input_data)
        
        if not parsed_data:
            return json.dumps({
                'success': False,
                'error': 'No tabular data found to extract columns from'
            }, indent=2)
        
        # Extract specified columns
        extracted_data = []
        for row in parsed_data:
            extracted_row = {}
            for col in columns:
                extracted_row[col] = row.get(col, '')
            extracted_data.append(extracted_row)
        
        return json.dumps({
            'success': True,
            'extracted_columns': columns,
            'row_count': len(extracted_data),
            'data': extracted_data,
            'note': 'This is a simple column extractor - actual p3-extract may have different functionality'
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'Column extraction failed: {str(e)}'
        }, indent=2)

# Job Status Tools (ENHANCED FROM COMPUTATIONAL SERVER)

@mcp.tool()
def p3_check_job_status(job_id: str) -> str:
    """
    Check current status of a computational job
    
    Args:
        job_id: Job ID to check
    
    Returns:
        JSON string with job status information
    """
    
    result = robust_p3_execution(['p3-job-status', job_id])
    
    if result['success']:
        status_output = result['stdout'].lower()
        
        if 'completed' in status_output or 'complete' in status_output:
            status = 'completed'
        elif 'failed' in status_output or 'error' in status_output:
            status = 'failed'
        elif 'running' in status_output:
            status = 'running'
        elif 'queued' in status_output:
            status = 'queued'
        else:
            status = 'unknown'
        
        return json.dumps({
            'success': True,
            'job_id': job_id,
            'status': status,
            'details': result['stdout'],
            'raw_output': result['stdout']
        }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'job_id': job_id,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# System and Directory Tools (MIXED)

@mcp.tool()
def p3_mkdir(workspace_path: str, parents: bool = True) -> str:
    """
    Create workspace directory (if p3-mkdir exists)
    
    Args:
        workspace_path: Directory path to create
        parents: Create parent directories if needed
    
    Returns:
        JSON string with operation result
    """
    
    command = ['p3-mkdir']
    
    if parents:
        command.append('-p')
    
    command.append(workspace_path)
    
    result = robust_p3_execution(command)
    
    return json.dumps({
        'success': result['success'],
        'workspace_path': workspace_path,
        'parents': parents,
        'message': 'Directory created successfully' if result['success'] else 'Directory creation failed',
        'error': result['stderr'] if not result['success'] else None,
        'diagnosis': result.get('diagnosis', {}) if not result['success'] else None,
        'note': 'p3-mkdir may not exist in all P3-Tools installations'
    }, indent=2)

# STUBBED TOOLS - Clear "not yet implemented" messages

@mcp.tool()
def p3_extract(input_data: str, extract_pattern: str, output_format: str = 'tsv') -> str:
    """
    Extract data using p3-extract (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input data to process
        extract_pattern: Extraction pattern/criteria
        output_format: Output format (tsv, csv, json)
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-extract', 
        'Advanced data extraction with patterns and multiple output formats')

@mcp.tool()
def p3_collate(input_files: List[str], collation_key: Optional[str] = None, 
               output_format: str = 'tsv') -> str:
    """
    Collate data from multiple sources (NOT YET IMPLEMENTED)
    
    Args:
        input_files: List of input file paths
        collation_key: Key field for collation
        output_format: Output format
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-collate',
        'Collate and merge data from multiple files based on key fields')

@mcp.tool()
def p3_format_results(input_data: str, output_format: str = 'table', 
                     headers: Optional[List[str]] = None, delimiter: str = '\t') -> str:
    """
    Format P3-Tools results for display (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Raw tabular data
        output_format: Output format (table, csv, json, html)
        headers: Column headers
        delimiter: Input delimiter
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-format-results',
        'Format tabular data into various output formats with proper alignment')

@mcp.tool()
def p3_aggregates_to_html(input_data: str, title: str = "P3-Tools Results", 
                         template: Optional[str] = None) -> str:
    """
    Convert aggregate results to HTML (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Tabular input data
        title: HTML page title
        template: HTML template file
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-aggregates-to-html',
        'Convert data aggregates to formatted HTML reports')

@mcp.tool()
def p3_head(input_data: str, lines: int = 10) -> str:
    """
    Display first N lines of data (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input data or file path
        lines: Number of lines to display
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-head',
        'Display first N lines of data like Unix head command')

@mcp.tool()
def p3_tail(input_data: str, lines: int = 10) -> str:
    """
    Display last N lines of data (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input data or file path
        lines: Number of lines to display
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-tail',
        'Display last N lines of data like Unix tail command')

@mcp.tool()
def p3_sort(input_data: str, columns: Optional[List[str]] = None, 
           unique: bool = False, reverse: bool = False) -> str:
    """
    Sort tabular data by specified columns (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input tabular data
        columns: Columns to sort by
        unique: Remove duplicate records
        reverse: Sort in descending order
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-sort',
        'Sort tabular data by specified columns with various options')

@mcp.tool()
def p3_join(left_data: str, right_data: str, join_column: str, 
           join_type: str = 'inner') -> str:
    """
    Join two data tables on specified column (NOT YET IMPLEMENTED)
    
    Args:
        left_data: Left table data
        right_data: Right table data
        join_column: Column to join on
        join_type: Type of join (inner, left, right)
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-join',
        'Join two tabular datasets on specified columns with various join types')

@mcp.tool()
def p3_pivot(input_data: str, index_column: str, value_column: str, 
            pivot_column: str) -> str:
    """
    Create pivot table from data (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input tabular data
        index_column: Column to use as row index
        value_column: Column containing values
        pivot_column: Column to pivot on
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-pivot',
        'Create pivot tables from tabular data with specified index and value columns')

@mcp.tool()
def p3_tbl_to_fasta(input_data: str, sequence_column: str, 
                   id_column: Optional[str] = None, 
                   description_column: Optional[str] = None) -> str:
    """
    Convert tabular data to FASTA format (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input tabular data
        sequence_column: Column containing sequences
        id_column: Column for sequence IDs
        description_column: Column for descriptions
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-tbl-to-fasta',
        'Convert tabular data to FASTA format with customizable ID and description fields')

@mcp.tool()
def p3_tbl_to_html(input_data: str, title: str = "P3-Tools Results", 
                  styling: str = "basic") -> str:
    """
    Convert tabular data to HTML table (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input tabular data
        title: HTML table title
        styling: Style type (basic, bootstrap, custom)
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-tbl-to-html',
        'Convert tabular data to formatted HTML tables with various styling options')

@mcp.tool()
def p3_stats(input_data: str, stat_column: str, 
            group_column: Optional[str] = None) -> str:
    """
    Calculate statistics for numeric columns (NOT YET IMPLEMENTED)
    
    Args:
        input_data: Input tabular data
        stat_column: Column to calculate statistics for
        group_column: Optional grouping column
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-stats',
        'Calculate descriptive statistics for numeric columns with optional grouping')

@mcp.tool()
def p3_qstat(detailed: bool = False) -> str:
    """
    Query P3-Tools system status and job queue (NOT YET IMPLEMENTED)
    
    Args:
        detailed: Get detailed status information
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-qstat',
        'Query BV-BRC system status, job queue information, and resource usage')

# Pipeline and Batch Utility Functions (IF TIME PERMITS)

@mcp.tool()
def batch_file_upload(local_files: List[str], workspace_dir: str) -> str:
    """
    Upload multiple files to workspace directory
    
    Args:
        local_files: List of local file paths to upload
        workspace_dir: Target workspace directory
    
    Returns:
        JSON string with batch upload results
    """
    
    results = {}
    successful = 0
    failed = 0
    
    for local_file in local_files:
        if not os.path.exists(local_file):
            results[local_file] = {'success': False, 'error': 'Local file does not exist'}
            failed += 1
            continue
        
        filename = Path(local_file).name
        workspace_path = f"{workspace_dir}/{filename}"
        
        # Use the existing p3_cp_to_workspace function
        upload_result = json.loads(p3_cp_to_workspace(local_file, workspace_path))
        results[local_file] = upload_result
        
        if upload_result.get('success'):
            successful += 1
        else:
            failed += 1
    
    return json.dumps({
        'success': True,
        'total_files': len(local_files),
        'successful_uploads': successful,
        'failed_uploads': failed,
        'workspace_dir': workspace_dir,
        'results': results
    }, indent=2)

@mcp.tool()
def batch_file_download(workspace_dir: str, local_dir: str, 
                       file_patterns: Optional[List[str]] = None) -> str:
    """
    Download multiple files from workspace directory
    
    Args:
        workspace_dir: Source workspace directory
        local_dir: Local destination directory
        file_patterns: File patterns to match (if None, downloads all)
    
    Returns:
        JSON string with batch download results
    """
    
    # First, list files in workspace directory
    ls_result = json.loads(p3_ls(workspace_dir))
    
    if not ls_result.get('success'):
        return json.dumps({
            'success': False,
            'error': 'Could not list workspace directory',
            'workspace_dir': workspace_dir,
            'details': ls_result.get('error')
        }, indent=2)
    
    files = ls_result.get('files', [])
    
    # Create local directory if it doesn't exist
    try:
        os.makedirs(local_dir, exist_ok=True)
    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'Could not create local directory: {str(e)}',
            'local_dir': local_dir
        }, indent=2)
    
    results = {}
    successful = 0
    failed = 0
    
    for file in files:
        # Check if file matches patterns
        should_download = True
        if file_patterns:
            should_download = any(pattern in file for pattern in file_patterns)
        
        if should_download:
            workspace_path = f"{workspace_dir}/{file}"
            local_path = f"{local_dir}/{file}"
            
            # Use existing p3_cp_from_workspace function
            download_result = json.loads(p3_cp_from_workspace(workspace_path, local_path))
            results[file] = download_result
            
            if download_result.get('success'):
                successful += 1
            else:
                failed += 1
    
    return json.dumps({
        'success': True,
        'workspace_dir': workspace_dir,
        'local_dir': local_dir,
        'file_patterns': file_patterns,
        'total_files_available': len(files),
        'files_downloaded_attempt': len(results),
        'successful_downloads': successful,
        'failed_downloads': failed,
        'results': results
    }, indent=2)

@mcp.tool()
def workspace_cleanup(workspace_dir: str, keep_patterns: Optional[List[str]] = None,
                     dry_run: bool = True) -> str:
    """
    Clean up workspace directory by removing files (with safety dry-run default)
    
    Args:
        workspace_dir: Workspace directory to clean
        keep_patterns: File patterns to preserve
        dry_run: If True, only report what would be deleted (SAFETY DEFAULT)
    
    Returns:
        JSON string with cleanup results
    """
    
    # First, list files in workspace directory
    ls_result = json.loads(p3_ls(workspace_dir))
    
    if not ls_result.get('success'):
        return json.dumps({
            'success': False,
            'error': 'Could not list workspace directory',
            'workspace_dir': workspace_dir
        }, indent=2)
    
    files = ls_result.get('files', [])
    
    files_to_remove = []
    files_to_keep = []
    
    for file in files:
        should_keep = False
        
        if keep_patterns:
            for pattern in keep_patterns:
                if pattern in file:
                    should_keep = True
                    break
        
        if should_keep:
            files_to_keep.append(file)
        else:
            files_to_remove.append(file)
    
    if dry_run:
        return json.dumps({
            'success': True,
            'dry_run': True,
            'workspace_dir': workspace_dir,
            'total_files': len(files),
            'files_to_keep': len(files_to_keep),
            'files_to_remove': len(files_to_remove),
            'files_that_would_be_removed': files_to_remove,
            'files_that_would_be_kept': files_to_keep,
            'message': 'DRY RUN - No files were actually removed. Set dry_run=False to perform actual cleanup.'
        }, indent=2)
    
    # Actual cleanup (only if dry_run=False)
    removal_results = {}
    successful_removals = 0
    
    for file in files_to_remove:
        workspace_path = f"{workspace_dir}/{file}"
        rm_result = json.loads(p3_rm(workspace_path, force=True))
        removal_results[file] = rm_result
        
        if rm_result.get('success'):
            successful_removals += 1
    
    return json.dumps({
        'success': True,
        'dry_run': False,
        'workspace_dir': workspace_dir,
        'total_files': len(files),
        'files_kept': len(files_to_keep),
        'files_removed_attempted': len(files_to_remove),
        'files_removed_successfully': successful_removals,
        'removal_results': removal_results
    }, indent=2)

if __name__ == "__main__":
    mcp.run()

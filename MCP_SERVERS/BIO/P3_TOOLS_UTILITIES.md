# P3-Tools Utilities Guide

**Part of the P3-Tools Programming Guide Series**

This guide focuses on utility tools for file management, data processing, and pipeline support. These tools provide essential infrastructure for P3-Tools workflows.

## 📚 **Guide Series Navigation**
- **P3_TOOLS_GUIDE_CORE.md** - Core patterns and cross-references
- **P3_TOOLS_DATA_RETRIEVAL.md** - Data retrieval tools (p3-all-*, p3-get-*)
- **P3_TOOLS_COMPUTATIONAL.md** - Computational services (p3-submit-*)
- **P3_TOOLS_UTILITIES.md** ← *You are here* - File management and processing
- **P3_TOOLS_SPECIALIZED.md** - Domain-specific analysis tools

## Table of Contents
- [Utilities Overview](#utilities-overview)
- [Authentication and System Tools](#authentication-and-system-tools)
- [Workspace File Management](#workspace-file-management)
- [Data Processing Tools](#data-processing-tools)
- [Output Formatting Tools](#output-formatting-tools)
- [Configuration and System Tools](#configuration-and-system-tools)
- [Best Practices](#best-practices)

## Utilities Overview

### Tool Categories

#### **Authentication and System Tools**
- `p3-login` - Authenticate with BV-BRC
- `p3-logout` - End authentication session
- `p3-whoami` - Check authentication status

#### **Workspace File Management**
- `p3-ls` - List workspace files and directories
- `p3-cp` - Copy files to/from workspace
- `p3-rm` - Remove workspace files
- `p3-cat` - Display workspace file contents

#### **Data Processing Tools**
- `p3-extract` - Extract/filter data from results
- `p3-collate` - Combine data from multiple sources
- `p3-count` - Count records and statistics
- `p3-compare-cols` - Compare data columns
- `p3-file-filter` - Filter files by criteria

#### **Output Formatting Tools**
- `p3-format-results` - Format tool output
- `p3-aggregates-to-html` - Convert results to HTML
- `p3-fasta-md5` - FASTA file utilities

### Import Required Functions
```python
import subprocess
import os
import tempfile
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# Core execution functions (from P3_TOOLS_GUIDE_CORE.md)
def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300) -> Dict[str, Any]:
    pass

def robust_p3_execution(command: List[str], input_data: str = None, 
                       timeout: int = 300, max_retries: int = 2) -> Dict[str, Any]:
    pass

def parse_p3_tabular_output(output: str) -> List[Dict[str, str]]:
    pass
```

## Authentication and System Tools

### p3-login - Authentication
```python
def p3_login(username: str = None, password: str = None) -> Dict[str, Any]:
    """
    Authenticate with BV-BRC using p3-login
    
    Args:
        username: BV-BRC username (if None, will prompt)
        password: BV-BRC password (if None, will prompt)
    
    Returns:
        Authentication result
    """
    
    command = ['p3-login']
    
    # Add credentials if provided (otherwise tool will prompt)
    if username:
        command.extend(['--user', username])
    if password:
        command.extend(['--password', password])
    
    result = run_p3_tool(command, timeout=60)
    
    if result['success']:
        return {
            'authenticated': True,
            'message': 'Successfully logged in to BV-BRC',
            'output': result['stdout']
        }
    else:
        return {
            'authenticated': False,
            'error': result['stderr'],
            'message': 'Authentication failed'
        }

def p3_logout() -> Dict[str, Any]:
    """
    End BV-BRC authentication session
    
    Returns:
        Logout result
    """
    
    result = run_p3_tool(['p3-logout'])
    
    return {
        'success': result['success'],
        'message': result['stdout'] if result['success'] else result['stderr']
    }

def p3_whoami() -> Dict[str, Any]:
    """
    Check current BV-BRC authentication status
    
    Returns:
        Authentication status and username
    """
    
    result = run_p3_tool(['p3-whoami'])
    
    if result['success']:
        return {
            'authenticated': True,
            'username': result['stdout'].strip(),
            'message': 'User is authenticated'
        }
    else:
        return {
            'authenticated': False,
            'username': None,
            'message': 'Not authenticated - run p3-login'
        }
```

## Workspace File Management

### p3-ls - List Workspace Files
```python
def p3_ls(path: str = "/workspace", 
          long_format: bool = False,
          recursive: bool = False) -> List[str]:
    """
    List workspace files using p3-ls
    
    Args:
        path: Workspace path to list
        long_format: Show detailed file information
        recursive: List subdirectories recursively
    
    Returns:
        List of file/directory names
    """
    
    command = ['p3-ls']
    
    if long_format:
        command.append('-l')
    if recursive:
        command.append('-R')
    
    command.append(path)
    
    result = run_p3_tool(command)
    
    if result['success']:
        return [line.strip() for line in result['stdout'].split('\n') if line.strip()]
    else:
        return []

def p3_ls_detailed(path: str = "/workspace") -> List[Dict[str, str]]:
    """
    Get detailed workspace file listing
    
    Args:
        path: Workspace path to list
    
    Returns:
        List of file information dictionaries
    """
    
    result = run_p3_tool(['p3-ls', '-l', path])
    
    if result['success']:
        files = []
        lines = result['stdout'].split('\n')
        
        for line in lines:
            if line.strip() and not line.startswith('total'):
                parts = line.split()
                if len(parts) >= 9:
                    files.append({
                        'permissions': parts[0],
                        'links': parts[1],
                        'owner': parts[2],
                        'group': parts[3],
                        'size': parts[4],
                        'date': ' '.join(parts[5:8]),
                        'name': ' '.join(parts[8:])
                    })
        return files
    else:
        return []
```

### p3-cp - Copy Files
```python
def p3_cp_to_workspace(local_path: str, workspace_path: str) -> bool:
    """
    Copy local file to workspace
    
    Args:
        local_path: Path to local file
        workspace_path: Destination workspace path
    
    Returns:
        Success status
    """
    
    command = ['p3-cp', local_path, workspace_path]
    result = run_p3_tool(command, timeout=120)
    
    return result['success']

def p3_cp_from_workspace(workspace_path: str, local_path: str) -> bool:
    """
    Copy workspace file to local system
    
    Args:
        workspace_path: Source workspace path
        local_path: Destination local path
    
    Returns:
        Success status
    """
    
    command = ['p3-cp', workspace_path, local_path]
    result = run_p3_tool(command, timeout=120)
    
    return result['success']

def p3_cp_within_workspace(source_path: str, dest_path: str) -> bool:
    """
    Copy file within workspace
    
    Args:
        source_path: Source workspace path
        dest_path: Destination workspace path
    
    Returns:
        Success status
    """
    
    command = ['p3-cp', source_path, dest_path]
    result = run_p3_tool(command)
    
    return result['success']
```

### p3-rm - Remove Files
```python
def p3_rm(workspace_path: str, recursive: bool = False, force: bool = False) -> bool:
    """
    Remove workspace files or directories
    
    Args:
        workspace_path: Workspace path to remove
        recursive: Remove directories recursively
        force: Force removal without confirmation
    
    Returns:
        Success status
    """
    
    command = ['p3-rm']
    
    if recursive:
        command.append('-r')
    if force:
        command.append('-f')
    
    command.append(workspace_path)
    
    result = run_p3_tool(command)
    return result['success']

def p3_rm_multiple(workspace_paths: List[str], recursive: bool = False) -> Dict[str, bool]:
    """
    Remove multiple workspace files/directories
    
    Args:
        workspace_paths: List of workspace paths to remove
        recursive: Remove directories recursively
    
    Returns:
        Dict mapping paths to success status
    """
    
    results = {}
    for path in workspace_paths:
        results[path] = p3_rm(path, recursive=recursive, force=True)
    
    return results
```

### p3-cat - Display File Contents
```python
def p3_cat(workspace_path: str, max_lines: int = None) -> str:
    """
    Display workspace file contents
    
    Args:
        workspace_path: Workspace file path
        max_lines: Maximum number of lines to show
    
    Returns:
        File contents as string
    """
    
    command = ['p3-cat', workspace_path]
    
    result = run_p3_tool(command)
    
    if result['success']:
        output = result['stdout']
        # Apply line limiting if specified (pipes don't work in subprocess)
        if max_lines:
            lines = output.split('\n')
            output = '\n'.join(lines[:max_lines])
        return output
    else:
        return ""

def p3_cat_multiple(workspace_paths: List[str]) -> str:
    """
    Concatenate multiple workspace files
    
    Args:
        workspace_paths: List of workspace file paths
    
    Returns:
        Combined file contents
    """
    
    command = ['p3-cat'] + workspace_paths
    result = run_p3_tool(command)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

## Data Processing Tools

### p3-extract - Data Extraction
```python
def p3_extract(input_data: str, 
               extract_pattern: str,
               output_format: str = 'tsv') -> str:
    """
    Extract data using p3-extract
    
    Args:
        input_data: Input data to process
        extract_pattern: Extraction pattern/criteria
        output_format: Output format (tsv, csv, json)
    
    Returns:
        Extracted data
    """
    
    command = ['p3-extract', '--pattern', extract_pattern, '--format', output_format]
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""

def p3_extract_columns(input_data: str, columns: List[str]) -> str:
    """
    Extract specific columns from tabular data
    
    Args:
        input_data: Tabular input data
        columns: List of column names to extract
    
    Returns:
        Extracted columns data
    """
    
    column_spec = ','.join(columns)
    command = ['p3-extract', '--columns', column_spec]
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

### p3-collate - Data Collation
```python
def p3_collate(input_files: List[str], 
               collation_key: str = None,
               output_format: str = 'tsv') -> str:
    """
    Collate data from multiple sources
    
    Args:
        input_files: List of input file paths
        collation_key: Key field for collation
        output_format: Output format
    
    Returns:
        Collated data
    """
    
    command = ['p3-collate', '--format', output_format]
    
    if collation_key:
        command.extend(['--key', collation_key])
    
    command.extend(input_files)
    
    result = run_p3_tool(command)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

### p3-count - Counting Operations
```python
def p3_count(input_data: str, 
             count_field: str = None,
             unique_only: bool = False) -> Dict[str, int]:
    """
    Count records or field values
    
    Args:
        input_data: Input data to count
        count_field: Specific field to count
        unique_only: Count only unique values
    
    Returns:
        Count results
    """
    
    command = ['p3-count']
    
    if count_field:
        command.extend(['--field', count_field])
    if unique_only:
        command.append('--unique')
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        counts = {}
        for line in result['stdout'].split('\n'):
            if line.strip():
                parts = line.split('\t')
                if len(parts) == 2:
                    counts[parts[0]] = int(parts[1])
        return counts
    else:
        return {}

def p3_count_families(family_type: str = 'plfam') -> int:
    """
    Count protein families using p3-count-families
    
    Args:
        family_type: Type of family to count (plfam, pgfam, figfam)
    
    Returns:
        Family count
    """
    
    command = ['p3-count-families', '--type', family_type]
    result = run_p3_tool(command)
    
    if result['success']:
        return int(result['stdout'].strip())
    else:
        return 0
```

## Output Formatting Tools

### p3-format-results - Result Formatting
```python
def p3_format_results(input_data: str,
                     output_format: str = 'table',
                     headers: List[str] = None,
                     delimiter: str = '\t') -> str:
    """
    Format P3-Tools results for display
    
    Args:
        input_data: Raw tabular data
        output_format: Output format (table, csv, json, html)
        headers: Column headers
        delimiter: Input delimiter
    
    Returns:
        Formatted output
    """
    
    command = ['p3-format-results', '--format', output_format]
    
    if headers:
        command.extend(['--headers', ','.join(headers)])
    if delimiter != '\t':
        command.extend(['--delimiter', delimiter])
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return input_data  # Return original if formatting fails

def p3_format_table(data: List[Dict[str, str]], 
                   columns: List[str] = None) -> str:
    """
    Format data as aligned table
    
    Args:
        data: List of data dictionaries
        columns: Specific columns to display
    
    Returns:
        Formatted table string
    """
    
    if not data:
        return ""
    
    # Use specified columns or all columns
    if columns:
        display_columns = columns
    else:
        display_columns = list(data[0].keys())
    
    # Create tabular input
    lines = ['\t'.join(display_columns)]  # Header
    for row in data:
        values = [row.get(col, '') for col in display_columns]
        lines.append('\t'.join(values))
    
    input_data = '\n'.join(lines)
    
    return p3_format_results(input_data, 'table', display_columns)
```

### p3-aggregates-to-html - HTML Conversion
```python
def p3_aggregates_to_html(input_data: str,
                         title: str = "P3-Tools Results",
                         template: str = None) -> str:
    """
    Convert aggregate results to HTML
    
    Args:
        input_data: Tabular input data
        title: HTML page title
        template: HTML template file
    
    Returns:
        HTML formatted output
    """
    
    command = ['p3-aggregates-to-html', '--title', title]
    
    if template:
        command.extend(['--template', template])
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

## Configuration and System Tools

### p3-config - Configuration Management
```python
def p3_config(key: str = None, value: str = None, 
             list_all: bool = False) -> Dict[str, Any]:
    """
    Manage P3-Tools configuration
    
    Args:
        key: Configuration key
        value: Configuration value (to set)
        list_all: List all configuration settings
    
    Returns:
        Configuration result
    """
    
    command = ['p3-config']
    
    if list_all:
        command.append('--list')
    elif key and value:
        command.extend(['--set', key, value])
    elif key:
        command.extend(['--get', key])
    
    result = run_p3_tool(command)
    
    if result['success']:
        if list_all:
            config = {}
            for line in result['stdout'].split('\n'):
                if '=' in line:
                    k, v = line.split('=', 1)
                    config[k.strip()] = v.strip()
            return {'success': True, 'config': config}
        else:
            return {'success': True, 'value': result['stdout'].strip()}
    else:
        return {'success': False, 'error': result['stderr']}
```

### p3-echo - Echo Utilities
```python
def p3_echo_simple(message: str) -> str:
    """
    Simple echo message using p3-echo
    
    Args:
        message: Message to echo
    
    Returns:
        Echo output
    """
    
    command = ['p3-echo', message]
    result = run_p3_tool(command)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

## Best Practices

### Workspace Management Patterns
```python
def ensure_workspace_directory(path: str) -> bool:
    """
    Ensure workspace directory exists
    
    Args:
        path: Workspace directory path
    
    Returns:
        Success status
    """
    
    # Try to list directory
    files = p3_ls(path)
    
    if not files and path != "/":
        # Directory might not exist, try to create
        # This would require workspace directory creation tools
        # For now, just check if parent exists
        parent = str(Path(path).parent)
        if parent != path:  # Avoid infinite recursion
            return ensure_workspace_directory(parent)
    
    return True

def cleanup_workspace_directory(path: str, keep_patterns: List[str] = None) -> int:
    """
    Clean up workspace directory
    
    Args:
        path: Workspace directory path
        keep_patterns: File patterns to keep
    
    Returns:
        Number of files removed
    """
    
    files = p3_ls(path)
    removed = 0
    
    for file in files:
        file_path = f"{path}/{file}"
        
        # Check if file should be kept
        should_keep = False
        if keep_patterns:
            for pattern in keep_patterns:
                if pattern in file:
                    should_keep = True
                    break
        
        if not should_keep:
            if p3_rm(file_path, force=True):
                removed += 1
    
    return removed
```

### Data Processing Pipelines
```python
def create_data_processing_pipeline(input_data: str, 
                                   processing_steps: List[Dict[str, Any]]) -> str:
    """
    Create data processing pipeline
    
    Args:
        input_data: Initial input data
        processing_steps: List of processing step configurations
    
    Returns:
        Final processed data
    """
    
    current_data = input_data
    
    for step in processing_steps:
        step_type = step.get('type')
        
        if step_type == 'extract':
            current_data = p3_extract(current_data, step.get('pattern', ''))
        elif step_type == 'format':
            current_data = p3_format_results(current_data, step.get('format', 'table'))
        elif step_type == 'count':
            counts = p3_count(current_data, step.get('field'))
            # Convert counts to tabular format
            lines = []
            for key, value in counts.items():
                lines.append(f"{key}\t{value}")
            current_data = '\n'.join(lines)
        
        # Add more processing types as needed
    
    return current_data

# Usage
pipeline_steps = [
    {'type': 'extract', 'pattern': 'CDS'},
    {'type': 'count', 'field': 'product'},
    {'type': 'format', 'format': 'table'}
]

processed_data = create_data_processing_pipeline(raw_data, pipeline_steps)
```

### Additional File Processing Tools

#### p3-head - Display First Lines
```python
def p3_head(input_data: str, lines: int = 10) -> str:
    """
    Display first N lines of data (like Unix head)
    
    Args:
        input_data: Input data or file path
        lines: Number of lines to display
    
    Returns:
        First N lines of data
    """
    
    command = ['p3-head', '--lines', str(lines)]
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

#### p3-tail - Display Last Lines
```python
def p3_tail(input_data: str, lines: int = 10) -> str:
    """
    Display last N lines of data (like Unix tail)
    
    Args:
        input_data: Input data or file path
        lines: Number of lines to display
    
    Returns:
        Last N lines of data
    """
    
    command = ['p3-tail', '--lines', str(lines)]
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

#### p3-sort - Sort Data  
```python
def p3_sort(input_data: str, 
           columns: List[str] = None,
           unique: bool = False,
           count_mode: bool = False,
           reverse: bool = False) -> str:
    """
    Sort tabular data by specified columns
    
    Args:
        input_data: Input tabular data
        columns: Columns to sort by (column names or indices)
        unique: Remove duplicate records
        count_mode: Count occurrences instead of sorting
        reverse: Sort in descending order
    
    Returns:
        Sorted data
    """
    
    command = ['p3-sort']
    
    # Add columns to sort by
    if columns:
        command.extend(columns)
    
    # Add options
    if unique:
        command.append('--unique')
    if count_mode:
        command.append('--count')
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

#### p3-stats - Calculate Statistics
```python
def p3_stats(input_data: str, 
            stat_column: str,
            group_column: str = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics for numeric columns
    
    Args:
        input_data: Input tabular data
        stat_column: Column to calculate statistics for
        group_column: Optional grouping column
    
    Returns:
        Statistics dictionary
    """
    
    command = ['p3-stats', stat_column]
    
    if group_column:
        command.extend(['--col', group_column])
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        # Parse statistics output
        stats_data = parse_p3_tabular_output(result['stdout'])
        return {row.get('group', 'overall'): row for row in stats_data}
    else:
        return {}
```

#### Data Manipulation Tools

#### p3-join - Join Data Tables
```python
def p3_join(left_data: str,
           right_data: str, 
           join_column: str,
           join_type: str = 'inner') -> str:
    """
    Join two data tables on specified column
    
    Args:
        left_data: Left table data
        right_data: Right table data  
        join_column: Column to join on
        join_type: Type of join (inner, left, right)
    
    Returns:
        Joined data table
    """
    
    # Write right data to temp file for p3-join
    right_file = create_temp_input_file(right_data, '.tsv')
    
    try:
        command = ['p3-join', '--col', join_column, right_file]
        
        result = run_p3_tool(command, input_data=left_data)
        
        if result['success']:
            return result['stdout']
        else:
            return ""
    
    finally:
        cleanup_temp_file(right_file)
```

#### p3-pivot - Pivot Table Operations
```python
def p3_pivot(input_data: str,
            index_column: str,
            value_column: str,
            pivot_column: str) -> str:
    """
    Create pivot table from data
    
    Args:
        input_data: Input tabular data
        index_column: Column to use as row index
        value_column: Column containing values
        pivot_column: Column to pivot on
    
    Returns:
        Pivoted data table
    """
    
    command = ['p3-pivot', '--index', index_column, '--value', value_column, '--pivot', pivot_column]
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

#### Data Conversion Tools

#### p3-tbl-to-fasta - Convert Table to FASTA
```python
def p3_tbl_to_fasta(input_data: str,
                   sequence_column: str,
                   id_column: str = None,
                   description_column: str = None) -> str:
    """
    Convert tabular data to FASTA format
    
    Args:
        input_data: Input tabular data
        sequence_column: Column containing sequences
        id_column: Column for sequence IDs
        description_column: Column for descriptions
    
    Returns:
        FASTA formatted sequences
    """
    
    command = ['p3-tbl-to-fasta', '--seq', sequence_column]
    
    if id_column:
        command.extend(['--id', id_column])
    if description_column:
        command.extend(['--desc', description_column])
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

#### p3-tbl-to-html - Convert Table to HTML  
```python
def p3_tbl_to_html(input_data: str,
                  title: str = "P3-Tools Results",
                  styling: str = "basic") -> str:
    """
    Convert tabular data to HTML table
    
    Args:
        input_data: Input tabular data
        title: HTML table title
        styling: Style type (basic, bootstrap, custom)
    
    Returns:
        HTML formatted table
    """
    
    command = ['p3-tbl-to-html', '--title', title, '--style', styling]
    
    result = run_p3_tool(command, input_data=input_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""
```

#### System and Directory Tools

#### p3-mkdir - Create Workspace Directories
```python
def p3_mkdir(workspace_path: str, parents: bool = True) -> bool:
    """
    Create workspace directory
    
    Args:
        workspace_path: Directory path to create
        parents: Create parent directories if needed
    
    Returns:
        True if successful
    """
    
    command = ['p3-mkdir', workspace_path]
    
    if parents:
        command.append('--parents')
    
    result = run_p3_tool(command)
    
    return result['success']
```

#### p3-qstat - Query System Status
```python
def p3_qstat(detailed: bool = False) -> Dict[str, Any]:
    """
    Query P3-Tools system status and job queue
    
    Args:
        detailed: Get detailed status information
    
    Returns:
        System status information
    """
    
    command = ['p3-qstat']
    
    if detailed:
        command.append('--detailed')
    
    result = run_p3_tool(command)
    
    if result['success']:
        try:
            # Try to parse as JSON first
            return json.loads(result['stdout'])
        except json.JSONDecodeError:
            # Fall back to tabular parsing
            status_data = parse_p3_tabular_output(result['stdout'])
            return {'jobs': status_data}
    else:
        return {'error': result['stderr']}
```

#### p3-job-status - Individual Job Status Check
```python
def p3_job_status(job_ids: List[str],
                 verbose: bool = False,
                 get_stdout: bool = False,
                 get_stderr: bool = False,
                 stdout_file: str = None,
                 stderr_file: str = None) -> List[Dict[str, Any]]:
    """
    Check status of individual BV-BRC jobs
    
    Args:
        job_ids: List of job IDs to check
        verbose: Get detailed job information
        get_stdout: Retrieve job stdout content
        get_stderr: Retrieve job stderr content  
        stdout_file: Write stdout to file (use '-' for terminal)
        stderr_file: Write stderr to file (use '-' for terminal)
    
    Returns:
        List of job status information
    """
    
    command = ['p3-job-status']
    
    # Add verbose flag
    if verbose:
        command.append('--verbose')
    
    # Add output redirection options
    if stdout_file:
        command.extend(['--stdout', stdout_file])
    if stderr_file:
        command.extend(['--stderr', stderr_file])
    
    # Add job IDs as positional arguments
    command.extend(job_ids)
    
    result = run_p3_tool(command, timeout=60)
    
    if result['success']:
        try:
            # Try to parse as structured data
            job_data = parse_p3_tabular_output(result['stdout'])
            if job_data:
                return job_data
            else:
                # Fallback to simple parsing
                return [{'job_id': jid, 'status': 'checked', 'output': result['stdout']} 
                       for jid in job_ids]
        except:
            # Return raw output if parsing fails
            return [{'job_id': jid, 'raw_output': result['stdout']} for jid in job_ids]
    else:
        return [{'job_id': jid, 'error': result['stderr']} for jid in job_ids]


def p3_job_status_single(job_id: str, 
                        get_logs: bool = False) -> Dict[str, Any]:
    """
    Check status of a single job with optional log retrieval
    
    Args:
        job_id: Job ID to check
        get_logs: Retrieve stdout/stderr logs
    
    Returns:
        Job status and optional logs
    """
    
    result = {'job_id': job_id}
    
    # Get basic job status
    status_info = p3_job_status([job_id], verbose=True)
    
    if status_info and status_info[0].get('status') != 'error':
        result.update(status_info[0])
        
        # Get logs if requested
        if get_logs:
            # Get stdout
            stdout_result = p3_job_status([job_id], stdout_file='-')
            if stdout_result and 'output' in stdout_result[0]:
                result['stdout'] = stdout_result[0]['output']
            
            # Get stderr  
            stderr_result = p3_job_status([job_id], stderr_file='-')
            if stderr_result and 'output' in stderr_result[0]:
                result['stderr'] = stderr_result[0]['output']
    
    return result


def p3_monitor_job(job_id: str,
                  poll_interval: int = 30,
                  max_wait_time: int = 3600) -> Dict[str, Any]:
    """
    Monitor a job until completion
    
    Args:
        job_id: Job ID to monitor
        poll_interval: Seconds between status checks
        max_wait_time: Maximum time to wait (seconds)
    
    Returns:
        Final job status
    """
    
    import time
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        status = p3_job_status_single(job_id, get_logs=False)
        
        # Check if job is complete (implementation depends on BV-BRC status format)
        if status.get('status') in ['completed', 'failed', 'error']:
            # Get final logs
            return p3_job_status_single(job_id, get_logs=True)
        
        print(f"Job {job_id} status: {status.get('status', 'unknown')} (waiting {poll_interval}s)")
        time.sleep(poll_interval)
    
    return {'job_id': job_id, 'status': 'timeout', 'message': f'Job monitoring timed out after {max_wait_time} seconds'}
```

#### p3-echo - Pipeline Echo Utility
```python  
def p3_echo(values: List[str],
           header: List[str] = None,
           output_file: str = None,
           no_header: bool = False,
           num_columns: int = None) -> str:
    """
    Echo values for pipeline processing (P3-specific echo utility)
    
    Args:
        values: Values to echo
        header: Header values for first record
        output_file: Output to file instead of stdout
        no_header: File has no header and specified number of columns
        num_columns: Number of columns when no header
    
    Returns:
        Echo output string
    """
    
    command = ['p3-echo']
    
    # Add header if specified
    if header:
        header_str = '\t'.join(header)
        command.extend(['--header', header_str])
    
    # Add no header option
    if no_header and num_columns:
        command.extend(['--nohead', str(num_columns)])
    
    # Add output file option
    if output_file:
        command.extend(['--data', output_file])
    
    # Add values as positional arguments
    command.extend(values)
    
    result = run_p3_tool(command)
    
    if result['success']:
        return result['stdout']
    else:
        return ""


def p3_echo_table(data: List[List[str]], 
                 headers: List[str] = None) -> str:
    """
    Echo tabular data using p3-echo
    
    Args:
        data: List of rows (each row is list of values)
        headers: Optional column headers
    
    Returns:
        Formatted table output
    """
    
    output_lines = []
    
    # Add headers if provided
    if headers:
        result = p3_echo(headers, header=headers)
        if result:
            output_lines.append(result)
    
    # Add data rows
    for row in data:
        row_result = p3_echo(row)
        if row_result:
            output_lines.append(row_result)
    
    return '\n'.join(output_lines)


def p3_echo_pipeline_data(input_data: Any, 
                         format_type: str = 'tabular') -> str:
    """
    Format data for pipeline consumption using p3-echo
    
    Args:
        input_data: Data to format (list, dict, etc.)
        format_type: Output format ('tabular', 'values')
    
    Returns:
        Pipeline-formatted data
    """
    
    if format_type == 'tabular' and isinstance(input_data, list):
        if input_data and isinstance(input_data[0], dict):
            # Convert list of dicts to table
            headers = list(input_data[0].keys())
            rows = []
            for item in input_data:
                row = [str(item.get(h, '')) for h in headers]
                rows.append(row)
            
            return p3_echo_table(rows, headers)
    
    elif format_type == 'values' and isinstance(input_data, list):
        # Simple value list
        str_values = [str(v) for v in input_data]
        return p3_echo(str_values)
    
    # Fallback to string representation
    return str(input_data)
```

### File Management Utilities
```python
def batch_upload_files(local_files: List[str], workspace_dir: str) -> Dict[str, bool]:
    """
    Upload multiple files to workspace
    
    Args:
        local_files: List of local file paths
        workspace_dir: Target workspace directory
    
    Returns:
        Upload results for each file
    """
    
    results = {}
    
    for local_file in local_files:
        filename = Path(local_file).name
        workspace_path = f"{workspace_dir}/{filename}"
        
        results[local_file] = p3_cp_to_workspace(local_file, workspace_path)
    
    return results

def batch_download_results(workspace_dir: str, local_dir: str, 
                          file_patterns: List[str] = None) -> int:
    """
    Download results matching patterns
    
    Args:
        workspace_dir: Source workspace directory
        local_dir: Local destination directory
        file_patterns: File patterns to match
    
    Returns:
        Number of files downloaded
    """
    
    files = p3_ls(workspace_dir)
    downloaded = 0
    
    os.makedirs(local_dir, exist_ok=True)
    
    for file in files:
        should_download = True
        
        if file_patterns:
            should_download = any(pattern in file for pattern in file_patterns)
        
        if should_download:
            workspace_path = f"{workspace_dir}/{file}"
            local_path = f"{local_dir}/{file}"
            
            if p3_cp_from_workspace(workspace_path, local_path):
                downloaded += 1
    
    return downloaded
```

---

**Next Steps**: For specialized analysis tools and remaining P3-Tools, see **P3_TOOLS_SPECIALIZED.md**.
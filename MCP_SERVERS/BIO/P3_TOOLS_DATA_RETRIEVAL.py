#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp",
# ]
# ///

"""
P3-Tools Data Retrieval MCP Server

This MCP server provides access to P3-Tools data retrieval functionality through
subprocess calls to the P3-Tools command-line interface. It implements the tools
documented in P3_TOOLS_DATA_RETRIEVAL.md, covering both p3-all-* and p3-get-* tools.

The server exposes 32 tools covering:
- p3-all-* tools (6 tools): Collection queries with filtering
- p3-get-* tools (26 tools): ID-based data retrieval via stdin
- Pipeline operations and batch processing
- Error handling and validation

All tools follow the subprocess execution patterns from P3_TOOLS_GUIDE_CORE.md.
"""

import json
import subprocess
import time
from typing import Any, Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("P3-Tools Data Retrieval Server")

# Core utility functions from P3_TOOLS_GUIDE_CORE.md

def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300, 
                limit_results: int = None) -> Dict[str, Any]:
    """
    Execute P3 tool with proper error handling and output parsing
    
    Args:
        command: List of command arguments (e.g., ['p3-all-genomes', '--eq', 'species,Escherichia coli'])
        input_data: Optional stdin data for tools that read from stdin
        timeout: Timeout in seconds (default 5 minutes)
        limit_results: Optional limit using head command (for p3-all-* tools)
    
    Returns:
        Dict with 'success', 'stdout', 'stderr', and 'returncode' keys
    """
    try:
        # Source the BV-BRC environment before running the command
        command_str = ' '.join(command)
        
        # Add shell piping for limiting results if specified
        if limit_results and limit_results > 0:
            command_str += f" | head -n {limit_results + 3}"  # +3 for welcome line, empty line, and header
        
        full_command = f"source /Applications/BV-BRC.app/user-env.sh && {command_str}"
        
        if input_data:
            result = subprocess.run(
                full_command,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                check=False  # Don't raise exception on non-zero exit
            )
        else:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                check=False  # Don't raise exception on non-zero exit
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
    except FileNotFoundError:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Command not found: {command[0]}',
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
    
    Args:
        result: Result dictionary from run_p3_tool
    
    Returns:
        Dict with error diagnosis and recommendations
    """
    stderr = result.get('stderr', '').lower()
    command = result.get('command', '')
    
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
    
    # Data not found
    elif any(phrase in stderr for phrase in ['no data', 'not found', 'empty result']):
        diagnosis.update({
            'error_type': 'no_data',
            'retry_recommended': False,
            'remediation': 'No matching data found - check filter criteria'
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
                       timeout: int = 300, max_retries: int = 2, 
                       limit_results: int = None) -> Dict[str, Any]:
    """
    Enhanced P3 tool execution with retry logic and error diagnosis
    
    Args:
        command: Command to execute
        input_data: Optional stdin data
        timeout: Timeout in seconds
        max_retries: Number of retry attempts for transient errors
        limit_results: Optional limit using head command
    
    Returns:
        Dict with execution results and diagnostic information
    """
    
    for attempt in range(max_retries + 1):
        result = run_p3_tool(command, input_data, timeout, limit_results)
        
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
    Filters out BV-BRC welcome messages and handles multi-line fields properly
    
    Args:
        output: Raw stdout from P3 tool
    
    Returns:
        List of dictionaries with column headers as keys
    """
    if not output.strip():
        return []
    
    lines = output.strip().split('\n')
    
    # Filter out welcome message and find header
    headers = None
    found_header = False
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        # Skip welcome messages
        if "Welcome to the BV-BRC" in line:
            continue
        
        # Skip empty lines
        if not line.strip():
            continue
            
        # Look for TSV header (contains dots like genome.genome_id or subsystem.)
        if not found_header and ('\t' in line and ('.' in line or line.count('\t') > 0)):
            # This is likely the header row
            headers = line.split('\t')
            found_header = True
            data_start_idx = i + 1
            break
    
    if not headers:
        return []
    
    # Clean up header names (remove prefixes like 'genome.' for readability)
    clean_headers = []
    for header in headers:
        clean_header = header
        for prefix in ['genome.', 'subsystem.', 'feature.', 'drug.', 'taxonomy.']:
            if header.startswith(prefix):
                clean_header = header.replace(prefix, '')
                break
        clean_headers.append(clean_header)
    
    # Get remaining data lines after header
    data_lines = lines[data_start_idx:]
    if not data_lines:
        return []
    
    # Strategy: Find record start points by looking for lines that start with non-whitespace
    # and contain at least some tabs, then parse each record by accumulating fields
    results = []
    num_cols = len(headers)
    import re
    
    # Identify potential record start lines
    # These should start with non-whitespace and have tabs
    potential_starts = []
    for i, line in enumerate(data_lines):
        # Skip empty lines
        if not line.strip():
            continue
        # Lines starting with just tabs are continuation lines
        if line.startswith('\t'):
            continue
        # Lines with no tabs are likely continuation text
        if '\t' not in line:
            continue
        # This looks like a record start
        potential_starts.append(i)
    
    # Process each potential record
    for i, start_idx in enumerate(potential_starts):
        # Determine end of this record (start of next record or end of data)
        if i + 1 < len(potential_starts):
            end_idx = potential_starts[i + 1]
        else:
            end_idx = len(data_lines)
        
        # Combine all lines for this record
        record_lines = data_lines[start_idx:end_idx]
        record_text = '\n'.join(record_lines)
        
        # Parse fields carefully
        # Start by splitting on tabs, but handle the last field specially
        # since it may contain embedded newlines
        parts = record_text.split('\t')
        
        # We need exactly num_cols fields
        if len(parts) >= num_cols:
            # Take the first num_cols-1 fields directly
            fields = parts[:num_cols-1]
            # The last field is everything remaining joined back with tabs
            last_field = '\t'.join(parts[num_cols-1:])
            fields.append(last_field)
        elif len(parts) == num_cols - 1:
            # Missing the last field - might be empty
            fields = parts + ['']
        else:
            # Not enough fields, skip this record
            continue
        
        # Clean up fields
        cleaned_fields = []
        for field in fields:
            # Strip leading/trailing whitespace and normalize internal whitespace
            cleaned_field = field.strip()
            # Replace multiple whitespace with single spaces
            cleaned_field = re.sub(r'\s+', ' ', cleaned_field)
            cleaned_fields.append(cleaned_field)
        
        # Create the record
        row_dict = dict(zip(clean_headers, cleaned_fields))
        results.append(row_dict)
    
    return results

def validate_genome_ids(genome_ids: List[str]) -> List[str]:
    """Validate and clean genome ID list"""
    valid_ids = []
    for gid in genome_ids:
        # Basic validation - genome IDs are typically numeric.numeric
        if isinstance(gid, str) and '.' in gid:
            parts = gid.split('.')
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                valid_ids.append(gid)
    return valid_ids

# p3-all-* Tools (Collection Query Tools)

@mcp.tool()
def p3_all_contigs(genome_id: Optional[str] = None, 
                   length_min: Optional[int] = None,
                   additional_filters: Optional[List[str]] = None,
                   attributes: Optional[List[str]] = None,
                   count_only: bool = False,
                   limit: int = 1000) -> str:
    """
    Get contig/sequence data using p3-all-contigs
    
    Args:
        genome_id: Filter by genome ID
        length_min: Minimum contig length
        additional_filters: Additional --eq filters (format: "field,value")
        attributes: Fields to return (if None, returns default set)
        count_only: Return count instead of records
        limit: Maximum number of results (implemented via shell piping)
    
    Returns:
        JSON string with contig data or error information
    """
    
    command = ['p3-all-contigs']
    
    # Add filters
    if genome_id:
        command.extend(['--eq', f'genome_id,{genome_id}'])
    if length_min:
        command.extend(['--ge', f'length,{length_min}'])
    if additional_filters:
        for filter_expr in additional_filters:
            command.extend(['--eq', filter_expr])
    
    
    # Add attributes or count
    if count_only:
        command.append('--count')
    else:
        if attributes:
            for attr in attributes:
                command.extend(['--attr', attr])
        else:
            # Default useful attributes
            default_attrs = ['genome_id', 'accession', 'length', 'gc_content', 
                           'description', 'sequence_type', 'topology']
            for attr in default_attrs:
                command.extend(['--attr', attr])
    
    # Use shell piping for limiting unless doing count_only
    limit_results = None if count_only else limit
    result = robust_p3_execution(command, limit_results=limit_results)
    
    if result['success']:
        if count_only:
            return json.dumps({'count': result['stdout'].strip()}, indent=2)
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_all_drugs(drug_name: Optional[str] = None, 
                 drug_class: Optional[str] = None,
                 additional_filters: Optional[List[str]] = None,
                 attributes: Optional[List[str]] = None,
                 count_only: bool = False,
                 limit: int = 1000) -> str:
    """
    Get drug/antimicrobial data using p3-all-drugs
    
    Args:
        drug_name: Filter by drug name
        drug_class: Filter by drug class
        additional_filters: Additional --eq filters
        attributes: Fields to return
        count_only: Return count instead of records
        limit: Maximum number of results (implemented via shell piping)
    
    Returns:
        JSON string with drug data or error information
    """
    
    command = ['p3-all-drugs']
    
    # Add filters
    if drug_name:
        command.extend(['--eq', f'drug_name,{drug_name}'])
    if drug_class:
        command.extend(['--eq', f'drug_class,{drug_class}'])
    if additional_filters:
        for filter_expr in additional_filters:
            command.extend(['--eq', filter_expr])
    
    # Add attributes or count
    if count_only:
        command.append('--count')
    else:
        if attributes:
            for attr in attributes:
                command.extend(['--attr', attr])
        else:
            # Default drug attributes
            default_attrs = ['drug_id', 'drug_name', 'drug_class', 'pubchem_cid', 
                           'cas_id', 'description']
            for attr in default_attrs:
                command.extend(['--attr', attr])
    
    # Use shell piping for limiting unless doing count_only
    limit_results = None if count_only else limit
    result = robust_p3_execution(command, limit_results=limit_results)
    
    if result['success']:
        if count_only:
            return json.dumps({'count': result['stdout'].strip()}, indent=2)
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_all_genomes(species: Optional[str] = None, 
                   strain: Optional[str] = None, 
                   genome_status: Optional[str] = None,
                   additional_filters: Optional[List[str]] = None,
                   attributes: Optional[List[str]] = None,
                   count_only: bool = False,
                   limit: int = 1000) -> str:
    """
    Get complete genome records using p3-all-genomes
    
    Args:
        species: Filter by species name
        strain: Filter by strain
        genome_status: Filter by genome status (Complete, WGS, etc.)
        additional_filters: Additional --eq filters
        attributes: Fields to return
        count_only: Return count instead of records
        limit: Maximum number of results (implemented via shell piping)
    
    Returns:
        JSON string with genome data or error information
    """
    
    command = ['p3-all-genomes']
    
    # Add filters
    if species:
        command.extend(['--eq', f'species,{species}'])
    if strain:
        command.extend(['--eq', f'strain,{strain}'])
    if genome_status:
        command.extend(['--eq', f'genome_status,{genome_status}'])
    if additional_filters:
        for filter_expr in additional_filters:
            command.extend(['--eq', filter_expr])
    
    
    # Add attributes or count
    if count_only:
        command.append('--count')
    else:
        if attributes:
            for attr in attributes:
                command.extend(['--attr', attr])
        else:
            # Default comprehensive genome attributes
            default_attrs = ['genome_id', 'genome_name', 'organism_name', 'species',
                           'strain', 'genome_status', 'genome_length', 'chromosomes',
                           'plasmids', 'contigs', 'taxon_id', 'completion_date']
            for attr in default_attrs:
                command.extend(['--attr', attr])
    
    # Use shell piping for limiting unless doing count_only
    limit_results = None if count_only else limit
    result = robust_p3_execution(command, limit_results=limit_results)
    
    if result['success']:
        if count_only:
            return json.dumps({'count': result['stdout'].strip()}, indent=2)
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_all_subsystem_roles(subsystem_name: Optional[str] = None, 
                          role_name: Optional[str] = None,
                          additional_filters: Optional[List[str]] = None,
                          attributes: Optional[List[str]] = None,
                          count_only: bool = False,
                          limit: int = 1000) -> str:
    """
    Get subsystem functional roles using p3-all-subsystem-roles
    
    Args:
        subsystem_name: Filter by subsystem name
        role_name: Filter by role name
        additional_filters: Additional --eq filters
        attributes: Fields to return
        count_only: Return count instead of records
        limit: Maximum number of results (implemented via shell piping)
    
    Returns:
        JSON string with subsystem role data or error information
    """
    
    command = ['p3-all-subsystem-roles']
    
    # Add filters
    if subsystem_name:
        command.extend(['--eq', f'subsystem_name,{subsystem_name}'])
    if role_name:
        command.extend(['--eq', f'role_name,{role_name}'])
    if additional_filters:
        for filter_expr in additional_filters:
            command.extend(['--eq', filter_expr])
    
    # Add attributes or count
    if count_only:
        command.append('--count')
    else:
        if attributes:
            for attr in attributes:
                command.extend(['--attr', attr])
        else:
            # Default subsystem role attributes
            default_attrs = ['subsystem_id', 'subsystem_name', 'role_id', 'role_name', 
                           'role_type', 'role_description']
            for attr in default_attrs:
                command.extend(['--attr', attr])
    
    # Use shell piping for limiting unless doing count_only
    limit_results = None if count_only else limit
    result = robust_p3_execution(command, limit_results=limit_results)
    
    if result['success']:
        if count_only:
            return json.dumps({'count': result['stdout'].strip()}, indent=2)
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_all_subsystems(subsystem_class: Optional[str] = None, 
                     subsystem_name: Optional[str] = None,
                     additional_filters: Optional[List[str]] = None,
                     attributes: Optional[List[str]] = None,
                     count_only: bool = False,
                     limit: int = 1000) -> str:
    """
    Get subsystem data using p3-all-subsystems
    
    Args:
        subsystem_class: Filter by subsystem class
        subsystem_name: Filter by subsystem name
        additional_filters: Additional --eq filters
        attributes: Fields to return
        count_only: Return count instead of records
        limit: Maximum number of results (implemented via shell piping)
    
    Returns:
        JSON string with subsystem data or error information
    """
    
    command = ['p3-all-subsystems']
    
    # Add filters
    if subsystem_class:
        command.extend(['--eq', f'class,{subsystem_class}'])
    if subsystem_name:
        command.extend(['--eq', f'subsystem_name,{subsystem_name}'])
    if additional_filters:
        for filter_expr in additional_filters:
            command.extend(['--eq', filter_expr])
    
    # Add attributes or count
    if count_only:
        command.append('--count')
    else:
        if attributes:
            for attr in attributes:
                command.extend(['--attr', attr])
        else:
            # Default subsystem attributes (including potentially problematic fields)
            default_attrs = ['subsystem_id', 'subsystem_name', 'class', 'subclass', 'description', 'role_count']
            for attr in default_attrs:
                command.extend(['--attr', attr])
    
    # Execute command without shell piping limit for count_only, apply limit after parsing for others
    limit_results = None if count_only else None  # Don't use shell piping for limiting
    result = robust_p3_execution(command, limit_results=limit_results)
    
    if result['success']:
        if count_only:
            return json.dumps({'count': result['stdout'].strip()}, indent=2)
        parsed_data = parse_p3_tabular_output(result['stdout'])
        # Apply limit by slicing the parsed data array to get exact number requested
        if limit and limit > 0:
            parsed_data = parsed_data[:limit]
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_all_taxonomies(taxon_name: Optional[str] = None, 
                     taxon_rank: Optional[str] = None,
                     taxon_id: Optional[int] = None,
                     additional_filters: Optional[List[str]] = None,
                     attributes: Optional[List[str]] = None,
                     count_only: bool = False,
                     limit: int = 1000) -> str:
    """
    Get taxonomic data using p3-all-taxonomies
    
    Args:
        taxon_name: Filter by taxonomic name
        taxon_rank: Filter by taxonomic rank (species, genus, family, etc.)
        taxon_id: Filter by NCBI taxon ID
        additional_filters: Additional --eq filters
        attributes: Fields to return
        count_only: Return count instead of records
        limit: Maximum number of results (implemented via shell piping)
    
    Returns:
        JSON string with taxonomic data or error information
    """
    
    command = ['p3-all-taxonomies']
    
    # Add filters
    if taxon_name:
        command.extend(['--eq', f'taxon_name,{taxon_name}'])
    if taxon_rank:
        command.extend(['--eq', f'taxon_rank,{taxon_rank}'])
    if taxon_id:
        command.extend(['--eq', f'taxon_id,{taxon_id}'])
    if additional_filters:
        for filter_expr in additional_filters:
            command.extend(['--eq', filter_expr])
    
    # Add attributes or count
    if count_only:
        command.append('--count')
    else:
        if attributes:
            for attr in attributes:
                command.extend(['--attr', attr])
        else:
            # Default taxonomy attributes
            default_attrs = ['taxon_id', 'taxon_name', 'taxon_rank', 'parent_id', 
                           'lineage_names', 'genetic_code']
            for attr in default_attrs:
                command.extend(['--attr', attr])
    
    # Use shell piping for limiting unless doing count_only
    limit_results = None if count_only else limit
    result = robust_p3_execution(command, limit_results=limit_results)
    
    if result['success']:
        if count_only:
            return json.dumps({'count': result['stdout'].strip()}, indent=2)
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# p3-get-* Tools (ID-Based Retrieval Tools)

@mcp.tool()
def p3_get_genome_data(genome_ids: List[str],
                       attributes: Optional[List[str]] = None) -> str:
    """
    Get detailed genome data using p3-get-genome-data
    
    Args:
        genome_ids: List of genome IDs to retrieve
        attributes: Fields to return (if None, returns default set)
    
    Returns:
        JSON string with genome data or error information
    """
    
    if not genome_ids:
        return json.dumps({'success': False, 'error': 'No genome IDs provided'}, indent=2)
    
    # Validate genome IDs
    valid_ids = validate_genome_ids(genome_ids)
    if not valid_ids:
        return json.dumps({'success': False, 'error': 'No valid genome IDs provided'}, indent=2)
    
    # Create stdin data
    # Bug fix: P3 tools seem to require multiple input lines even for single genomes
    input_lines = [f"genome.genome_id\t{gid}" for gid in valid_ids]
    if len(input_lines) == 1:
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    command = ['p3-get-genome-data']
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default comprehensive genome attributes
        default_attrs = ['genome_id', 'genome_name', 'organism_name', 'taxon_id',
                        'genome_status', 'strain', 'serovar', 'biovar', 'pathovar',
                        'mlst', 'other_typing', 'culture_collection', 'type_strain',
                        'completion_date', 'publication', 'genome_length', 'chromosomes',
                        'plasmids', 'contigs', 'sequences', 'genome_quality',
                        'checkm_completeness', 'checkm_contamination']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_genome_contigs(genome_ids: List[str],
                         attributes: Optional[List[str]] = None) -> str:
    """
    Get contigs for specific genomes using p3-get-genome-contigs
    
    Args:
        genome_ids: List of genome IDs
        attributes: Fields to return
    
    Returns:
        JSON string with contig data or error information
    """
    
    if not genome_ids:
        return json.dumps({'success': False, 'error': 'No genome IDs provided'}, indent=2)
    
    valid_ids = validate_genome_ids(genome_ids)
    if not valid_ids:
        return json.dumps({'success': False, 'error': 'No valid genome IDs provided'}, indent=2)
    
    # Bug fix: P3 tools seem to require multiple input lines even for single genomes
    input_lines = [f"genome.genome_id\t{gid}" for gid in valid_ids]
    if len(input_lines) == 1:
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    command = ['p3-get-genome-contigs']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['genome_id', 'sequence_id', 'accession', 'length',
                        'gc_content', 'description', 'sequence_type']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_genome_features(genome_ids: List[str],
                          feature_type: Optional[str] = None,
                          gene_name: Optional[str] = None,
                          attributes: Optional[List[str]] = None) -> str:
    """
    Get genome features using p3-get-genome-features
    
    Args:
        genome_ids: List of genome IDs
        feature_type: Filter by feature type (CDS, rRNA, etc.)
        gene_name: Filter by gene name
        attributes: Fields to return
    
    Returns:
        JSON string with feature data or error information
    """
    
    if not genome_ids:
        return json.dumps({'success': False, 'error': 'No genome IDs provided'}, indent=2)
    
    valid_ids = validate_genome_ids(genome_ids)
    if not valid_ids:
        return json.dumps({'success': False, 'error': 'No valid genome IDs provided'}, indent=2)
    
    command = ['p3-get-genome-features']
    
    # Add filters
    if feature_type:
        command.extend(['--eq', f'feature_type,{feature_type}'])
    if gene_name:
        command.extend(['--eq', f'gene,{gene_name}'])
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['genome_id', 'patric_id', 'gene', 'product', 'feature_type', 
                        'start', 'end', 'strand', 'aa_length']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    # Provide genome IDs via stdin
    # Bug fix: P3 tools seem to require multiple input lines even for single genomes
    # So we ensure at least 2 lines are present
    input_lines = [f"genome.genome_id\t{gid}" for gid in valid_ids]
    if len(input_lines) == 1:
        # Duplicate the single genome ID to ensure proper processing
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_feature_data(feature_ids: List[str],
                       attributes: Optional[List[str]] = None) -> str:
    """
    Get detailed feature data using p3-get-feature-data
    
    Args:
        feature_ids: List of feature IDs (patric_id, refseq_locus_tag, etc.)
        attributes: Fields to return
    
    Returns:
        JSON string with feature data or error information
    """
    
    if not feature_ids:
        return json.dumps({'success': False, 'error': 'No feature IDs provided'}, indent=2)
    
    # Bug fix: P3 tools seem to require multiple input lines even for single items
    input_lines = [f"genome_feature.patric_id\t{fid}" for fid in feature_ids]
    if len(input_lines) == 1:
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    command = ['p3-get-feature-data']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['patric_id', 'genome_id', 'sequence_id', 'feature_type',
                        'start', 'end', 'strand', 'gene', 'product', 'aa_length',
                        'gene_id', 'go_terms', 'pathway', 'plfam_id', 'figfam_id']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_feature_sequence(feature_ids: List[str],
                           sequence_type: str = 'dna') -> str:
    """
    Get sequences for features using p3-get-feature-sequence
    
    Args:
        feature_ids: List of feature IDs
        sequence_type: 'dna' or 'protein' 
    
    Returns:
        JSON string with sequence data or error information
    """
    
    if not feature_ids:
        return json.dumps({'success': False, 'error': 'No feature IDs provided'}, indent=2)
    
    # Bug fix: P3 tools seem to require multiple input lines even for single items
    input_lines = [f"genome_feature.patric_id\t{fid}" for fid in feature_ids]
    if len(input_lines) == 1:
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    command = ['p3-get-feature-sequence']
    
    command.append('--' + sequence_type)
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        return json.dumps({'success': True, 'sequences': result['stdout']}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_feature_subsystems(feature_ids: List[str],
                            attributes: Optional[List[str]] = None) -> str:
    """
    Get subsystem associations for features using p3-get-feature-subsystems
    
    Args:
        feature_ids: List of feature IDs
        attributes: Fields to return
    
    Returns:
        JSON string with subsystem data or error information
    """
    
    if not feature_ids:
        return json.dumps({'success': False, 'error': 'No feature IDs provided'}, indent=2)
    
    # Bug fix: P3 tools seem to require multiple input lines even for single items
    input_lines = [f"genome_feature.patric_id\t{fid}" for fid in feature_ids]
    if len(input_lines) == 1:
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    command = ['p3-get-feature-subsystems']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['feature_id', 'subsystem_id', 'subsystem_name',
                        'role_id', 'role_name', 'class', 'subclass']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_subsystem_features(subsystem_ids: List[str],
                            genome_ids: Optional[List[str]] = None,
                            attributes: Optional[List[str]] = None) -> str:
    """
    Get features that participate in specific subsystems using p3-get-subsystem-features
    
    Args:
        subsystem_ids: List of subsystem IDs
        genome_ids: Filter by specific genomes (optional)
        attributes: Fields to return
    
    Returns:
        JSON string with feature data or error information
    """
    
    if not subsystem_ids:
        return json.dumps({'success': False, 'error': 'No subsystem IDs provided'}, indent=2)
    
    stdin_data = '\n'.join([f"subsystem.subsystem_id\t{sid}" for sid in subsystem_ids])
    
    command = ['p3-get-subsystem-features']
    
    # Add genome filter if specified
    if genome_ids:
        valid_ids = validate_genome_ids(genome_ids)
        for gid in valid_ids:
            command.extend(['--eq', f'genome_id,{gid}'])
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['subsystem_id', 'genome_id', 'patric_id', 'gene',
                        'product', 'role_name', 'active']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_drug_genomes(drug_ids: List[str],
                       attributes: Optional[List[str]] = None) -> str:
    """
    Get genomes with drug resistance/susceptibility data using p3-get-drug-genomes
    
    Args:
        drug_ids: List of drug IDs
        attributes: Fields to return
    
    Returns:
        JSON string with drug-genome association data or error information
    """
    
    if not drug_ids:
        return json.dumps({'success': False, 'error': 'No drug IDs provided'}, indent=2)
    
    stdin_data = '\n'.join([f"drug.drug_id\t{did}" for did in drug_ids])
    
    command = ['p3-get-drug-genomes']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['drug_id', 'genome_id', 'genome_name', 'resistant_phenotype',
                        'measurement', 'measurement_unit', 'testing_standard',
                        'laboratory_typing_method', 'laboratory_typing_platform']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_family_data(family_ids: List[str],
                      family_type: str = 'plfam',
                      attributes: Optional[List[str]] = None) -> str:
    """
    Get protein family data using p3-get-family-data
    
    Args:
        family_ids: List of family IDs
        family_type: 'plfam', 'pgfam', or 'figfam'
        attributes: Fields to return
    
    Returns:
        JSON string with family data or error information
    """
    
    if not family_ids:
        return json.dumps({'success': False, 'error': 'No family IDs provided'}, indent=2)
    
    # Format stdin based on family type
    if family_type in ['plfam', 'pgfam', 'figfam']:
        stdin_data = '\n'.join([f"protein_family_ref.family_id\t{fid}" for fid in family_ids])
    else:
        return json.dumps({'success': False, 'error': "family_type must be 'plfam', 'pgfam', or 'figfam'"}, indent=2)
    
    command = ['p3-get-family-data']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['family_id', 'family_type', 'product', 'aa_length_avg',
                        'aa_length_std', 'feature_count']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_taxonomy_data(taxon_ids: List[int],
                        attributes: Optional[List[str]] = None) -> str:
    """
    Get detailed taxonomic data for taxon IDs using p3-get-taxonomy-data
    
    Args:
        taxon_ids: List of NCBI taxon IDs
        attributes: Fields to return
    
    Returns:
        JSON string with taxonomy data or error information
    """
    
    if not taxon_ids:
        return json.dumps({'success': False, 'error': 'No taxon IDs provided'}, indent=2)
    
    stdin_data = '\n'.join([f"taxonomy.taxon_id\t{tid}" for tid in taxon_ids])
    
    command = ['p3-get-taxonomy-data']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['taxon_id', 'taxon_name', 'taxon_rank', 'genetic_code',
                        'parent_id', 'division', 'lineage_ids', 'lineage_names']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({'success': True, 'data': parsed_data, 'count': len(parsed_data)}, indent=2)
    else:
        return json.dumps({
            'success': False, 
            'error': result['stderr'], 
            'command': result['command'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# Pipeline and Batch Processing Tools

@mcp.tool()
def genome_to_features_pipeline(species: str, 
                               feature_type: str = "CDS") -> str:
    """
    Pipeline: Find genomes by species -> Get features
    
    Args:
        species: Species name to search for
        feature_type: Type of features to retrieve
    
    Returns:
        JSON string with pipeline results or error information
    """
    
    # Step 1: Get genomes for species using p3-all-genomes
    genomes_result = json.loads(p3_all_genomes(species=species, attributes=['genome_id']))
    
    if not genomes_result.get('success') or not genomes_result.get('data'):
        return json.dumps({
            'success': False, 
            'error': f'No genomes found for species: {species}',
            'step': 'genome_query'
        }, indent=2)
    
    genome_ids = [g['genome_id'] for g in genomes_result['data']]
    
    # Step 2: Get features for these genomes
    features_result = json.loads(p3_get_genome_features(
        genome_ids=genome_ids, 
        feature_type=feature_type,
        attributes=['patric_id', 'product', 'gene', 'genome_id']
    ))
    
    if not features_result.get('success'):
        return json.dumps({
            'success': False,
            'error': 'Failed to retrieve features',
            'step': 'feature_query',
            'genome_count': len(genome_ids)
        }, indent=2)
    
    return json.dumps({
        'success': True,
        'species': species,
        'feature_type': feature_type,
        'genomes_found': len(genome_ids),
        'features_found': features_result.get('count', 0),
        'features': features_result.get('data', [])
    }, indent=2)

@mcp.tool()
def find_genes_by_name_across_species(gene_name: str, 
                                     species: str) -> str:
    """
    Find genes by name within a specific species
    
    Args:
        gene_name: Gene name to search for
        species: Species to limit search to
    
    Returns:
        JSON string with gene search results or error information
    """
    
    # First get genomes for the species
    genomes_result = json.loads(p3_all_genomes(species=species, attributes=['genome_id']))
    
    if not genomes_result.get('success') or not genomes_result.get('data'):
        return json.dumps({
            'success': False,
            'error': f'No genomes found for species: {species}'
        }, indent=2)
    
    genome_ids = [g['genome_id'] for g in genomes_result['data']]
    
    # Then get features for those genomes with the specific gene name
    features_result = json.loads(p3_get_genome_features(
        genome_ids=genome_ids,
        gene_name=gene_name,
        attributes=['patric_id', 'genome_id', 'gene', 'product', 'start', 'end']
    ))
    
    if not features_result.get('success'):
        return json.dumps({
            'success': False,
            'error': f'Failed to find gene {gene_name} in {species}',
            'genomes_searched': len(genome_ids)
        }, indent=2)
    
    return json.dumps({
        'success': True,
        'gene_name': gene_name,
        'species': species,
        'genomes_searched': len(genome_ids),
        'gene_instances_found': features_result.get('count', 0),
        'genes': features_result.get('data', [])
    }, indent=2)

@mcp.tool()
def batch_process_genomes(genome_ids: List[str], 
                         batch_size: int = 100) -> str:
    """
    Process large genome lists in manageable batches
    
    Args:
        genome_ids: List of genome IDs to process
        batch_size: Number of genomes per batch
    
    Returns:
        JSON string with batch processing results
    """
    
    if not genome_ids:
        return json.dumps({'success': False, 'error': 'No genome IDs provided'}, indent=2)
    
    valid_ids = validate_genome_ids(genome_ids)
    if not valid_ids:
        return json.dumps({'success': False, 'error': 'No valid genome IDs provided'}, indent=2)
    
    results = []
    batch_count = 0
    
    for i in range(0, len(valid_ids), batch_size):
        batch = valid_ids[i:i + batch_size]
        batch_count += 1
        
        # Get essential genome data for this batch
        batch_result = json.loads(p3_get_genome_data(
            batch, 
            attributes=['genome_id', 'genome_name', 'organism_name', 'genome_length']
        ))
        
        if batch_result.get('success'):
            results.extend(batch_result.get('data', []))
        
        # Brief pause between batches to be respectful
        if batch_count > 1:
            time.sleep(0.5)
    
    return json.dumps({
        'success': True,
        'total_genomes_requested': len(genome_ids),
        'valid_genomes': len(valid_ids),
        'batches_processed': batch_count,
        'batch_size': batch_size,
        'results_count': len(results),
        'results': results
    }, indent=2)

if __name__ == "__main__":
    mcp.run()

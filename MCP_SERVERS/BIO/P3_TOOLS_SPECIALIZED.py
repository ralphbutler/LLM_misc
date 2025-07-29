#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp",
# ]
# ///

"""
P3-Tools Specialized Analysis MCP Server

This MCP server provides access to P3-Tools specialized analysis functions for
comparative genomics, k-mer analysis, feature analysis, and domain-specific
computational tasks. It implements the tools documented in P3_TOOLS_SPECIALIZED.md.

The server exposes specialized analysis tools covering:
- Comparative Genomics Tools: p3-blast, p3-discriminating-kmers, p3-co-occur, p3-genome-distance, etc.
- K-mer Analysis Tools: p3-build-kmer-db, k-mer database construction
- Feature Analysis Tools: p3-find-features, p3-find-couples, p3-feature-gap, p3-feature-upstream
- Signature Analysis Tools: p3-signature-families, p3-signature-clusters
- Data Export Tools: p3-dump-genomes, p3-role-features
- 61+ additional specialized tools (stubbed for future implementation)

12 core specialized tools are fully implemented with robust error handling.
Advanced tools are stubbed with clear "not yet implemented" messages.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("P3-Tools Specialized Analysis Server")

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

def create_fasta_input(sequences: Dict[str, str]) -> str:
    """Create FASTA file content from sequence dictionary"""
    lines = []
    for seq_id, sequence in sequences.items():
        lines.append(f">{seq_id}")
        lines.append(sequence)
    return '\n'.join(lines)

# Helper function for not-yet-implemented tools
def not_yet_implemented(tool_name: str, description: str = "", category: str = "specialized") -> str:
    """Return standard message for not-yet-implemented tools"""
    return json.dumps({
        'implemented': False,
        'tool_name': tool_name,
        'category': category,
        'status': 'not_yet_implemented',
        'message': f'{tool_name} is not yet implemented in this MCP server',
        'description': description,
        'note': 'This is a stub - the tool may or may not exist in the actual P3-Tools suite'
    }, indent=2)

# Comparative Genomics Tools (IMPLEMENTED)

@mcp.tool()
def p3_blast(query_sequences: Dict[str, str],
             database: str = 'nr',
             program: str = 'blastp',
             evalue: float = 0.001,
             max_target_seqs: int = 100,
             output_format: int = 6,
             additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Perform direct BLAST search using p3-blast
    
    Args:
        query_sequences: Dict of sequence_id -> sequence
        database: BLAST database (nr, nt, refseq_protein, etc.)
        program: BLAST program (blastp, blastn, blastx, etc.)
        evalue: E-value threshold
        max_target_seqs: Maximum number of target sequences
        output_format: BLAST output format (6 for tabular)
        additional_params: Additional BLAST parameters
    
    Returns:
        JSON string with BLAST results
    """
    
    # Create query file
    fasta_content = create_fasta_input(query_sequences)
    query_file = create_temp_input_file(fasta_content, '.fasta')
    
    try:
        command = ['p3-blast']
        
        # Core parameters
        command.extend(['-query', query_file])
        command.extend(['-db', database])
        command.extend(['-evalue', str(evalue)])
        command.extend(['-max_target_seqs', str(max_target_seqs)])
        command.extend(['-outfmt', str(output_format)])
        
        # Program-specific parameter
        if program != 'blastp':
            command.extend(['-program', program])
        
        # Additional parameters
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'-{param}', str(value)])
        
        result = robust_p3_execution(command, timeout=600)  # BLAST can take time
        
        if result['success']:
            return json.dumps({
                'success': True,
                'program': program,
                'database': database,
                'query_count': len(query_sequences),
                'evalue': evalue,
                'results': result['stdout'],
                'output_format': output_format
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-blast'
            }, indent=2)
    
    finally:
        cleanup_temp_file(query_file)

@mcp.tool()
def p3_discriminating_kmers(genome_groups: List[List[str]],
                           kmer_length: int = 8,
                           min_reps: int = 1,
                           max_reps: int = 1,
                           output_format: str = 'table') -> str:
    """
    Find discriminating k-mers between genome groups
    
    Args:
        genome_groups: List of genome ID groups to compare
        kmer_length: Length of k-mers to analyze
        min_reps: Minimum repetitions in target group
        max_reps: Maximum repetitions in other groups
        output_format: Output format
    
    Returns:
        JSON string with discriminating k-mers results
    """
    
    # Create genome group files
    group_files = []
    
    try:
        for i, group in enumerate(genome_groups):
            group_content = '\n'.join(group)
            group_file = create_temp_input_file(group_content, f'_group{i}.txt')
            group_files.append(group_file)
        
        command = ['p3-discriminating-kmers']
        
        # Add parameters
        command.extend(['--kmer-length', str(kmer_length)])
        command.extend(['--min-reps', str(min_reps)])
        command.extend(['--max-reps', str(max_reps)])
        command.extend(['--output-format', output_format])
        
        # Add group files
        for group_file in group_files:
            command.extend(['--group', group_file])
        
        result = robust_p3_execution(command, timeout=300)
        
        if result['success']:
            return json.dumps({
                'success': True,
                'kmer_length': kmer_length,
                'groups_analyzed': len(genome_groups),
                'total_genomes': sum(len(group) for group in genome_groups),
                'min_reps': min_reps,
                'max_reps': max_reps,
                'results': result['stdout']
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-discriminating-kmers'
            }, indent=2)
    
    finally:
        for group_file in group_files:
            cleanup_temp_file(group_file)

@mcp.tool()
def p3_co_occur(genome_ids: List[str],
               feature_type: str = 'CDS',
               distance_threshold: int = 5000,
               min_genomes: int = 2) -> str:
    """
    Analyze feature co-occurrence patterns across genomes
    
    Args:
        genome_ids: List of genome IDs to analyze
        feature_type: Type of features to analyze
        distance_threshold: Maximum distance for co-occurrence
        min_genomes: Minimum genomes for pattern to be reported
    
    Returns:
        JSON string with co-occurrence patterns
    """
    
    # Create genome list file
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-co-occur']
        
        command.extend(['--genome-file', genome_file])
        command.extend(['--feature-type', feature_type])
        command.extend(['--distance', str(distance_threshold)])
        command.extend(['--min-genomes', str(min_genomes)])
        
        result = robust_p3_execution(command, timeout=300)
        
        if result['success']:
            parsed_data = parse_p3_tabular_output(result['stdout'])
            return json.dumps({
                'success': True,
                'genome_count': len(genome_ids),
                'feature_type': feature_type,
                'distance_threshold': distance_threshold,
                'min_genomes': min_genomes,
                'patterns_found': len(parsed_data),
                'patterns': parsed_data
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-co-occur'
            }, indent=2)
    
    finally:
        cleanup_temp_file(genome_file)

@mcp.tool()
def p3_genome_distance(base_genome_id: str,
                      comparison_genome_ids: List[str],
                      kmer_size: int = 8,
                      dna_mode: bool = False) -> str:
    """
    Calculate k-mer based distances between genomes
    
    Args:
        base_genome_id: Reference genome ID
        comparison_genome_ids: List of genomes to compare against base
        kmer_size: Size of k-mers for comparison (default 8)
        dna_mode: Use DNA k-mers instead of protein k-mers
    
    Returns:
        JSON string with genome distance measurements
    """
    
    # Prepare input: comparison genomes via stdin
    # Bug fix: P3 tools seem to require multiple input lines even for single genomes
    input_lines = [f"genome.genome_id\t{gid}" for gid in comparison_genome_ids]
    if len(input_lines) == 1:
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    command = ['p3-genome-distance']
    
    # Add base genome as positional argument
    command.append(base_genome_id)
    
    # Add k-mer size parameter
    command.extend(['--kmer', str(kmer_size)])
    
    # Add DNA mode if specified
    if dna_mode:
        command.append('--dna')
    
    result = robust_p3_execution(command, input_data=stdin_data, timeout=300)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({
            'success': True,
            'base_genome': base_genome_id,
            'comparison_genomes': comparison_genome_ids,
            'kmer_size': kmer_size,
            'dna_mode': dna_mode,
            'distances_calculated': len(parsed_data),
            'distances': parsed_data
        }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {}),
            'tool': 'p3-genome-distance'
        }, indent=2)

# K-mer Analysis Tools (IMPLEMENTED)

@mcp.tool()
def p3_build_kmer_db(genome_ids: List[str],
                    kmer_length: int = 8,
                    database_name: str = 'kmer_db',
                    output_path: str = '/rbutler@bvbrc/home/kmer_dbs') -> str:
    """
    Build k-mer database from genomes
    
    Args:
        genome_ids: List of genome IDs
        kmer_length: K-mer length
        database_name: Name for the database
        output_path: Output directory path
    
    Returns:
        JSON string with database construction result
    """
    
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-build-kmer-db']
        
        command.extend(['--genome-file', genome_file])
        command.extend(['--kmer-length', str(kmer_length)])
        command.extend(['--database-name', database_name])
        command.extend(['--output-path', output_path])
        
        result = robust_p3_execution(command, timeout=600)
        
        if result['success']:
            return json.dumps({
                'success': True,
                'database_path': f"{output_path}/{database_name}",
                'kmer_length': kmer_length,
                'genome_count': len(genome_ids),
                'database_name': database_name,
                'output': result['stdout']
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-build-kmer-db'
            }, indent=2)
    
    finally:
        cleanup_temp_file(genome_file)

# Signature Analysis Tools (IMPLEMENTED)

@mcp.tool()
def p3_signature_families(genomes_with_property: List[str],
                         genomes_without_property: List[str],
                         min_fraction_with: float = 0.8,
                         max_fraction_without: float = 0.2) -> str:
    """
    Find protein families that serve as signatures for a genome property
    
    Args:
        genomes_with_property: Genome IDs that have the target property
        genomes_without_property: Genome IDs that lack the target property  
        min_fraction_with: Minimum fraction of "with" genomes that must have family
        max_fraction_without: Maximum fraction of "without" genomes that can have family
    
    Returns:
        JSON string with signature protein families and statistics
    """
    
    # Create genome group files
    with_content = '\n'.join([f"genome.genome_id\t{gid}" for gid in genomes_with_property])
    without_content = '\n'.join([f"genome.genome_id\t{gid}" for gid in genomes_without_property])
    
    with_file = create_temp_input_file(with_content, '_with.txt')
    without_file = create_temp_input_file(without_content, '_without.txt')
    
    try:
        command = ['p3-signature-families']
        
        # Add genome group files
        command.extend(['--gs1', with_file])  # genomes with property
        command.extend(['--gs2', without_file])  # genomes without property
        
        # Add threshold parameters  
        command.extend(['--min', str(min_fraction_with)])
        command.extend(['--max', str(max_fraction_without)])
        
        result = robust_p3_execution(command, timeout=600)
        
        if result['success']:
            parsed_data = parse_p3_tabular_output(result['stdout'])
            return json.dumps({
                'success': True,
                'genomes_with_property': len(genomes_with_property),
                'genomes_without_property': len(genomes_without_property),
                'min_fraction_with': min_fraction_with,
                'max_fraction_without': max_fraction_without,
                'signature_families_found': len(parsed_data),
                'signature_families': parsed_data
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-signature-families'
            }, indent=2)
    
    finally:
        cleanup_temp_file(with_file)
        cleanup_temp_file(without_file)

@mcp.tool()
def p3_signature_clusters(genome_ids: List[str],
                         cluster_method: str = 'hierarchical',
                         distance_metric: str = 'jaccard',
                         min_cluster_size: int = 3) -> str:
    """
    Find signature clusters in genome sets
    
    Args:
        genome_ids: List of genome IDs
        cluster_method: Clustering method
        distance_metric: Distance metric for clustering
        min_cluster_size: Minimum cluster size
    
    Returns:
        JSON string with signature cluster results
    """
    
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-signature-clusters']
        
        command.extend(['--genome-file', genome_file])
        command.extend(['--method', cluster_method])
        command.extend(['--distance', distance_metric])
        command.extend(['--min-size', str(min_cluster_size)])
        
        result = robust_p3_execution(command, timeout=300)
        
        if result['success']:
            parsed_data = parse_p3_tabular_output(result['stdout'])
            return json.dumps({
                'success': True,
                'genome_count': len(genome_ids),
                'cluster_method': cluster_method,
                'distance_metric': distance_metric,
                'min_cluster_size': min_cluster_size,
                'clusters_found': len(parsed_data),
                'clusters': parsed_data
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-signature-clusters'
            }, indent=2)
    
    finally:
        cleanup_temp_file(genome_file)

# Feature Analysis Tools (IMPLEMENTED)

@mcp.tool()
def p3_find_features(search_criteria: Optional[Dict[str, Any]] = None,
                    keyword: Optional[str] = None,
                    attributes: Optional[List[str]] = None,
                    count_only: bool = False) -> str:
    """
    Search for features across the entire BV-BRC feature database
    
    Args:
        search_criteria: Dictionary of field filters (e.g., {"feature_type": "CDS", "organism_name": "Escherichia"})
        keyword: Keyword to search for in any field
        attributes: Specific attributes to return
        count_only: Return count only instead of records
    
    Returns:
        JSON string with matching features or count
    """
    
    command = ['p3-find-features']
    
    # Add search criteria as --eq parameters
    if search_criteria:
        for field, value in search_criteria.items():
            command.extend(['--eq', f'{field},{value}'])
    
    # Add keyword search
    if keyword:
        command.extend(['--keyword', keyword])
    
    # Add count mode
    if count_only:
        command.append('--count')
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default attributes for features
        default_attrs = ['patric_id', 'feature_type', 'product', 'genome_name', 'start', 'end']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    # Add positional argument (key name)  
    command.append('patric_id')
    
    result = robust_p3_execution(command, timeout=300)
    
    if result['success']:
        if count_only:
            # Parse count result
            try:
                count = int(result['stdout'].strip())
                return json.dumps({
                    'success': True,
                    'count': count,
                    'search_criteria': search_criteria,
                    'keyword': keyword
                }, indent=2)
            except ValueError:
                return json.dumps({
                    'success': True,
                    'count': 0,
                    'search_criteria': search_criteria,
                    'keyword': keyword
                }, indent=2)
        else:
            parsed_data = parse_p3_tabular_output(result['stdout'])
            return json.dumps({
                'success': True,
                'features_found': len(parsed_data),
                'search_criteria': search_criteria,
                'keyword': keyword,
                'features': parsed_data
            }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {}),
            'tool': 'p3-find-features'
        }, indent=2)

@mcp.tool()
def p3_find_couples(input_data: str,
                   category_column: str,
                   min_count: int = 2,
                   max_gap: int = 1000,
                   location_column: Optional[str] = None,
                   sequence_column: Optional[str] = None) -> str:
    """
    Find features that co-occur (coupled features like operons)
    
    Args:
        input_data: Input data with features to analyze
        category_column: Column containing feature categories
        min_count: Minimum occurrence count for couples
        max_gap: Maximum gap between coupled features
        location_column: Column with feature locations
        sequence_column: Column with sequence/contig IDs
    
    Returns:
        JSON string with coupled feature pairs and statistics
    """
    
    command = ['p3-find-couples', category_column]
    
    # Add minimum count
    command.extend(['--min', str(min_count)])
    
    # Add maximum gap
    if max_gap:
        command.extend(['--maxgap', str(max_gap)])
    
    # Add location column if specified
    if location_column:
        command.extend(['--location', location_column])
    
    # Add sequence column if specified
    if sequence_column:
        command.extend(['--sequence', sequence_column])
    
    result = robust_p3_execution(command, input_data=input_data, timeout=300)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({
            'success': True,
            'category_column': category_column,
            'min_count': min_count,
            'max_gap': max_gap,
            'couples_found': len(parsed_data),
            'couples': parsed_data
        }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {}),
            'tool': 'p3-find-couples'
        }, indent=2)

@mcp.tool()
def p3_feature_gap(genome_ids: List[str],
                  reference_genome_id: str,
                  gap_threshold: int = 1000,
                  feature_types: Optional[List[str]] = None) -> str:
    """
    Analyze gaps in feature coverage between genomes
    
    Args:
        genome_ids: List of genome IDs to analyze
        reference_genome_id: Reference genome for comparison
        gap_threshold: Minimum gap size to report
        feature_types: Types of features to analyze
    
    Returns:
        JSON string with gap analysis results
    """
    
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-feature-gap']
        
        command.extend(['--genome-file', genome_file])
        command.extend(['--reference', reference_genome_id])
        command.extend(['--gap-threshold', str(gap_threshold)])
        
        if feature_types:
            for ft in feature_types:
                command.extend(['--feature-type', ft])
        
        result = robust_p3_execution(command, timeout=300)
        
        if result['success']:
            parsed_data = parse_p3_tabular_output(result['stdout'])
            return json.dumps({
                'success': True,
                'genome_count': len(genome_ids),
                'reference_genome': reference_genome_id,
                'gap_threshold': gap_threshold,
                'feature_types': feature_types,
                'gaps_found': len(parsed_data),
                'gaps': parsed_data
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-feature-gap'
            }, indent=2)
    
    finally:
        cleanup_temp_file(genome_file)

@mcp.tool()
def p3_feature_upstream(feature_ids: List[str],
                       upstream_length: int = 500,
                       include_sequence: bool = False) -> str:
    """
    Analyze upstream regions of features
    
    Args:
        feature_ids: List of feature IDs
        upstream_length: Length of upstream region to analyze
        include_sequence: Whether to include sequence data
    
    Returns:
        JSON string with upstream region analysis results
    """
    
    feature_content = '\n'.join([f"genome_feature.patric_id\t{fid}" for fid in feature_ids])
    
    command = ['p3-feature-upstream']
    
    command.extend(['--upstream-length', str(upstream_length)])
    
    if include_sequence:
        command.append('--include-sequence')
    
    result = robust_p3_execution(command, input_data=feature_content, timeout=180)
    
    if result['success']:
        parsed_data = parse_p3_tabular_output(result['stdout'])
        return json.dumps({
            'success': True,
            'feature_count': len(feature_ids),
            'upstream_length': upstream_length,
            'include_sequence': include_sequence,
            'upstream_regions_found': len(parsed_data),
            'upstream_regions': parsed_data
        }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {}),
            'tool': 'p3-feature-upstream'
        }, indent=2)

# Data Export Tools (IMPLEMENTED)

@mcp.tool()
def p3_dump_genomes(genome_ids: List[str],
                   output_format: str = 'genbank',
                   include_features: bool = True,
                   output_dir: str = '/rbutler@bvbrc/home/genome_dumps') -> str:
    """
    Dump genome data in various formats
    
    Args:
        genome_ids: List of genome IDs
        output_format: Output format (genbank, gff, fasta)
        include_features: Whether to include feature annotations
        output_dir: Output directory
    
    Returns:
        JSON string with dump operation result
    """
    
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-dump-genomes']
        
        command.extend(['--genome-file', genome_file])
        command.extend(['--format', output_format])
        command.extend(['--output-dir', output_dir])
        
        if include_features:
            command.append('--include-features')
        
        result = robust_p3_execution(command, timeout=600)
        
        if result['success']:
            return json.dumps({
                'success': True,
                'output_dir': output_dir,
                'format': output_format,
                'genome_count': len(genome_ids),
                'include_features': include_features,
                'files_created': result['stdout'].split('\n') if result['stdout'] else [],
                'output': result['stdout']
            }, indent=2)
        else:
            return json.dumps({
                'success': False,
                'error': result['stderr'],
                'diagnosis': result.get('diagnosis', {}),
                'tool': 'p3-dump-genomes'
            }, indent=2)
    
    finally:
        cleanup_temp_file(genome_file)

@mcp.tool()
def p3_role_features(role_names: List[str],
                    search_criteria: Optional[Dict[str, Any]] = None,
                    attributes: Optional[List[str]] = None,
                    count_only: bool = False) -> str:
    """
    Find features based on functional roles
    
    Args:
        role_names: List of functional role names to search for
        search_criteria: Additional search criteria (organism, feature_type, etc.)
        attributes: Specific attributes to return
        count_only: Return count only instead of records
    
    Returns:
        JSON string with features having specified roles
    """
    
    # Prepare stdin data with role names
    # Bug fix: P3 tools seem to require multiple input lines even for single items
    input_lines = [f"role\t{role}" for role in role_names]
    if len(input_lines) == 1:
        input_lines.append(input_lines[0])
    stdin_data = '\n'.join(input_lines)
    
    command = ['p3-role-features']
    
    # Add search criteria
    if search_criteria:
        for field, value in search_criteria.items():
            command.extend(['--eq', f'{field},{value}'])
    
    # Add count mode
    if count_only:
        command.append('--count')
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default attributes for role-based features
        default_attrs = ['patric_id', 'feature_type', 'product', 'genome_name', 'role']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = robust_p3_execution(command, input_data=stdin_data, timeout=300)
    
    if result['success']:
        if count_only:
            try:
                count = int(result['stdout'].strip())
                return json.dumps({
                    'success': True,
                    'count': count,
                    'role_names': role_names,
                    'search_criteria': search_criteria
                }, indent=2)
            except ValueError:
                return json.dumps({
                    'success': True,
                    'count': 0,
                    'role_names': role_names,
                    'search_criteria': search_criteria
                }, indent=2)
        else:
            parsed_data = parse_p3_tabular_output(result['stdout'])
            return json.dumps({
                'success': True,
                'features_found': len(parsed_data),
                'role_names': role_names,
                'search_criteria': search_criteria,
                'features': parsed_data
            }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {}),
            'tool': 'p3-role-features'
        }, indent=2)

# STUBBED TOOLS - The remaining 61+ specialized tools mentioned in the documentation

# Comparative Genomics Tools (Additional ~15 tools - STUBBED)

@mcp.tool()
def p3_synteny_analysis(genome_ids: List[str], reference_genome: str) -> str:
    """
    Analyze synteny (gene order conservation) between genomes (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs to analyze
        reference_genome: Reference genome for synteny comparison
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-synteny-analysis', 
        'Analyze gene order conservation and synteny blocks between genomes', 'comparative_genomics')

@mcp.tool()
def p3_ortholog_analysis(genome_ids: List[str], method: str = 'reciprocal_best_hit') -> str:
    """
    Find orthologous genes between genomes (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        method: Ortholog detection method
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-ortholog-analysis',
        'Identify orthologous and paralogous genes across genomes', 'comparative_genomics')

@mcp.tool()
def p3_genome_alignment(genome_ids: List[str], alignment_method: str = 'progressiveMauve') -> str:
    """
    Perform whole genome alignment (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs to align
        alignment_method: Alignment algorithm
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-genome-alignment',
        'Perform whole genome multiple sequence alignment', 'comparative_genomics')

# Metabolic Analysis Tools (~10 tools - STUBBED)

@mcp.tool()
def p3_pathway_reconstruction(genome_ids: List[str], pathway_database: str = 'KEGG') -> str:
    """
    Reconstruct metabolic pathways in genomes (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        pathway_database: Pathway database to use
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-pathway-reconstruction',
        'Reconstruct and analyze metabolic pathways from genome annotations', 'metabolic_analysis')

@mcp.tool()
def p3_metabolic_network(genome_id: str, network_type: str = 'bipartite') -> str:
    """
    Construct metabolic networks (NOT YET IMPLEMENTED)
    
    Args:
        genome_id: Genome ID for network construction
        network_type: Type of network to construct
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-metabolic-network',
        'Construct and analyze metabolic networks from pathway data', 'metabolic_analysis')

@mcp.tool()
def p3_enzyme_classification(protein_sequences: Dict[str, str]) -> str:
    """
    Classify enzymes and predict EC numbers (NOT YET IMPLEMENTED)
    
    Args:
        protein_sequences: Dict of protein_id -> sequence
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-enzyme-classification',
        'Classify enzymes and predict EC numbers from protein sequences', 'metabolic_analysis')

# Protein Analysis Tools (~12 tools - STUBBED)

@mcp.tool()
def p3_protein_structure_prediction(protein_sequences: Dict[str, str], method: str = 'alphafold') -> str:
    """
    Predict protein 3D structures (NOT YET IMPLEMENTED)
    
    Args:
        protein_sequences: Dict of protein_id -> sequence
        method: Structure prediction method
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-protein-structure-prediction',
        'Predict 3D protein structures using various computational methods', 'protein_analysis')

@mcp.tool()
def p3_domain_architecture(protein_sequences: Dict[str, str]) -> str:
    """
    Analyze protein domain architectures (NOT YET IMPLEMENTED)
    
    Args:
        protein_sequences: Dict of protein_id -> sequence
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-domain-architecture',
        'Analyze protein domain architectures and functional organization', 'protein_analysis')

@mcp.tool()
def p3_protein_interaction_prediction(genome_id: str, confidence_threshold: float = 0.7) -> str:
    """
    Predict protein-protein interactions (NOT YET IMPLEMENTED)
    
    Args:
        genome_id: Genome ID for interaction prediction
        confidence_threshold: Minimum confidence score
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-protein-interaction-prediction',
        'Predict protein-protein interactions using computational methods', 'protein_analysis')

# Advanced Phylogenetics Tools (~8 tools - STUBBED)

@mcp.tool()
def p3_species_tree_construction(genome_ids: List[str], method: str = 'concat', marker_genes: int = 50) -> str:
    """
    Construct species phylogenetic trees (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        method: Tree construction method
        marker_genes: Number of marker genes to use
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-species-tree-construction',
        'Construct species trees using concatenated alignments or other methods', 'phylogenetics')

@mcp.tool()
def p3_hgt_detection(genome_ids: List[str], reference_tree: str = None) -> str:
    """
    Detect horizontal gene transfer events (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        reference_tree: Reference species tree
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-hgt-detection',
        'Detect horizontal gene transfer events by comparing gene and species trees', 'phylogenetics')

@mcp.tool()
def p3_phylogenetic_profiling(gene_families: List[str], genome_ids: List[str]) -> str:
    """
    Perform phylogenetic profiling analysis (NOT YET IMPLEMENTED)
    
    Args:
        gene_families: List of gene family IDs
        genome_ids: List of genome IDs
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-phylogenetic-profiling',
        'Analyze phylogenetic profiles of gene families across genomes', 'phylogenetics')

# Specialized Genomics Tools (~10 tools - STUBBED)

@mcp.tool()
def p3_mobile_element_detection(genome_ids: List[str], element_types: List[str] = ['transposon', 'integron']) -> str:
    """
    Detect mobile genetic elements (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        element_types: Types of mobile elements to detect
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-mobile-element-detection',
        'Detect and classify mobile genetic elements in genomes', 'specialized_genomics')

@mcp.tool()
def p3_crispr_analysis(genome_ids: List[str], spacer_analysis: bool = True) -> str:
    """
    Analyze CRISPR systems and spacers (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        spacer_analysis: Whether to analyze spacer sequences
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-crispr-analysis',
        'Identify and analyze CRISPR systems and their spacer sequences', 'specialized_genomics')

@mcp.tool()
def p3_prophage_detection(genome_ids: List[str], sensitivity: str = 'medium') -> str:
    """
    Detect prophage sequences in genomes (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        sensitivity: Detection sensitivity level
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-prophage-detection',
        'Detect and analyze prophage sequences integrated in bacterial genomes', 'specialized_genomics')

@mcp.tool()
def p3_amr_prediction(genome_ids: List[str], prediction_method: str = 'amrfinder') -> str:
    """
    Predict antibiotic resistance genes and phenotypes (NOT YET IMPLEMENTED)
    
    Args:
        genome_ids: List of genome IDs
        prediction_method: AMR prediction method
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-amr-prediction',
        'Predict antibiotic resistance genes and resistance phenotypes', 'specialized_genomics')

# Workflow Integration Tools (~6 tools - STUBBED)

@mcp.tool()
def p3_pipeline_orchestrator(pipeline_config: Dict[str, Any]) -> str:
    """
    Orchestrate complex multi-step analysis pipelines (NOT YET IMPLEMENTED)
    
    Args:
        pipeline_config: Pipeline configuration dictionary
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-pipeline-orchestrator',
        'Orchestrate and execute complex multi-step bioinformatics pipelines', 'workflow_integration')

@mcp.tool()
def p3_data_integration(data_sources: List[Dict[str, Any]], integration_method: str = 'join') -> str:
    """
    Integrate data from multiple P3-Tools analyses (NOT YET IMPLEMENTED)
    
    Args:
        data_sources: List of data source configurations
        integration_method: Method for data integration
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-data-integration',
        'Integrate and merge results from multiple P3-Tools analyses', 'workflow_integration')

@mcp.tool()
def p3_visualization_generator(analysis_results: Dict[str, Any], plot_type: str = 'auto') -> str:
    """
    Generate visualizations from analysis results (NOT YET IMPLEMENTED)
    
    Args:
        analysis_results: Analysis results to visualize
        plot_type: Type of visualization to generate
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-visualization-generator',
        'Generate interactive visualizations from P3-Tools analysis results', 'workflow_integration')

@mcp.tool()
def p3_report_generator(analysis_data: Dict[str, Any], report_format: str = 'html') -> str:
    """
    Generate comprehensive analysis reports (NOT YET IMPLEMENTED)
    
    Args:
        analysis_data: Analysis data to include in report
        report_format: Output format for report
    
    Returns:
        JSON string indicating this tool is not yet implemented
    """
    return not_yet_implemented('p3-report-generator',
        'Generate comprehensive HTML/PDF reports from analysis results', 'workflow_integration')

# Workflow and Batch Processing Utilities (IMPLEMENTED)

@mcp.tool()
def create_comparative_analysis_workflow(genome_groups: List[List[str]]) -> str:
    """
    Create comprehensive comparative analysis workflow combining multiple specialized tools
    
    Args:
        genome_groups: Groups of genomes for comparison
    
    Returns:
        JSON string with workflow execution results
    """
    
    results = {}
    errors = []
    
    try:
        # Step 1: K-mer discriminating analysis
        kmer_result = json.loads(p3_discriminating_kmers(genome_groups))
        results['discriminating_kmers'] = kmer_result
        
        # Step 2: Co-occurrence analysis (flatten groups)
        all_genomes = [genome for group in genome_groups for genome in group]
        if len(all_genomes) <= 50:  # Reasonable limit
            cooccur_result = json.loads(p3_co_occur(all_genomes))
            results['co_occurrence'] = cooccur_result
        else:
            results['co_occurrence'] = {'skipped': True, 'reason': 'Too many genomes for co-occurrence analysis'}
        
        # Step 3: Signature clustering
        if len(all_genomes) <= 30:  # Reasonable limit for clustering
            cluster_result = json.loads(p3_signature_clusters(all_genomes))
            results['signature_clusters'] = cluster_result
        else:
            results['signature_clusters'] = {'skipped': True, 'reason': 'Too many genomes for clustering analysis'}
        
        return json.dumps({
            'success': True,
            'workflow': 'comparative_analysis',
            'groups_analyzed': len(genome_groups),
            'total_genomes': len(all_genomes),
            'steps_completed': len([k for k, v in results.items() if not v.get('skipped', False)]),
            'results': results,
            'errors': errors
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            'success': False,
            'workflow': 'comparative_analysis',
            'error': str(e),
            'partial_results': results
        }, indent=2)

@mcp.tool()
def batch_genome_analysis(genome_ids: List[str], analysis_types: List[str], 
                         batch_size: int = 10) -> str:
    """
    Perform batch analysis of multiple genomes with specified analysis types
    
    Args:
        genome_ids: List of genome IDs to analyze
        analysis_types: Types of analyses to perform ('distance', 'features', 'dump')
        batch_size: Number of genomes to process per batch
    
    Returns:
        JSON string with batch analysis results
    """
    
    total_genomes = len(genome_ids)
    batches = [genome_ids[i:i+batch_size] for i in range(0, total_genomes, batch_size)]
    results = {}
    
    for batch_num, batch in enumerate(batches):
        batch_results = {}
        
        # Genome distance analysis
        if 'distance' in analysis_types and len(batch) > 1:
            base_genome = batch[0]
            comparison_genomes = batch[1:]
            distance_result = json.loads(p3_genome_distance(base_genome, comparison_genomes))
            batch_results['distance'] = distance_result
        
        # Feature finding
        if 'features' in analysis_types:
            feature_result = json.loads(p3_find_features({'genome_id': batch[0]}, count_only=True))
            batch_results['features'] = feature_result
        
        # Genome dumping
        if 'dump' in analysis_types:
            dump_result = json.loads(p3_dump_genomes(batch, 'fasta', False))
            batch_results['dump'] = dump_result
        
        results[f'batch_{batch_num + 1}'] = batch_results
    
    return json.dumps({
        'success': True,
        'total_genomes': total_genomes,
        'batches_processed': len(batches),
        'batch_size': batch_size,
        'analysis_types': analysis_types,
        'results': results
    }, indent=2)

if __name__ == "__main__":
    mcp.run()

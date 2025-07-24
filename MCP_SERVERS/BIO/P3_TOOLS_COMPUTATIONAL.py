#!/usr/bin/env python3
"""
P3-Tools Computational Services MCP Server

This MCP server provides access to P3-Tools computational services through
subprocess calls to the P3-Tools command-line interface. It implements the tools
documented in P3_TOOLS_COMPUTATIONAL.md, covering asynchronous computational jobs.

The server exposes 16 computational service tools covering:
- Sequence Analysis Services (2 tools): BLAST, MSA
- Phylogenetic Analysis Services (2 tools): gene trees, codon trees
- Genome Analysis Services (2 tools): annotation, assembly
- Comparative Analysis Services (1 tool): proteome comparison
- Specialized Analysis Services (2 tools): variation analysis, RNA-seq
- Comprehensive Analysis Services (1 tool): CGA
- Metagenomic Analysis Services (4 tools): fastqutils, taxonomic classification, binning, read mapping
- SARS-CoV-2 Services (2 tools): analysis, assembly

All tools follow the subprocess execution patterns with job monitoring and result retrieval.
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
mcp = FastMCP("P3-Tools Computational Services Server")

# Core utility functions from P3_TOOLS_GUIDE_CORE.md and existing servers

def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300) -> Dict[str, Any]:
    """
    Execute P3 tool with proper error handling and output parsing
    
    Args:
        command: List of command arguments
        input_data: Optional stdin data
        timeout: Timeout in seconds (default 5 minutes)
    
    Returns:
        Dict with 'success', 'stdout', 'stderr', and 'returncode' keys
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

# Job Management Utilities

def create_temp_input_file(content: str, suffix: str = '.tmp') -> str:
    """Create temporary input file for job submission"""
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
        pass  # Ignore cleanup failures

def create_fasta_input(sequences: Dict[str, str]) -> str:
    """Create FASTA file content from sequence dictionary"""
    lines = []
    for seq_id, sequence in sequences.items():
        lines.append(f">{seq_id}")
        lines.append(sequence)
    return '\n'.join(lines)

def parse_job_output(stdout: str) -> Dict[str, Any]:
    """Parse job submission output for job ID and status"""
    job_info = {'submitted': True, 'raw_output': stdout}
    
    # Look for common job ID patterns
    lines = stdout.split('\n')
    for line in lines:
        line_lower = line.strip().lower()
        if 'job' in line_lower and ('id' in line_lower or any(char.isdigit() for char in line)):
            # Extract job ID
            import re
            matches = re.findall(r'\d+', line)
            if matches:
                job_info['job_id'] = matches[-1]
        elif 'submitted' in line_lower:
            job_info['status'] = 'submitted'
        elif 'queued' in line_lower:
            job_info['status'] = 'queued'
    
    return job_info

def monitor_job_completion(job_id: str, 
                         poll_interval: int = 30,
                         max_wait_time: int = 3600) -> Dict[str, Any]:
    """
    Monitor job until completion or timeout
    
    Args:
        job_id: Job ID to monitor
        poll_interval: Seconds between status checks
        max_wait_time: Maximum seconds to wait
    
    Returns:
        Job completion status
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        # Check job status using p3-job-status
        result = run_p3_tool(['p3-job-status', job_id])
        
        if result['success']:
            status_output = result['stdout'].lower()
            
            if 'completed' in status_output or 'complete' in status_output:
                return {
                    'completed': True,
                    'status': 'completed',
                    'job_id': job_id,
                    'message': 'Job completed successfully'
                }
            elif 'failed' in status_output or 'error' in status_output:
                return {
                    'completed': True,
                    'status': 'failed',
                    'job_id': job_id,
                    'message': 'Job failed',
                    'details': result['stdout']
                }
            elif 'running' in status_output:
                # Job still running, continue monitoring
                pass
        
        time.sleep(poll_interval)
    
    return {
        'completed': False,
        'status': 'timeout',
        'job_id': job_id,
        'message': f'Job monitoring timeout after {max_wait_time} seconds'
    }

# Sequence Analysis Services (2 tools)

@mcp.tool()
def p3_submit_blast(query_sequences: Dict[str, str],
                   output_path: str,
                   output_name: str,
                   program: str = 'blastp',
                   database: str = 'nr', 
                   evalue: float = 0.001,
                   max_target_seqs: int = 100,
                   additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit BLAST similarity search job
    
    Args:
        query_sequences: Dict of sequence_id -> sequence
        output_path: Workspace output directory
        output_name: Job name/output prefix
        program: BLAST program (blastp, blastn, blastx, tblastn, tblastx)
        database: Target database (nr, nt, refseq_protein, etc.)
        evalue: E-value threshold
        max_target_seqs: Maximum number of target sequences
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    # Create query file
    fasta_content = create_fasta_input(query_sequences)
    query_file = create_temp_input_file(fasta_content, '.fasta')
    
    try:
        command = ['p3-submit-BLAST', output_path, output_name]
        
        # Core parameters
        command.extend(['--program', program])
        command.extend(['--database', database])
        command.extend(['--evalue', str(evalue)])
        command.extend(['--max-target-seqs', str(max_target_seqs)])
        command.extend(['--query-file', query_file])
        
        # Additional parameters
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'--{param}', str(value)])
        
        result = robust_p3_execution(command, timeout=120)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'BLAST',
                'program': program,
                'database': database,
                'output_path': output_path,
                'output_name': output_name,
                'sequence_count': len(query_sequences)
            })
            return json.dumps(job_info, indent=2)
        else:
            return json.dumps({
                'submitted': False,
                'error': result['stderr'],
                'service': 'BLAST',
                'diagnosis': result.get('diagnosis', {})
            }, indent=2)
    
    finally:
        cleanup_temp_file(query_file)

@mcp.tool()
def p3_submit_msa(sequences: Dict[str, str],
                 output_path: str,
                 output_name: str,
                 alignment_method: str = 'muscle',
                 additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit Multiple Sequence Alignment job
    
    Args:
        sequences: Dict of sequence_id -> sequence
        output_path: Workspace output directory  
        output_name: Job name/output prefix
        alignment_method: Alignment algorithm (kept for API compatibility, but P3-MSA uses default method)
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    fasta_content = create_fasta_input(sequences)
    sequence_file = create_temp_input_file(fasta_content, '.fasta')
    
    try:
        # Upload FASTA file to workspace first
        workspace_fasta_path = f"{output_path}/msa_input_{output_name}.fasta"
        upload_command = ['p3-cp', sequence_file, f'ws:{workspace_fasta_path}']
        upload_result = robust_p3_execution(upload_command, timeout=60)
        
        if not upload_result['success']:
            return json.dumps({
                'submitted': False,
                'error': f"Failed to upload FASTA file to workspace: {upload_result['stderr']}",
                'service': 'MSA',
                'diagnosis': {'error_type': 'upload_failed', 'retry_recommended': True}
            }, indent=2)
        
        # Submit MSA job using workspace file
        command = ['p3-submit-MSA', output_path, output_name]
        command.extend(['--fasta-file', f'ws:{workspace_fasta_path}'])
        
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'--{param}', str(value)])
        
        result = robust_p3_execution(command, timeout=120)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'MSA',
                'alignment_method': alignment_method,
                'output_path': output_path,
                'output_name': output_name,
                'sequence_count': len(sequences),
                'input_file': workspace_fasta_path
            })
            return json.dumps(job_info, indent=2)
        else:
            return json.dumps({
                'submitted': False,
                'error': result['stderr'],
                'service': 'MSA',
                'diagnosis': result.get('diagnosis', {})
            }, indent=2)
    
    finally:
        cleanup_temp_file(sequence_file)

# Phylogenetic Analysis Services (2 tools)

@mcp.tool()
def p3_submit_gene_tree(gene_sequences: Dict[str, str],
                       output_path: str,
                       output_name: str,
                       tree_method: str = 'FastTree',
                       alignment_method: str = 'muscle',
                       additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit gene phylogenetic tree construction job
    
    Args:
        gene_sequences: Dict of sequence_id -> gene sequence
        output_path: Workspace output directory
        output_name: Job name/output prefix  
        tree_method: Tree construction method (FastTree, RAxML, etc.)
        alignment_method: Sequence alignment method
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    fasta_content = create_fasta_input(gene_sequences)
    gene_file = create_temp_input_file(fasta_content, '.fasta')
    
    try:
        command = ['p3-submit-gene-tree', output_path, output_name]
        
        command.extend(['--tree-method', tree_method])
        command.extend(['--alignment-method', alignment_method])
        command.extend(['--gene-file', gene_file])
        
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'--{param}', str(value)])
        
        result = robust_p3_execution(command, timeout=120)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'gene-tree',
                'tree_method': tree_method,
                'alignment_method': alignment_method,
                'output_path': output_path,
                'output_name': output_name,
                'sequence_count': len(gene_sequences)
            })
            return json.dumps(job_info, indent=2)
        else:
            return json.dumps({
                'submitted': False,
                'error': result['stderr'],
                'service': 'gene-tree',
                'diagnosis': result.get('diagnosis', {})
            }, indent=2)
    
    finally:
        cleanup_temp_file(gene_file)

@mcp.tool()
def p3_submit_codon_tree(gene_sequences: Dict[str, str],
                        output_path: str,
                        output_name: str,
                        codon_position: str = '123',
                        tree_method: str = 'FastTree',
                        additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit codon-based phylogenetic tree job
    
    Args:
        gene_sequences: Dict of sequence_id -> coding sequence
        output_path: Workspace output directory
        output_name: Job name/output prefix
        codon_position: Codon positions to use ('1', '2', '3', '12', '123')
        tree_method: Tree construction method
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    fasta_content = create_fasta_input(gene_sequences)
    sequence_file = create_temp_input_file(fasta_content, '.fasta')
    
    try:
        command = ['p3-submit-codon-tree', output_path, output_name]
        
        command.extend(['--codon-position', codon_position])
        command.extend(['--tree-method', tree_method])
        command.extend(['--sequence-file', sequence_file])
        
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'--{param}', str(value)])
        
        result = robust_p3_execution(command, timeout=120)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'codon-tree',
                'codon_position': codon_position,
                'tree_method': tree_method,
                'output_path': output_path,
                'output_name': output_name,
                'sequence_count': len(gene_sequences)
            })
            return json.dumps(job_info, indent=2)
        else:
            return json.dumps({
                'submitted': False,
                'error': result['stderr'],
                'service': 'codon-tree',
                'diagnosis': result.get('diagnosis', {})
            }, indent=2)
    
    finally:
        cleanup_temp_file(sequence_file)

# Genome Analysis Services (2 tools)

@mcp.tool()
def p3_submit_genome_annotation(contigs_file: str,
                               output_path: str,
                               output_name: str,
                               domain: str = 'Bacteria',
                               taxonomy_id: Optional[int] = None,
                               annotation_scheme: str = 'RAST',
                               additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit genome annotation job using RAST
    
    Args:
        contigs_file: Path to contigs FASTA file
        output_path: Workspace output directory
        output_name: Job name/output prefix
        domain: Taxonomic domain (Bacteria, Archaea, Virus)
        taxonomy_id: NCBI taxonomy ID for closest organism
        annotation_scheme: Annotation scheme (RAST, etc.)
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-genome-annotation', output_path, output_name]
    
    # Core parameters
    command.extend(['--domain', domain])
    command.extend(['--annotation-scheme', annotation_scheme])
    command.extend(['--contigs-file', contigs_file])
    
    if taxonomy_id:
        command.extend(['--taxonomy-id', str(taxonomy_id)])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=300)  # Annotation jobs can be longer
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'genome-annotation',
            'domain': domain,
            'annotation_scheme': annotation_scheme,
            'taxonomy_id': taxonomy_id,
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'genome-annotation',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_submit_genome_assembly(reads_files: List[str],
                             output_path: str,
                             output_name: str,
                             assembly_strategy: str = 'auto',
                             recipe: str = 'unicycler',
                             additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit genome assembly job
    
    Args:
        reads_files: List of paths to reads files (FASTQ)
        output_path: Workspace output directory
        output_name: Job name/output prefix
        assembly_strategy: Assembly strategy (auto, single_cell, etc.)
        recipe: Assembly recipe/pipeline (unicycler, spades, canu)
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-genome-assembly', output_path, output_name]
    
    # Core parameters
    command.extend(['--assembly-strategy', assembly_strategy])
    command.extend(['--recipe', recipe])
    
    # Add reads files
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'genome-assembly',
            'assembly_strategy': assembly_strategy,
            'recipe': recipe,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'genome-assembly',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# Comparative Analysis Services (1 tool)

@mcp.tool()
def p3_submit_proteome_comparison(genome_ids: List[str],
                                 output_path: str,
                                 output_name: str,
                                 comparison_type: str = 'bidirectional',
                                 evalue_cutoff: float = 1e-5,
                                 additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit proteome comparison job (bidirectional BLAST)
    
    Args:
        genome_ids: List of genome IDs to compare
        output_path: Workspace output directory
        output_name: Job name/output prefix
        comparison_type: Type of comparison (bidirectional, all-vs-all)
        evalue_cutoff: E-value threshold for matches
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    # Create genome ID file
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-submit-proteome-comparison', output_path, output_name]
        
        command.extend(['--comparison-type', comparison_type])
        command.extend(['--evalue-cutoff', str(evalue_cutoff)])
        command.extend(['--genome-file', genome_file])
        
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'--{param}', str(value)])
        
        result = robust_p3_execution(command, timeout=300)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'proteome-comparison',
                'comparison_type': comparison_type,
                'genome_count': len(genome_ids),
                'evalue_cutoff': evalue_cutoff,
                'output_path': output_path,
                'output_name': output_name
            })
            return json.dumps(job_info, indent=2)
        else:
            return json.dumps({
                'submitted': False,
                'error': result['stderr'],
                'service': 'proteome-comparison',
                'diagnosis': result.get('diagnosis', {})
            }, indent=2)
    
    finally:
        cleanup_temp_file(genome_file)

# Specialized Analysis Services (2 tools)

@mcp.tool()
def p3_submit_variation_analysis(reads_files: List[str],
                                reference_genome: str,
                                output_path: str,
                                output_name: str,
                                mapper: str = 'bwa',
                                caller: str = 'gatk',
                                additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit variation analysis job (SNP/variant calling)
    
    Args:
        reads_files: List of paths to reads files
        reference_genome: Reference genome ID or file path
        output_path: Workspace output directory
        output_name: Job name/output prefix
        mapper: Read mapping tool (bwa, bowtie2)
        caller: Variant caller (gatk, bcftools, freebayes)
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-variation-analysis', output_path, output_name]
    
    command.extend(['--reference-genome', reference_genome])
    command.extend(['--mapper', mapper])
    command.extend(['--caller', caller])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'variation-analysis',
            'mapper': mapper,
            'caller': caller,
            'reference_genome': reference_genome,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'variation-analysis',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_submit_rnaseq(reads_files: List[str],
                    reference_genome: str,
                    output_path: str,
                    output_name: str,
                    analysis_type: str = 'differential',
                    mapper: str = 'hisat2',
                    additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit RNA-Seq analysis job
    
    Args:
        reads_files: List of paths to RNA-Seq reads files
        reference_genome: Reference genome ID
        output_path: Workspace output directory
        output_name: Job name/output prefix
        analysis_type: Analysis type (differential, host_response)
        mapper: Read mapping tool (hisat2, star, bowtie2)
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-rnaseq', output_path, output_name]
    
    command.extend(['--reference-genome', reference_genome])
    command.extend(['--analysis-type', analysis_type])
    command.extend(['--mapper', mapper])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'rnaseq',
            'analysis_type': analysis_type,
            'mapper': mapper,
            'reference_genome': reference_genome,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'rnaseq',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# Comprehensive Analysis Services (1 tool)

@mcp.tool()
def p3_submit_cga(reads_files: List[str],
                 output_path: str,
                 output_name: str,
                 recipe: str = 'comprehensive',
                 reference_genome: Optional[str] = None,
                 additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit Comprehensive Genome Analysis (CGA) job
    Combines multiple analysis steps: assembly, annotation, analysis
    
    Args:
        reads_files: List of paths to reads files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        recipe: Analysis recipe (comprehensive, minimal)
        reference_genome: Reference genome for comparison
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-CGA', output_path, output_name]
    
    command.extend(['--recipe', recipe])
    
    if reference_genome:
        command.extend(['--reference-genome', reference_genome])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=600)  # CGA jobs can be very long
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'CGA',
            'recipe': recipe,
            'reference_genome': reference_genome,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'CGA',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# Metagenomic Analysis Services (4 tools)

@mcp.tool()
def p3_submit_fastqutils(reads_files: List[str],
                        output_path: str,
                        output_name: str,
                        operation: str = 'trim',
                        quality_threshold: int = 20,
                        additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit FASTQ utility operations job (trimming, filtering, etc.)
    
    Args:
        reads_files: List of paths to FASTQ files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        operation: Operation type (trim, filter, stats, etc.)
        quality_threshold: Quality score threshold
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-fastqutils', output_path, output_name]
    
    command.extend(['--operation', operation])
    command.extend(['--quality-threshold', str(quality_threshold)])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=120)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'fastqutils',
            'operation': operation,
            'quality_threshold': quality_threshold,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'fastqutils',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_submit_taxonomic_classification(reads_files: List[str],
                                     output_path: str,
                                     output_name: str,
                                     classifier: str = 'kraken2',
                                     database: str = 'standard',
                                     additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit taxonomic classification job
    
    Args:
        reads_files: List of paths to reads files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        classifier: Classification tool (kraken2, metaphlan, etc.)
        database: Reference database
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-taxonomic-classification', output_path, output_name]
    
    command.extend(['--classifier', classifier])
    command.extend(['--database', database])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=180)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'taxonomic-classification',
            'classifier': classifier,
            'database': database,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'taxonomic-classification',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_submit_metagenome_binning(contigs_file: str,
                                reads_files: List[str],
                                output_path: str,
                                output_name: str,
                                binning_method: str = 'metabat2',
                                additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit metagenome binning job
    
    Args:
        contigs_file: Path to assembled contigs file
        reads_files: List of paths to reads files for coverage calculation
        output_path: Workspace output directory
        output_name: Job name/output prefix
        binning_method: Binning algorithm (metabat2, maxbin2, concoct)
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-metagenome-binning', output_path, output_name]
    
    command.extend(['--contigs-file', contigs_file])
    command.extend(['--binning-method', binning_method])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'metagenome-binning',
            'binning_method': binning_method,
            'contigs_file': contigs_file,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'metagenome-binning',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_submit_metagenomic_read_mapping(reads_files: List[str],
                                      reference_genomes: List[str],
                                      output_path: str,
                                      output_name: str,
                                      mapper: str = 'bwa',
                                      additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit metagenomic read mapping job
    
    Args:
        reads_files: List of paths to reads files
        reference_genomes: List of reference genome IDs or paths
        output_path: Workspace output directory
        output_name: Job name/output prefix
        mapper: Read mapping tool (bwa, bowtie2, minimap2)
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    # Create reference genomes file
    ref_content = '\n'.join(reference_genomes)
    ref_file = create_temp_input_file(ref_content, '.txt')
    
    try:
        command = ['p3-submit-metagenomic-read-mapping', output_path, output_name]
        
        command.extend(['--mapper', mapper])
        command.extend(['--reference-file', ref_file])
        
        for reads_file in reads_files:
            command.extend(['--reads-file', reads_file])
        
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'--{param}', str(value)])
        
        result = robust_p3_execution(command, timeout=240)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'metagenomic-read-mapping',
                'mapper': mapper,
                'reference_count': len(reference_genomes),
                'reads_files_count': len(reads_files),
                'output_path': output_path,
                'output_name': output_name
            })
            return json.dumps(job_info, indent=2)
        else:
            return json.dumps({
                'submitted': False,
                'error': result['stderr'],
                'service': 'metagenomic-read-mapping',
                'diagnosis': result.get('diagnosis', {})
            }, indent=2)
    
    finally:
        cleanup_temp_file(ref_file)

# SARS-CoV-2 Services (2 tools)

@mcp.tool()
def p3_submit_sars2_analysis(reads_files: List[str],
                            output_path: str,
                            output_name: str,
                            analysis_type: str = 'comprehensive',
                            reference_genome: str = 'NC_045512.2',
                            additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit SARS-CoV-2 specialized analysis job
    
    Args:
        reads_files: List of paths to reads files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        analysis_type: Analysis type (comprehensive, variants, lineage)
        reference_genome: SARS-CoV-2 reference genome
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-sars2-analysis', output_path, output_name]
    
    command.extend(['--analysis-type', analysis_type])
    command.extend(['--reference-genome', reference_genome])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=180)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'sars2-analysis',
            'analysis_type': analysis_type,
            'reference_genome': reference_genome,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'sars2-analysis',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_submit_sars2_assembly(reads_files: List[str],
                            output_path: str,
                            output_name: str,
                            assembly_strategy: str = 'consensus',
                            reference_genome: str = 'NC_045512.2',
                            additional_params: Optional[Dict[str, Any]] = None) -> str:
    """
    Submit SARS-CoV-2 specialized assembly job
    
    Args:
        reads_files: List of paths to reads files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        assembly_strategy: Assembly strategy (consensus, denovo)
        reference_genome: SARS-CoV-2 reference genome
        additional_params: Additional parameters
    
    Returns:
        JSON string with job submission result
    """
    
    command = ['p3-submit-sars2-assembly', output_path, output_name]
    
    command.extend(['--assembly-strategy', assembly_strategy])
    command.extend(['--reference-genome', reference_genome])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = robust_p3_execution(command, timeout=120)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'sars2-assembly',
            'assembly_strategy': assembly_strategy,
            'reference_genome': reference_genome,
            'reads_files_count': len(reads_files),
            'output_path': output_path,
            'output_name': output_name
        })
        return json.dumps(job_info, indent=2)
    else:
        return json.dumps({
            'submitted': False,
            'error': result['stderr'],
            'service': 'sars2-assembly',
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

# Job Monitoring and Management Tools

@mcp.tool()
def p3_monitor_job(job_id: str,
                   poll_interval: int = 30,
                   max_wait_time: int = 3600) -> str:
    """
    Monitor a computational job until completion or timeout
    
    Args:
        job_id: Job ID to monitor
        poll_interval: Seconds between status checks (default 30)
        max_wait_time: Maximum seconds to wait (default 1 hour)
    
    Returns:
        JSON string with monitoring results
    """
    
    result = monitor_job_completion(job_id, poll_interval, max_wait_time)
    return json.dumps(result, indent=2)

@mcp.tool()
def p3_check_job_status(job_id: str) -> str:
    """
    Check current status of a computational job
    
    Args:
        job_id: Job ID to check
    
    Returns:
        JSON string with job status information
    """
    
    result = run_p3_tool(['p3-job-status', job_id])
    
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
            'details': result['stdout']
        }, indent=2)
    else:
        return json.dumps({
            'success': False,
            'job_id': job_id,
            'error': result['stderr'],
            'diagnosis': result.get('diagnosis', {})
        }, indent=2)

@mcp.tool()
def p3_get_job_results(job_name: str, result_type: str = 'summary') -> str:
    """
    Retrieve and parse computational job results
    
    Args:
        job_name: Name of the completed job
        result_type: Type of results to retrieve ('summary', 'detailed', 'files')
    
    Returns:
        JSON string with job results
    """
    
    # Check the hidden results directory (following OLD_bvbrc_p3tools.py pattern)
    results_base = f"/rbutler@bvbrc/home/.{job_name}"
    
    # List available result files
    list_result = run_p3_tool(['p3-ls', results_base])
    
    if not list_result['success']:
        return json.dumps({
            'success': False,
            'error': f'Could not access results for job {job_name}',
            'details': list_result['stderr']
        }, indent=2)
    
    available_files = list_result['stdout'].strip().split('\n')
    
    results_info = {
        'success': True,
        'job_name': job_name,
        'results_directory': results_base,
        'available_files': available_files,
        'file_count': len(available_files)
    }
    
    if result_type == 'files':
        return json.dumps(results_info, indent=2)
    
    # For summary or detailed, try to read key result files
    key_files = ['output.txt', 'results.json', 'summary.txt', 'report.html']
    file_contents = {}
    
    for filename in key_files:
        if filename in available_files:
            file_path = f"{results_base}/{filename}"
            file_result = run_p3_tool(['p3-cat', file_path])
            
            if file_result['success']:
                content = file_result['stdout']
                if result_type == 'summary':
                    # Truncate for summary
                    file_contents[filename] = content[:1000] + '...' if len(content) > 1000 else content
                else:
                    file_contents[filename] = content
    
    results_info['file_contents'] = file_contents
    return json.dumps(results_info, indent=2)

if __name__ == "__main__":
    mcp.run()
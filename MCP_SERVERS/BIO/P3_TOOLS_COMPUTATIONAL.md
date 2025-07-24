# P3-Tools Computational Services Guide

**Part of the P3-Tools Programming Guide Series**

This guide focuses on computational services (p3-submit-* tools) that submit asynchronous jobs to the BV-BRC compute infrastructure. These tools wrap the BV-BRC App Service JSON-RPC API for complex bioinformatics analyses.

## 📚 **Guide Series Navigation**
- **P3_TOOLS_GUIDE_CORE.md** - Core patterns and cross-references
- **P3_TOOLS_DATA_RETRIEVAL.md** - Data retrieval tools (p3-all-*, p3-get-*)
- **P3_TOOLS_COMPUTATIONAL.md** ← *You are here* - Computational services
- **P3_TOOLS_UTILITIES.md** - File management and processing tools
- **P3_TOOLS_SPECIALIZED.md** - Domain-specific analysis tools

## Table of Contents
- [Computational Services Overview](#computational-services-overview)
- [Core Service Infrastructure](#core-service-infrastructure)
- [Sequence Analysis Services (2 tools)](#sequence-analysis-services-2-tools)
- [Phylogenetic Analysis Services (2 tools)](#phylogenetic-analysis-services-2-tools)
- [Genome Analysis Services (2 tools)](#genome-analysis-services-2-tools)
- [Comparative Analysis Services (1 tool)](#comparative-analysis-services-1-tool)
- [Specialized Analysis Services (2 tools)](#specialized-analysis-services-2-tools)
- [Comprehensive Analysis Services (1 tool)](#comprehensive-analysis-services-1-tool)
- [Metagenomic Analysis Services (4 tools)](#metagenomic-analysis-services-4-tools)
- [SARS-CoV-2 Services (2 tools)](#sars-cov-2-services-2-tools)
- [Workspace Management](#workspace-management)
- [Best Practices](#best-practices)

## Computational Services Overview

### Service Characteristics
Computational services are asynchronous tools that submit jobs to the BV-BRC compute infrastructure. They follow a consistent pattern:

1. **Job Submission** → Submit analysis parameters and input data
2. **Job Monitoring** → Check status until completion 
3. **Result Retrieval** → Download and parse results

All computational services require:
- **Authentication**: Active p3-login session
- **Workspace Access**: Valid workspace path for outputs
- **Input Data**: Properly formatted input files or data

### Import Required Functions
All examples assume these core functions are available (from P3_TOOLS_GUIDE_CORE.md):

```python
import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Core execution functions
def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300) -> Dict[str, Any]:
    # Implementation from P3_TOOLS_GUIDE_CORE.md
    pass

def robust_p3_execution(command: List[str], input_data: str = None, 
                       timeout: int = 300, max_retries: int = 2) -> Dict[str, Any]:
    # Implementation from P3_TOOLS_GUIDE_CORE.md
    pass
```

## Core Service Infrastructure

### Job Management Utilities
```python
def create_temp_input_file(content: str, suffix: str = '.tmp') -> str:
    """Create temporary input file for job submission"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
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
        line = line.strip().lower()
        if 'job' in line and 'id' in line:
            job_info['job_id'] = line
        elif 'submitted' in line:
            job_info['status'] = 'submitted'
        elif 'queued' in line:
            job_info['status'] = 'queued'
    
    return job_info

def monitor_job_completion(job_info: Dict[str, Any], 
                         poll_interval: int = 30,
                         max_wait_time: int = 3600) -> Dict[str, Any]:
    """
    Monitor job until completion or timeout
    
    Args:
        job_info: Job information from submission
        poll_interval: Seconds between status checks
        max_wait_time: Maximum seconds to wait
    
    Returns:
        Job completion status
    """
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        # This would use actual job monitoring tools when available
        # For now, return a placeholder
        time.sleep(poll_interval)
        
        # Placeholder - real implementation would check job status
        return {
            'completed': True,
            'status': 'finished',
            'message': 'Job monitoring requires p3-job-status or similar tools'
        }
    
    return {
        'completed': False,
        'status': 'timeout',
        'message': f'Job monitoring timeout after {max_wait_time} seconds'
    }
```

## Sequence Analysis Services (2 tools)

### p3-submit-BLAST - BLAST Similarity Search
```python
def p3_submit_blast(query_sequences: Dict[str, str],
                   output_path: str,
                   output_name: str,
                   program: str = 'blastp',
                   database: str = 'nr', 
                   evalue: float = 0.001,
                   max_target_seqs: int = 100,
                   additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
        Job submission result
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
        
        result = run_p3_tool(command, timeout=120)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'BLAST',
                'output_path': output_path,
                'output_name': output_name
            })
            return job_info
        else:
            return {
                'submitted': False,
                'error': result['stderr'],
                'service': 'BLAST'
            }
    
    finally:
        cleanup_temp_file(query_file)

# Usage
blast_job = p3_submit_blast(
    {'seq1': 'MKAIFVLKFG...'},
    '/workspace/blast_results',
    'ecoli_blast',
    program='blastp',
    database='nr'
)
```

### p3-submit-MSA - Multiple Sequence Alignment
```python
def p3_submit_msa(sequences: Dict[str, str],
                 output_path: str,
                 output_name: str,
                 alignment_method: str = 'muscle',
                 additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit Multiple Sequence Alignment job
    
    Args:
        sequences: Dict of sequence_id -> sequence
        output_path: Workspace output directory  
        output_name: Job name/output prefix
        alignment_method: Alignment algorithm (kept for API compatibility, but P3-MSA uses default method)
        additional_params: Additional parameters
    
    Returns:
        Job submission result
    """
    
    fasta_content = create_fasta_input(sequences)
    sequence_file = create_temp_input_file(fasta_content, '.fasta')
    
    try:
        # Upload FASTA file to workspace first
        workspace_fasta_path = f"{output_path}/msa_input_{output_name}.fasta"
        upload_command = ['p3-cp', sequence_file, f'ws:{workspace_fasta_path}']
        upload_result = run_p3_tool(upload_command, timeout=60)
        
        if not upload_result['success']:
            return {
                'submitted': False,
                'error': f"Failed to upload FASTA file to workspace: {upload_result['stderr']}",
                'service': 'MSA'
            }
        
        # Submit MSA job using workspace file
        command = ['p3-submit-MSA', output_path, output_name]
        command.extend(['--fasta-file', f'ws:{workspace_fasta_path}'])
        
        if additional_params:
            for param, value in additional_params.items():
                command.extend([f'--{param}', str(value)])
        
        result = run_p3_tool(command, timeout=120)
        
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
            return job_info
        else:
            return {
                'submitted': False,
                'error': result['stderr'],
                'service': 'MSA'
            }
    
    finally:
        cleanup_temp_file(sequence_file)
```

## Phylogenetic Analysis Services (2 tools)

### p3-submit-gene-tree - Gene Phylogenetic Tree
```python
def p3_submit_gene_tree(gene_sequences: Dict[str, str],
                       output_path: str,
                       output_name: str,
                       tree_method: str = 'FastTree',
                       alignment_method: str = 'muscle',
                       additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
        Job submission result
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
        
        result = run_p3_tool(command, timeout=120)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'gene-tree',
                'tree_method': tree_method,
                'output_path': output_path,
                'output_name': output_name
            })
            return job_info
        else:
            return {
                'submitted': False,
                'error': result['stderr'],
                'service': 'gene-tree'
            }
    
    finally:
        cleanup_temp_file(gene_file)
```

### p3-submit-codon-tree - Codon-Based Phylogenetic Tree
```python
def p3_submit_codon_tree(gene_sequences: Dict[str, str],
                        output_path: str,
                        output_name: str,
                        codon_position: str = '123',
                        tree_method: str = 'FastTree',
                        additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit codon-based phylogenetic tree job
    
    Args:
        gene_sequences: Dict of sequence_id -> coding sequence
        output_path: Workspace output directory
        output_name: Job name/output prefix
        codon_position: Codon positions to use ('1', '2', '3', '12', '123')
        tree_method: Tree construction method
        additional_params: Additional parameters
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
        
        result = run_p3_tool(command, timeout=120)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'codon-tree',
                'codon_position': codon_position,
                'output_path': output_path,
                'output_name': output_name
            })
            return job_info
        else:
            return {
                'submitted': False,
                'error': result['stderr'],
                'service': 'codon-tree'
            }
    
    finally:
        cleanup_temp_file(sequence_file)
```

## Genome Analysis Services (2 tools)

### p3-submit-genome-annotation - Genome Annotation
```python
def p3_submit_genome_annotation(contigs_file: str,
                               output_path: str,
                               output_name: str,
                               domain: str = 'Bacteria',
                               taxonomy_id: int = None,
                               annotation_scheme: str = 'RAST',
                               additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
    
    result = run_p3_tool(command, timeout=300)  # Annotation jobs can be longer
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'genome-annotation',
            'domain': domain,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'genome-annotation'
        }
```

### p3-submit-genome-assembly - Genome Assembly
```python
def p3_submit_genome_assembly(reads_files: List[str],
                             output_path: str,
                             output_name: str,
                             assembly_strategy: str = 'auto',
                             recipe: str = 'unicycler',
                             additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit genome assembly job
    
    Args:
        reads_files: List of paths to reads files (FASTQ)
        output_path: Workspace output directory
        output_name: Job name/output prefix
        assembly_strategy: Assembly strategy (auto, single_cell, etc.)
        recipe: Assembly recipe/pipeline (unicycler, spades, canu)
        additional_params: Additional parameters
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
    
    result = run_p3_tool(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'genome-assembly',
            'assembly_strategy': assembly_strategy,
            'recipe': recipe,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'genome-assembly'
        }
```

## Comparative Analysis Services (1 tool)

### p3-submit-proteome-comparison - Proteome Comparison
```python
def p3_submit_proteome_comparison(genome_ids: List[str],
                                 output_path: str,
                                 output_name: str,
                                 comparison_type: str = 'bidirectional',
                                 evalue_cutoff: float = 1e-5,
                                 additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit proteome comparison job (bidirectional BLAST)
    
    Args:
        genome_ids: List of genome IDs to compare
        output_path: Workspace output directory
        output_name: Job name/output prefix
        comparison_type: Type of comparison (bidirectional, all-vs-all)
        evalue_cutoff: E-value threshold for matches
        additional_params: Additional parameters
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
        
        result = run_p3_tool(command, timeout=300)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'proteome-comparison',
                'comparison_type': comparison_type,
                'genome_count': len(genome_ids),
                'output_path': output_path,
                'output_name': output_name
            })
            return job_info
        else:
            return {
                'submitted': False,
                'error': result['stderr'],
                'service': 'proteome-comparison'
            }
    
    finally:
        cleanup_temp_file(genome_file)
```

## Specialized Analysis Services (2 tools)

### p3-submit-variation-analysis - Variation Analysis  
```python
def p3_submit_variation_analysis(reads_files: List[str],
                                reference_genome: str,
                                output_path: str,
                                output_name: str,
                                mapper: str = 'bwa',
                                caller: str = 'gatk',
                                additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
    
    result = run_p3_tool(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'variation-analysis',
            'mapper': mapper,
            'caller': caller,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'variation-analysis'
        }
```

### p3-submit-rnaseq - RNA-Seq Analysis
```python
def p3_submit_rnaseq(reads_files: List[str],
                    reference_genome: str,
                    output_path: str,
                    output_name: str,
                    analysis_type: str = 'differential',
                    mapper: str = 'hisat2',
                    additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
    
    result = run_p3_tool(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'rnaseq',
            'analysis_type': analysis_type,
            'mapper': mapper,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'rnaseq'
        }
```

## Comprehensive Analysis Services (1 tool)

### p3-submit-CGA - Comprehensive Genome Analysis
```python
def p3_submit_cga(reads_files: List[str],
                 output_path: str,
                 output_name: str,
                 recipe: str = 'comprehensive',
                 reference_genome: str = None,
                 additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
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
    
    result = run_p3_tool(command, timeout=600)  # CGA jobs can be very long
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'CGA',
            'recipe': recipe,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'CGA'
        }
```

## Metagenomic Analysis Services (4 tools)

### p3-submit-fastqutils - FASTQ Utility Operations
```python
def p3_submit_fastqutils(reads_files: List[str],
                        output_path: str,
                        output_name: str,
                        operation: str = 'trim',
                        quality_threshold: int = 20,
                        additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit FASTQ utility operations job (trimming, filtering, etc.)
    
    Args:
        reads_files: List of paths to FASTQ files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        operation: Operation type (trim, filter, stats, etc.)
        quality_threshold: Quality score threshold
        additional_params: Additional parameters
    """
    
    command = ['p3-submit-fastqutils', output_path, output_name]
    
    command.extend(['--operation', operation])
    command.extend(['--quality-threshold', str(quality_threshold)])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = run_p3_tool(command, timeout=120)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'fastqutils',
            'operation': operation,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'fastqutils'
        }
```

### p3-submit-taxonomic-classification - Taxonomic Classification
```python
def p3_submit_taxonomic_classification(reads_files: List[str],
                                     output_path: str,
                                     output_name: str,
                                     classifier: str = 'kraken2',
                                     database: str = 'standard',
                                     additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit taxonomic classification job
    
    Args:
        reads_files: List of paths to reads files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        classifier: Classification tool (kraken2, metaphlan, etc.)
        database: Reference database
        additional_params: Additional parameters
    """
    
    command = ['p3-submit-taxonomic-classification', output_path, output_name]
    
    command.extend(['--classifier', classifier])
    command.extend(['--database', database])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = run_p3_tool(command, timeout=180)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'taxonomic-classification',
            'classifier': classifier,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'taxonomic-classification'
        }
```

### p3-submit-metagenome-binning - Metagenome Binning
```python
def p3_submit_metagenome_binning(contigs_file: str,
                                reads_files: List[str],
                                output_path: str,
                                output_name: str,
                                binning_method: str = 'metabat2',
                                additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit metagenome binning job
    
    Args:
        contigs_file: Path to assembled contigs file
        reads_files: List of paths to reads files for coverage calculation
        output_path: Workspace output directory
        output_name: Job name/output prefix
        binning_method: Binning algorithm (metabat2, maxbin2, concoct)
        additional_params: Additional parameters
    """
    
    command = ['p3-submit-metagenome-binning', output_path, output_name]
    
    command.extend(['--contigs-file', contigs_file])
    command.extend(['--binning-method', binning_method])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = run_p3_tool(command, timeout=300)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'metagenome-binning',
            'binning_method': binning_method,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'metagenome-binning'
        }
```

### p3-submit-metagenomic-read-mapping - Metagenomic Read Mapping
```python
def p3_submit_metagenomic_read_mapping(reads_files: List[str],
                                      reference_genomes: List[str],
                                      output_path: str,
                                      output_name: str,
                                      mapper: str = 'bwa',
                                      additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit metagenomic read mapping job
    
    Args:
        reads_files: List of paths to reads files
        reference_genomes: List of reference genome IDs or paths
        output_path: Workspace output directory
        output_name: Job name/output prefix
        mapper: Read mapping tool (bwa, bowtie2, minimap2)
        additional_params: Additional parameters
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
        
        result = run_p3_tool(command, timeout=240)
        
        if result['success']:
            job_info = parse_job_output(result['stdout'])
            job_info.update({
                'service': 'metagenomic-read-mapping',
                'mapper': mapper,
                'reference_count': len(reference_genomes),
                'output_path': output_path,
                'output_name': output_name
            })
            return job_info
        else:
            return {
                'submitted': False,
                'error': result['stderr'],
                'service': 'metagenomic-read-mapping'
            }
    
    finally:
        cleanup_temp_file(ref_file)
```

## SARS-CoV-2 Services (2 tools)

### p3-submit-sars2-analysis - SARS-CoV-2 Analysis
```python
def p3_submit_sars2_analysis(reads_files: List[str],
                            output_path: str,
                            output_name: str,
                            analysis_type: str = 'comprehensive',
                            reference_genome: str = 'NC_045512.2',
                            additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit SARS-CoV-2 specialized analysis job
    
    Args:
        reads_files: List of paths to reads files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        analysis_type: Analysis type (comprehensive, variants, lineage)
        reference_genome: SARS-CoV-2 reference genome
        additional_params: Additional parameters
    """
    
    command = ['p3-submit-sars2-analysis', output_path, output_name]
    
    command.extend(['--analysis-type', analysis_type])
    command.extend(['--reference-genome', reference_genome])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = run_p3_tool(command, timeout=180)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'sars2-analysis',
            'analysis_type': analysis_type,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'sars2-analysis'
        }
```

### p3-submit-sars2-assembly - SARS-CoV-2 Assembly
```python
def p3_submit_sars2_assembly(reads_files: List[str],
                            output_path: str,
                            output_name: str,
                            assembly_strategy: str = 'consensus',
                            reference_genome: str = 'NC_045512.2',
                            additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit SARS-CoV-2 specialized assembly job
    
    Args:
        reads_files: List of paths to reads files
        output_path: Workspace output directory
        output_name: Job name/output prefix
        assembly_strategy: Assembly strategy (consensus, denovo)
        reference_genome: SARS-CoV-2 reference genome
        additional_params: Additional parameters
    """
    
    command = ['p3-submit-sars2-assembly', output_path, output_name]
    
    command.extend(['--assembly-strategy', assembly_strategy])
    command.extend(['--reference-genome', reference_genome])
    
    for reads_file in reads_files:
        command.extend(['--reads-file', reads_file])
    
    if additional_params:
        for param, value in additional_params.items():
            command.extend([f'--{param}', str(value)])
    
    result = run_p3_tool(command, timeout=120)
    
    if result['success']:
        job_info = parse_job_output(result['stdout'])
        job_info.update({
            'service': 'sars2-assembly',
            'assembly_strategy': assembly_strategy,
            'output_path': output_path,
            'output_name': output_name
        })
        return job_info
    else:
        return {
            'submitted': False,
            'error': result['stderr'],
            'service': 'sars2-assembly'
        }
```

## Workspace Management

### Basic Workspace Operations
```python
def list_workspace_files(path: str = "/workspace") -> List[str]:
    """List files in workspace using p3-ls"""
    command = ['p3-ls', path]
    result = run_p3_tool(command)
    
    if result['success']:
        return result['stdout'].split('\n')
    else:
        return []

def download_workspace_file(workspace_path: str, local_path: str) -> bool:
    """Download file from workspace using p3-cp"""
    command = ['p3-cp', workspace_path, local_path]
    result = run_p3_tool(command)
    return result['success']

def upload_to_workspace(local_path: str, workspace_path: str) -> bool:
    """Upload file to workspace using p3-cp"""  
    command = ['p3-cp', local_path, workspace_path]
    result = run_p3_tool(command)
    return result['success']
```

### Job Result Management
```python
def download_job_results(output_path: str, output_name: str, local_dir: str) -> bool:
    """
    Download all results from a computational job
    
    Args:
        output_path: Workspace output directory from job submission
        output_name: Job output name from job submission
        local_dir: Local directory to store results
    
    Returns:
        Success status
    """
    
    # List all files in job output directory
    job_dir = f"{output_path}/{output_name}"
    files = list_workspace_files(job_dir)
    
    success = True
    for file in files:
        if file.strip():  # Skip empty lines
            workspace_file = f"{job_dir}/{file}"
            local_file = f"{local_dir}/{file}"
            if not download_workspace_file(workspace_file, local_file):
                print(f"Failed to download {workspace_file}")
                success = False
    
    return success
```

## Best Practices

### Job Submission Patterns
```python
def submit_job_with_monitoring(submit_func, *args, monitor: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Submit job and optionally monitor completion
    
    Args:
        submit_func: Job submission function
        *args: Arguments for submission function
        monitor: Whether to monitor job completion
        **kwargs: Keyword arguments for submission function
    
    Returns:
        Job result with completion status
    """
    
    # Submit job
    job_result = submit_func(*args, **kwargs)
    
    if not job_result.get('submitted', False):
        return job_result
    
    # Monitor if requested
    if monitor:
        completion_status = monitor_job_completion(job_result)
        job_result['completion'] = completion_status
    
    return job_result

# Usage
blast_result = submit_job_with_monitoring(
    p3_submit_blast,
    {'seq1': 'MKLAVF...'},
    '/workspace/analysis',
    'blast_job1',
    monitor=True
)
```

### Batch Job Submission
```python
def submit_multiple_jobs(job_configs: List[Dict[str, Any]], 
                        max_concurrent: int = 5) -> List[Dict[str, Any]]:
    """
    Submit multiple jobs with concurrency control
    
    Args:
        job_configs: List of job configuration dictionaries
        max_concurrent: Maximum concurrent jobs
    
    Returns:
        List of job submission results
    """
    
    results = []
    submitted_count = 0
    
    for config in job_configs:
        if submitted_count >= max_concurrent:
            # Wait before submitting more
            time.sleep(30)
            submitted_count = 0
        
        submit_func = config.pop('submit_function')
        result = submit_func(**config)
        results.append(result)
        
        if result.get('submitted', False):
            submitted_count += 1
    
    return results
```

### Error Handling for Computational Services
```python
def robust_service_submission(submit_func, *args, max_retries: int = 2, **kwargs) -> Dict[str, Any]:
    """
    Submit computational service with retry logic
    
    Args:
        submit_func: Service submission function
        *args: Positional arguments
        max_retries: Maximum retry attempts
        **kwargs: Keyword arguments
    
    Returns:
        Job submission result
    """
    
    for attempt in range(max_retries + 1):
        try:
            result = submit_func(*args, **kwargs)
            
            if result.get('submitted', False):
                return result
            elif attempt < max_retries:
                # Wait before retry
                time.sleep(5 * (2 ** attempt))  # Exponential backoff
                continue
            else:
                return result
        
        except Exception as e:
            if attempt == max_retries:
                return {
                    'submitted': False,
                    'error': f"Exception after {max_retries + 1} attempts: {str(e)}",
                    'service': getattr(submit_func, '__name__', 'unknown')
                }
            time.sleep(5 * (2 ** attempt))
    
    return {'submitted': False, 'error': 'Max retries exceeded'}
```

### Resource Management
```python
def cleanup_job_files(temp_files: List[str], job_result: Dict[str, Any]) -> None:
    """Clean up temporary files after job submission"""
    
    for temp_file in temp_files:
        cleanup_temp_file(temp_file)
    
    # Log job submission
    if job_result.get('submitted', False):
        print(f"Job submitted: {job_result.get('service', 'unknown')} -> {job_result.get('output_path', 'unknown')}")
    else:
        print(f"Job failed: {job_result.get('error', 'unknown error')}")
```

## Usage Examples Summary

```python
# Usage examples for computational services
blast_results = p3_submit_blast({'query1': 'MKLAVF...'}, '/workspace/blast', 'my_blast')
tree_job = p3_submit_gene_tree({'gene1': 'ATGAAA...', 'gene2': 'ATGCCC...'}, '/workspace/trees', 'phylo_analysis')
assembly_job = p3_submit_genome_assembly(['/data/reads_1.fastq', '/data/reads_2.fastq'], '/workspace/assembly', 'genome_v1')
taxonomy_job = p3_submit_taxonomic_classification(['/data/metagenome.fastq'], '/workspace/taxonomy', 'sample1_classification')
sars2_job = p3_submit_sars2_analysis(['/data/covid_reads.fastq'], '/workspace/sars2', 'covid_sample1')
```

---

**Next Steps**: For utility tools and file management, see **P3_TOOLS_UTILITIES.md**. For specialized analysis tools, see **P3_TOOLS_SPECIALIZED.md**.
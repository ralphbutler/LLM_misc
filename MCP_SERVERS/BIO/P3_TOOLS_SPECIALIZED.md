# P3-Tools Specialized Analysis Guide

**Part of the P3-Tools Programming Guide Series**

This guide focuses on specialized analysis tools for comparative genomics, k-mer analysis, feature analysis, and domain-specific computational tasks. These tools complement the core data retrieval and computational services.

## 📚 **Guide Series Navigation**
- **P3_TOOLS_GUIDE_CORE.md** - Core patterns and cross-references
- **P3_TOOLS_DATA_RETRIEVAL.md** - Data retrieval tools (p3-all-*, p3-get-*)
- **P3_TOOLS_COMPUTATIONAL.md** - Computational services (p3-submit-*)
- **P3_TOOLS_UTILITIES.md** - File management and processing tools
- **P3_TOOLS_SPECIALIZED.md** ← *You are here* - Specialized analysis

## Table of Contents
- [Specialized Analysis Overview](#specialized-analysis-overview)
- [Comparative Genomics Tools](#comparative-genomics-tools)
- [K-mer Analysis Tools](#k-mer-analysis-tools)
- [Feature Analysis Tools](#feature-analysis-tools)
- [Direct Analysis Tools](#direct-analysis-tools)
- [Phylogenetic and Tree Tools](#phylogenetic-and-tree-tools)
- [Workflow and Pipeline Tools](#workflow-and-pipeline-tools)
- [Future Tools (61 remaining)](#future-tools-61-remaining)
- [Best Practices](#best-practices)

## Specialized Analysis Overview

### Current Status
**Documented in this guide**: 12 specialized analysis tools  
**Remaining to document**: 61 tools across various categories  
**Total P3-Tools coverage**: 73 of 134 tools (54% complete)

### Tool Categories

#### **Currently Documented**
- **Comparative Genomics**: Direct BLAST, discriminating k-mers, co-occurrence analysis
- **K-mer Analysis**: K-mer database construction and analysis
- **Feature Analysis**: Gap analysis, upstream regions, genome dumping
- **Direct Analysis**: Command-line BLAST operations

#### **Remaining Tools** (61 tools)
- Additional comparative genomics tools
- Advanced phylogenetic analysis
- Metabolic pathway analysis
- Protein structure analysis
- Workflow orchestration tools

### Import Required Functions
```python
import subprocess
import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Core execution functions (from P3_TOOLS_GUIDE_CORE.md)
def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300) -> Dict[str, Any]:
    pass

def robust_p3_execution(command: List[str], input_data: str = None, 
                       timeout: int = 300, max_retries: int = 2) -> Dict[str, Any]:
    pass

def parse_p3_tabular_output(output: str) -> List[Dict[str, str]]:
    pass

def create_temp_input_file(content: str, suffix: str = '.tmp') -> str:
    pass

def cleanup_temp_file(filepath: str) -> None:
    pass
```

## Comparative Genomics Tools

### p3-blast - Direct BLAST Operations
```python
def p3_blast(query_file: str,
             database: str = 'nr',
             program: str = 'blastp',
             evalue: float = 0.001,
             max_target_seqs: int = 100,
             output_format: int = 6,
             additional_params: Dict[str, Any] = None) -> str:
    """
    Perform direct BLAST search using p3-blast
    
    Args:
        query_file: Path to query sequences file
        database: BLAST database (nr, nt, refseq_protein, etc.)
        program: BLAST program (blastp, blastn, blastx, etc.)
        evalue: E-value threshold
        max_target_seqs: Maximum number of target sequences
        output_format: BLAST output format (6 for tabular)
        additional_params: Additional BLAST parameters
    
    Returns:
        BLAST results as string
    """
    
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
    
    result = run_p3_tool(command, timeout=600)  # BLAST can take time
    
    if result['success']:
        return result['stdout']
    else:
        return ""

def p3_blast_sequences(sequences: Dict[str, str],
                      database: str = 'nr',
                      program: str = 'blastp') -> str:
    """
    BLAST sequences directly without file I/O
    
    Args:
        sequences: Dict of sequence_id -> sequence
        database: BLAST database
        program: BLAST program
    
    Returns:
        BLAST results
    """
    
    # Create temporary FASTA file
    fasta_content = []
    for seq_id, sequence in sequences.items():
        fasta_content.append(f">{seq_id}")
        fasta_content.append(sequence)
    
    query_file = create_temp_input_file('\n'.join(fasta_content), '.fasta')
    
    try:
        return p3_blast(query_file, database, program)
    finally:
        cleanup_temp_file(query_file)
```

### p3-discriminating-kmers - K-mer Analysis
```python
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
        Discriminating k-mers results
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
        
        result = run_p3_tool(command, timeout=300)
        
        if result['success']:
            return result['stdout']
        else:
            return ""
    
    finally:
        for group_file in group_files:
            cleanup_temp_file(group_file)
```

### p3-co-occur - Co-occurrence Analysis
```python
def p3_co_occur(genome_ids: List[str],
               feature_type: str = 'CDS',
               distance_threshold: int = 5000,
               min_genomes: int = 2) -> List[Dict[str, str]]:
    """
    Analyze feature co-occurrence patterns across genomes
    
    Args:
        genome_ids: List of genome IDs to analyze
        feature_type: Type of features to analyze
        distance_threshold: Maximum distance for co-occurrence
        min_genomes: Minimum genomes for pattern to be reported
    
    Returns:
        Co-occurrence patterns
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
        
        result = run_p3_tool(command, timeout=300)
        
        if result['success']:
            return parse_p3_tabular_output(result['stdout'])
        else:
            return []
    
    finally:
        cleanup_temp_file(genome_file)
```

### p3-genome-distance - Genome Distance Calculation
```python
def p3_genome_distance(base_genome_id: str,
                      comparison_genome_ids: List[str],
                      kmer_size: int = 8,
                      dna_mode: bool = False) -> List[Dict[str, str]]:
    """
    Calculate k-mer based distances between genomes
    
    Args:
        base_genome_id: Reference genome ID
        comparison_genome_ids: List of genomes to compare against base
        kmer_size: Size of k-mers for comparison (default 8)
        dna_mode: Use DNA k-mers instead of protein k-mers
    
    Returns:
        List of genome distance measurements
    """
    
    # Prepare input: comparison genomes via stdin
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in comparison_genome_ids])
    
    command = ['p3-genome-distance']
    
    # Add base genome as positional argument
    command.append(base_genome_id)
    
    # Add k-mer size parameter
    command.extend(['--kmer', str(kmer_size)])
    
    # Add DNA mode if specified
    if dna_mode:
        command.append('--dna')
    
    result = run_p3_tool(command, input_data=stdin_data, timeout=300)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []


def p3_genome_distance_batch(genome_pairs: List[tuple], 
                           kmer_size: int = 8) -> List[Dict[str, str]]:
    """
    Calculate distances for multiple genome pairs
    
    Args:
        genome_pairs: List of (base_genome, comparison_genome) tuples
        kmer_size: K-mer size for comparison
    
    Returns:
        Combined distance results
    """
    
    all_results = []
    for base_genome, comparison_genome in genome_pairs:
        distances = p3_genome_distance(base_genome, [comparison_genome], kmer_size)
        all_results.extend(distances)
    
    return all_results
```

### p3-signature-families - Signature Family Analysis  
```python
def p3_signature_families(genomes_with_property: List[str],
                         genomes_without_property: List[str],
                         min_fraction_with: float = 0.8,
                         max_fraction_without: float = 0.2) -> List[Dict[str, str]]:
    """
    Find protein families that serve as signatures for a genome property
    
    Args:
        genomes_with_property: Genome IDs that have the target property
        genomes_without_property: Genome IDs that lack the target property  
        min_fraction_with: Minimum fraction of "with" genomes that must have family
        max_fraction_without: Maximum fraction of "without" genomes that can have family
    
    Returns:
        List of signature protein families with statistics
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
        
        result = run_p3_tool(command, timeout=600)
        
        if result['success']:
            return parse_p3_tabular_output(result['stdout'])
        else:
            return []
    
    finally:
        cleanup_temp_file([with_file, without_file])  # Fix function name


def p3_signature_families_from_stdin(genome_property_data: List[Tuple[str, bool]],
                                   min_fraction_with: float = 0.8,
                                   max_fraction_without: float = 0.2) -> List[Dict[str, str]]:
    """
    Find signature families using genome property data from stdin format
    
    Args:
        genome_property_data: List of (genome_id, has_property) tuples
        min_fraction_with: Minimum fraction threshold for genomes with property
        max_fraction_without: Maximum fraction threshold for genomes without property
    
    Returns:
        Signature family results
    """
    
    # Separate genomes by property
    with_property = [gid for gid, has_prop in genome_property_data if has_prop]
    without_property = [gid for gid, has_prop in genome_property_data if not has_prop]
    
    return p3_signature_families(with_property, without_property, 
                               min_fraction_with, max_fraction_without)
```

### p3-signature-clusters - Signature Cluster Analysis
```python
def p3_signature_clusters(genome_ids: List[str],
                         cluster_method: str = 'hierarchical',
                         distance_metric: str = 'jaccard',
                         min_cluster_size: int = 3) -> List[Dict[str, str]]:
    """
    Find signature clusters in genome sets
    
    Args:
        genome_ids: List of genome IDs
        cluster_method: Clustering method
        distance_metric: Distance metric for clustering
        min_cluster_size: Minimum cluster size
    
    Returns:
        Signature cluster results
    """
    
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-signature-clusters']
        
        command.extend(['--genome-file', genome_file])
        command.extend(['--method', cluster_method])
        command.extend(['--distance', distance_metric])
        command.extend(['--min-size', str(min_cluster_size)])
        
        result = run_p3_tool(command, timeout=300)
        
        if result['success']:
            return parse_p3_tabular_output(result['stdout'])
        else:
            return []
    
    finally:
        cleanup_temp_file(genome_file)
```

## K-mer Analysis Tools

### p3-build-kmer-db - K-mer Database Construction
```python
def p3_build_kmer_db(genome_ids: List[str],
                    kmer_length: int = 8,
                    database_name: str = 'kmer_db',
                    output_path: str = '/workspace/kmer_dbs') -> Dict[str, Any]:
    """
    Build k-mer database from genomes
    
    Args:
        genome_ids: List of genome IDs
        kmer_length: K-mer length
        database_name: Name for the database
        output_path: Output directory path
    
    Returns:
        Database construction result
    """
    
    genome_content = '\n'.join(genome_ids)
    genome_file = create_temp_input_file(genome_content, '.txt')
    
    try:
        command = ['p3-build-kmer-db']
        
        command.extend(['--genome-file', genome_file])
        command.extend(['--kmer-length', str(kmer_length)])
        command.extend(['--database-name', database_name])
        command.extend(['--output-path', output_path])
        
        result = run_p3_tool(command, timeout=600)
        
        if result['success']:
            return {
                'success': True,
                'database_path': f"{output_path}/{database_name}",
                'kmer_length': kmer_length,
                'genome_count': len(genome_ids),
                'output': result['stdout']
            }
        else:
            return {
                'success': False,
                'error': result['stderr']
            }
    
    finally:
        cleanup_temp_file(genome_file)
```

## Feature Analysis Tools

### p3-find-features - Global Feature Search
```python
def p3_find_features(search_criteria: Dict[str, Any] = None,
                    keyword: str = None,
                    attributes: List[str] = None,
                    count_only: bool = False) -> List[Dict[str, str]]:
    """
    Search for features across the entire BV-BRC feature database
    
    Args:
        search_criteria: Dictionary of field filters (e.g., {"feature_type": "CDS", "organism_name": "Escherichia"})
        keyword: Keyword to search for in any field
        attributes: Specific attributes to return
        count_only: Return count only instead of records
    
    Returns:
        List of matching features or count
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
    
    result = run_p3_tool(command, timeout=300)
    
    if result['success']:
        if count_only:
            # Parse count result
            try:
                return [{'count': int(result['stdout'].strip())}]
            except ValueError:
                return [{'count': 0}]
        else:
            return parse_p3_tabular_output(result['stdout'])
    else:
        return []


def p3_find_features_by_product(product_keyword: str, 
                               organism: str = None,
                               feature_type: str = "CDS") -> List[Dict[str, str]]:
    """
    Find features by product description (common use case)
    
    Args:
        product_keyword: Keyword to search in product descriptions
        organism: Optional organism name filter
        feature_type: Feature type filter (default CDS)
    
    Returns:
        Matching features with product containing keyword
    """
    
    search_criteria = {'feature_type': feature_type}
    if organism:
        search_criteria['organism_name'] = organism
    
    return p3_find_features(search_criteria=search_criteria, 
                          keyword=product_keyword)


def p3_find_features_in_region(genome_id: str,
                              contig_id: str,
                              start: int,
                              end: int,
                              feature_type: str = None) -> List[Dict[str, str]]:
    """
    Find features in a genomic region using global search
    
    Args:
        genome_id: Genome ID
        contig_id: Contig ID
        start: Start position
        end: End position  
        feature_type: Optional feature type filter
    
    Returns:
        Features in the specified region
    """
    
    search_criteria = {
        'genome_id': genome_id,
        'sequence_id': contig_id
    }
    
    if feature_type:
        search_criteria['feature_type'] = feature_type
    
    # Note: Position filtering would need additional --ge and --le parameters
    command = ['p3-find-features']
    
    for field, value in search_criteria.items():
        command.extend(['--eq', f'{field},{value}'])
    
    # Add position constraints
    command.extend(['--ge', f'start,{start}'])
    command.extend(['--le', f'end,{end}'])
    
    # Default attributes
    default_attrs = ['patric_id', 'feature_type', 'product', 'start', 'end', 'strand']
    for attr in default_attrs:
        command.extend(['--attr', attr])
    
    command.append('patric_id')
    
    result = run_p3_tool(command, timeout=300)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

### p3-find-couples - Find Coupled Features
```python
def p3_find_couples(input_data: str,
                   category_column: str,
                   min_count: int = 2,
                   max_gap: int = 1000,
                   location_column: str = None,
                   sequence_column: str = None) -> List[Dict[str, str]]:
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
        List of coupled feature pairs with statistics
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
    
    result = run_p3_tool(command, input_data=input_data, timeout=300)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []


def p3_find_gene_couples_in_genomes(genome_ids: List[str],
                                   max_gap: int = 500) -> List[Dict[str, str]]:
    """
    Find coupled genes (potential operons) in specified genomes
    
    Args:
        genome_ids: List of genome IDs to analyze
        max_gap: Maximum gap between genes to consider coupling
    
    Returns:
        Coupled gene pairs across all genomes
    """
    
    # First get features for genomes
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
    
    # Get CDS features with location info
    get_features_cmd = ['p3-get-genome-features', 
                       '--eq', 'feature_type,CDS',
                       '--attr', 'patric_id', 
                       '--attr', 'product',
                       '--attr', 'start',
                       '--attr', 'end', 
                       '--attr', 'sequence_id',
                       '--attr', 'genome_id']
    
    features_result = run_p3_tool(get_features_cmd, input_data=stdin_data)
    
    if not features_result['success']:
        return []
    
    # Now find couples in the feature data
    couples = p3_find_couples(features_result['stdout'],
                             category_column='product',
                             max_gap=max_gap,
                             location_column='start',
                             sequence_column='sequence_id')
    
    return couples
```

### p3-feature-gap - Feature Gap Analysis
```python
def p3_feature_gap(genome_ids: List[str],
                  reference_genome_id: str,
                  gap_threshold: int = 1000,
                  feature_types: List[str] = None) -> List[Dict[str, str]]:
    """
    Analyze gaps in feature coverage between genomes
    
    Args:
        genome_ids: List of genome IDs to analyze
        reference_genome_id: Reference genome for comparison
        gap_threshold: Minimum gap size to report
        feature_types: Types of features to analyze
    
    Returns:
        Gap analysis results
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
        
        result = run_p3_tool(command, timeout=300)
        
        if result['success']:
            return parse_p3_tabular_output(result['stdout'])
        else:
            return []
    
    finally:
        cleanup_temp_file(genome_file)
```

### p3-feature-upstream - Upstream Region Analysis
```python
def p3_feature_upstream(feature_ids: List[str],
                       upstream_length: int = 500,
                       include_sequence: bool = False) -> List[Dict[str, str]]:
    """
    Analyze upstream regions of features
    
    Args:
        feature_ids: List of feature IDs
        upstream_length: Length of upstream region to analyze
        include_sequence: Whether to include sequence data
    
    Returns:
        Upstream region analysis results
    """
    
    feature_content = '\n'.join([f"genome_feature.patric_id\t{fid}" for fid in feature_ids])
    
    command = ['p3-feature-upstream']
    
    command.extend(['--upstream-length', str(upstream_length)])
    
    if include_sequence:
        command.append('--include-sequence')
    
    result = run_p3_tool(command, input_data=feature_content, timeout=180)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

### p3-dump-genomes - Genome Data Export
```python
def p3_dump_genomes(genome_ids: List[str],
                   output_format: str = 'genbank',
                   include_features: bool = True,
                   output_dir: str = '/workspace/genome_dumps') -> Dict[str, Any]:
    """
    Dump genome data in various formats
    
    Args:
        genome_ids: List of genome IDs
        output_format: Output format (genbank, gff, fasta)
        include_features: Whether to include feature annotations
        output_dir: Output directory
    
    Returns:
        Dump operation result
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
        
        result = run_p3_tool(command, timeout=600)
        
        if result['success']:
            return {
                'success': True,
                'output_dir': output_dir,
                'format': output_format,
                'genome_count': len(genome_ids),
                'files_created': result['stdout'].split('\n')
            }
        else:
            return {
                'success': False,
                'error': result['stderr']
            }
    
    finally:
        cleanup_temp_file(genome_file)
```

### p3-role-features - Find Features by Functional Role
```python
def p3_role_features(role_names: List[str],
                    search_criteria: Dict[str, Any] = None,
                    attributes: List[str] = None,
                    count_only: bool = False) -> List[Dict[str, str]]:
    """
    Find features based on functional roles
    
    Args:
        role_names: List of functional role names to search for
        search_criteria: Additional search criteria (organism, feature_type, etc.)
        attributes: Specific attributes to return
        count_only: Return count only instead of records
    
    Returns:
        List of features with specified roles
    """
    
    # Prepare stdin data with role names
    stdin_data = '\n'.join([f"role\t{role}" for role in role_names])
    
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
    
    result = run_p3_tool(command, input_data=stdin_data, timeout=300)
    
    if result['success']:
        if count_only:
            try:
                return [{'count': int(result['stdout'].strip())}]
            except ValueError:
                return [{'count': 0}]
        else:
            return parse_p3_tabular_output(result['stdout'])
    else:
        return []


def p3_role_features_by_category(role_category: str,
                                organism: str = None,
                                feature_type: str = "CDS") -> List[Dict[str, str]]:
    """
    Find features by role category (common use case)
    
    Args:
        role_category: Role category to search for (e.g., "kinase", "transporter")
        organism: Optional organism filter
        feature_type: Feature type filter (default CDS)
    
    Returns:
        Features with roles containing the category
    """
    
    search_criteria = {'feature_type': feature_type}
    if organism:
        search_criteria['organism_name'] = organism
    
    command = ['p3-role-features']
    
    # Add search criteria
    for field, value in search_criteria.items():
        command.extend(['--eq', f'{field},{value}'])
    
    # Use keyword search for role category
    command.extend(['--keyword', role_category])
    
    # Default attributes
    default_attrs = ['patric_id', 'feature_type', 'product', 'genome_name', 'role']
    for attr in default_attrs:
        command.extend(['--attr', attr])
    
    result = run_p3_tool(command, timeout=300)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []


def p3_role_features_in_genomes(role_names: List[str],
                               genome_ids: List[str],
                               feature_type: str = "CDS") -> List[Dict[str, str]]:
    """
    Find features with specific roles in specific genomes
    
    Args:
        role_names: List of functional role names
        genome_ids: List of genome IDs to search within
        feature_type: Feature type filter
    
    Returns:
        Features with specified roles in specified genomes
    """
    
    # Prepare stdin data with role names
    stdin_data = '\n'.join([f"role\t{role}" for role in role_names])
    
    command = ['p3-role-features']
    
    # Add genome filter
    genome_list = ','.join(genome_ids)
    command.extend(['--in', f'genome_id,{genome_list}'])
    
    # Add feature type filter
    command.extend(['--eq', f'feature_type,{feature_type}'])
    
    # Default attributes with genome info
    default_attrs = ['patric_id', 'feature_type', 'product', 'genome_id', 'genome_name', 'role']
    for attr in default_attrs:
        command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data, timeout=300)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

## Direct Analysis Tools

### Additional Direct Analysis Tools

#### p3-find-couples - Paired Data Finding
# Duplicate p3_find_couples definition removed - already defined above

## Phylogenetic and Tree Tools

### Advanced Tree Construction (Future Implementation)
```python
def p3_advanced_tree_analysis(sequences: Dict[str, str],
                             tree_method: str = 'maximum_likelihood',
                             bootstrap_reps: int = 100) -> str:
    """
    Placeholder for advanced phylogenetic analysis tools
    
    Args:
        sequences: Sequence data
        tree_method: Tree construction method
        bootstrap_reps: Bootstrap replicates
    
    Returns:
        Tree analysis results
    
    Note:
        This is a template for future phylogenetic tools
    """
    
    # Template for future tools like:
    # p3-maximum-likelihood-tree
    # p3-bootstrap-analysis
    # p3-tree-comparison
    
    return "Advanced tree tools not yet implemented"
```

## Workflow and Pipeline Tools

### Configuration and Workflow Management
```python
def p3_workflow_config(workflow_name: str,
                      config_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configure workflow parameters
    
    Args:
        workflow_name: Name of workflow
        config_params: Configuration parameters
    
    Returns:
        Configuration result
    """
    
    # Template for workflow configuration tools
    return {
        'workflow': workflow_name,
        'configured': True,
        'params': config_params
    }

def p3_pipeline_orchestrator(pipeline_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Orchestrate multi-step analysis pipelines
    
    Args:
        pipeline_steps: List of pipeline step configurations
    
    Returns:
        Pipeline execution result
    """
    
    # Template for pipeline orchestration tools
    results = []
    
    for step in pipeline_steps:
        step_result = {
            'step': step.get('name', 'unnamed'),
            'tool': step.get('tool', 'unknown'),
            'status': 'placeholder'
        }
        results.append(step_result)
    
    return {
        'pipeline_completed': False,
        'steps': results,
        'message': 'Pipeline orchestration tools not yet implemented'
    }
```

## Future Tools (61 remaining)

### Tool Categories Requiring Documentation

The following categories contain tools that need to be documented in future sessions:

#### **Comparative Genomics** (~15 tools)
- Advanced genome comparison tools
- Synteny analysis tools  
- Ortholog/paralog analysis tools
- Genome alignment tools

#### **Metabolic Analysis** (~10 tools)
- Pathway reconstruction tools
- Metabolic network analysis
- Enzyme classification tools
- Biochemical reaction analysis

#### **Protein Analysis** (~12 tools)
- Protein structure prediction
- Domain architecture analysis
- Protein-protein interaction tools
- Function prediction tools

#### **Advanced Phylogenetics** (~8 tools)
- Species tree construction
- Gene tree/species tree reconciliation
- Horizontal gene transfer detection
- Phylogenetic profiling

#### **Specialized Genomics** (~10 tools)
- Mobile genetic element analysis
- CRISPR system analysis
- Prophage detection
- Antibiotic resistance prediction

#### **Workflow and Integration** (~6 tools)
- Pipeline orchestration
- Data integration tools
- Visualization tools
- Report generation

### Implementation Strategy for Remaining Tools

```python
def implement_remaining_tools():
    """
    Strategy for documenting remaining 61 tools
    
    Priority order:
    1. High-impact comparative genomics tools
    2. Essential metabolic analysis tools
    3. Core protein analysis capabilities
    4. Advanced phylogenetic tools
    5. Specialized analysis tools
    6. Workflow integration tools
    """
    
    remaining_categories = {
        'comparative_genomics': 15,
        'metabolic_analysis': 10, 
        'protein_analysis': 12,
        'phylogenetics': 8,
        'specialized_genomics': 10,
        'workflow_integration': 6
    }
    
    return {
        'total_remaining': sum(remaining_categories.values()),
        'categories': remaining_categories,
        'priority': 'comparative_genomics'
    }
```

## Best Practices

### Specialized Analysis Workflows
```python
def create_comparative_analysis_workflow(genome_groups: List[List[str]]) -> Dict[str, Any]:
    """
    Create comprehensive comparative analysis workflow
    
    Args:
        genome_groups: Groups of genomes for comparison
    
    Returns:
        Workflow execution results
    """
    
    results = {}
    
    # Step 1: K-mer analysis
    discriminating_kmers = p3_discriminating_kmers(genome_groups)
    results['kmers'] = discriminating_kmers
    
    # Step 2: Co-occurrence analysis (flatten groups)
    all_genomes = [genome for group in genome_groups for genome in group]
    co_occurrence = p3_co_occur(all_genomes)
    results['co_occurrence'] = co_occurrence
    
    # Step 3: Signature clusters
    signature_clusters = p3_signature_clusters(all_genomes)
    results['clusters'] = signature_clusters
    
    return results

def optimize_analysis_parameters(genome_count: int, analysis_type: str) -> Dict[str, Any]:
    """
    Optimize analysis parameters based on dataset size
    
    Args:
        genome_count: Number of genomes
        analysis_type: Type of analysis
    
    Returns:
        Optimized parameters
    """
    
    params = {}
    
    if analysis_type == 'kmer':
        if genome_count < 10:
            params['kmer_length'] = 8
            params['timeout'] = 300
        elif genome_count < 100:
            params['kmer_length'] = 10
            params['timeout'] = 600
        else:
            params['kmer_length'] = 12
            params['timeout'] = 1200
    
    elif analysis_type == 'comparative':
        params['batch_size'] = min(20, genome_count // 2)
        params['timeout'] = 60 * genome_count  # Scale with genome count
    
    return params
```

### Error Handling for Specialized Tools
```python
def robust_specialized_analysis(analysis_func, *args, **kwargs) -> Dict[str, Any]:
    """
    Robust wrapper for specialized analysis tools
    
    Args:
        analysis_func: Specialized analysis function
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Analysis result with error handling
    """
    
    try:
        result = analysis_func(*args, **kwargs)
        
        if result:
            return {
                'success': True,
                'result': result,
                'tool': analysis_func.__name__
            }
        else:
            return {
                'success': False,
                'error': 'No results returned',
                'tool': analysis_func.__name__
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'tool': analysis_func.__name__
        }
```

---

**Status**: 12 specialized tools documented. 61 tools remaining for complete P3-Tools coverage.

**Next Priority**: Comparative genomics tools, metabolic analysis tools, and protein analysis capabilities.

**Complete P3-Tools Guide Series**: All 5 modular guides now available for comprehensive P3-Tools subprocess programming.
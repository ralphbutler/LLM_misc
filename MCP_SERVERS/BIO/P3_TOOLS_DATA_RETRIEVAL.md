# P3-Tools Data Retrieval Guide

**Part of the P3-Tools Programming Guide Series**

This guide focuses on data retrieval tools (p3-all-* and p3-get-*) that provide direct access to BV-BRC data collections via subprocess calls. These tools wrap the BV-BRC REST API for fast, real-time data access.

## 📚 **Guide Series Navigation**
- **P3_TOOLS_GUIDE_CORE.md** - Core patterns and cross-references
- **P3_TOOLS_DATA_RETRIEVAL.md** ← *You are here* - Data retrieval tools
- **P3_TOOLS_COMPUTATIONAL.md** - Computational services (p3-submit-*)
- **P3_TOOLS_UTILITIES.md** - File management and processing tools
- **P3_TOOLS_SPECIALIZED.md** - Domain-specific analysis tools

## Table of Contents
- [Data Retrieval Overview](#data-retrieval-overview)
- [Common Usage Patterns](#common-usage-patterns)
- [p3-all-* Tools (6 tools)](#p3-all--tools-6-tools)
- [p3-get-* Tools (26 tools)](#p3-get--tools-26-tools)
- [Pipeline Operations](#pipeline-operations)
- [Best Practices](#best-practices)

## Data Retrieval Overview

### Tool Categories

#### **p3-all-* Tools** (Direct Collection Access)
These tools query entire data collections with filtering capabilities:
- `p3-all-contigs` - Contig/sequence data
- `p3-all-drugs` - Drug/antimicrobial data  
- `p3-all-genomes` - Complete genome records
- `p3-all-subsystem-roles` - Functional role data
- `p3-all-subsystems` - Subsystem pathway data
- `p3-all-taxonomies` - Taxonomic classification data

#### **p3-get-* Tools** (ID-Based Retrieval)
These tools retrieve detailed data based on input IDs via stdin:
- Core genome/feature tools (8 tools)
- Protein and sequence tools (5 tools)
- Functional analysis tools (4 tools)
- Specialized data tools (5 tools)
- Regional/group analysis tools (4 tools)

### Import Required Functions
All examples assume these core functions are available (from P3_TOOLS_GUIDE_CORE.md):

```python
import subprocess
import json
from typing import List, Dict, Any, Optional

# Core execution functions
def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300,
                limit_results: int = None) -> Dict[str, Any]:
    # Implementation from P3_TOOLS_GUIDE_CORE.md
    # limit_results uses shell piping with head command
    pass

def robust_p3_execution(command: List[str], input_data: str = None, 
                       timeout: int = 300, max_retries: int = 2,
                       limit_results: int = None) -> Dict[str, Any]:
    # Implementation from P3_TOOLS_GUIDE_CORE.md
    # limit_results parameter added for token management
    pass

def parse_p3_tabular_output(output: str) -> List[Dict[str, str]]:
    # Implementation from P3_TOOLS_GUIDE_CORE.md
    pass
```

## Common Usage Patterns

### Basic Genome Queries
```python
def get_genomes(species: str = None, strain: str = None, 
                additional_filters: List[str] = None) -> List[Dict[str, str]]:
    """Get genomes using p3-all-genomes"""
    
    command = ['p3-all-genomes']
    
    # Add filters
    if species:
        command.extend(['--eq', f'species,{species}'])
    if strain:
        command.extend(['--eq', f'strain,{strain}'])
    if additional_filters:
        for filter_expr in additional_filters:
            command.extend(['--eq', filter_expr])
    
    # Add attributes (fields to return)
    attrs = ['genome_id', 'genome_name', 'species', 'strain', 'genome_status', 'genome_length']
    for attr in attrs:
        command.extend(['--attr', attr])
    
    result = run_p3_tool(command)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

def get_genomes_by_ids(genome_ids: List[str]) -> List[Dict[str, str]]:
    """Get specific genomes by their IDs"""
    
    # Create input for stdin (genome IDs)
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
    
    command = [
        'p3-get-genomes',
        '--attr', 'genome_id', '--attr', 'genome_name', 
        '--attr', 'species', '--attr', 'strain', '--attr', 'genome_status'
    ]
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage examples
ecoli_genomes = get_genomes(species="Escherichia coli")
k12_genomes = get_genomes(species="Escherichia coli", strain="K-12")
specific_genomes = get_genomes_by_ids(["83333.111", "511145.12"])
```

### Feature Queries
```python
def get_genome_features(genome_ids: List[str] = None, feature_type: str = None,
                       gene_name: str = None) -> List[Dict[str, str]]:
    """Get genome features using p3-get-genome-features"""
    
    command = ['p3-get-genome-features']
    
    # Add filters
    if feature_type:
        command.extend(['--eq', f'feature_type,{feature_type}'])
    if gene_name:
        command.extend(['--eq', f'gene,{gene_name}'])
    
    # Add attributes
    attrs = ['genome_id', 'patric_id', 'gene', 'product', 'feature_type', 
             'start', 'end', 'strand', 'aa_length']
    for attr in attrs:
        command.extend(['--attr', attr])
    
    # Handle input
    if genome_ids:
        # Provide genome IDs via stdin
        stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
        result = run_p3_tool(command, input_data=stdin_data)
    else:
        # No stdin input - tool will get features from all genomes (be careful!)
        result = run_p3_tool(command)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

def find_genes_by_name(gene_name: str, species: str = None) -> List[Dict[str, str]]:
    """Find genes by name, optionally filtered by species"""
    
    if species:
        # First get genomes for the species
        genomes = get_genomes(species=species)
        genome_ids = [g['genome_id'] for g in genomes]
        
        # Then get features for those genomes
        return get_genome_features(genome_ids=genome_ids, gene_name=gene_name)
    else:
        # Search across all genomes (use with caution)
        return get_genome_features(gene_name=gene_name)

# Usage examples
all_cds = get_genome_features(genome_ids=["83333.111"], feature_type="CDS")
pheS_genes = find_genes_by_name("pheS", species="Escherichia coli")
```

## p3-all-* Tools (6 tools)

These tools provide direct access to BV-BRC data collections. All follow consistent parameter patterns for filtering, attribute selection, and output formatting.

### p3-all-contigs - Contig/Sequence Data
```python
def p3_all_contigs(genome_id: str = None, length_min: int = None, 
                   additional_filters: List[str] = None,
                   attributes: List[str] = None,
                   count_only: bool = False) -> List[Dict[str, str]]:
    """
    Get contig/sequence data using p3-all-contigs
    
    Args:
        genome_id: Filter by genome ID
        length_min: Minimum contig length
        additional_filters: Additional --eq filters (format: "field,value")
        attributes: Fields to return (if None, returns default set)
        count_only: Return count instead of records
    
    Returns:
        List of dictionaries with contig data, or empty list on error
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
    
    result = run_p3_tool(command)
    
    if result['success']:
        if count_only:
            return [{'count': result['stdout'].strip()}]
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage examples
all_contigs = p3_all_contigs(genome_id="83333.111")
large_contigs = p3_all_contigs(length_min=100000, attributes=['genome_id', 'length', 'gc_content'])
contig_count = p3_all_contigs(genome_id="83333.111", count_only=True)
```

### p3-all-drugs - Drug/Antimicrobial Data
```python
def p3_all_drugs(drug_name: str = None, drug_class: str = None,
                 additional_filters: List[str] = None,
                 attributes: List[str] = None,
                 count_only: bool = False,
                 limit: int = 1000) -> List[Dict[str, str]]:
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
        List of dictionaries with drug data
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
            return [{'count': result['stdout'].strip()}]
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage examples
all_drugs = p3_all_drugs(limit=500)  # Limit to 500 results
antibiotics = p3_all_drugs(drug_class="Antibiotic", limit=200)
penicillin_drugs = p3_all_drugs(drug_name="Penicillin", limit=50)
```

### p3-all-genomes - Complete Genome Records
```python
def p3_all_genomes(species: str = None, strain: str = None, genome_status: str = None,
                   additional_filters: List[str] = None,
                   attributes: List[str] = None,
                   count_only: bool = False) -> List[Dict[str, str]]:
    """
    Get complete genome records using p3-all-genomes
    
    Args:
        species: Filter by species name
        strain: Filter by strain
        genome_status: Filter by genome status (Complete, WGS, etc.)
        additional_filters: Additional --eq filters
        attributes: Fields to return
        count_only: Return count instead of records
    
    Returns:
        List of dictionaries with genome data
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
    
    result = run_p3_tool(command)
    
    if result['success']:
        if count_only:
            return [{'count': result['stdout'].strip()}]
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage examples  
ecoli_genomes = p3_all_genomes(species="Escherichia coli")
complete_genomes = p3_all_genomes(genome_status="Complete")
k12_genomes = p3_all_genomes(species="Escherichia coli", strain="K-12")
```

### p3-all-subsystem-roles - Functional Role Data
```python
def p3_all_subsystem_roles(subsystem_name: str = None, role_name: str = None,
                          additional_filters: List[str] = None,
                          attributes: List[str] = None,
                          count_only: bool = False,
                          limit: int = 1000) -> List[Dict[str, str]]:
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
        List of dictionaries with subsystem role data
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
            return [{'count': result['stdout'].strip()}]
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage examples
all_roles = p3_all_subsystem_roles(limit=500)  # Limit to 500 results
glycolysis_roles = p3_all_subsystem_roles(subsystem_name="Glycolysis", limit=100)
kinase_roles = p3_all_subsystem_roles(role_name="kinase", limit=50)
```

### p3-all-subsystems - Subsystem Data
```python
def p3_all_subsystems(subsystem_class: str = None, subsystem_name: str = None,
                     additional_filters: List[str] = None,
                     attributes: List[str] = None,
                     count_only: bool = False,
                     limit: int = 1000) -> List[Dict[str, str]]:
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
        List of dictionaries with subsystem data
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
            # Default subsystem attributes (includes all comprehensive fields)
            # Note: Enhanced parsing now handles multi-line descriptions and empty fields
            default_attrs = ['subsystem_id', 'subsystem_name', 'class', 'subclass', 
                           'description', 'role_count']
            for attr in default_attrs:
                command.extend(['--attr', attr])
    
    # Use shell piping for limiting unless doing count_only
    limit_results = None if count_only else limit
    result = robust_p3_execution(command, limit_results=limit_results)
    
    if result['success']:
        if count_only:
            return [{'count': result['stdout'].strip()}]
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage examples
all_subsystems = p3_all_subsystems(limit=500)  # Limit to 500 results
metabolism_subsystems = p3_all_subsystems(subsystem_class="Metabolism", limit=200)
amino_acid_subsystems = p3_all_subsystems(additional_filters=["subclass,Amino Acids and Derivatives"], limit=100)
```

### p3-all-taxonomies - Taxonomic Data
```python
def p3_all_taxonomies(taxon_name: str = None, taxon_rank: str = None,
                     taxon_id: int = None,
                     additional_filters: List[str] = None,
                     attributes: List[str] = None,
                     count_only: bool = False,
                     limit: int = 1000) -> List[Dict[str, str]]:
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
        List of dictionaries with taxonomic data
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
            return [{'count': result['stdout'].strip()}]
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage examples
all_species = p3_all_taxonomies(taxon_rank="species", limit=500)  # Limit to 500 results
ecoli_taxonomy = p3_all_taxonomies(taxon_name="Escherichia coli", limit=100)  # Limit to 100 results
bacteria_taxonomy = p3_all_taxonomies(taxon_id=2, limit=50)  # Bacteria kingdom, limited to 50
# To avoid token limits with large result sets:
mycobacterium_limited = p3_all_taxonomies(taxon_name="Mycobacterium tuberculosis", limit=100)
```

## p3-get-* Tools (26 tools)

These tools retrieve data based on input provided via stdin (typically IDs or other identifiers). They follow a consistent pattern of taking input data and returning detailed information.

### Core Genome and Feature Tools

#### p3-get-genome-data - Detailed Genome Information
```python
def p3_get_genome_data(genome_ids: List[str],
                       attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get detailed genome data using p3-get-genome-data
    
    Args:
        genome_ids: List of genome IDs to retrieve
        attributes: Fields to return (if None, returns default set)
    
    Returns:
        List of dictionaries with genome data
    """
    
    if not genome_ids:
        return []
    
    # Create stdin data
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
    
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
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []

# Usage
detailed_genomes = p3_get_genome_data(["83333.111", "511145.12"])
```

#### p3-get-genome-contigs - Genome Contig Information
```python
def p3_get_genome_contigs(genome_ids: List[str],
                         attributes: List[str] = None) -> List[Dict[str, str]]:
    """Get contigs for specific genomes using p3-get-genome-contigs"""
    
    if not genome_ids:
        return []
    
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
    
    command = ['p3-get-genome-contigs']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['genome_id', 'sequence_id', 'accession', 'length',
                        'gc_content', 'description', 'sequence_type']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-feature-data - Detailed Feature Information
```python
def p3_get_feature_data(feature_ids: List[str],
                       attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get detailed feature data using p3-get-feature-data
    
    Args:
        feature_ids: List of feature IDs (patric_id, refseq_locus_tag, etc.)
        attributes: Fields to return
    
    Returns:
        List of dictionaries with feature data
    """
    
    if not feature_ids:
        return []
    
    stdin_data = '\n'.join([f"genome_feature.patric_id\t{fid}" for fid in feature_ids])
    
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
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

### Protein and Sequence Tools

#### p3-get-feature-sequence - Feature Sequences
```python
def p3_get_feature_sequence(feature_ids: List[str],
                           sequence_type: str = 'dna',
                           fasta_format: bool = True) -> str:
    """
    Get sequences for features using p3-get-feature-sequence
    
    Args:
        feature_ids: List of feature IDs
        sequence_type: 'dna' or 'protein' 
        fasta_format: Return in FASTA format
    
    Returns:
        Sequence data (FASTA format if requested)
    """
    
    if not feature_ids:
        return ""
    
    stdin_data = '\n'.join([f"genome_feature.patric_id\t{fid}" for fid in feature_ids])
    
    command = ['p3-get-feature-sequence']
    
    if sequence_type == 'protein':
        command.append('--protein')
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return result['stdout']
    else:
        return ""

# Usage
dna_seqs = p3_get_feature_sequence(["PATRIC.83333.111.1", "PATRIC.83333.111.2"])
protein_seqs = p3_get_feature_sequence(["PATRIC.83333.111.1"], sequence_type='protein')
```

#### p3-get-feature-protein-structures - Protein Structure Data
```python
def p3_get_feature_protein_structures(feature_ids: List[str],
                                    attributes: List[str] = None) -> List[Dict[str, str]]:
    """Get protein structure data for features"""
    
    if not feature_ids:
        return []
    
    stdin_data = '\n'.join([f"genome_feature.patric_id\t{fid}" for fid in feature_ids])
    
    command = ['p3-get-feature-protein-structures']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['feature_id', 'pdb_id', 'method', 'resolution',
                        'pmid', 'institution', 'structure_id', 'file_path']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

### Functional Analysis Tools

#### p3-get-feature-subsystems - Subsystem Associations
```python
def p3_get_feature_subsystems(feature_ids: List[str],
                            attributes: List[str] = None) -> List[Dict[str, str]]:
    """Get subsystem associations for features"""
    
    if not feature_ids:
        return []
    
    stdin_data = '\n'.join([f"genome_feature.patric_id\t{fid}" for fid in feature_ids])
    
    command = ['p3-get-feature-subsystems']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['feature_id', 'subsystem_id', 'subsystem_name',
                        'role_id', 'role_name', 'class', 'subclass']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-subsystem-features - Features in Subsystems
```python
def p3_get_subsystem_features(subsystem_ids: List[str],
                            genome_ids: List[str] = None,
                            attributes: List[str] = None) -> List[Dict[str, str]]:
    """Get features that participate in specific subsystems"""
    
    if not subsystem_ids:
        return []
    
    stdin_data = '\n'.join([f"subsystem.subsystem_id\t{sid}" for sid in subsystem_ids])
    
    command = ['p3-get-subsystem-features']
    
    # Add genome filter if specified
    if genome_ids:
        for gid in genome_ids:
            command.extend(['--eq', f'genome_id,{gid}'])
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['subsystem_id', 'genome_id', 'patric_id', 'gene',
                        'product', 'role_name', 'active']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

### Specialized Data Tools

#### p3-get-drug-genomes - Drug Resistance Data
```python
def p3_get_drug_genomes(drug_ids: List[str],
                       attributes: List[str] = None) -> List[Dict[str, str]]:
    """Get genomes with drug resistance/susceptibility data"""
    
    if not drug_ids:
        return []
    
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
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-family-data - Protein Family Information
```python
def p3_get_family_data(family_ids: List[str],
                      family_type: str = 'plfam',
                      attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get protein family data
    
    Args:
        family_ids: List of family IDs
        family_type: 'plfam', 'pgfam', or 'figfam'
        attributes: Fields to return
    """
    
    if not family_ids:
        return []
    
    # Format stdin based on family type
    if family_type == 'plfam':
        stdin_data = '\n'.join([f"protein_family_ref.family_id\t{fid}" for fid in family_ids])
    elif family_type == 'pgfam':
        stdin_data = '\n'.join([f"protein_family_ref.family_id\t{fid}" for fid in family_ids])
    elif family_type == 'figfam':
        stdin_data = '\n'.join([f"protein_family_ref.family_id\t{fid}" for fid in family_ids])
    else:
        raise ValueError("family_type must be 'plfam', 'pgfam', or 'figfam'")
    
    command = ['p3-get-family-data']
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['family_id', 'family_type', 'product', 'aa_length_avg',
                        'aa_length_std', 'feature_count']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

### Advanced Search and Analysis Tools

#### p3-get-features-by-sequence - Sequence-Based Feature Search
```python
def p3_get_features_by_sequence(sequences: List[str],
                               search_type: str = 'similar',
                               identity_threshold: float = 0.8,
                               attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Find features by sequence similarity
    
    Args:
        sequences: List of query sequences (protein or DNA)
        search_type: 'similar' or 'identical'
        identity_threshold: Minimum identity for matches
        attributes: Fields to return
    """
    
    if not sequences:
        return []
    
    # Create FASTA-formatted input
    stdin_lines = []
    for i, seq in enumerate(sequences):
        stdin_lines.append(f">query_{i}")
        stdin_lines.append(seq)
    stdin_data = '\n'.join(stdin_lines)
    
    command = ['p3-get-features-by-sequence']
    
    # Add search parameters
    if search_type == 'identical':
        command.append('--identical')
    if identity_threshold != 0.8:
        command.extend(['--min-identity', str(identity_threshold)])
    
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        default_attrs = ['patric_id', 'genome_id', 'product', 'identity', 
                        'query_coverage', 'subject_coverage']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-taxonomy-data - Taxonomic Information
```python
def p3_get_taxonomy_data(taxon_ids: List[int],
                        attributes: List[str] = None) -> List[Dict[str, str]]:
    """Get detailed taxonomic data for taxon IDs"""
    
    if not taxon_ids:
        return []
    
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
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### Extended Data Retrieval Tools

#### p3-get-feature-regions - Feature Genomic Regions
```python
def p3_get_feature_regions(feature_ids: List[str],
                          margin: int = 500,
                          consolidated: bool = False,
                          attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get genomic regions around features
    
    Args:
        feature_ids: List of feature IDs to get regions for
        margin: Distance to show around feature (default 500bp)
        consolidated: Extend regions to include overlapping features
        attributes: Specific attributes to return
    
    Returns:
        List of genomic region data
    """
    
    # Prepare stdin data
    stdin_data = '\n'.join([f"feature.patric_id\t{fid}" for fid in feature_ids])
    
    command = ['p3-get-feature-regions']
    
    # Add margin parameter
    command.extend(['--margin', str(margin)])
    
    # Add consolidated mode if requested
    if consolidated:
        command.append('--consolidated')
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-features-in-regions - Features Within Genomic Regions
```python
def p3_get_features_in_regions(region_data: List[Dict[str, str]],
                              feature_type: str = None,
                              attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get features within specified genomic regions
    
    Args:
        region_data: List of regions with genome_id, contig_id, start, end
        feature_type: Filter by feature type (CDS, rRNA, etc.)
        attributes: Specific attributes to return
    
    Returns:
        List of features within the regions
    """
    
    # Prepare stdin data with required columns
    stdin_lines = []
    for region in region_data:
        line = f"{region['genome_id']}\t{region['contig_id']}\t{region['start']}\t{region['end']}"
        stdin_lines.append(line)
    
    stdin_data = '\n'.join(stdin_lines)
    
    command = ['p3-get-features-in-regions', 'genome_id', 'contig_id', 'start', 'end']
    
    # Add feature type filter
    if feature_type:
        command.extend(['--eq', f'feature_type,{feature_type}'])
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default attributes
        default_attrs = ['patric_id', 'feature_type', 'product', 'start', 'end']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-genome-refseq-features - RefSeq Feature Annotations  
```python
def p3_get_genome_refseq_features(genome_ids: List[str],
                                 feature_type: str = None,
                                 attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get RefSeq feature annotations for genomes
    
    Args:
        genome_ids: List of genome IDs
        feature_type: Filter by feature type (CDS, rRNA, etc.)
        attributes: Specific attributes to return
    
    Returns:
        List of RefSeq features
    """
    
    # Prepare stdin data
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
    
    command = ['p3-get-genome-refseq-features']
    
    # Add feature type filter
    if feature_type:
        command.extend(['--eq', f'feature_type,{feature_type}'])
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default RefSeq attributes
        default_attrs = ['refseq_locus_tag', 'feature_type', 'product', 'gene', 'start', 'end']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-feature-protein-regions - Feature Protein Regions
```python  
def p3_get_feature_protein_regions(feature_ids: List[str],
                                  attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get protein regions for features
    
    Args:
        feature_ids: List of feature IDs
        attributes: Specific attributes to return
    
    Returns:
        List of protein region data
    """
    
    # Prepare stdin data
    stdin_data = '\n'.join([f"feature.patric_id\t{fid}" for fid in feature_ids])
    
    command = ['p3-get-feature-protein-regions']
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default protein region attributes
        default_attrs = ['patric_id', 'region_name', 'start', 'end', 'description']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### Group Management Tools

#### p3-get-feature-group - Feature Group Data
```python
def p3_get_feature_group(group_names: List[str],
                        attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get feature group data
    
    Args:
        group_names: List of feature group names
        attributes: Specific attributes to return
    
    Returns:
        List of feature group data
    """
    
    # Prepare stdin data
    stdin_data = '\n'.join([f"feature_group.name\t{name}" for name in group_names])
    
    command = ['p3-get-feature-group']
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default group attributes
        default_attrs = ['name', 'description', 'feature_count', 'owner']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-genome-group - Genome Group Data
```python
def p3_get_genome_group(group_names: List[str],
                       attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get genome group data
    
    Args:
        group_names: List of genome group names
        attributes: Specific attributes to return
    
    Returns:
        List of genome group data
    """
    
    # Prepare stdin data  
    stdin_data = '\n'.join([f"genome_group.name\t{name}" for name in group_names])
    
    command = ['p3-get-genome-group']
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default group attributes
        default_attrs = ['name', 'description', 'genome_count', 'owner']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-genome-protein-regions - Genome Protein Regions
```python
def p3_get_genome_protein_regions(genome_ids: List[str],
                                 attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get protein regions for genomes
    
    Args:
        genome_ids: List of genome IDs
        attributes: Specific attributes to return
    
    Returns:
        List of protein region data for genomes
    """
    
    # Prepare stdin data
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
    
    command = ['p3-get-genome-protein-regions']
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default protein region attributes
        default_attrs = ['genome_id', 'patric_id', 'region_name', 'start', 'end', 'description']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

#### p3-get-genome-protein-structures - Genome Protein Structures  
```python
def p3_get_genome_protein_structures(genome_ids: List[str],
                                   attributes: List[str] = None) -> List[Dict[str, str]]:
    """
    Get protein structures for genomes
    
    Args:
        genome_ids: List of genome IDs
        attributes: Specific attributes to return
    
    Returns:
        List of protein structure data for genomes
    """
    
    # Prepare stdin data
    stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
    
    command = ['p3-get-genome-protein-structures']
    
    # Add attributes
    if attributes:
        for attr in attributes:
            command.extend(['--attr', attr])
    else:
        # Default protein structure attributes
        default_attrs = ['genome_id', 'patric_id', 'pdb_id', 'method', 'resolution', 'description']
        for attr in default_attrs:
            command.extend(['--attr', attr])
    
    result = run_p3_tool(command, input_data=stdin_data)
    
    if result['success']:
        return parse_p3_tabular_output(result['stdout'])
    else:
        return []
```

## Pipeline Operations

### Genome to Features Pipeline
```python
def genome_to_features_pipeline(species: str, feature_type: str = "CDS") -> List[Dict[str, str]]:
    """Pipeline: Find genomes -> Get features (equivalent to Bob's example)"""
    
    # Step 1: Get genomes for species
    command1 = ['p3-all-genomes', '--eq', f'species,{species}']
    result1 = run_p3_tool(command1)
    
    if not result1['success']:
        return []
    
    # Step 2: Pipe to get features
    command2 = ['p3-get-genome-features', '--eq', f'feature_type,{feature_type}',
                '--attr', 'patric_id', '--attr', 'product']
    
    result2 = run_p3_tool(command2, input_data=result1['stdout'])
    
    if result2['success']:
        return parse_p3_tabular_output(result2['stdout'])
    else:
        return []

# Bob's example: p3-all-genomes --eq genus,Rickettsia | p3-get-genome-features --eq feature_type,CDS --attr patric_id --attr product
rickettsia_cds = genome_to_features_pipeline("Rickettsia")
```

### Multi-Step Analysis Pipeline
```python
def comprehensive_genome_analysis(species: str, genome_limit: int = 10) -> Dict[str, Any]:
    """
    Comprehensive analysis pipeline combining multiple data retrieval tools
    
    Returns:
        Dictionary with genomes, features, subsystems, and drug resistance data
    """
    
    results = {
        'genomes': [],
        'features': [],
        'subsystems': [],
        'drug_resistance': []
    }
    
    # Step 1: Get genomes for species
    genomes = p3_all_genomes(species=species)
    results['genomes'] = genomes
    
    if not genomes:
        return results
    
    genome_ids = [g['genome_id'] for g in genomes]
    
    # Step 2: Get all CDS features for these genomes
    features = get_genome_features(genome_ids=genome_ids, feature_type="CDS")
    results['features'] = features
    
    # Step 3: Get subsystem data for these genomes
    subsystems = p3_get_genome_subsystems(genome_ids)
    results['subsystems'] = subsystems
    
    # Step 4: Get drug resistance data
    drug_data = p3_get_genome_drugs(genome_ids)
    results['drug_resistance'] = drug_data
    
    return results

# Usage
ecoli_analysis = comprehensive_genome_analysis("Escherichia coli")
```

## Best Practices

### Managing Large Result Sets and Token Limits

**Important Note**: Some queries can return very large datasets that exceed MCP token limits (25,000 tokens). Use the `limit` parameter to control result size:

```python
# Token Limit Management Examples

# BAD: This may exceed token limits
all_mycobacterium = p3_all_taxonomies(taxon_name="Mycobacterium tuberculosis")

# GOOD: Use limit to control result size
mycobacterium_sample = p3_all_taxonomies(taxon_name="Mycobacterium tuberculosis", limit=100)

# For exploration, start small and increase as needed
small_sample = p3_all_taxonomies(taxon_name="Escherichia coli", limit=10)
larger_sample = p3_all_taxonomies(taxon_name="Escherichia coli", limit=50)

# Use count_only to check total results before retrieving data
result_count = p3_all_taxonomies(taxon_name="Mycobacterium tuberculosis", count_only=True)
print(f"Total results available: {result_count[0]['count']}")

# Then use appropriate limit based on count
if int(result_count[0]['count']) > 1000:
    # Large dataset - use smaller limit
    data = p3_all_taxonomies(taxon_name="Mycobacterium tuberculosis", limit=100)
else:
    # Smaller dataset - can use default limit
    data = p3_all_taxonomies(taxon_name="Mycobacterium tuberculosis")
```

**Functions with `limit` parameter support**:
- `p3_all_contigs(limit=1000)` 
- `p3_all_genomes(limit=1000)`
- `p3_all_taxonomies(limit=1000)` ✨ *Recently added*
- `p3_all_drugs(limit=1000)` ✨ *Recently added*
- `p3_all_subsystem_roles(limit=1000)` ✨ *Recently added*
- `p3_all_subsystems(limit=1000)` ✨ *Recently added*

**All `p3_all_*` functions now support the `limit` parameter!**

### Efficient Data Retrieval
```python
# Good: Specific attribute selection
def get_essential_genome_data(genome_ids: List[str]) -> List[Dict]:
    """Get only essential genome attributes"""
    essential_attrs = ['genome_id', 'genome_name', 'organism_name', 'genome_length']
    return p3_get_genome_data(genome_ids, attributes=essential_attrs)

# Avoid: Retrieving all attributes when only few are needed
```

### Batch Processing for Large Datasets
```python
def process_large_genome_list(genome_ids: List[str], batch_size: int = 100):
    """Process large genome lists in manageable batches"""
    
    results = []
    for i in range(0, len(genome_ids), batch_size):
        batch = genome_ids[i:i + batch_size]
        batch_results = get_essential_genome_data(batch)
        results.extend(batch_results)
        
        # Brief pause between batches
        import time
        time.sleep(0.5)
    
    return results
```

### Error Handling for Data Retrieval
```python
def robust_data_retrieval(retrieval_func, *args, **kwargs):
    """Wrapper for robust data retrieval with error handling"""
    
    try:
        result = retrieval_func(*args, **kwargs)
        if result:
            return result
        else:
            print(f"No data returned from {retrieval_func.__name__}")
            return []
    except Exception as e:
        print(f"Error in {retrieval_func.__name__}: {e}")
        return []

# Usage
genomes = robust_data_retrieval(p3_all_genomes, species="Escherichia coli")
```

### Parameter Validation
```python
def validate_genome_ids(genome_ids: List[str]) -> List[str]:
    """Validate and clean genome ID list"""
    
    valid_ids = []
    for gid in genome_ids:
        # Basic validation - genome IDs are typically numeric.numeric
        if isinstance(gid, str) and '.' in gid:
            parts = gid.split('.')
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                valid_ids.append(gid)
            else:
                print(f"Warning: Invalid genome ID format: {gid}")
        else:
            print(f"Warning: Invalid genome ID: {gid}")
    
    return valid_ids

# Usage in functions
def safe_get_genome_data(genome_ids: List[str], **kwargs):
    valid_ids = validate_genome_ids(genome_ids)
    if valid_ids:
        return p3_get_genome_data(valid_ids, **kwargs)
    else:
        return []
```

---

**Next Steps**: For computational services and job submission patterns, see **P3_TOOLS_COMPUTATIONAL.md**. For utility and pipeline tools, see **P3_TOOLS_UTILITIES.md**.
# P3-Tools Programming Guide - Core Fundamentals

This is the master reference for P3-Tools subprocess programming. All other guides in this series build upon these fundamentals.

## 📚 **P3-Tools Guide Series Navigation**

### **Complete Guide Structure**
- **P3_TOOLS_GUIDE_CORE.md** ← *You are here* - Core patterns and cross-references
- **P3_TOOLS_DATA_RETRIEVAL.md** - Data retrieval tools (p3-all-*, p3-get-*)
- **P3_TOOLS_COMPUTATIONAL.md** - Computational services (p3-submit-*)
- **P3_TOOLS_UTILITIES.md** - File management and processing tools  
- **P3_TOOLS_SPECIALIZED.md** - Domain-specific analysis tools

### **Tool Coverage Summary**
- **Total P3-Tools**: 134 tools
- **Currently Documented**: 73 tools (54% complete)
- **Remaining**: 61 tools to be documented

## Table of Contents
- [P3-Tools Overview](#p3-tools-overview)
- [Tool Category Cross-Reference](#tool-category-cross-reference)
- [Subprocess Execution Fundamentals](#subprocess-execution-fundamentals)
- [Authentication and Setup](#authentication-and-setup)
- [Programming Patterns](#programming-patterns)
- [Error Handling Framework](#error-handling-framework)
- [Best Practices](#best-practices)
- [Quick Reference](#quick-reference)

## P3-Tools Overview

### What are P3-Tools?
P3-Tools are command-line scripts that provide access to BV-BRC (Bacterial and Viral Bioinformatics Resource Center) data and computational services. They wrap the BV-BRC REST API and App Service JSON-RPC API with user-friendly command-line interfaces.

### Tool Categories

#### **Data Retrieval Tools** (REST API Wrappers)
- Fast, direct data access
- Real-time results
- Simple parameter passing
- Examples: `p3-all-genomes`, `p3-get-genome-features`, `p3-all-contigs`

#### **Computational Services** (JSON-RPC Wrappers)
- Asynchronous job submission
- Require workspace management
- Complex parameter handling
- Examples: `p3-submit-BLAST`, `p3-submit-proteome-comparison`

#### **Utility Tools**
- Authentication and workspace management
- Data format conversion
- Pipeline support
- Examples: `p3-login`, `p3-format-results`

## Tool Category Cross-Reference

### **Data Retrieval Tools** → [P3_TOOLS_DATA_RETRIEVAL.md]
**p3-all-* Tools (6 tools):**
- `p3-all-contigs` - Contig sequence data retrieval
- `p3-all-drugs` - Drug compound information
- `p3-all-genomes` - Complete genome records
- `p3-all-subsystem-roles` - Subsystem functional roles
- `p3-all-subsystems` - Subsystem pathway data
- `p3-all-taxonomies` - Taxonomic classification data

**p3-get-* Tools (26 tools):**
- `p3-get-genome-data` - Genome metadata and statistics
- `p3-get-feature-data` - Gene/protein feature annotations
- `p3-get-feature-sequence` - DNA/protein sequences
- `p3-get-genome-contigs` - Individual genome contigs
- `p3-get-drug-genomes` - Drug-genome associations
- `p3-get-family-data` - Protein family information
- `p3-get-subsystem-features` - Features in metabolic pathways
- [... and 19 more tools documented in P3_TOOLS_DATA_RETRIEVAL.md]

### **Computational Services** → [P3_TOOLS_COMPUTATIONAL.md]
**p3-submit-* Tools (16 tools):**
- `p3-submit-BLAST` - Sequence homology search
- `p3-submit-MSA` - Multiple sequence alignment
- `p3-submit-gene-tree` - Phylogenetic tree construction
- `p3-submit-codon-tree` - Codon-based phylogeny
- `p3-submit-genome-annotation` - Automated genome annotation
- `p3-submit-genome-assembly` - De novo genome assembly
- `p3-submit-proteome-comparison` - Comparative proteomics
- `p3-submit-variation-analysis` - SNP and variation calling
- `p3-submit-rnaseq` - RNA-seq differential expression
- `p3-submit-CGA` - Comparative genome analysis
- `p3-submit-fastqutils` - FASTQ quality processing
- `p3-submit-taxonomic-classification` - Metagenomic classification
- `p3-submit-metagenome-binning` - Metagenomic binning
- `p3-submit-metagenomic-read-mapping` - Read alignment
- `p3-submit-sars2-analysis` - SARS-CoV-2 analysis
- `p3-submit-sars2-assembly` - SARS-CoV-2 genome assembly

### **Utility Tools** → [P3_TOOLS_UTILITIES.md]
**File Management:**
- `p3-ls` - Workspace file listing
- `p3-cp` - Workspace file operations
- `p3-rm` - Workspace file deletion
- `p3-cat` - File concatenation

**Data Processing:**
- `p3-extract` - Data extraction utilities
- `p3-collate` - Data collation
- `p3-count` - Counting operations
- `p3-format-results` - Output formatting

**System Tools:**
- `p3-login` - Authentication
- `p3-logout` - Session management
- `p3-whoami` - User identification
- `p3-config` - Configuration management

### **Specialized Analysis Tools** → [P3_TOOLS_SPECIALIZED.md]
**Comparative Genomics:**
- `p3-blast` - Direct BLAST operations
- `p3-discriminating-kmers` - K-mer analysis
- `p3-co-occur` - Co-occurrence analysis
- `p3-signature-clusters` - Signature cluster analysis

**Feature Analysis:**
- `p3-feature-gap` - Feature gap analysis
- `p3-feature-upstream` - Upstream region analysis
- `p3-dump-genomes` - Genome data export

## Input Method Design Patterns

### Critical Tool Type Distinction

**IMPORTANT:** P3-Tools follow two distinct input patterns based on their function:

#### `p3-all-*` Tools (Collection Query Tools)
- **Input Method**: Direct parameters (`--eq`, `--in`, `--attr`)
- **Purpose**: Query entire data collections with real-time filtering
- **Examples**: `p3-all-genomes`, `p3-all-subsystems`, `p3-all-contigs`
- **Correct Usage**:
  ```bash
  p3-all-genomes --eq "species,Escherichia coli" --attr genome_id,genome_name
  p3-all-genomes --in "genome_status,Complete,WGS"
  ```

#### `p3-get-*` Tools (ID-Based Retrieval Tools)  
- **Input Method**: ID lists via **stdin** (tab-delimited format)
- **Purpose**: Retrieve detailed data for specific entities by their IDs
- **Examples**: `p3-get-genome-features`, `p3-get-genome-data`, `p3-get-feature-sequence`
- **Correct Usage**:
  ```python
  # Prepare stdin data
  stdin_data = '\n'.join([f"genome.genome_id\t{gid}" for gid in genome_ids])
  result = run_p3_tool(['p3-get-genome-features', '--attr', 'patric_id,product'], 
                       input_data=stdin_data)
  ```

### Common Error: Mixing Input Methods

**❌ INCORRECT** - Using `--in` with get-type tools:
```bash
p3-get-genome-features --in "genome_id,511145.12,83333.1" --attr patric_id,product
```

**✅ CORRECT** - Using stdin with get-type tools:
```python
stdin_data = "genome.genome_id\t511145.12\ngenome.genome_id\t83333.1"
result = run_p3_tool(['p3-get-genome-features', '--attr', 'patric_id,product'], 
                     input_data=stdin_data)
```

### Design Rationale
- **`p3-all-*` tools**: Designed for exploratory queries and filtering large collections
- **`p3-get-*` tools**: Designed for bulk retrieval of specific records via pipeline processing
- **`--in` parameter**: Only valid for collection query tools, not ID-based retrieval tools

## Subprocess Execution Fundamentals

### Basic Subprocess Pattern
```python
import subprocess
import json
import sys
from typing import List, Dict, Any, Optional

def run_p3_tool(command: List[str], input_data: str = None, timeout: int = 300) -> Dict[str, Any]:
    """
    Execute P3 tool with proper error handling and output parsing
    
    Args:
        command: List of command arguments (e.g., ['p3-all-genomes', '--eq', 'species,Escherichia coli'])
        input_data: Optional stdin data for tools that read from stdin
        timeout: Timeout in seconds (default 5 minutes)
    
    Returns:
        Dict with 'success', 'stdout', 'stderr', and 'returncode' keys
    """
    try:
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False  # Don't raise exception on non-zero exit
        )
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'returncode': result.returncode,
            'command': ' '.join(command)
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
```

### Enhanced Subprocess Pattern with Diagnostics
```python
def robust_p3_execution(command: List[str], input_data: str = None, 
                       timeout: int = 300, max_retries: int = 2) -> Dict[str, Any]:
    """
    Enhanced P3 tool execution with retry logic and error diagnosis
    
    Args:
        command: Command to execute
        input_data: Optional stdin data
        timeout: Timeout in seconds
        max_retries: Number of retry attempts for transient errors
    
    Returns:
        Dict with execution results and diagnostic information
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
            import time
            time.sleep(2 ** attempt)  # Exponential backoff
            continue
        
        # No more retries or non-transient error
        return result
    
    return result

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
```

## Authentication and Setup

### Authentication Check
```python
def check_p3_authentication() -> Dict[str, Any]:
    """
    Check if user is authenticated with BV-BRC
    
    Returns:
        Dict with authentication status and user information
    """
    result = run_p3_tool(['p3-whoami'])
    
    if result['success']:
        return {
            'authenticated': True,
            'username': result['stdout'].strip(),
            'message': 'Successfully authenticated'
        }
    else:
        return {
            'authenticated': False,
            'username': None,
            'message': 'Not authenticated - run p3-login'
        }

def ensure_p3_authentication():
    """
    Ensure P3 authentication, prompt for login if needed
    
    Raises:
        Exception: If authentication fails
    """
    auth_status = check_p3_authentication()
    
    if not auth_status['authenticated']:
        print("P3-Tools authentication required.")
        print("Please run 'p3-login' to authenticate with BV-BRC")
        raise Exception("Authentication required")
    
    return auth_status
```

### Environment Validation
```python
def validate_p3_environment() -> Dict[str, Any]:
    """
    Validate P3-Tools environment and dependencies
    
    Returns:
        Dict with validation results
    """
    validation = {
        'p3_tools_available': False,
        'authentication_status': None,
        'issues': []
    }
    
    # Check if p3-whoami is available (basic tool availability)
    result = run_p3_tool(['p3-whoami'], timeout=10)
    
    if result['returncode'] == -1 and 'Command not found' in result['stderr']:
        validation['issues'].append('P3-Tools not installed or not in PATH')
        return validation
    
    validation['p3_tools_available'] = True
    
    # Check authentication
    auth_status = check_p3_authentication()
    validation['authentication_status'] = auth_status
    
    if not auth_status['authenticated']:
        validation['issues'].append('Not authenticated with BV-BRC - run p3-login')
    
    return validation
```

## Programming Patterns

### Retry Logic with Exponential Backoff
```python
def retry_with_backoff(func, max_retries=3, base_delay=1):
    """
    Execute function with exponential backoff retry
    
    Args:
        func: Function to execute
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
    
    Returns:
        Function result
    """
    import time
    
    for attempt in range(max_retries + 1):
        try:
            result = func()
            if isinstance(result, dict) and result.get('success'):
                return result
            elif attempt == max_retries:
                return result
            
        except Exception as e:
            if attempt == max_retries:
                raise e
        
        # Exponential backoff
        delay = base_delay * (2 ** attempt)
        time.sleep(delay)
    
    return None
```

### Batch Processing Pattern
```python
def process_in_batches(items: List[Any], process_func, batch_size: int = 100):
    """
    Process items in batches to avoid overwhelming the API
    
    Args:
        items: List of items to process
        process_func: Function to process each batch
        batch_size: Items per batch
    
    Yields:
        Results from each batch
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield process_func(batch)
```

### Pipeline Composition Pattern
```python
def create_pipeline(*functions):
    """
    Create a data processing pipeline from multiple functions
    
    Args:
        *functions: Functions to chain together
    
    Returns:
        Composed pipeline function
    """
    def pipeline(data):
        result = data
        for func in functions:
            result = func(result)
        return result
    
    return pipeline

# Example usage:
# genome_analysis_pipeline = create_pipeline(
#     get_genome_data,
#     extract_features,
#     analyze_features,
#     format_results
# )
```

## Error Handling Framework

### Common P3-Tools Error Patterns

#### Authentication Errors
```python
def handle_auth_error(result: Dict[str, Any]) -> Dict[str, Any]:
    """Handle authentication-related errors"""
    return {
        'error_type': 'authentication',
        'message': 'Please run p3-login to authenticate',
        'action_required': 'login'
    }
```

#### Network/Server Errors
```python
def handle_network_error(result: Dict[str, Any]) -> Dict[str, Any]:
    """Handle network and server errors"""
    return {
        'error_type': 'network',
        'message': 'Server or network error - retry may succeed',
        'action_required': 'retry'
    }
```

#### Parameter Validation Errors
```python
def handle_parameter_error(result: Dict[str, Any]) -> Dict[str, Any]:
    """Handle parameter validation errors"""
    return {
        'error_type': 'parameter',
        'message': 'Invalid parameters provided',
        'action_required': 'fix_parameters'
    }
```

### Comprehensive Error Handler
```python
def handle_p3_error(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive P3-Tools error handler
    
    Args:
        result: Result from P3 tool execution
    
    Returns:
        Error handling information
    """
    if result.get('success'):
        return {'handled': False, 'message': 'No error to handle'}
    
    stderr = result.get('stderr', '').lower()
    
    # Authentication errors
    if any(term in stderr for term in ['not logged in', 'authentication', 'unauthorized']):
        return handle_auth_error(result)
    
    # Network errors
    elif any(term in stderr for term in ['connection', 'timeout', 'server error']):
        return handle_network_error(result)
    
    # Parameter errors
    elif any(term in stderr for term in ['invalid', 'unknown option', 'missing required']):
        return handle_parameter_error(result)
    
    # Generic error
    else:
        return {
            'error_type': 'unknown',
            'message': f"Unknown error: {result.get('stderr', 'No error message')}",
            'action_required': 'investigate'
        }
```

## Best Practices

### 1. Tool Selection Guidelines

#### Use Data Retrieval Tools When:
- You need real-time data access
- Query parameters are simple
- Results are needed immediately
- Working with small to medium datasets

#### Use Computational Services When:
- Analysis requires significant compute time
- Working with large datasets
- Need reproducible analysis workflows
- Results can be retrieved asynchronously

### 2. Performance Optimization

#### Efficient Data Retrieval
```python
# Good: Specific attribute selection
def get_essential_genome_data(genome_ids: List[str]) -> List[Dict]:
    """Get only essential genome attributes"""
    command = ['p3-get-genome-data']
    
    # Add attribute selection (p3-get-* tools use --attr for output fields)
    for attr in ['genome_id', 'genome_name', 'organism_name', 'genome_length']:
        command.extend(['--attr', attr])
    
    # p3-get-* tools use stdin input, not --eq parameters
    stdin_data = '\n'.join(genome_ids) + '\n'
    
    result = robust_p3_execution(command, stdin=stdin_data)
    return parse_p3_tabular_output(result['stdout']) if result['success'] else []

# Avoid: Retrieving all attributes when only few are needed
```

#### Batch Processing for Large Datasets
```python
def process_large_genome_list(genome_ids: List[str], batch_size: int = 100):
    """Process large genome lists in manageable batches"""
    
    results = []
    for i in range(0, len(genome_ids), batch_size):
        batch = genome_ids[i:i + batch_size]
        batch_results = get_essential_genome_data(batch)
        results.extend(batch_results)
        
        # Brief pause between batches
        time.sleep(0.5)
    
    return results
```

### 3. Resource Management

#### Memory Management for Large Results
```python
def process_large_results_streaming(command: List[str], process_func):
    """Process large results without loading everything into memory"""
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Process line by line
    for line in process.stdout:
        if line.strip():  # Skip empty lines
            processed = process_func(line.strip())
            yield processed
    
    process.wait()
```

#### Temporary File Management
```python
import tempfile
import os

def with_temp_file(content: str, suffix: str = '.txt'):
    """Context manager for temporary files"""
    
    class TempFileManager:
        def __init__(self, content, suffix):
            self.content = content
            self.suffix = suffix
            self.temp_file = None
        
        def __enter__(self):
            self.temp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix=self.suffix, delete=False
            )
            self.temp_file.write(self.content)
            self.temp_file.close()
            return self.temp_file.name
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.temp_file:
                os.unlink(self.temp_file.name)
    
    return TempFileManager(content, suffix)

# Usage:
# with with_temp_file(sequence_data, '.fasta') as temp_path:
#     result = run_p3_tool(['p3-submit-BLAST', workspace_path, job_name, temp_path])
```

### 4. Data Parsing Utilities

#### Tabular Data Parser
```python
def parse_p3_tabular_output(output: str) -> List[Dict[str, str]]:
    """
    Parse standard P3-Tools tabular output into list of dictionaries
    
    Args:
        output: Raw stdout from P3 tool
    
    Returns:
        List of dictionaries with column headers as keys
    """
    if not output.strip():
        return []
    
    lines = output.strip().split('\n')
    if len(lines) < 2:  # Need at least header + one data row
        return []
    
    headers = lines[0].split('\t')
    results = []
    
    for line in lines[1:]:
        values = line.split('\t')
        if len(values) == len(headers):
            row_dict = dict(zip(headers, values))
            results.append(row_dict)
    
    return results
```

#### JSON Data Parser
```python
def parse_p3_json_output(output: str) -> Any:
    """
    Parse P3-Tools JSON output
    
    Args:
        output: Raw JSON stdout from P3 tool
    
    Returns:
        Parsed JSON data
    """
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return None
```

## Quick Reference

### Essential Commands
```bash
# Authentication
p3-login                    # Authenticate with BV-BRC
p3-whoami                  # Check authentication status
p3-logout                  # End session

# Data Retrieval Examples
p3-all-genomes --eq "species,Escherichia coli" --attr genome_id,genome_name
echo "511145.12" | p3-get-genome-features --attr patric_id,product

# Computational Services Examples
p3-submit-BLAST /username/results blast_job input.fasta
p3-submit-genome-annotation /username/results annotation_job contigs.fasta
```

### Common Parameter Patterns
```python
# Equality filters
['--eq', 'field_name,value']

# Attribute selection  
['--attr', 'field1', '--attr', 'field2']

# Multiple values
['--in', 'field_name,value1,value2,value3']

# Note: p3-all-* tools do not support --limit parameter
# Use shell commands like head or tail to limit output if needed
```

### File Locations in Guide Series
- **Core fundamentals**: P3_TOOLS_GUIDE_CORE.md (this file)
- **Data retrieval tools**: P3_TOOLS_DATA_RETRIEVAL.md
- **Computational services**: P3_TOOLS_COMPUTATIONAL.md  
- **Utility tools**: P3_TOOLS_UTILITIES.md
- **Specialized analysis**: P3_TOOLS_SPECIALIZED.md

---

**Next Steps**: Refer to the specific guide files for detailed tool documentation and examples. Each guide builds upon the patterns established in this core reference.
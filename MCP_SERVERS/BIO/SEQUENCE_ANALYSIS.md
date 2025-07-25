# Sequence Analysis MCP Server

## Overview

The Sequence Analysis MCP Server provides comprehensive tools for DNA sequence analysis including basic statistics, reading frame analysis, codon usage calculation, and NCBI BLAST homology searches. This server is designed for molecular biology and bioinformatics applications requiring detailed sequence characterization.

## Features

- **Basic Sequence Statistics**: Calculate length, GC content, and base composition
- **Reading Frame Analysis**: Find open reading frames (ORFs) in all 6 reading frames
- **Codon Usage Analysis**: Calculate codon frequencies and amino acid usage patterns
- **BLAST Integration**: Submit sequences to NCBI BLAST and retrieve homology search results
- **Protein Translation**: Translate DNA sequences to protein using the standard genetic code

## Installation & Requirements

### Dependencies
- Python 3.12+
- `mcp` library
- `requests` library

### Setup
```bash
pip install mcp requests
```

## Available Tools

### 1. basic_sequence_stats(sequence)

Calculate basic statistics for a DNA sequence.

**Parameters:**
- `sequence` (str): DNA sequence string containing only A, T, C, G characters

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `sequence_length`: Total length of the sequence
- `gc_content_percent`: GC content as percentage
- `base_counts`: Count of each nucleotide (A, T, C, G)
- `base_percentages`: Percentage of each nucleotide

**Example:**
```python
basic_sequence_stats("ATCGATCGATCG")
```

**Sample Output:**
```json
{
  "status": "success",
  "sequence_length": 12,
  "gc_content_percent": 50.0,
  "base_counts": {"A": 3, "T": 3, "C": 3, "G": 3},
  "base_percentages": {"A": 25.0, "T": 25.0, "C": 25.0, "G": 25.0}
}
```

### 2. analyze_reading_frames(sequence, min_orf_length)

Find and analyze open reading frames (ORFs) in all 6 reading frames.

**Parameters:**
- `sequence` (str): DNA sequence string (A, T, C, G characters)
- `min_orf_length` (int, optional): Minimum ORF length in base pairs (default: 90)

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `sequence_length`: Length of input sequence
- `min_orf_length`: Minimum ORF length threshold used
- `total_orfs_found`: Number of ORFs identified
- `orfs`: List of ORF details

**ORF Details Include:**
- `frame`: Reading frame (+1, +2, +3, -1, -2, -3)
- `start_pos`: Start position in nucleotides
- `end_pos`: End position in nucleotides
- `length_bp`: Length in base pairs
- `length_aa`: Length in amino acids
- `protein_sequence`: Translated protein sequence

**Reading Frames:**
- **+1, +2, +3**: Forward strand (frames 1, 2, 3)
- **-1, -2, -3**: Reverse complement strand (frames 1, 2, 3)

**Example:**
```python
analyze_reading_frames("ATGAAATTTAAATAG", 9)
```

### 3. calculate_codon_usage(sequence)

Calculate codon usage frequencies for a DNA sequence.

**Parameters:**
- `sequence` (str): DNA sequence string (A, T, C, G characters)

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `total_codons`: Total number of codons analyzed
- `codon_counts`: Count of each codon
- `codon_frequencies`: Frequency of each codon (0-1)
- `amino_acid_counts`: Count of each amino acid
- `amino_acid_frequencies`: Frequency of each amino acid (0-1)

**Note:** Analyzes sequence in reading frame +1 only. Use `analyze_reading_frames()` first to identify the correct reading frame for comprehensive analysis.

**Example:**
```python
calculate_codon_usage("ATGAAATTTAAATAG")
```

**Sample Output:**
```json
{
  "status": "success",
  "total_codons": 5,
  "codon_counts": {"ATG": 1, "AAA": 1, "TTT": 1, "AAA": 1, "TAG": 1},
  "codon_frequencies": {"ATG": 0.2, "AAA": 0.4, "TTT": 0.2, "TAG": 0.2},
  "amino_acid_counts": {"M": 1, "K": 2, "F": 1, "*": 1},
  "amino_acid_frequencies": {"M": 0.2, "K": 0.4, "F": 0.2, "*": 0.2}
}
```

### 4. start_blast_search(sequence, database, program)

Submit a sequence to NCBI BLAST for homology search.

**Parameters:**
- `sequence` (str): DNA or protein sequence to search
- `database` (str, optional): BLAST database (default: "nt")
  - `"nt"`: Nucleotide database
  - `"nr"`: Protein database
- `program` (str, optional): BLAST program (default: "blastn")
  - `"blastn"`: Nucleotide-nucleotide BLAST
  - `"blastp"`: Protein-protein BLAST
  - `"blastx"`: Translated nucleotide vs protein database

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `job_id`: BLAST job identifier for checking results
- `estimated_time`: Estimated completion time
- `database`: Database used for search
- `program`: BLAST program used

**Example:**
```python
start_blast_search("ATCGATCGATCG", "nt", "blastn")
```

**Sample Output:**
```json
{
  "status": "success",
  "job_id": "ABC123XYZ",
  "estimated_time": "60-120 seconds",
  "database": "nt",
  "program": "blastn"
}
```

### 5. check_blast_results(job_id)

Check the status and retrieve results of a BLAST search job.

**Parameters:**
- `job_id` (str): BLAST job ID returned from `start_blast_search()`

**Returns:**
JSON object containing:
- `status`: "success", "running", or "error"
- `message`: Status description
- `job_id`: Original job identifier
- `total_hits`: Number of hits found (when completed)
- `top_hits`: List of top 5 hits with details (when completed)

**Hit Details Include:**
- `title`: Description of the matched sequence
- `accession`: Database accession number
- `length`: Length of the matched sequence
- `hsps`: High-scoring segment pairs (alignments)

**HSP Details Include:**
- `score`: Alignment score
- `evalue`: Expectation value
- `identity`: Number of identical matches
- `identity_percent`: Percentage identity
- `align_length`: Length of alignment
- `query_coverage`: Query sequence coverage
- `subject_coverage`: Subject sequence coverage

**Example:**
```python
check_blast_results("ABC123XYZ")
```

**When Running:**
```json
{
  "status": "running",
  "message": "BLAST job is still processing. Please wait and check again.",
  "job_id": "ABC123XYZ"
}
```

**When Complete:**
```json
{
  "status": "success",
  "message": "BLAST search completed successfully.",
  "job_id": "ABC123XYZ",
  "total_hits": 25,
  "top_hits": [...]
}
```

## Genetic Code & Translation

The server uses the standard genetic code for protein translation:

### Start and Stop Codons
- **Start Codon**: ATG (Methionine)
- **Stop Codons**: TAA, TAG, TGA (represented as '*')

### Translation Features
- Supports all 6 reading frames (3 forward, 3 reverse complement)
- Uses single-letter amino acid codes
- Invalid codons are represented as 'X'

## Sequence Processing

### Input Validation
- Only accepts A, T, C, G characters
- Automatically converts to uppercase
- Removes whitespace and formatting
- Rejects sequences with invalid characters

### Reading Frame Analysis
The server analyzes sequences in all 6 possible reading frames:
- **Forward Strand**: +1, +2, +3
- **Reverse Complement**: -1, -2, -3

### ORF Detection
- Searches for ORFs starting with ATG (Methionine)
- Ends at first stop codon (TAA, TAG, TGA) or sequence end
- Filters by minimum length threshold
- Returns results sorted by length (longest first)

## BLAST Integration

### Supported Databases
- **nt**: NCBI nucleotide database
- **nr**: NCBI protein database
- Custom databases (if available)

### Supported Programs
- **blastn**: Nucleotide vs nucleotide database
- **blastp**: Protein vs protein database
- **blastx**: Translated nucleotide vs protein database
- **tblastn**: Protein vs translated nucleotide database
- **tblastx**: Translated nucleotide vs translated nucleotide database

### Search Parameters
- Default E-value threshold: 10
- Maximum hits returned: 10
- Output format: JSON2
- Top 5 hits displayed with top 2 alignments each

## Configuration

### Required Settings
```python
NCBI_API_EMAIL = "your.email@example.com"  # Replace with your email
NCBI_API_TOOL = "sequence_analysis_demo"
```

**Important**: Update the email address to comply with NCBI API usage policies.

## Usage Examples

### Complete Sequence Analysis Workflow
```python
# 1. Get basic statistics
stats = basic_sequence_stats("ATGAAATTTAAATAG")

# 2. Find all ORFs
orfs = analyze_reading_frames("ATGAAATTTAAATAG", 9)

# 3. Calculate codon usage
codons = calculate_codon_usage("ATGAAATTTAAATAG")

# 4. Search for homologous sequences
blast_job = start_blast_search("ATGAAATTTAAATAG", "nt", "blastn")

# 5. Check BLAST results (after waiting)
blast_results = check_blast_results(blast_job["job_id"])
```

### Gene Sequence Analysis
```python
# Analyze a gene sequence
gene_sequence = "ATGACCGGTCGTCGTCGTAAATAG"

# Basic characterization
stats = basic_sequence_stats(gene_sequence)
print(f"Length: {stats['sequence_length']} bp")
print(f"GC Content: {stats['gc_content_percent']}%")

# Find potential coding regions
orfs = analyze_reading_frames(gene_sequence, 60)
print(f"Found {orfs['total_orfs_found']} ORFs")

# Analyze codon usage pattern
codons = calculate_codon_usage(gene_sequence)
print(f"Total codons: {codons['total_codons']}")
```

### BLAST Search Pipeline
```python
# Submit BLAST search
sequence = "ATCGATCGATCGATCG"
job = start_blast_search(sequence, "nt", "blastn")

if job["status"] == "success":
    job_id = job["job_id"]
    print(f"BLAST job submitted: {job_id}")
    
    # Poll for results
    import time
    while True:
        results = check_blast_results(job_id)
        if results["status"] == "success":
            print(f"Found {results['total_hits']} hits")
            break
        elif results["status"] == "running":
            print("Still processing...")
            time.sleep(30)
        else:
            print("Error occurred")
            break
```

## Error Handling

### Common Errors
- **Invalid Sequence**: Non-ATCG characters in input
- **BLAST Submission Failed**: Network issues or API problems
- **Job Not Found**: Expired or invalid BLAST job ID
- **Parsing Errors**: Unexpected response formats

### Debugging Features
- Detailed logging for BLAST operations
- Debug information in error responses
- Response format detection and handling

## Performance Considerations

### Sequence Length Limits
- Basic statistics: No practical limit
- ORF analysis: Efficient for sequences up to several Mb
- Codon usage: Processes entire sequence in memory
- BLAST: NCBI limits apply (typically <1Mb for web BLAST)

### BLAST Processing Times
- Short sequences (<100 bp): 30-60 seconds
- Medium sequences (100-1000 bp): 60-120 seconds
- Long sequences (>1000 bp): 2-5 minutes
- Database load affects timing

## Rate Limiting & Best Practices

### NCBI API Guidelines
- Provide valid email address
- Limit concurrent requests
- Use appropriate delays between requests
- Monitor job status periodically

### Recommended Usage Patterns
- Submit BLAST searches during off-peak hours
- Cache results for frequently analyzed sequences
- Use appropriate minimum ORF lengths to reduce noise
- Validate input sequences before processing

## Output Formats

All tools return JSON-formatted strings with consistent structure:
```json
{
  "status": "success|error",
  "message": "Descriptive message (optional)",
  "data": "Tool-specific results"
}
```

## Troubleshooting

### Sequence Issues
- Ensure sequences contain only A, T, C, G characters
- Remove any whitespace or formatting characters
- Check for proper FASTA format if copying from files

### BLAST Issues
- Verify internet connectivity
- Check NCBI service status
- Ensure job ID is correctly copied
- Wait adequate time before checking results

### Performance Issues
- Use appropriate minimum lengths for ORF analysis
- Consider sequence complexity for BLAST searches
- Monitor memory usage for very long sequences

## License & Attribution

This server integrates with NCBI BLAST services. Please ensure compliance with NCBI usage policies and cite NCBI BLAST in any publications using results from this server.

### NCBI BLAST Citation
Altschul, S.F., Gish, W., Miller, W., Myers, E.W. & Lipman, D.J. (1990) "Basic local alignment search tool." J. Mol. Biol. 215:403-410.
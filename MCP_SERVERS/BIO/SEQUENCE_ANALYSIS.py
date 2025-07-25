# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp",
#     "requests",
# ]
# ///

"""
Sequence Analysis MCP Server

Provides tools for DNA sequence analysis including reading frame analysis,
codon usage calculation, basic sequence statistics, and NCBI BLAST searches.
"""

import requests
import time
import json
import re
import zipfile
import io
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("Sequence Analysis")

# Configuration constants
NCBI_API_EMAIL = "your.email@example.com"
NCBI_API_TOOL = "sequence_analysis_demo"

# Standard genetic code
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}

# ===== HELPER FUNCTIONS =====

def clean_sequence(sequence: str) -> str:
    """Clean and validate DNA sequence."""
    # Remove whitespace and convert to uppercase
    clean_seq = re.sub(r'\s+', '', sequence.upper())
    
    # Check for valid DNA characters
    if not re.match(r'^[ATCG]+$', clean_seq):
        return ""
    
    return clean_seq

def reverse_complement(sequence: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in sequence[::-1])

def translate_dna(sequence: str, frame: int = 0) -> str:
    """Translate DNA sequence to protein in specified frame (0, 1, or 2)."""
    # Adjust for reading frame
    seq = sequence[frame:]
    
    # Translate codons
    protein = ""
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if len(codon) == 3:
            protein += GENETIC_CODE.get(codon, 'X')
    
    return protein

def find_orfs_in_sequence(sequence: str, min_length: int = 90) -> list:
    """Find all ORFs in a DNA sequence (all 6 reading frames)."""
    orfs = []
    
    # Check all 6 reading frames (3 forward, 3 reverse complement)
    sequences = [sequence, reverse_complement(sequence)]
    strand_names = ['+', '-']
    
    for strand_idx, seq in enumerate(sequences):
        strand = strand_names[strand_idx]
        
        for frame in range(3):
            frame_name = f"{strand}{frame + 1}"
            protein = translate_dna(seq, frame)
            
            # Find ORFs (start with M, end with *)
            start_pos = 0
            while start_pos < len(protein):
                # Find next methionine
                m_pos = protein.find('M', start_pos)
                if m_pos == -1:
                    break
                
                # Find next stop codon
                stop_pos = protein.find('*', m_pos)
                if stop_pos == -1:
                    stop_pos = len(protein)
                
                # Check if ORF meets minimum length
                orf_length = (stop_pos - m_pos) * 3
                if orf_length >= min_length:
                    orf_protein = protein[m_pos:stop_pos]
                    
                    # Calculate nucleotide positions
                    if strand == '+':
                        nt_start = m_pos * 3 + frame + 1
                        nt_end = stop_pos * 3 + frame
                    else:
                        # For reverse complement, positions are relative to original sequence
                        nt_start = len(sequence) - (stop_pos * 3 + frame)
                        nt_end = len(sequence) - (m_pos * 3 + frame) + 1
                    
                    orfs.append({
                        'frame': frame_name,
                        'start_pos': nt_start,
                        'end_pos': nt_end,
                        'length_bp': orf_length,
                        'length_aa': len(orf_protein),
                        'protein_sequence': orf_protein
                    })
                
                start_pos = m_pos + 1
    
    return sorted(orfs, key=lambda x: x['length_bp'], reverse=True)

# ===== MCP TOOLS =====

@mcp.tool()
def basic_sequence_stats(sequence: str) -> str:
    """Calculate basic statistics for a DNA sequence.
    
    Args:
        sequence: DNA sequence string (A, T, C, G characters)
    
    Returns:
        JSON with sequence length, GC content, base composition, and other basic statistics
        
    Example:
        basic_sequence_stats("ATCGATCG") -> returns length, GC%, base counts, etc.
    """
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        return json.dumps({
            "status": "error",
            "message": "Invalid DNA sequence. Only A, T, C, G characters allowed."
        })
    
    # Calculate base counts
    base_counts = {
        'A': clean_seq.count('A'),
        'T': clean_seq.count('T'),
        'C': clean_seq.count('C'),
        'G': clean_seq.count('G')
    }
    
    total_length = len(clean_seq)
    gc_count = base_counts['G'] + base_counts['C']
    gc_content = (gc_count / total_length) * 100 if total_length > 0 else 0
    
    # Calculate base percentages
    base_percentages = {base: (count / total_length) * 100 for base, count in base_counts.items()}
    
    return json.dumps({
        "status": "success",
        "sequence_length": total_length,
        "gc_content_percent": round(gc_content, 2),
        "base_counts": base_counts,
        "base_percentages": {base: round(pct, 2) for base, pct in base_percentages.items()}
    })

@mcp.tool()
def analyze_reading_frames(sequence: str, min_orf_length: int = 90) -> str:
    """Find and analyze open reading frames (ORFs) in all 6 reading frames.
    
    Args:
        sequence: DNA sequence string (A, T, C, G characters)
        min_orf_length: Minimum ORF length in base pairs (default: 90)
    
    Returns:
        JSON with all ORFs found, including position, length, and translated protein sequence
        
    Example:
        analyze_reading_frames("ATGATC...") -> returns list of ORFs with details
        
    Note:
        Searches both forward and reverse complement strands in frames +1, +2, +3, -1, -2, -3
    """
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        return json.dumps({
            "status": "error",
            "message": "Invalid DNA sequence. Only A, T, C, G characters allowed."
        })
    
    orfs = find_orfs_in_sequence(clean_seq, min_orf_length)
    
    return json.dumps({
        "status": "success",
        "sequence_length": len(clean_seq),
        "min_orf_length": min_orf_length,
        "total_orfs_found": len(orfs),
        "orfs": orfs
    })

@mcp.tool()
def calculate_codon_usage(sequence: str) -> str:
    """Calculate codon usage frequencies for a DNA sequence.
    
    Args:
        sequence: DNA sequence string (A, T, C, G characters)
    
    Returns:
        JSON with codon frequencies and usage statistics
        
    Example:
        calculate_codon_usage("ATGATC...") -> returns codon frequencies and amino acid usage
        
    Note:
        Analyzes the sequence in reading frame +1 only. For comprehensive analysis,
        use analyze_reading_frames() first to identify the correct reading frame.
    """
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        return json.dumps({
            "status": "error",
            "message": "Invalid DNA sequence. Only A, T, C, G characters allowed."
        })
    
    # Count codons in reading frame 1
    codon_counts = {}
    amino_acid_counts = {}
    total_codons = 0
    
    for i in range(0, len(clean_seq) - 2, 3):
        codon = clean_seq[i:i+3]
        if len(codon) == 3:
            total_codons += 1
            codon_counts[codon] = codon_counts.get(codon, 0) + 1
            
            # Count amino acids
            aa = GENETIC_CODE.get(codon, 'X')
            amino_acid_counts[aa] = amino_acid_counts.get(aa, 0) + 1
    
    # Calculate frequencies
    codon_frequencies = {codon: count / total_codons for codon, count in codon_counts.items()}
    aa_frequencies = {aa: count / total_codons for aa, count in amino_acid_counts.items()}
    
    return json.dumps({
        "status": "success",
        "total_codons": total_codons,
        "codon_counts": codon_counts,
        "codon_frequencies": {codon: round(freq, 4) for codon, freq in codon_frequencies.items()},
        "amino_acid_counts": amino_acid_counts,
        "amino_acid_frequencies": {aa: round(freq, 4) for aa, freq in aa_frequencies.items()}
    })

@mcp.tool()
def start_blast_search(sequence: str, database: str = "nt", program: str = "blastn") -> str:
    """Submit a sequence to NCBI BLAST for homology search.
    
    Args:
        sequence: DNA or protein sequence to search
        database: BLAST database ("nt" for nucleotide, "nr" for protein, default: "nt")
        program: BLAST program ("blastn", "blastp", "blastx", default: "blastn")
    
    Returns:
        JSON with BLAST job ID for checking results later, or error if submission failed
        
    Example:
        start_blast_search("ATCGATCG", "nt") -> returns {"job_id": "ABC123", "estimated_time": "60 seconds"}
        
    Note:
        This starts the search but doesn't wait for results. Use check_blast_results() to poll for completion.
    """
    clean_seq = clean_sequence(sequence)
    if not clean_seq:
        return json.dumps({
            "status": "error",
            "message": "Invalid DNA sequence. Only A, T, C, G characters allowed."
        })
    
    print(f"INFO: Submitting BLAST search: {len(clean_seq)} bp sequence to {database} database...")
    
    # NCBI BLAST Common URL API endpoint
    submit_url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
    
    # Prepare BLAST parameters
    params = {
        'CMD': 'Put',
        'PROGRAM': program,
        'DATABASE': database,
        'QUERY': clean_seq,
        'EXPECT': '10',
        'HITLIST_SIZE': '10',
        'FORMAT_TYPE': 'JSON2'
    }
    
    response = requests.post(submit_url, data=params)
    if response.status_code != 200:
        return json.dumps({
            "status": "error",
            "message": f"BLAST submission failed: HTTP {response.status_code}"
        })
    
    # Extract job ID from response
    response_text = response.text.strip()
    
    # Parse RID from the response
    rid = None
    rtoe = None
    
    # Look for RID in response (format varies)
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if 'RID =' in line:
            rid = line.split('=')[1].strip()
        elif 'RTOE =' in line:
            rtoe = line.split('=')[1].strip()
    
    if rid:
        estimated_time = f"{rtoe} seconds" if rtoe else "60-120 seconds"
        print(f"INFO: BLAST job submitted successfully. Job ID: {rid}")
        return json.dumps({
            "status": "success",
            "job_id": rid,
            "estimated_time": estimated_time,
            "database": database,
            "program": program
        })
    
    return json.dumps({
        "status": "error",
        "message": "Failed to parse BLAST submission response"
    })

@mcp.tool()
def check_blast_results(job_id: str) -> str:
    """Check the status and retrieve results of a BLAST search job.
    
    Args:
        job_id: BLAST job ID returned from start_blast_search()
    
    Returns:
        JSON with job status and results (if completed), or status message if still running
        
    Example:
        check_blast_results("ABC123") -> returns results or {"status": "running", "message": "Job still processing..."}
        
    Note:
        If the job is still running, you'll need to call this function again after waiting.
        Typical BLAST jobs take 30-120 seconds depending on sequence length and database.
    """
    print(f"INFO: Checking BLAST job status for ID: {job_id}")
    
    # Check job status using Common URL API
    status_url = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"
    status_params = {
        'CMD': 'Get',
        'RID': job_id,
        'FORMAT_TYPE': 'JSON2'
    }
    
    print(f"DEBUG: Making request to {status_url} with params: {status_params}")
    
    response = requests.get(status_url, params=status_params)
    print(f"DEBUG: Response status code: {response.status_code}")
    
    if response.status_code != 200:
        return json.dumps({
            "status": "error",
            "message": f"Failed to check BLAST status: HTTP {response.status_code}",
            "debug_response": response.text[:200]
        })
    
    response_text = response.text.strip()
    print(f"DEBUG: Response length: {len(response_text)}")
    print(f"DEBUG: First 200 chars: {response_text[:200]}")
    print(f"DEBUG: Contains Status=WAITING? {'Status=WAITING' in response_text}")
    print(f"DEBUG: Contains Status=READY? {'Status=READY' in response_text}")
    print(f"DEBUG: Starts with {{? {response_text.startswith('{')}")
    
    # Check if results are ready using Common URL API format
    if "Status=WAITING" in response_text or "ThereAreHits=maybe" in response_text:
        return json.dumps({
            "status": "running",
            "message": "BLAST job is still processing. Please wait and check again.",
            "job_id": job_id
        })
    elif "Status=FAILED" in response_text:
        return json.dumps({
            "status": "error",
            "message": "BLAST job failed. Please try submitting a new search."
        })
    elif "Status=UNKNOWN" in response_text:
        return json.dumps({
            "status": "error",
            "message": "BLAST job ID not found. It may have expired."
        })
    elif "Status=READY" in response_text or response_text.startswith('{') or response.content.startswith(b'PK'):
        # Results are ready, check if it's a ZIP file or JSON
        if response.content.startswith(b'PK'):
            # NCBI returned a ZIP file containing JSON results
            print("DEBUG: Response is a ZIP file, extracting contents...")
            
            # Extract ZIP contents
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                file_list = zip_file.namelist()
                print(f"DEBUG: ZIP contains files: {file_list}")
                
                # Look for the main results file (usually ends with _1.json)
                results_file = None
                for filename in file_list:
                    if filename.endswith('_1.json'):
                        results_file = filename
                        break
                
                if results_file:
                    print(f"DEBUG: Reading results from {results_file}")
                    with zip_file.open(results_file) as json_file:
                        results_data = json.load(json_file)
                else:
                    return json.dumps({
                        "status": "error",
                        "message": "Could not find results file in ZIP archive."
                    })
        
        elif response_text.startswith('{'):
            # Regular JSON response
            results_data = json.loads(response_text)
        
        else:
            return json.dumps({
                "status": "error",
                "message": "Could not parse BLAST results format."
            })
        
        # Extract key information from JSON2 format with robust parsing
        print(f"DEBUG: JSON structure keys: {list(results_data.keys())}")
        
        # Handle different possible JSON structures
        if 'BlastOutput2' in results_data:
            if isinstance(results_data['BlastOutput2'], list) and results_data['BlastOutput2']:
                blast_output = results_data['BlastOutput2'][0]
            else:
                blast_output = results_data['BlastOutput2']
            
            report = blast_output.get('report', {})
            program = report.get('program', 'BLAST')
            search = report.get('results', {}).get('search', {})
            hits = search.get('hits', [])
        else:
            # Fallback to direct structure if BlastOutput2 isn't present
            report = results_data.get('report', {})
            program = report.get('program', 'BLAST')
            search = results_data.get('results', {}).get('search', {})
            hits = search.get('hits', [])
        
        print(f"DEBUG: BLAST Program: {program}")
        print(f"DEBUG: Total hits found: {len(hits)}")
        
        simplified_hits = []
        for hit in hits[:5]:  # Top 5 hits
            hit_data = {
                'title': hit.get('description', [{}])[0].get('title', 'N/A'),
                'accession': hit.get('description', [{}])[0].get('accession', 'N/A'),
                'length': hit.get('len', 'N/A'),
                'hsps': []
            }
            
            # Get alignment details
            for hsp in hit.get('hsps', [])[:2]:  # Top 2 alignments per hit
                score = hsp.get('score', 'N/A')
                evalue = hsp.get('evalue', 'N/A')
                identity = hsp.get('identity', 'N/A')
                align_len = hsp.get('align_len', 'N/A')
                
                # Calculate identity percentage
                if align_len and identity and align_len != 'N/A' and identity != 'N/A':
                    identity_pct = round((identity / align_len) * 100, 1)
                else:
                    identity_pct = 'N/A'
                
                hsp_data = {
                    'score': score,
                    'evalue': evalue,
                    'identity': identity,
                    'identity_percent': identity_pct,
                    'align_length': align_len,
                    'query_coverage': f"{hsp.get('query_from', 'N/A')}-{hsp.get('query_to', 'N/A')}",
                    'subject_coverage': f"{hsp.get('hit_from', 'N/A')}-{hsp.get('hit_to', 'N/A')}"
                }
                hit_data['hsps'].append(hsp_data)
            
            simplified_hits.append(hit_data)
        
        print(f"INFO: BLAST search completed. Found {len(hits)} hits.")
        return json.dumps({
            "status": "success",
            "message": f"BLAST search completed successfully.",
            "job_id": job_id,
            "total_hits": len(hits),
            "top_hits": simplified_hits
        })
    
    return json.dumps({
        "status": "error",
        "message": "Unknown BLAST response format.",
        "debug_response": response_text[:500],
        "debug_contains": {
            "waiting": "Status=WAITING" in response_text,
            "ready": "Status=READY" in response_text,
            "failed": "Status=FAILED" in response_text,
            "json": response_text.startswith('{')
        }
    })

# Run the server when executed directly
if __name__ == "__main__":
    mcp.run(transport="stdio")

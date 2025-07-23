# BV-BRC REST API Programming Guide

This document provides a comprehensive programming reference for the BV-BRC (Bacterial and Viral Bioinformatics Resource Center) REST API. It covers the core endpoints, programming patterns, and best practices needed to build applications that interact with BV-BRC data.

## Table of Contents
- [API Fundamentals](#api-fundamentals)
- [Core Endpoints](#core-endpoints)
- [Programming Patterns](#programming-patterns)
- [Complete Code Examples](#complete-code-examples)
- [Best Practices](#best-practices)
- [Error Handling](#error-handling)
- [Authentication](#authentication)

## API Fundamentals

### Base Configuration
```python
BASE_URL = "https://www.bv-brc.org/api/"
HEADERS = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}
```

### Response Format
- All endpoints return JSON arrays of objects
- Each object represents one record
- Missing fields are omitted (not null)
- Response is always an array, even for single results

### RQL (Resource Query Language)

BV-BRC uses RQL for filtering and querying data. Here's the complete syntax reference:

#### Basic Operators
```python
# Equality
eq(field,value)                    # field equals value
ne(field,value)                    # field not equals value

# List membership  
in(field,(value1,value2,value3))   # field matches any value in list

# Comparison
lt(field,value)                    # less than
le(field,value)                    # less than or equal
gt(field,value)                    # greater than
ge(field,value)                    # greater than or equal

# Text search
keyword(value)                     # searches across multiple fields
```

#### Logical Operators
```python
# Combine expressions
and(expr1,expr2)                   # logical AND
or(expr1,expr2)                    # logical OR
```

#### Query Modifiers
```python
# Field selection
select(field1,field2,field3)       # return only specified fields

# Pagination
limit(count)                       # limit number of results
limit(count,start)                 # limit with offset

# Sorting
sort(+field)                       # ascending sort
sort(-field)                       # descending sort
```

#### URL Encoding Rules
```python
from urllib.parse import quote

# CORRECT: Encode values individually
species_query = quote("Escherichia coli")
query = f"eq(species,{species_query})"

# WRONG: Don't encode the entire expression
query = quote("eq(species,Escherichia coli)")  # Breaks RQL operators
```

#### Complex Query Examples
```python
# E. coli K-12 genomes
species_q = quote('Escherichia coli')
strain_q = quote('K-12')
query = f"and(eq(species,{species_q}),eq(strain,{strain_q}))"

# Multiple genome IDs
genome_ids = "83333.111,511145.12,316407.7"
query = f"in(genome_id,({genome_ids}))"

# Combine filters with field selection
query = f"and(eq(species,{species_q}),eq(genome_status,Complete))&select(genome_id,genome_name,strain)"
```

## Core Endpoints

### 1. Genome Endpoint (`/api/genome/`)

**Purpose**: Retrieve genome metadata and information
**URL**: `https://www.bv-brc.org/api/genome/`

#### Key Fields
- `genome_id` - Unique identifier for the genome
- `genome_name` - Descriptive name of the genome
- `species` - Species name
- `strain` - Strain designation
- `genome_status` - Complete, Draft, etc.
- `genome_length` - Total genome size in base pairs
- `gc_content` - GC content percentage
- `completion_date` - When genome sequencing was completed
- `chromosomes` - Number of chromosomes
- `plasmids` - Number of plasmids
- `contigs` - Number of contigs
- `cds` - Number of coding sequences
- `host_name` - Host organism (if applicable)
- `geographic_location` - Where sample was collected
- `collection_date` - Sample collection date

#### Common Queries
```python
import requests
from urllib.parse import quote

def get_genomes_by_species(species, limit=50):
    """Get genomes for a specific species"""
    url = "https://www.bv-brc.org/api/genome/"
    species_query = quote(species)
    params = f"eq(species,{species_query})&limit({limit})"
    params += "&select(genome_id,genome_name,strain,genome_status,genome_length)"
    
    response = requests.get(f"{url}?{params}", headers=HEADERS)
    return response.json() if response.status_code == 200 else None

def get_complete_genomes(species, limit=20):
    """Get only complete genomes for a species"""
    url = "https://www.bv-brc.org/api/genome/"
    species_query = quote(species)
    params = f"and(eq(species,{species_query}),eq(genome_status,Complete))&limit({limit})"
    
    response = requests.get(f"{url}?{params}", headers=HEADERS)
    return response.json() if response.status_code == 200 else None

def get_genomes_by_ids(genome_ids):
    """Get specific genomes by their IDs"""
    url = "https://www.bv-brc.org/api/genome/"
    ids_str = ','.join(genome_ids)
    params = f"in(genome_id,({ids_str}))"
    
    response = requests.get(f"{url}?{params}", headers=HEADERS)
    return response.json() if response.status_code == 200 else None
```

### 2. Genome Feature Endpoint (`/api/genome_feature/`)

**Purpose**: Retrieve gene annotations and features
**URL**: `https://www.bv-brc.org/api/genome_feature/`

#### Key Fields
- `genome_id` - Parent genome identifier
- `patric_id` - Unique feature identifier in BV-BRC
- `feature_id` - Alternative feature identifier
- `gene` - Gene name/symbol
- `product` - Protein product description
- `start` - Start coordinate (1-based)
- `end` - End coordinate (1-based)
- `strand` - Strand orientation ("fwd" or "rev")
- `feature_type` - Type (CDS, rRNA, tRNA, etc.)
- `go_terms` - Gene Ontology terms
- `ec_number` - Enzyme Commission number
- `pathway` - Metabolic pathway information
- `protein_id` - Protein identifier
- `aa_length` - Amino acid sequence length
- `na_length` - Nucleotide sequence length

#### Common Queries
```python
def get_genes_by_name(gene_name, genome_ids=None, limit=100):
    """Find genes by name across genomes"""
    url = "https://www.bv-brc.org/api/genome_feature/"
    gene_query = quote(gene_name)
    
    if genome_ids:
        ids_str = ','.join(genome_ids)
        params = f"and(in(genome_id,({ids_str})),eq(gene,{gene_query}))&limit({limit})"
    else:
        params = f"eq(gene,{gene_query})&limit({limit})"
    
    params += "&select(genome_id,patric_id,gene,product,start,end,strand)"
    
    response = requests.get(f"{url}?{params}", headers=HEADERS)
    return response.json() if response.status_code == 200 else None

def get_cds_features(genome_id, limit=1000):
    """Get all CDS features for a genome"""
    url = "https://www.bv-brc.org/api/genome_feature/"
    params = f"and(eq(genome_id,{genome_id}),eq(feature_type,CDS))&limit({limit})"
    params += "&select(patric_id,gene,product,start,end,strand,aa_length)"
    
    response = requests.get(f"{url}?{params}", headers=HEADERS)
    return response.json() if response.status_code == 200 else None

def search_genes_by_product(product_keyword, genome_ids=None, limit=100):
    """Search genes by product description"""
    url = "https://www.bv-brc.org/api/genome_feature/"
    keyword_query = quote(product_keyword)
    
    if genome_ids:
        ids_str = ','.join(genome_ids)
        base_query = f"in(genome_id,({ids_str}))"
        params = f"and({base_query},keyword({keyword_query}))&limit({limit})"
    else:
        params = f"keyword({keyword_query})&limit({limit})"
    
    response = requests.get(f"{url}?{params}", headers=HEADERS)
    return response.json() if response.status_code == 200 else None
```

### 3. Genome Sequence Endpoint (`/api/genome_sequence/`)

**Purpose**: Retrieve full genome sequences
**URL**: `https://www.bv-brc.org/api/genome_sequence/`

#### Key Fields
- `genome_id` - Genome identifier
- `sequence` - Complete genome sequence string
- `sequence_type` - Type of sequence (chromosome, plasmid, etc.)
- `topology` - Circular or linear
- `gi` - GenBank identifier
- `accession` - GenBank accession number
- `length` - Sequence length in base pairs
- `gc_content` - GC content of sequence
- `md5` - MD5 hash of sequence

#### Common Queries
```python
def get_genome_sequence(genome_id):
    """Get complete genome sequence"""
    url = "https://www.bv-brc.org/api/genome_sequence/"
    params = f"eq(genome_id,{genome_id})&select(sequence,topology,length,gc_content)"
    
    response = requests.get(f"{url}?{params}", headers=HEADERS)
    data = response.json() if response.status_code == 200 else None
    
    if data and len(data) > 0:
        return data[0]  # Usually one sequence per genome
    return None

def extract_gene_sequence(genome_id, start, end, strand):
    """Extract gene sequence using coordinates"""
    # Get full genome sequence
    genome_data = get_genome_sequence(genome_id)
    if not genome_data or 'sequence' not in genome_data:
        return None
    
    genome_seq = genome_data['sequence']
    
    # Extract subsequence (convert to 0-based indexing)
    gene_seq = genome_seq[start-1:end]
    
    # Handle reverse strand
    if strand == 'rev':
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        gene_seq = ''.join(complement.get(base.upper(), base) for base in gene_seq[::-1])
    
    return gene_seq
```

## Programming Patterns

### Error Handling Pattern
```python
def safe_api_request(url, params=None):
    """Make API request with proper error handling"""
    try:
        full_url = f"{url}?{params}" if params else url
        response = requests.get(full_url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return {"success": True, "data": data, "count": len(data)}
        else:
            return {
                "success": False, 
                "error": f"HTTP {response.status_code}", 
                "message": response.text
            }
            
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    except ValueError as e:
        return {"success": False, "error": f"JSON parsing failed: {str(e)}"}
```

### Pagination Pattern
```python
def get_all_results(base_url, base_params, batch_size=1000, max_results=None):
    """Get all results using pagination"""
    all_results = []
    offset = 0
    
    while True:
        # Add pagination to params
        if base_params:
            params = f"{base_params}&limit({batch_size},{offset})"
        else:
            params = f"limit({batch_size},{offset})"
        
        result = safe_api_request(base_url, params)
        
        if not result["success"]:
            break
            
        batch_data = result["data"]
        if not batch_data:  # No more results
            break
            
        all_results.extend(batch_data)
        
        # Check limits
        if max_results and len(all_results) >= max_results:
            all_results = all_results[:max_results]
            break
            
        # Prepare for next batch
        offset += batch_size
        
        # Rate limiting
        time.sleep(0.1)  # Be polite to the API
    
    return all_results
```

### Batch Processing Pattern
```python
def process_genomes_in_batches(genome_ids, batch_size=10):
    """Process genome IDs in batches to avoid URL length limits"""
    results = []
    
    for i in range(0, len(genome_ids), batch_size):
        batch = genome_ids[i:i + batch_size]
        batch_results = get_genomes_by_ids(batch)
        
        if batch_results:
            results.extend(batch_results)
        
        # Rate limiting between batches
        time.sleep(0.5)
    
    return results
```

## Complete Code Examples

### Example 1: Find Similar Genomes by Species and Strain
```python
import requests
import time
from urllib.parse import quote

def find_similar_genomes(target_species, target_strain=None, min_genome_length=None):
    """Find genomes similar to target criteria"""
    url = "https://www.bv-brc.org/api/genome/"
    
    # Build query
    species_query = quote(target_species)
    query_parts = [f"eq(species,{species_query})"]
    
    if target_strain:
        strain_query = quote(target_strain)
        query_parts.append(f"eq(strain,{strain_query})")
    
    if min_genome_length:
        query_parts.append(f"ge(genome_length,{min_genome_length})")
    
    # Combine query parts
    if len(query_parts) > 1:
        params = f"and({','.join(query_parts)})"
    else:
        params = query_parts[0]
    
    params += "&select(genome_id,genome_name,strain,genome_length,completion_date)"
    params += "&sort(-genome_length)&limit(100)"
    
    result = safe_api_request(url, params)
    return result["data"] if result["success"] else []

# Usage
ecoli_genomes = find_similar_genomes("Escherichia coli", "K-12", 4000000)
for genome in ecoli_genomes[:5]:
    print(f"{genome['genome_name']} - {genome['genome_length']} bp")
```

### Example 2: Comprehensive Gene Analysis
```python
def analyze_gene_across_genomes(gene_name, species, max_genomes=50):
    """Analyze a specific gene across multiple genomes"""
    
    # Step 1: Find genomes for the species
    genomes = get_complete_genomes(species, max_genomes)
    if not genomes:
        return {"error": f"No genomes found for {species}"}
    
    genome_ids = [g['genome_id'] for g in genomes]
    
    # Step 2: Find the gene in these genomes
    gene_features = get_genes_by_name(gene_name, genome_ids)
    if not gene_features:
        return {"error": f"Gene {gene_name} not found in {species} genomes"}
    
    # Step 3: Extract sequences for each gene instance
    gene_sequences = []
    for feature in gene_features:
        sequence = extract_gene_sequence(
            feature['genome_id'],
            feature['start'],
            feature['end'],
            feature['strand']
        )
        
        if sequence:
            gene_sequences.append({
                'genome_id': feature['genome_id'],
                'gene_name': feature['gene'],
                'product': feature['product'],
                'sequence': sequence,
                'length': len(sequence)
            })
    
    # Step 4: Analysis summary
    analysis = {
        "gene_name": gene_name,
        "species": species,
        "genomes_searched": len(genomes),
        "gene_instances_found": len(gene_features),
        "sequences_extracted": len(gene_sequences),
        "sequence_lengths": [s['length'] for s in gene_sequences],
        "sequences": gene_sequences
    }
    
    if gene_sequences:
        analysis["avg_length"] = sum(analysis["sequence_lengths"]) / len(analysis["sequence_lengths"])
        analysis["min_length"] = min(analysis["sequence_lengths"])
        analysis["max_length"] = max(analysis["sequence_lengths"])
    
    return analysis

# Usage
pheS_analysis = analyze_gene_across_genomes("pheS", "Escherichia coli", 20)
print(f"Found {pheS_analysis['gene_instances_found']} pheS genes")
print(f"Average length: {pheS_analysis.get('avg_length', 0):.1f} bp")
```

### Example 3: Genome Feature Summary
```python
def genome_feature_summary(genome_id):
    """Generate comprehensive feature summary for a genome"""
    
    # Get genome metadata
    genome_data = get_genomes_by_ids([genome_id])
    if not genome_data:
        return {"error": "Genome not found"}
    
    genome_info = genome_data[0]
    
    # Get all features
    all_features = get_cds_features(genome_id, limit=10000)
    if not all_features:
        return {"error": "No features found"}
    
    # Analyze features
    feature_types = {}
    gene_count = 0
    hypothetical_count = 0
    total_aa_length = 0
    
    for feature in all_features:
        # Count by type
        ftype = feature.get('feature_type', 'unknown')
        feature_types[ftype] = feature_types.get(ftype, 0) + 1
        
        # Gene analysis
        if feature.get('gene'):
            gene_count += 1
        
        product = feature.get('product', '').lower()
        if 'hypothetical' in product:
            hypothetical_count += 1
        
        # Length analysis
        aa_len = feature.get('aa_length', 0)
        if aa_len:
            total_aa_length += aa_len
    
    summary = {
        "genome_info": {
            "genome_id": genome_info['genome_id'],
            "genome_name": genome_info['genome_name'],
            "genome_length": genome_info.get('genome_length', 0),
            "species": genome_info.get('species', 'unknown')
        },
        "feature_summary": {
            "total_features": len(all_features),
            "feature_types": feature_types,
            "named_genes": gene_count,
            "hypothetical_proteins": hypothetical_count,
            "avg_protein_length": total_aa_length / len(all_features) if all_features else 0
        },
        "coding_density": len(all_features) / (genome_info.get('genome_length', 1) / 1000)  # genes per kb
    }
    
    return summary

# Usage
summary = genome_feature_summary("83333.111")  # E. coli K-12
print(f"Genome: {summary['genome_info']['genome_name']}")
print(f"Total features: {summary['feature_summary']['total_features']}")
print(f"Coding density: {summary['coding_density']:.2f} genes/kb")
```

## Best Practices

### 1. Rate Limiting and Politeness
```python
import time

# Add delays between requests
time.sleep(0.1)  # 100ms delay for light usage
time.sleep(0.5)  # 500ms delay for batch processing
time.sleep(1.0)  # 1s delay for heavy usage
```

### 2. Efficient Field Selection
```python
# BAD: Get all fields (slow, large response)
params = "eq(species,Escherichia coli)"

# GOOD: Get only needed fields (fast, small response)
params = "eq(species,Escherichia coli)&select(genome_id,genome_name,strain)"
```

### 3. Proper URL Encoding
```python
from urllib.parse import quote

# Always encode values that might contain spaces or special characters
species = quote("Escherichia coli")
strain = quote("K-12 MG1655")
query = f"and(eq(species,{species}),eq(strain,{strain}))"
```

### 4. Reasonable Limits
```python
# Start with small limits for exploration
params = f"eq(species,{species_query})&limit(50)"

# Increase for production, but be mindful of performance
params = f"eq(species,{species_query})&limit(1000)"
```

### 5. Result Validation
```python
def validate_response(data, expected_fields):
    """Validate API response has expected structure"""
    if not data:
        return False
    
    if not isinstance(data, list):
        return False
    
    if len(data) == 0:
        return True  # Empty result is valid
    
    # Check first record has expected fields
    sample = data[0]
    for field in expected_fields:
        if field not in sample:
            return False
    
    return True

# Usage
genomes = get_genomes_by_species("Escherichia coli")
if validate_response(genomes, ['genome_id', 'genome_name']):
    print("Valid response received")
```

## Error Handling

### Common HTTP Status Codes
- **200**: Success - request completed successfully
- **400**: Bad Request - malformed RQL query
- **404**: Not Found - endpoint doesn't exist
- **429**: Too Many Requests - rate limit exceeded
- **500**: Internal Server Error - server-side problem

### Error Recovery Strategies
```python
def robust_api_call(url, params, max_retries=3):
    """API call with retry logic"""
    for attempt in range(max_retries):
        result = safe_api_request(url, params)
        
        if result["success"]:
            return result
        
        # Handle specific errors
        if "429" in result.get("error", ""):  # Rate limit
            time.sleep(2 ** attempt)  # Exponential backoff
            continue
        elif "timeout" in result.get("error", "").lower():
            time.sleep(1)
            continue
        else:
            break  # Don't retry for other errors
    
    return result
```

## Authentication

### Public Data Access
Most BV-BRC data is publicly accessible without authentication:
```python
# No authentication needed for public data
headers = {
    'Accept': 'application/json'
}
```

### BV-BRC Service Authentication
For authenticated app services like BLAST submission and private workspace access, BV-BRC uses token-based authentication:

```python
import requests

def get_bv_brc_token(username, password):
    """Get authentication token from BV-BRC"""
    response = requests.post(
        'https://user.patricbrc.org/authenticate',
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        data={'username': username, 'password': password}
    )
    
    if response.status_code == 200:
        return response.text.strip()
    else:
        raise Exception(f"Authentication failed: {response.status_code}")

def make_authenticated_request(endpoint, params, token):
    """Make authenticated request to BV-BRC app services"""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "id": "1",
        "method": endpoint,
        "params": params,
        "jsonrpc": "2.0"
    }
    
    response = requests.post(
        "https://p3.theseed.org/services/app_service", 
        json=payload, 
        headers=headers
    )
    
    return response.json()
```

### Service Discovery
BV-BRC provides service discovery capabilities for available app services:

```python
def discover_available_services():
    """Discover all available BV-BRC app services"""
    payload = {
        "id": "1",
        "method": "AppService.enumerate_apps",
        "params": [],
        "jsonrpc": "2.0"
    }
    
    response = requests.post(
        "https://p3.theseed.org/services/app_service",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        return result.get("result", [])
    else:
        return None

# Example usage
services = discover_available_services()
if services:
    print(f"Found {len(services)} available services")
    for service_group in services[:3]:  # Show first 3 groups
        if isinstance(service_group, list):
            for app in service_group[:2]:  # Show first 2 apps per group
                if isinstance(app, dict):
                    print(f"- {app.get('id')}: {app.get('label')}")
```

### Workspace Integration
Some operations may require BV-BRC workspace access:
```python
def check_authentication_required(error_response):
    """Check if error indicates authentication is needed"""
    error_msg = error_response.get("message", "").lower()
    return "unauthorized" in error_msg or "login" in error_msg
```

## Additional Specialized Endpoints

### 6. Taxonomy Endpoint (`/api/taxonomy/`)

**Purpose**: Access taxonomic classification data and lineage information
**URL**: `https://www.bv-brc.org/api/taxonomy/`

#### Key Fields
- `taxon_id` - NCBI taxonomy identifier
- `taxon_name` - Scientific name of the taxon
- `division` - Taxonomic division (e.g., "Bacteria", "Plants")
- `genetic_code` - Genetic code table number
- `lineage_ids` - Array of NCBI taxonomy IDs in lineage
- `lineage_names` - Array of taxonomic names in lineage
- `lineage_ranks` - Array of taxonomic ranks (kingdom, phylum, etc.)

#### Common Queries
```python
def get_taxonomy_info(taxon_id=None, taxon_name=None):
    """Retrieve taxonomic information by ID or name"""
    base_url = "https://www.bv-brc.org/api/taxonomy/"
    
    if taxon_id:
        query = f"eq(taxon_id,{taxon_id})"
    elif taxon_name:
        query = f"eq(taxon_name,*{taxon_name}*)"
    else:
        raise ValueError("Must provide either taxon_id or taxon_name")
    
    url = f"{base_url}?{query}&limit(10)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed: {response.status_code}"}

def get_lineage_for_organism(organism_name):
    """Get complete taxonomic lineage for an organism"""
    base_url = "https://www.bv-brc.org/api/taxonomy/"
    query = f"eq(taxon_name,*{organism_name}*)&select(taxon_name,lineage_names,lineage_ranks)"
    
    url = f"{base_url}?{query}&limit(5)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            result = data[0]
            lineage = list(zip(result['lineage_names'], result['lineage_ranks']))
            return {
                "organism": result['taxon_name'],
                "lineage": lineage
            }
    return {"error": "Organism not found"}
```

### 7. Subsystem Endpoint (`/api/subsystem/`)

**Purpose**: Access subsystem functional classifications and pathway data
**URL**: `https://www.bv-brc.org/api/subsystem/`

#### Key Fields
- `genome_id` - BV-BRC genome identifier
- `genome_name` - Full genome name
- `patric_id` - PATRIC feature identifier
- `product` - Protein product description
- `subsystem_name` - Name of the subsystem
- `role_name` - Specific functional role within subsystem
- `active` - Activity status

#### Common Queries
```python
def get_subsystems_for_genome(genome_id):
    """Get all subsystems present in a genome"""
    base_url = "https://www.bv-brc.org/api/subsystem/"
    query = f"eq(genome_id,{genome_id})&select(subsystem_name,role_name,product)&limit(100)"
    
    url = f"{base_url}?{query}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        # Group by subsystem
        subsystems = {}
        for record in data:
            subsys_name = record.get('subsystem_name', 'Unknown')
            if subsys_name not in subsystems:
                subsystems[subsys_name] = []
            subsystems[subsys_name].append({
                'role': record.get('role_name'),
                'product': record.get('product')
            })
        return subsystems
    return {"error": f"Request failed: {response.status_code}"}

def find_genomes_with_subsystem(subsystem_name):
    """Find genomes containing a specific subsystem"""
    base_url = "https://www.bv-brc.org/api/subsystem/"
    query = f"eq(subsystem_name,*{subsystem_name}*)&select(genome_id,genome_name)&limit(50)"
    
    url = f"{base_url}?{query}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        # Remove duplicates
        genomes = {}
        for record in data:
            gid = record.get('genome_id')
            if gid:
                genomes[gid] = record.get('genome_name')
        return [{'genome_id': k, 'genome_name': v} for k, v in genomes.items()]
    return {"error": f"Request failed: {response.status_code}"}
```

### 8. Specialty Gene Endpoint (`/api/sp_gene/`)

**Purpose**: Access specialty gene annotations (antibiotic resistance, virulence factors, etc.)
**URL**: `https://www.bv-brc.org/api/sp_gene/`

#### Key Fields
- `genome_id` - BV-BRC genome identifier
- `patric_id` - PATRIC feature identifier
- `gene` - Gene name
- `organism` - Source organism
- `product` - Protein product description
- `antibiotics` - Array of associated antibiotics
- `evidence` - Evidence type (e.g., "K-mer Search")
- `source` - Data source
- `property` - Specialty property type

#### Common Queries
```python
def get_antibiotic_resistance_genes(genome_id):
    """Get antibiotic resistance genes for a genome"""
    base_url = "https://www.bv-brc.org/api/sp_gene/"
    query = f"eq(genome_id,{genome_id})&eq(property,*Antibiotic*)&select(gene,product,antibiotics,evidence)"
    
    url = f"{base_url}?{query}&limit(100)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    return {"error": f"Request failed: {response.status_code}"}

def find_genes_for_antibiotic(antibiotic_name):
    """Find genes associated with resistance to specific antibiotic"""
    base_url = "https://www.bv-brc.org/api/sp_gene/"
    query = f"eq(antibiotics,*{antibiotic_name}*)&select(gene,organism,product,genome_id)"
    
    url = f"{base_url}?{query}&limit(50)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    return {"error": f"Request failed: {response.status_code}"}

def get_virulence_factors(genome_id):
    """Get virulence factor genes for a genome"""
    base_url = "https://www.bv-brc.org/api/sp_gene/"
    query = f"eq(genome_id,{genome_id})&eq(property,*Virulence*)&select(gene,product,evidence)"
    
    url = f"{base_url}?{query}&limit(100)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    return {"error": f"Request failed: {response.status_code}"}
```

### 9. Genome AMR Endpoint (`/api/genome_amr/`)

**Purpose**: Access computational antimicrobial resistance predictions
**URL**: `https://www.bv-brc.org/api/genome_amr/`

#### Key Fields
- `genome_id` - BV-BRC genome identifier
- `genome_name` - Full genome name
- `antibiotic` - Antibiotic tested
- `resistant_phenotype` - Predicted resistance ("Susceptible", "Resistant")
- `computational_method` - ML algorithm used
- `computational_method_performance` - Performance metrics (accuracy, F1, AUC)

#### Common Queries
```python
def get_amr_profile(genome_id):
    """Get complete AMR profile for a genome"""
    base_url = "https://www.bv-brc.org/api/genome_amr/"
    query = f"eq(genome_id,{genome_id})&select(antibiotic,resistant_phenotype,computational_method_performance)"
    
    url = f"{base_url}?{query}&limit(100)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        # Organize by antibiotic
        amr_profile = {}
        for record in data:
            antibiotic = record.get('antibiotic')
            if antibiotic:
                amr_profile[antibiotic] = {
                    'phenotype': record.get('resistant_phenotype'),
                    'performance': record.get('computational_method_performance', {})
                }
        return amr_profile
    return {"error": f"Request failed: {response.status_code}"}

def compare_amr_methods():
    """Compare performance of different AMR prediction methods"""
    base_url = "https://www.bv-brc.org/api/genome_amr/"
    query = "select(computational_method,computational_method_performance)&limit(1000)"
    
    url = f"{base_url}?{query}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        methods = {}
        for record in data:
            method = record.get('computational_method')
            perf = record.get('computational_method_performance', {})
            
            if method and perf:
                if method not in methods:
                    methods[method] = []
                methods[method].append(perf)
        
        # Calculate average performance per method
        avg_performance = {}
        for method, perfs in methods.items():
            if perfs:
                avg_performance[method] = {
                    'avg_accuracy': sum(p.get('accuracy', 0) for p in perfs) / len(perfs),
                    'avg_f1': sum(p.get('f1_score', 0) for p in perfs) / len(perfs),
                    'sample_count': len(perfs)
                }
        return avg_performance
    return {"error": f"Request failed: {response.status_code}"}
```

### 10. Protein Feature Endpoint (`/api/protein_feature/`)

**Purpose**: Access protein domain and functional feature annotations
**URL**: `https://www.bv-brc.org/api/protein_feature/`

#### Key Fields
- `patric_id` - PATRIC feature identifier
- `genome_id` - BV-BRC genome identifier
- `description` - Feature description
- `start` - Start position in protein
- `end` - End position in protein
- `e_value` - Statistical significance
- `aa_sequence_md5` - MD5 hash of amino acid sequence

#### Common Queries
```python
def get_protein_domains(patric_id):
    """Get all protein domains for a specific gene"""
    base_url = "https://www.bv-brc.org/api/protein_feature/"
    query = f"eq(patric_id,{patric_id})&select(description,start,end,e_value)"
    
    url = f"{base_url}?{query}&limit(50)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    return {"error": f"Request failed: {response.status_code}"}

def find_proteins_with_domain(domain_description):
    """Find proteins containing a specific domain"""
    base_url = "https://www.bv-brc.org/api/protein_feature/"
    query = f"eq(description,*{domain_description}*)&select(patric_id,genome_id,start,end,e_value)"
    
    url = f"{base_url}?{query}&limit(100)"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    return {"error": f"Request failed: {response.status_code}"}
```

## Advanced Multi-Endpoint Workflows

### Comprehensive Genome Analysis Pipeline
```python
def comprehensive_genome_analysis(genome_id):
    """Complete analysis combining multiple endpoints"""
    analysis = {
        'genome_id': genome_id,
        'basic_info': {},
        'taxonomy': {},
        'genes': [],
        'subsystems': {},
        'specialty_genes': [],
        'amr_profile': {}
    }
    
    # 1. Get basic genome information
    genome_data = get_genome_by_id(genome_id)
    if genome_data and not genome_data.get('error'):
        analysis['basic_info'] = genome_data[0] if genome_data else {}
        organism_name = analysis['basic_info'].get('organism_name', '')
        
        # 2. Get taxonomic information
        if organism_name:
            analysis['taxonomy'] = get_lineage_for_organism(organism_name)
        
        # 3. Get gene features (sample)
        gene_data = get_genes_by_genome(genome_id, limit=10)
        if gene_data and not gene_data.get('error'):
            analysis['genes'] = gene_data
        
        # 4. Get subsystems
        analysis['subsystems'] = get_subsystems_for_genome(genome_id)
        
        # 5. Get specialty genes
        analysis['specialty_genes'] = get_antibiotic_resistance_genes(genome_id)
        
        # 6. Get AMR profile
        analysis['amr_profile'] = get_amr_profile(genome_id)
    
    return analysis

def functional_comparison(genome_ids):
    """Compare functional profiles across multiple genomes"""
    comparison = {}
    
    for genome_id in genome_ids:
        comparison[genome_id] = {
            'subsystems': get_subsystems_for_genome(genome_id),
            'amr_genes': get_antibiotic_resistance_genes(genome_id),
            'amr_profile': get_amr_profile(genome_id)
        }
    
    # Find common and unique subsystems
    all_subsystems = set()
    for data in comparison.values():
        if isinstance(data['subsystems'], dict):
            all_subsystems.update(data['subsystems'].keys())
    
    subsystem_matrix = {}
    for subsys in all_subsystems:
        subsystem_matrix[subsys] = {}
        for genome_id in genome_ids:
            has_subsystem = subsys in comparison[genome_id].get('subsystems', {})
            subsystem_matrix[subsys][genome_id] = has_subsystem
    
    comparison['subsystem_matrix'] = subsystem_matrix
    return comparison
```

### 11. Pathway Endpoint (`/api/pathway/`)

**Purpose**: Access metabolic pathway and enzyme information
**URL**: `https://www.bv-brc.org/api/pathway/`

#### Key Fields
- `accession` - Feature accession identifier
- `genome_name` - Full genome name
- `pathway_name` - Name of metabolic pathway
- `ec_number` - Enzyme Commission number
- `ec_description` - Enzyme description
- `product` - Enzyme product description
- `taxon_id` - NCBI taxonomy identifier
- `annotation` - Annotation source

#### Common Queries
```python
def get_pathways_for_genome(genome_name):
    """Get all pathways present in a genome"""
    base_url = "https://www.bv-brc.org/api/pathway/"
    query = f"eq(genome_name,*{genome_name}*)&select(pathway_name,ec_number,product)&limit(100)"
    
    url = f"{base_url}?{query}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        # Group by pathway
        pathways = {}
        for record in data:
            pathway = record.get('pathway_name', 'Unknown')
            if pathway not in pathways:
                pathways[pathway] = []
            pathways[pathway].append({
                'ec_number': record.get('ec_number'),
                'product': record.get('product')
            })
        return pathways
    return {"error": f"Request failed: {response.status_code}"}

def find_genomes_with_pathway(pathway_name):
    """Find genomes containing a specific pathway"""
    base_url = "https://www.bv-brc.org/api/pathway/"
    query = f"eq(pathway_name,*{pathway_name}*)&select(genome_name,taxon_id)&limit(50)"
    
    url = f"{base_url}?{query}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        # Remove duplicates
        genomes = {}
        for record in data:
            genome = record.get('genome_name')
            if genome:
                genomes[genome] = record.get('taxon_id')
        return [{'genome_name': k, 'taxon_id': v} for k, v in genomes.items()]
    return {"error": f"Request failed: {response.status_code}"}
```

---

This guide provides comprehensive coverage of the major BV-BRC REST API endpoints. The documented endpoints cover genome data, taxonomic information, functional classifications, specialty genes, antimicrobial resistance predictions, and metabolic pathways - enabling complex bioinformatics workflows and analyses.
# Mutation Variant Analysis MCP Server

## Overview

The Mutation Variant Analysis MCP Server provides comprehensive tools for cancer mutation analysis, protein structure discovery, pathway analysis, and therapy/clinical trial search. This server integrates with multiple biological databases to deliver detailed insights about genetic variants and their clinical significance.

## Features

- **ClinVar Integration**: Search for variant pathogenicity and clinical significance
- **Protein Structure Discovery**: Find experimental (PDB) and predicted (AlphaFold) protein structures
- **Pathway Analysis**: Retrieve KEGG biological pathways associated with genes
- **Therapy & Clinical Trial Search**: Find targeted therapies and active clinical trials
- **UniProt Integration**: Get canonical protein IDs for human genes

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

### 1. get_variant_details(query_string, gene_to_verify)

Search ClinVar database for variant details and pathogenicity assessment.

**Parameters:**
- `query_string` (str): Full variant notation including gene name
  - Examples: "BRCA1 c.185delAG", "TP53 R273H", "PIK3CA H1047R"
- `gene_to_verify` (str): Gene symbol to verify correct match
  - Examples: "BRCA1", "TP53", "PIK3CA"

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `clinvar_id`: ClinVar variant ID
- `official_name`: Official variant nomenclature
- `pathogenicity`: Clinical significance (e.g., "Pathogenic", "Benign")
- `review_status`: Review status from ClinVar
- `phenotypes`: Associated phenotypes/diseases
- `url`: Direct link to ClinVar entry

**Example:**
```python
get_variant_details("BRCA1 c.185delAG", "BRCA1")
```

### 2. get_uniprot_id(gene_name)

Get UniProt accession ID for a human gene.

**Parameters:**
- `gene_name` (str): Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `uniprot_id`: UniProt accession ID for canonical human protein isoform

**Example:**
```python
get_uniprot_id("BRCA1")
# Returns: {"status": "success", "uniprot_id": "P38398"}
```

### 3. get_protein_structure_info(gene_name)

Find protein structure information from PDB or AlphaFold databases.

**Parameters:**
- `gene_name` (str): Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `source`: "PDB (Experimental)" or "AlphaFold (Predicted)"
- `id`: Structure ID (PDB ID or UniProt ID)
- `data_url`: Direct download URL for structure file
- `viewer_url`: Web viewer URL for 3D visualization

**Search Priority:**
1. PDB database (experimental structures)
2. AlphaFold database (predicted structures)

**Example:**
```python
get_protein_structure_info("BRCA1")
```

### 4. get_pathways_for_gene(gene_name)

Find KEGG biological pathways associated with a gene.

**Parameters:**
- `gene_name` (str): Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `pathways`: Dictionary of pathway IDs and descriptive names

**Process:**
1. Retrieves UniProt ID for the gene
2. Converts UniProt ID to KEGG gene ID
3. Extracts associated biological pathways

**Example:**
```python
get_pathways_for_gene("BRCA1")
# Returns pathways like "Homologous recombination", "Fanconi anemia pathway"
```

### 5. search_therapies_and_trials(gene_name, disease_context)

Search for targeted therapies and clinical trials related to a gene and disease.

**Parameters:**
- `gene_name` (str): Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")
- `disease_context` (str, optional): Disease context (default: "cancer")
  - Examples: "breast cancer", "lung cancer", "ovarian cancer"

**Returns:**
JSON object containing:
- `status`: "success" or "error"
- `pubmed_articles`: List of relevant PubMed articles with PMID, title, and URL
- `clinical_trials`: List of active clinical trials with NCT ID, title, and URL

**Data Sources:**
- PubMed: Targeted therapy literature and clinical trial publications
- ClinicalTrials.gov: Actively recruiting clinical trials

**Example:**
```python
search_therapies_and_trials("BRCA1", "cancer")
# Returns PARP inhibitor studies and BRCA1-related trials

search_therapies_and_trials("EGFR", "lung cancer")
# Returns EGFR inhibitor trials specific to lung cancer
```

## API Endpoints & Data Sources

### External APIs Used
- **ClinVar**: Variant pathogenicity and clinical significance
  - Endpoint: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
- **UniProt**: Protein information and accession IDs
  - Endpoint: `https://rest.uniprot.org/uniprotkb/`
- **PDB**: Experimental protein structures
  - Endpoint: `https://search.rcsb.org/rcsbsearch/v2/query`
- **AlphaFold**: Predicted protein structures
  - Endpoint: `https://alphafold.ebi.ac.uk/api/`
- **KEGG**: Biological pathways
  - Endpoint: `http://rest.kegg.jp/`
- **PubMed**: Scientific literature
  - Endpoint: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
- **ClinicalTrials.gov**: Clinical trial information
  - Endpoint: `https://clinicaltrials.gov/api/v2/`

## Configuration

### Required Settings
```python
NCBI_API_EMAIL = "your.email@example.com"  # Replace with your email
NCBI_API_TOOL = "ai_gene_analysis_demo"
```

**Important**: Update the email address in the configuration to comply with NCBI API usage policies.

## Error Handling

The server implements comprehensive error handling:
- **Network Errors**: Graceful handling of API timeouts and connection issues
- **Data Validation**: Verification of gene names and variant formats
- **Rate Limiting**: Built-in delays to respect API rate limits
- **Fallback Mechanisms**: Alternative data sources when primary sources fail

## Usage Examples

### Complete Variant Analysis Workflow
```python
# 1. Search for variant details
variant_info = get_variant_details("BRCA1 c.185delAG", "BRCA1")

# 2. Get protein structure information
structure_info = get_protein_structure_info("BRCA1")

# 3. Find associated biological pathways
pathways = get_pathways_for_gene("BRCA1")

# 4. Search for therapies and clinical trials
therapies = search_therapies_and_trials("BRCA1", "breast cancer")
```

### Gene Analysis Pipeline
```python
# Analyze multiple aspects of a gene
gene = "TP53"
uniprot_id = get_uniprot_id(gene)
structure = get_protein_structure_info(gene)
pathways = get_pathways_for_gene(gene)
treatments = search_therapies_and_trials(gene, "cancer")
```

## Rate Limiting & Best Practices

- **API Delays**: 0.4-second delays between consecutive requests
- **Result Limits**: Maximum 3 results per search to optimize performance
- **Caching**: Consider implementing local caching for frequently accessed data
- **Email Requirements**: Always provide a valid email for NCBI API requests

## Output Format

All tools return JSON-formatted strings with consistent structure:
```json
{
  "status": "success|error",
  "message": "Descriptive message",
  "data": "Tool-specific results"
}
```

## Troubleshooting

### Common Issues
1. **Email Configuration**: Ensure NCBI_API_EMAIL is set to a valid email address
2. **Network Connectivity**: Verify internet access to external APIs
3. **Gene Symbol Format**: Use official HGNC gene symbols
4. **Variant Notation**: Follow standard variant nomenclature (HGVS format recommended)

### Debug Information
The server provides detailed logging with INFO and WARNING messages for debugging API interactions and data retrieval processes.

## License & Attribution

This server integrates with multiple public biological databases. Please ensure compliance with individual database usage policies and cite appropriate sources when using the data in research or publications.
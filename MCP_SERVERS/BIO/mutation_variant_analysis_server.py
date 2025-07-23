# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp",
#     "requests",
# ]
# ///

"""
Biology Analysis MCP Server

Provides tools for cancer mutation analysis, protein structure discovery,
pathway analysis, and therapy/clinical trial search.
"""

import requests
import time
import json
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("Biology Analysis")

# Configuration constants
NCBI_API_EMAIL = "your.email@example.com"
NCBI_API_TOOL = "ai_gene_analysis_demo"

# ===== HELPER FUNCTIONS =====

def get_uniprot_id_helper(gene_name: str) -> dict:
    """Helper function to get UniProt ID for a gene."""
    print(f"INFO: Getting UniProt ID for gene '{gene_name}'...")
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = { 
        "query": f'(gene_exact:"{gene_name}") AND (organism_id:9606) AND (reviewed:true)', 
        "fields": "accession,id", 
        "size": "1" 
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("results"):
            accession = data["results"][0]["primaryAccession"]
            print(f"INFO: Found canonical UniProt ID: {accession}")
            return {"status": "success", "uniprot_id": accession}
    except requests.exceptions.RequestException as e:
        print(f"WARNING: UniProt search failed: {e}")
    return {"status": "error", "message": f"Could not find a reviewed UniProt ID for {gene_name}."}

# ===== MCP TOOLS =====

@mcp.tool()
def get_variant_details(query_string: str, gene_to_verify: str) -> str:
    """Search ClinVar database for variant details and pathogenicity assessment.
    
    Args:
        query_string: Full variant notation including gene name (e.g., "BRCA1 c.185delAG", "TP53 R273H", "PIK3CA H1047R")
        gene_to_verify: Gene symbol to verify correct match (e.g., "BRCA1", "TP53", "PIK3CA")
    
    Returns:
        JSON with variant details including pathogenicity, ClinVar ID, review status, and associated phenotypes
        
    Example:
        get_variant_details("BRCA1 c.185delAG", "BRCA1") -> returns ClinVar data for this BRCA1 mutation
    """
    print(f"INFO: Searching ClinVar for '{query_string}'...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_params = { 
        "db": "clinvar", 
        "term": query_string, 
        "retmode": "json", 
        "retmax": "3", 
        "email": NCBI_API_EMAIL, 
        "tool": NCBI_API_TOOL 
    }
    
    try:
        search_response = requests.get(f"{base_url}esearch.fcgi", params=search_params)
        search_response.raise_for_status()
        search_data = search_response.json()
        id_list = search_data.get("esearchresult", {}).get("idlist")
        
        if not id_list: 
            return json.dumps({
                "query": query_string, 
                "status": "error", 
                "message": f"Variant '{query_string}' not found in ClinVar."
            })
        
        print(f"INFO: Found {len(id_list)} candidate(s): {id_list}")
    except requests.exceptions.RequestException as e: 
        return json.dumps({
            "status": "error", 
            "message": f"API request failed during search: {e}"
        })
    
    for variant_id in id_list:
        print(f"INFO: Checking candidate ID: {variant_id}...")
        time.sleep(0.4)
        summary_params = { 
            "db": "clinvar", 
            "id": variant_id, 
            "retmode": "json", 
            "email": NCBI_API_EMAIL, 
            "tool": NCBI_API_TOOL 
        }
        
        try:
            summary_response = requests.get(f"{base_url}esummary.fcgi", params=summary_params)
            summary_response.raise_for_status()
            summary_data = summary_response.json()
            result = summary_data.get("result", {})
            
            if not result or variant_id not in result: 
                print(f"WARNING: Could not retrieve summary for ID {variant_id}. Skipping.")
                continue
            
            variant_info = result[variant_id]
            official_name = variant_info.get("title", "N/A")
            gene_to_verify_upper = gene_to_verify.upper()
            
            if gene_to_verify_upper in official_name.upper():
                print(f"INFO: SUCCESS! ID {variant_id} verified for gene '{gene_to_verify_upper}'.")
                germline_info = variant_info.get("germline_classification", {})
                pathogenicity = germline_info.get("description", "N/A")
                review_status = germline_info.get("review_status", "N/A")
                
                if pathogenicity == "N/A":
                    cs = variant_info.get("clinical_significance_list", [{}])[0]
                    pathogenicity = cs.get("description", "N/A")
                    review_status = cs.get("review_status", "N/A")
                
                traits = variant_info.get("trait_set", [])
                phenotypes = sorted([t.get('trait_name') for t in traits if t.get('trait_name')])
                
                return json.dumps({ 
                    "query": query_string, 
                    "status": "success", 
                    "message": "Variant found and details retrieved.", 
                    "clinvar_id": variant_id, 
                    "official_name": official_name, 
                    "pathogenicity": pathogenicity, 
                    "review_status": review_status, 
                    "phenotypes": phenotypes, 
                    "url": f"https://www.ncbi.nlm.nih.gov/clinvar/variation/{variant_id}/" 
                })
            else: 
                print(f"INFO: ID {variant_id} is for a different gene ('{official_name}'). Checking next candidate.")
        except requests.exceptions.RequestException as e: 
            print(f"WARNING: API request for ID {variant_id} failed: {e}. Skipping.")
            continue
    
    return json.dumps({ 
        "query": query_string, 
        "status": "error", 
        "message": f"Could not find a verified match for gene '{gene_to_verify}' among the top search results." 
    })

@mcp.tool()
def get_uniprot_id(gene_name: str) -> str:
    """Get UniProt accession ID for a human gene.
    
    Args:
        gene_name: Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")
    
    Returns:
        JSON with UniProt accession ID for the canonical human protein isoform
        
    Example:
        get_uniprot_id("BRCA1") -> returns {"status": "success", "uniprot_id": "P38398"}
    """
    result = get_uniprot_id_helper(gene_name)
    return json.dumps(result)

@mcp.tool()
def get_protein_structure_info(gene_name: str) -> str:
    """Find protein structure information from PDB or AlphaFold databases.
    
    Args:
        gene_name: Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")
    
    Returns:
        JSON with structure information including source (PDB experimental or AlphaFold predicted),
        structure ID, data download URL, and web viewer URL
        
    Example:
        get_protein_structure_info("BRCA1") -> returns AlphaFold structure info for BRCA1 protein
        
    Note:
        Searches PDB first for experimental structures, falls back to AlphaFold predictions
    """
    print(f"INFO: Searching for protein structure for '{gene_name}'...")
    print("INFO: Attempting to find experimental structure in PDB...")
    
    pdb_url = f"https://search.rcsb.org/rcsbsearch/v2/query"
    query = { 
        "query": { 
            "type": "group", 
            "logical_operator": "and", 
            "nodes": [ 
                {"type": "terminal", "service": "text", "parameters": {"attribute": "rcsb_entity_source_organism.taxonomy_id", "operator": "exact_match", "value": "9606"}}, 
                {"type": "terminal", "service": "text", "parameters": {"attribute": "struct_gene.gene_name", "operator": "exact_match", "value": gene_name.upper()}} 
            ]
        }, 
        "return_type": "entry", 
        "request_options": {"pager": {"start": 0, "rows": 1}} 
    }
    
    try:
        response = requests.post(pdb_url, json=query)
        response.raise_for_status()
        pdb_data = response.json()
        if pdb_data.get("total_count", 0) > 0:
            structure_id = pdb_data["result_set"][0]["identifier"]
            print(f"INFO: SUCCESS! Found PDB ID: {structure_id}")
            return json.dumps({ 
                "status": "success", 
                "source": "PDB (Experimental)", 
                "id": structure_id, 
                "data_url": f"https://files.rcsb.org/download/{structure_id}.pdb", 
                "viewer_url": f"https://www.rcsb.org/3d-view/{structure_id}" 
            })
    except requests.exceptions.RequestException as e: 
        print(f"WARNING: PDB search request failed: {e}")
    
    print("INFO: Not found in PDB or PDB search failed. Trying AlphaFold (predicted structure)...")
    uniprot_result = get_uniprot_id_helper(gene_name)
    if uniprot_result["status"] == "success":
        uniprot_id = uniprot_result["uniprot_id"]
        alphafold_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
        try:
            response = requests.get(alphafold_url)
            if response.status_code == 200:
                alphafold_data = response.json()
                if alphafold_data and "errorMessage" not in str(alphafold_data):
                    cif_url = alphafold_data[0]['cifUrl']
                    print(f"INFO: SUCCESS! Found AlphaFold prediction for UniProt ID: {uniprot_id}")
                    return json.dumps({ 
                        "status": "success", 
                        "source": "AlphaFold (Predicted)", 
                        "id": uniprot_id, 
                        "data_url": cif_url, 
                        "viewer_url": f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}" 
                    })
        except requests.exceptions.RequestException as e: 
            print(f"WARNING: AlphaFold search request failed: {e}")
    
    return json.dumps({
        "status": "error", 
        "message": f"Could not find a structure for '{gene_name}' in PDB or AlphaFold."
    })

@mcp.tool()
def get_pathways_for_gene(gene_name: str) -> str:
    """Find KEGG biological pathways associated with a gene.
    
    Args:
        gene_name: Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")
    
    Returns:
        JSON with pathway information including pathway IDs and descriptive names
        
    Example:
        get_pathways_for_gene("BRCA1") -> returns pathways like "Homologous recombination", "Fanconi anemia pathway"
        
    Note:
        Uses UniProt -> KEGG conversion pipeline to find associated biological pathways
    """
    print(f"INFO: Searching for KEGG pathways for '{gene_name}'...")
    uniprot_result = get_uniprot_id_helper(gene_name)
    
    if uniprot_result["status"] != "success":
        return json.dumps({
            "status": "error", 
            "message": "Failed to get UniProt ID, cannot proceed with KEGG search."
        })
    
    uniprot_id = uniprot_result["uniprot_id"]
    try:
        conv_url = f"http://rest.kegg.jp/conv/genes/uniprot:{uniprot_id}"
        conv_response = requests.get(conv_url)
        conv_response.raise_for_status()
        
        if not conv_response.text.strip():
            return json.dumps({
                "status": "error", 
                "message": f"Could not convert UniProt ID {uniprot_id} to a KEGG ID."
            })
        
        kegg_id = conv_response.text.strip().split('\t')[1]
        print(f"INFO: Converted UniProt:{uniprot_id} to KEGG ID: {kegg_id}. Getting pathways...")
        
        get_url = f"http://rest.kegg.jp/get/{kegg_id}"
        get_response = requests.get(get_url)
        get_response.raise_for_status()
        
        pathways = {}
        in_pathway_section = False
        for line in get_response.text.strip().split('\n'):
            if line.startswith("PATHWAY"): 
                in_pathway_section = True
                line_content = line[12:].strip()
            elif in_pathway_section and line.startswith(" "): 
                line_content = line.strip()
            elif in_pathway_section: 
                break
            else: 
                continue
            
            parts = line_content.split('  ', 1)
            if len(parts) == 2:
                path_id, path_name = parts
                pathways[path_id] = path_name.replace(" - Homo sapiens (human)", "")
        
        if not pathways: 
            return json.dumps({
                "status": "error", 
                "message": f"No pathways found for KEGG ID '{kegg_id}'."
            })
        
        return json.dumps({"status": "success", "pathways": pathways})
        
    except requests.exceptions.RequestException as e: 
        return json.dumps({
            "status": "error", 
            "message": f"KEGG API request failed: {e}"
        })

@mcp.tool()
def search_therapies_and_trials(gene_name: str, disease_context: str = "cancer") -> str:
    """Search for targeted therapies and clinical trials related to a gene and disease.
    
    Args:
        gene_name: Official gene symbol (e.g., "BRCA1", "TP53", "PIK3CA")
        disease_context: Disease or condition context (default: "cancer", could be "breast cancer", "lung cancer", etc.)
    
    Returns:
        JSON with PubMed articles about targeted therapies and actively recruiting clinical trials
        
    Example:
        search_therapies_and_trials("BRCA1", "cancer") -> returns PARP inhibitor studies and BRCA1-related trials
        search_therapies_and_trials("EGFR", "lung cancer") -> returns EGFR inhibitor trials specific to lung cancer
        
    Note:
        Searches PubMed for targeted therapy literature and ClinicalTrials.gov for recruiting studies
    """
    print(f"INFO: Searching for therapies and trials for '{gene_name}' in the context of '{disease_context}'...")
    
    # Step 5: Search PubMed for targeted therapies
    pubmed_results = []
    try:
        # A more targeted query for review articles or clinical trials
        search_term = f'("{gene_name}"[Gene Name]) AND ("{disease_context}"[MeSH Terms]) AND (targeted therapy OR clinical trial[Publication Type])'
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={search_term}&retmax=3&retmode=json&sort=relevance"
        search_resp = requests.get(search_url)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        id_list = search_data.get("esearchresult", {}).get("idlist")
        
        if id_list:
            summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(id_list)}&retmode=json"
            summary_resp = requests.get(summary_url)
            summary_resp.raise_for_status()
            summary_data = summary_resp.json().get("result", {})
            for pmid, details in summary_data.items():
                if pmid == 'uids': 
                    continue
                pubmed_results.append({
                    "pmid": pmid,
                    "title": details.get("title", "N/A"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
    except requests.exceptions.RequestException as e:
        print(f"WARNING: PubMed search failed: {e}")

    # Step 6: Search ClinicalTrials.gov
    trials_results = []
    try:
        search_term = f"{gene_name}+{disease_context}"
        trials_url = f"https://clinicaltrials.gov/api/v2/studies?query.term={search_term}&filter.overallStatus=RECRUITING&pageSize=3"
        trials_resp = requests.get(trials_url)
        trials_resp.raise_for_status()
        trials_data = trials_resp.json().get("studies", [])
        for trial in trials_data:
            protocol = trial.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            trials_results.append({
                "nct_id": id_module.get("nctId", "N/A"),
                "title": protocol.get("briefTitle", "N/A"),
                "url": f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}"
            })
    except requests.exceptions.RequestException as e:
        print(f"WARNING: ClinicalTrials.gov search failed: {e}")

    return json.dumps({
        "status": "success",
        "pubmed_articles": pubmed_results,
        "clinical_trials": trials_results
    })

# Run the server when executed directly
if __name__ == "__main__":
    try:
        mcp.run(transport="stdio")
    except Exception as e:
        print(f"Error running server: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

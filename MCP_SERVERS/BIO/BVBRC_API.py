#!/usr/bin/env python3
"""
BV-BRC API MCP Server

This MCP server provides access to the BV-BRC (Bacterial and Viral Bioinformatics Resource Center) 
REST API through standardized tools. It implements the comprehensive functionality documented in 
the BVBRC_REST_API_GUIDE.md.

The server exposes ~26 tools covering:
- Genome queries and metadata
- Gene/feature annotations  
- Sequence retrieval
- Taxonomic information
- Functional analysis (subsystems, pathways)
- Specialty genes (AMR, virulence factors)
- Complex analytical workflows
"""

import json
import time
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from mcp.server.fastmcp import FastMCP

# BV-BRC API Configuration
BASE_URL = "https://www.bv-brc.org/api/"
HEADERS = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

# Initialize FastMCP
mcp = FastMCP("BV-BRC API Server")

# Utility Functions
async def safe_api_request(url: str, params: Optional[str] = None, timeout: int = 30) -> Dict[str, Any]:
    """Make API request with proper error handling"""
    try:
        full_url = f"{url}?{params}" if params else url
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(full_url, headers=HEADERS)
            
            if response.status_code == 200:
                data = response.json()
                return {"success": True, "data": data, "count": len(data)}
            else:
                return {
                    "success": False, 
                    "error": f"HTTP {response.status_code}", 
                    "message": response.text
                }
                
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timeout"}
    except httpx.RequestError as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}
    except ValueError as e:
        return {"success": False, "error": f"JSON parsing failed: {str(e)}"}

# Genome Query Tools

@mcp.tool()
async def get_genomes_by_species(species: str, limit: int = 50, offset: int = 0) -> str:
    """Get genomes for a specific species
    
    Args:
        species: Species name (e.g., 'Escherichia coli')
        limit: Maximum number of results (default: 50)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}genome/"
    species_query = quote(species)
    params = f"eq(species,{species_query})&limit({limit},{offset})"
    params += "&select(genome_id,genome_name,strain,genome_status,genome_length)"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_complete_genomes(species: str, limit: int = 20, offset: int = 0) -> str:
    """Get only complete genomes for a species
    
    Args:
        species: Species name
        limit: Maximum number of results (default: 20)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}genome/"
    species_query = quote(species)
    params = f"and(eq(species,{species_query}),eq(genome_status,Complete))&limit({limit},{offset})"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_genomes_by_ids(genome_ids: List[str]) -> str:
    """Get specific genomes by their IDs
    
    Args:
        genome_ids: List of genome IDs
    """
    url = f"{BASE_URL}genome/"
    ids_str = ','.join(genome_ids)
    params = f"in(genome_id,({ids_str}))"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def find_similar_genomes(target_species: str, target_strain: Optional[str] = None, 
                             min_genome_length: Optional[int] = None) -> str:
    """Find genomes similar to target criteria
    
    Args:
        target_species: Target species name
        target_strain: Target strain (optional)
        min_genome_length: Minimum genome length in bp (optional)
    """
    url = f"{BASE_URL}genome/"
    
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
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def genome_feature_summary(genome_id: str) -> str:
    """Generate comprehensive feature summary for a genome
    
    Args:
        genome_id: BV-BRC genome ID
    """
    # Get genome metadata
    genome_result = await get_genomes_by_ids([genome_id])
    genome_data = json.loads(genome_result)
    
    if not genome_data.get("success") or not genome_data.get("data"):
        return json.dumps({"error": "Genome not found"}, indent=2)
    
    genome_info = genome_data["data"][0]
    
    # Get all features
    features_result = await get_cds_features(genome_id, limit=10000)
    features_data = json.loads(features_result)
    
    if not features_data.get("success") or not features_data.get("data"):
        return json.dumps({"error": "No features found"}, indent=2)
    
    all_features = features_data["data"]
    
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
    
    return json.dumps(summary, indent=2)

# Gene/Feature Query Tools

@mcp.tool()
async def get_genes_by_name(gene_name: str, genome_ids: Optional[List[str]] = None, 
                          limit: int = 100, offset: int = 0) -> str:
    """Find genes by name across genomes
    
    Args:
        gene_name: Gene name/symbol
        genome_ids: Limit to specific genomes (optional)
        limit: Maximum number of results (default: 100)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}genome_feature/"
    gene_query = quote(gene_name)
    
    if genome_ids:
        ids_str = ','.join(genome_ids)
        params = f"and(in(genome_id,({ids_str})),eq(gene,{gene_query}))&limit({limit},{offset})"
    else:
        params = f"eq(gene,{gene_query})&limit({limit},{offset})"
    
    params += "&select(genome_id,patric_id,gene,product,start,end,strand)"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_cds_features(genome_id: str, limit: int = 1000, offset: int = 0) -> str:
    """Get all CDS features for a genome
    
    Args:
        genome_id: BV-BRC genome ID
        limit: Maximum number of results (default: 1000)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}genome_feature/"
    params = f"and(eq(genome_id,{genome_id}),eq(feature_type,CDS))&limit({limit},{offset})"
    params += "&select(patric_id,gene,product,start,end,strand,aa_length)"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def search_genes_by_product(product_keyword: str, genome_ids: Optional[List[str]] = None, 
                                 limit: int = 100, offset: int = 0) -> str:
    """Search genes by product description
    
    Args:
        product_keyword: Product description keyword
        genome_ids: Limit to specific genomes (optional)
        limit: Maximum number of results (default: 100)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}genome_feature/"
    keyword_query = quote(product_keyword)
    
    if genome_ids:
        ids_str = ','.join(genome_ids)
        base_query = f"in(genome_id,({ids_str}))"
        params = f"and({base_query},keyword({keyword_query}))&limit({limit},{offset})"
    else:
        params = f"keyword({keyword_query})&limit({limit},{offset})"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def analyze_gene_across_genomes(gene_name: str, species: str, max_genomes: int = 50) -> str:
    """Analyze a specific gene across multiple genomes
    
    Args:
        gene_name: Gene name/symbol
        species: Species to search within
        max_genomes: Maximum genomes to analyze (default: 50)
    """
    # Step 1: Find genomes for the species
    genomes_result = await get_complete_genomes(species, max_genomes)
    genomes_data = json.loads(genomes_result)
    
    if not genomes_data.get("success") or not genomes_data.get("data"):
        return json.dumps({"error": f"No genomes found for {species}"}, indent=2)
    
    genomes = genomes_data["data"]
    genome_ids = [g['genome_id'] for g in genomes]
    
    # Step 2: Find the gene in these genomes
    gene_features_result = await get_genes_by_name(gene_name, genome_ids)
    gene_features_data = json.loads(gene_features_result)
    
    if not gene_features_data.get("success") or not gene_features_data.get("data"):
        return json.dumps({"error": f"Gene {gene_name} not found in {species} genomes"}, indent=2)
    
    gene_features = gene_features_data["data"]
    
    # Step 3: Extract sequences for each gene instance
    gene_sequences = []
    for feature in gene_features:
        sequence_result = await extract_gene_sequence(
            feature['genome_id'],
            feature['start'],
            feature['end'],
            feature['strand']
        )
        sequence_data = json.loads(sequence_result)
        
        if sequence_data.get("success") and sequence_data.get("sequence"):
            sequence = sequence_data["sequence"]
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
    
    return json.dumps(analysis, indent=2)

@mcp.tool()
async def get_protein_domains(patric_id: str) -> str:
    """Get all protein domains for a specific gene
    
    Args:
        patric_id: PATRIC feature ID
    """
    url = f"{BASE_URL}protein_feature/"
    params = f"eq(patric_id,{patric_id})&select(description,start,end,e_value)"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def find_proteins_with_domain(domain_description: str, limit: int = 100, offset: int = 0) -> str:
    """Find proteins containing a specific domain
    
    Args:
        domain_description: Domain description to search for
        limit: Maximum number of results (default: 100)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}protein_feature/"
    params = f"eq(description,*{domain_description}*)&select(patric_id,genome_id,start,end,e_value)&limit({limit},{offset})"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

# Sequence Retrieval Tools

@mcp.tool()
async def get_genome_sequence(genome_id: str) -> str:
    """Get complete genome sequence
    
    Args:
        genome_id: BV-BRC genome ID
    """
    url = f"{BASE_URL}genome_sequence/"
    params = f"eq(genome_id,{genome_id})&select(sequence,topology,length,gc_content)"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def extract_gene_sequence(genome_id: str, start: int, end: int, strand: str) -> str:
    """Extract gene sequence using coordinates
    
    Args:
        genome_id: BV-BRC genome ID
        start: Start coordinate (1-based)
        end: End coordinate (1-based)
        strand: Strand orientation (fwd or rev)
    """
    # Get full genome sequence
    genome_result = await get_genome_sequence(genome_id)
    genome_data = json.loads(genome_result)
    
    if not genome_data.get("success") or not genome_data.get("data") or len(genome_data["data"]) == 0:
        return json.dumps({"success": False, "error": "Genome sequence not found"}, indent=2)
    
    genome_seq = genome_data["data"][0].get('sequence')
    if not genome_seq:
        return json.dumps({"success": False, "error": "Sequence field not found"}, indent=2)
    
    # Extract subsequence (convert to 0-based indexing)
    gene_seq = genome_seq[start-1:end]
    
    # Handle reverse strand
    if strand == 'rev':
        complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        gene_seq = ''.join(complement.get(base.upper(), base) for base in gene_seq[::-1])
    
    result = {
        "success": True,
        "sequence": gene_seq,
        "length": len(gene_seq),
        "strand": strand,
        "coordinates": f"{start}-{end}"
    }
    
    return json.dumps(result, indent=2)

# Taxonomy Tools

@mcp.tool()
async def get_taxonomy_info(taxon_id: Optional[str] = None, taxon_name: Optional[str] = None) -> str:
    """Retrieve taxonomic information by ID or name
    
    Args:
        taxon_id: NCBI taxonomy ID (optional)
        taxon_name: Taxonomic name (optional)
    """
    if not taxon_id and not taxon_name:
        return json.dumps({"error": "Must provide either taxon_id or taxon_name"}, indent=2)
    
    url = f"{BASE_URL}taxonomy/"
    
    if taxon_id:
        query = f"eq(taxon_id,{taxon_id})"
    else:
        query = f"eq(taxon_name,*{taxon_name}*)"
    
    params = f"{query}&limit(10)"
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_lineage_for_organism(organism_name: str) -> str:
    """Get complete taxonomic lineage for an organism
    
    Args:
        organism_name: Organism name
    """
    url = f"{BASE_URL}taxonomy/"
    params = f"eq(taxon_name,*{organism_name}*)&select(taxon_name,lineage_names,lineage_ranks)&limit(5)"
    
    result = await safe_api_request(url, params)
    
    # Process result to create lineage
    if result.get("success") and result.get("data") and len(result["data"]) > 0:
        data = result["data"][0]
        lineage = list(zip(data.get('lineage_names', []), data.get('lineage_ranks', [])))
        processed_result = {
            "success": True,
            "organism": data.get('taxon_name'),
            "lineage": lineage
        }
        return json.dumps(processed_result, indent=2)
    
    return json.dumps({"error": "Organism not found"}, indent=2)

# Functional Analysis Tools

@mcp.tool()
async def get_subsystems_for_genome(genome_id: str) -> str:
    """Get all subsystems present in a genome
    
    Args:
        genome_id: BV-BRC genome ID
    """
    url = f"{BASE_URL}subsystem/"
    params = f"eq(genome_id,{genome_id})&select(subsystem_name,role_name,product)&limit(100)"
    
    result = await safe_api_request(url, params)
    
    # Group by subsystem
    if result.get("success") and result.get("data"):
        data = result["data"]
        subsystems = {}
        for record in data:
            subsys_name = record.get('subsystem_name', 'Unknown')
            if subsys_name not in subsystems:
                subsystems[subsys_name] = []
            subsystems[subsys_name].append({
                'role': record.get('role_name'),
                'product': record.get('product')
            })
        processed_result = {
            "success": True,
            "subsystems": subsystems,
            "count": len(subsystems)
        }
        return json.dumps(processed_result, indent=2)
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def find_genomes_with_subsystem(subsystem_name: str, limit: int = 50, offset: int = 0) -> str:
    """Find genomes containing a specific subsystem
    
    Args:
        subsystem_name: Subsystem name to search for
        limit: Maximum number of results (default: 50)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}subsystem/"
    params = f"eq(subsystem_name,*{subsystem_name}*)&select(genome_id,genome_name)&limit({limit},{offset})"
    
    result = await safe_api_request(url, params)
    
    # Remove duplicates
    if result.get("success") and result.get("data"):
        data = result["data"]
        genomes = {}
        for record in data:
            gid = record.get('genome_id')
            if gid:
                genomes[gid] = record.get('genome_name')
        processed_result = {
            "success": True,
            "genomes": [{'genome_id': k, 'genome_name': v} for k, v in genomes.items()],
            "count": len(genomes)
        }
        return json.dumps(processed_result, indent=2)
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_pathways_for_genome(genome_name: str) -> str:
    """Get all pathways present in a genome
    
    Args:
        genome_name: Genome name (not ID)
    """
    url = f"{BASE_URL}pathway/"
    params = f"eq(genome_name,*{genome_name}*)&select(pathway_name,ec_number,product)&limit(100)"
    
    result = await safe_api_request(url, params)
    
    # Group by pathway
    if result.get("success") and result.get("data"):
        data = result["data"]
        pathways = {}
        for record in data:
            pathway = record.get('pathway_name', 'Unknown')
            if pathway not in pathways:
                pathways[pathway] = []
            pathways[pathway].append({
                'ec_number': record.get('ec_number'),
                'product': record.get('product')
            })
        processed_result = {
            "success": True,
            "pathways": pathways,
            "count": len(pathways)
        }
        return json.dumps(processed_result, indent=2)
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def find_genomes_with_pathway(pathway_name: str, limit: int = 50, offset: int = 0) -> str:
    """Find genomes containing a specific pathway
    
    Args:
        pathway_name: Pathway name to search for
        limit: Maximum number of results (default: 50)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}pathway/"
    params = f"eq(pathway_name,*{pathway_name}*)&select(genome_name,taxon_id)&limit({limit},{offset})"
    
    result = await safe_api_request(url, params)
    
    # Remove duplicates
    if result.get("success") and result.get("data"):
        data = result["data"]
        genomes = {}
        for record in data:
            genome = record.get('genome_name')
            if genome:
                genomes[genome] = record.get('taxon_id')
        processed_result = {
            "success": True,
            "genomes": [{'genome_name': k, 'taxon_id': v} for k, v in genomes.items()],
            "count": len(genomes)
        }
        return json.dumps(processed_result, indent=2)
    
    return json.dumps(result, indent=2)

# Specialty Genes Tools

@mcp.tool()
async def get_antibiotic_resistance_genes(genome_id: str) -> str:
    """Get antibiotic resistance genes for a genome
    
    Args:
        genome_id: BV-BRC genome ID
    """
    url = f"{BASE_URL}sp_gene/"
    params = f"eq(genome_id,{genome_id})&eq(property,*Antibiotic*)&select(gene,product,antibiotics,evidence)&limit(100)"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def find_genes_for_antibiotic(antibiotic_name: str, limit: int = 50, offset: int = 0) -> str:
    """Find genes associated with resistance to specific antibiotic
    
    Args:
        antibiotic_name: Antibiotic name
        limit: Maximum number of results (default: 50)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}sp_gene/"
    params = f"eq(antibiotics,*{antibiotic_name}*)&select(gene,organism,product,genome_id)&limit({limit},{offset})"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

@mcp.tool()
async def get_virulence_factors(genome_id: str) -> str:
    """Get virulence factor genes for a genome
    
    Args:
        genome_id: BV-BRC genome ID
    """
    url = f"{BASE_URL}sp_gene/"
    params = f"eq(genome_id,{genome_id})&eq(property,*Virulence*)&select(gene,product,evidence)&limit(100)"
    
    result = await safe_api_request(url, params)
    return json.dumps(result, indent=2)

# AMR Analysis Tools

@mcp.tool()
async def get_amr_profile(genome_id: str) -> str:
    """Get complete AMR profile for a genome
    
    Args:
        genome_id: BV-BRC genome ID
    """
    url = f"{BASE_URL}genome_amr/"
    params = f"eq(genome_id,{genome_id})&select(antibiotic,resistant_phenotype,computational_method_performance)&limit(100)"
    
    result = await safe_api_request(url, params)
    
    # Organize by antibiotic
    if result.get("success") and result.get("data"):
        data = result["data"]
        amr_profile = {}
        for record in data:
            antibiotic = record.get('antibiotic')
            if antibiotic:
                amr_profile[antibiotic] = {
                    'phenotype': record.get('resistant_phenotype'),
                    'performance': record.get('computational_method_performance', {})
                }
        processed_result = {
            "success": True,
            "amr_profile": amr_profile,
            "antibiotic_count": len(amr_profile)
        }
        return json.dumps(processed_result, indent=2)
    
    return json.dumps(result, indent=2)

@mcp.tool()
async def compare_amr_methods(limit: int = 1000, offset: int = 0) -> str:
    """Compare performance of different AMR prediction methods
    
    Args:
        limit: Sample size for comparison (default: 1000)
        offset: Starting offset for pagination (default: 0)
    """
    url = f"{BASE_URL}genome_amr/"
    params = f"select(computational_method,computational_method_performance)&limit({limit},{offset})"
    
    result = await safe_api_request(url, params)
    
    if result.get("success") and result.get("data"):
        data = result["data"]
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
        
        processed_result = {
            "success": True,
            "method_comparison": avg_performance,
            "total_samples": len(data)
        }
        return json.dumps(processed_result, indent=2)
    
    return json.dumps(result, indent=2)

# Complex Workflow Tools

@mcp.tool()
async def comprehensive_genome_analysis(genome_id: str) -> str:
    """Complete analysis combining multiple endpoints
    
    Args:
        genome_id: BV-BRC genome ID
    """
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
    genome_result = await get_genomes_by_ids([genome_id])
    genome_data = json.loads(genome_result)
    if genome_data.get("success") and genome_data.get("data"):
        analysis['basic_info'] = genome_data["data"][0] if genome_data["data"] else {}
        organism_name = analysis['basic_info'].get('organism_name', '')
        
        # 2. Get taxonomic information
        if organism_name:
            taxonomy_result = await get_lineage_for_organism(organism_name)
            analysis['taxonomy'] = json.loads(taxonomy_result)
        
        # 3. Get gene features (sample)
        gene_result = await get_cds_features(genome_id, limit=10)
        gene_data = json.loads(gene_result)
        if gene_data.get("success") and gene_data.get("data"):
            analysis['genes'] = gene_data["data"]
        
        # 4. Get subsystems
        subsystems_result = await get_subsystems_for_genome(genome_id)
        analysis['subsystems'] = json.loads(subsystems_result)
        
        # 5. Get specialty genes
        specialty_result = await get_antibiotic_resistance_genes(genome_id)
        analysis['specialty_genes'] = json.loads(specialty_result)
        
        # 6. Get AMR profile
        amr_result = await get_amr_profile(genome_id)
        analysis['amr_profile'] = json.loads(amr_result)
    
    return json.dumps(analysis, indent=2)

@mcp.tool()
async def functional_comparison(genome_ids: List[str]) -> str:
    """Compare functional profiles across multiple genomes
    
    Args:
        genome_ids: List of genome IDs to compare
    """
    comparison = {}
    
    for genome_id in genome_ids:
        subsystems_result = await get_subsystems_for_genome(genome_id)
        amr_genes_result = await get_antibiotic_resistance_genes(genome_id)
        amr_profile_result = await get_amr_profile(genome_id)
        
        comparison[genome_id] = {
            'subsystems': json.loads(subsystems_result),
            'amr_genes': json.loads(amr_genes_result),
            'amr_profile': json.loads(amr_profile_result)
        }
    
    # Find common and unique subsystems
    all_subsystems = set()
    for data in comparison.values():
        subsystems_data = data['subsystems']
        if subsystems_data.get("success") and isinstance(subsystems_data.get('subsystems'), dict):
            all_subsystems.update(subsystems_data['subsystems'].keys())
    
    subsystem_matrix = {}
    for subsys in all_subsystems:
        subsystem_matrix[subsys] = {}
        for genome_id in genome_ids:
            subsystems_data = comparison[genome_id]['subsystems']
            has_subsystem = False
            if (subsystems_data.get("success") and 
                isinstance(subsystems_data.get('subsystems'), dict)):
                has_subsystem = subsys in subsystems_data['subsystems']
            subsystem_matrix[subsys][genome_id] = has_subsystem
    
    comparison['subsystem_matrix'] = subsystem_matrix
    comparison['summary'] = {
        'genomes_compared': len(genome_ids),
        'total_subsystems_found': len(all_subsystems)
    }
    
    return json.dumps(comparison, indent=2)

if __name__ == "__main__":
    mcp.run()
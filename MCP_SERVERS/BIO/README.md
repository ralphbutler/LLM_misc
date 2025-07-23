# Bioinformatics MCP Server Collection

This repository contains MCP (Model Context Protocol) servers providing programmatic access to major bioinformatics platforms: **BV-BRC** (Bacterial and Viral Bioinformatics Resource Center) and **P3-Tools**.

## 🧬 BV-BRC API Server

**File:** `BVBRC_API.py` | **Documentation:** `BVBRC_API.md`

REST API-based MCP server providing access to the BV-BRC platform's comprehensive microbial genomics database.

**Key Features:**
- **Genome queries** - Search and retrieve genome metadata
- **Gene/feature annotations** - Access detailed gene information
- **Sequence retrieval** - Download genome and gene sequences  
- **Taxonomic information** - Navigate microbial taxonomy
- **Functional analysis** - Explore subsystems and metabolic pathways
- **Specialty genes** - Query antimicrobial resistance (AMR) and virulence factors
- **Complex workflows** - Multi-step analytical pipelines

**Tools Available:** ~26 specialized tools covering the complete BV-BRC REST API

## 🛠️ P3-Tools Server Collection

**Files:** `P3_TOOLS_*.py` | **Documentation:** `P3_TOOLS_*.md` + `P3_TOOLS_GUIDE.md`

Subprocess-based MCP servers providing access to the P3-Tools command-line suite for advanced bioinformatics analysis.

### P3-Tools Components

1. **Data Retrieval** (`P3_TOOLS_DATA_RETRIEVAL.py`)
   - Genome and feature data queries
   - Batch data processing
   - ID-based record retrieval

2. **Computational Services** (`P3_TOOLS_COMPUTATIONAL.py`)
   - BLAST similarity searches
   - Phylogenetic tree construction
   - Genome assembly and annotation
   - RNA-Seq and variation analysis

3. **Specialized Analysis** (`P3_TOOLS_SPECIALIZED.py`)
   - K-mer analysis and genome distances
   - Protein family signatures
   - Co-occurrence pattern detection
   - Direct BLAST execution

4. **Utilities** (`P3_TOOLS_UTILITIES.py`)
   - File management and workspace operations
   - Authentication and session management
   - Data formatting and processing

**Total Tools:** 73+ tools across all categories

## 🚀 Usage

These MCP servers are designed to integrate with Claude and other AI systems to provide seamless access to bioinformatics data and analysis capabilities. Each server handles authentication, error management, and provides structured JSON responses.

## 📖 Documentation

Each component includes comprehensive documentation covering API endpoints, programming patterns, authentication methods, and complete code examples for building bioinformatics applications.
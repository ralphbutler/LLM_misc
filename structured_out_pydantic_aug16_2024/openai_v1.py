import os

from enum import Enum
from typing import Union

from pydantic import BaseModel

from openai import OpenAI

class DomainNames(str, Enum):
    """
    Enumeration of domain names.

    This enum provides different domains of organisms that can be referenced.
    """
    bacteria = "Bacteria"
    archaea = "Archaea"
    eukarya = "Eukarya"

class PhylumNames(str, Enum):
    """
    Enumeration of phylum names.

    This enum represents different phylum names that can be referenced.
    """
    actinobacteria = "Actinobacteria"
    proteobacteria = "Proteobacteria"
    cyanobacteria = "Cyanobacteria"

class Contig(BaseModel):
    """
    Represents a single contig.

    This model captures basic information about a single contig in a genome including contig name, and the name of the genome it occurs in.
    """
    name: str
    genome_name: str

class Genome(BaseModel):
    """
    Represents a single genome.

    This model captures basic information about a genome including name, number of contigs, domain, phylum, and list of contigs in the genome.
    """
    name: str
    domain: str
    phylum: str
    num_contigs: str
    contigs: list[Contig]

class Feature(BaseModel):
    """
    Represents a single feature of a genome.

    This model captures basic information about a feature of a genome including the name of the genome it is in, the contig it lies on, the strand (+ or -), location, and associated protein.
    """
    genome_name: str
    contig: str
    strand: str
    location: str
    protein: str

class Protein(BaseModel):
    """
    Represents a single protein.

    This model captures basic information about a protein including name, product, and number of amino acids.
    """
    name: str
    product: str
    num_amino_acids: str

class QueryResponse(BaseModel):
    """
    Represents the structured response for a query for info about genomes and their attriburtes.

    This model aggregates information about genomes, features of each genome, proteins, and contigs on each genome.  It should also include a summary of the text being queried.
    """
    genomes: list[Genome]
    features: list[Feature]
    proteins: list[Protein]
    contigs: list[Contig]
    summary: str


with open("511145.12.first8") as f:
    text = f.read()

client = OpenAI()

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that scans text for \
        info about genomes, features, proteins, and contigs."},
        {"role": "user", "content": text},
    ],
    response_format=QueryResponse,
)

message = completion.choices[0].message
if message.parsed:
    print(message.parsed.genomes)
    print("-"*50)
    print(message.parsed.features)
    print("-"*50)
    print(f"\n\nSummary: {message.parsed.summary}")
    print("-"*50)
    print("GENOMES LEN", len(message.parsed.genomes))
    for genome in message.parsed.genomes:
        print("GENOME:", genome)
    print("-"*50)
    print("FEATURES LEN", len(message.parsed.features))
    for feature in message.parsed.features:
        print("FEATURE:", feature)
    print("-"*50)
    print("CONTIGS LEN", len(message.parsed.contigs))
    for contig in message.parsed.contigs:
        print("CONTIG:", contig)
    print("-"*50)
    print("PROTEINS LEN", len(message.parsed.proteins))
    for protein in message.parsed.proteins:
        print("PROTEIN:", protein)
else:
    print(message.refusal)

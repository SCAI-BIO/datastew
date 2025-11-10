"""
Example: Export mappings from an in-memory Weaviate repository to JSONL

This script demonstrates how to:
1) Initialize an in-memory WeaviateRepository (no external service needed).
2) Create a small SNOMED CT terminology with several concepts.
3) Generate embeddings and store Mapping objects.
4) Export the repository contents to JSONL for downstream use.

Run:
    python examples/export_weaviate_jsonl.py
"""

from datastew.embedding import Vectorizer
from datastew.process.jsonl_adapter import WeaviateJsonlConverter
from datastew.repository import WeaviateRepository
from datastew.repository.model import Concept, Mapping, Terminology

# --------------------------------------------------------------------
# 1) Repository (in-memory) and converter
# --------------------------------------------------------------------
repo = WeaviateRepository(mode="memory", path="localhost", port=8080)
converter = WeaviateJsonlConverter(dest_dir="export")

# --------------------------------------------------------------------
# 2) Vectorizer (use default or specify a model/API key if desired)
# --------------------------------------------------------------------
# Example for OpenAI:
# vectorizer = Vectorizer("text-embedding-3-small", api_key="your_openai_api_key")
vectorizer = Vectorizer()

# --------------------------------------------------------------------
# 3) Define terminology, concepts, and mappings
# --------------------------------------------------------------------
terminology = Terminology("snomed CT", "SNOMED")

# Concept ID -> Preferred label
concepts_dict = {
    "11893007": "Diabetes mellitus (disorder)",
    "73211009": "Hypertension (disorder)",
    "195967001": "Asthma",
    "22298006": "Heart attack",
    "13260007": "Common cold",
    "422504002": "Stroke",
    "386098009": "Migraine",
    "57386000": "Influenza",
    "399206004": "Osteoarthritis",
    "386584008": "Depression",
}

concepts = []
mappings = []

for cid, label in concepts_dict.items():
    concept = Concept(terminology, pref_label=label, concept_identifier=cid)
    embedding = vectorizer.get_embedding(label)
    mapping = Mapping(
        concept=concept,
        text=label,
        embedding=embedding,
        sentence_embedder=vectorizer.model_name,
    )
    concepts.append(concept)
    mappings.append(mapping)

# --------------------------------------------------------------------
# 4) Store data and export to JSONL
# --------------------------------------------------------------------
try:
    repo.store_all([terminology, *concepts, *mappings])
    converter.from_repository(repo)
    print("Export complete. JSONL files written to 'export/'.")
finally:
    repo.shut_down()

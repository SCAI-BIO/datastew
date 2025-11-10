"""
Example: Store a small SNOMED CT terminology in PostgreSQL using datastew

This example shows how to:
1- Start a local PostgreSQL database (via Docker)
2- Initialize datastew with a vectorizer and repository
3- Insert a small set of medical concepts and text embeddings

---

Quick Start

# 1. Run PostgreSQL locally (in a new terminal):
docker run -d \
  --name datastew-postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  postgres:15

# 2. Run this script:
python examples/store_snomed_baseline.py

# 3. (Optional) Retrieve embeddings later using:
python examples/get_closest_mappings.py
"""

from datastew.embedding import Vectorizer
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import Concept, Mapping, Terminology

# --------------------------------------------------------------------
# 1) Connect to PostgreSQL
# --------------------------------------------------------------------
POSTGRES_USER = "user"
POSTGRES_PASSWORD = "password"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DB = "testdb"

connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Use OpenAI embeddings if you have an API key for higher-quality results:
# vectorizer = Vectorizer("text-embedding-3-small", key="your_openai_api_key")
vectorizer = Vectorizer()
repository = PostgreSQLRepository(connection_string, vectorizer=vectorizer)

# --------------------------------------------------------------------
# 2) Define a terminology namespace
# --------------------------------------------------------------------
terminology = Terminology("snomed CT", "SNOMED")

# --------------------------------------------------------------------
# 3) Define example medical concepts
# --------------------------------------------------------------------
concept_texts = {
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

# --------------------------------------------------------------------
# 4) Convert to datastew concepts + mappings
# --------------------------------------------------------------------
concepts, mappings = [], []
for concept_id, label in concept_texts.items():
    concept = Concept(terminology, label, concept_id)
    embedding = vectorizer.get_embedding(label)
    mapping = Mapping(concept, label, embedding, vectorizer.model_name)
    concepts.append(concept)
    mappings.append(mapping)

# --------------------------------------------------------------------
# 5) Store everything in the repository
# --------------------------------------------------------------------
repository.store_all([terminology, *concepts, *mappings])

print(f"Stored {len(concepts)} concepts and mappings in PostgreSQL.")
print("You can now query them using `repository.get_closest_mappings()`.")

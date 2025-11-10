"""
Example: Retrieve the closest text embeddings and their similarities.

This script shows how to:
1. Start a local PostgreSQL instance via Docker.
2. Store two simple SNOMED CT concepts in the database.
3. Retrieve the closest mappings for an input phrase.

---

Quick start

# Start PostgreSQL via Docker:
docker run -d \
  --name datastew-postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# Run this script:
python examples/get_closest_mappings.py
"""

from datastew.embedding import Vectorizer
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import Concept, Mapping, MappingResult, Terminology

# --------------------------------------------------------------------
# 1) Connect to PostgreSQL
# --------------------------------------------------------------------
POSTGRES_USER = "user"
POSTGRES_PASSWORD = "password"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DB = "testdb"

connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# You can use an OpenAI model if you have an API key:
# vectorizer = Vectorizer("text-embedding-3-small", api_key="your_openai_api_key")
vectorizer = Vectorizer()
repository = PostgreSQLRepository(connection_string, vectorizer=vectorizer)

# --------------------------------------------------------------------
# 2) Add a small SNOMED CT baseline
# --------------------------------------------------------------------
terminology = Terminology("snomed CT", "SNOMED")

text1 = "Diabetes mellitus (disorder)"
concept1 = Concept(terminology, text1, "Concept ID: 11893007")
mapping1 = Mapping(concept1, text1, vectorizer.get_embedding(text1), vectorizer.model_name)

text2 = "Hypertension (disorder)"
concept2 = Concept(terminology, text2, "Concept ID: 73211009")
mapping2 = Mapping(concept2, text2, vectorizer.get_embedding(text2), vectorizer.model_name)

repository.store_all([terminology, concept1, mapping1, concept2, mapping2])

# --------------------------------------------------------------------
# 3) Find closest mappings for a new phrase
# --------------------------------------------------------------------
query_text = "Sugar sickness"  # semantically similar to "Diabetes mellitus (disorder)"
embedding = vectorizer.get_embedding(query_text)
results = repository.get_closest_mappings(embedding, similarities=True, limit=2)

# --------------------------------------------------------------------
# 4) Display the results
# --------------------------------------------------------------------
print(f'Query: "{query_text}"\n')
for r in results:
    # If similarities=True, repo returns MappingResult; else Mapping.
    if isinstance(r, MappingResult):
        print(r)
    else:
        print(str(r))

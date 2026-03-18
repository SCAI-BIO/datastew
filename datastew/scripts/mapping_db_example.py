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
from datastew.repository.model import MappingResult

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
terminology = repository.add_terminology(name="snomed CT", short_name="SNOMED")

text1 = "Diabetes mellitus (disorder)"
concept1 = repository.add_concept(
    terminology_id=terminology.id, pref_label=text1, concept_identifier="Concept ID: 11893007"
)
repository.add_mapping(concept_id=concept1.id, text=text1)

text2 = "Hypertension (disorder)"
concept2 = repository.add_concept(
    terminology_id=terminology.id, pref_label=text2, concept_identifier="Concept ID: 73211009"
)
repository.add_mapping(concept_id=concept2.id, text=text2)

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
    if isinstance(r, MappingResult):
        print(r)
    else:
        print(str(r))

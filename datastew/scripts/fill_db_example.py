"""
Example: Store a small SNOMED CT terminology in PostgreSQL using datastew

This example shows how to:
1- Start a local PostgreSQL database (via Docker)
2- Initialize the database schema
3- Inject a session into the repository
4- Insert a small set of medical concepts and text embeddings

---

Quick Start

# 1. Run PostgreSQL locally (in a new terminal):
docker run -d \
  --name datastew-postgres \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  pgvector/pgvector:pg17

# 2. Run this script:
python examples/store_snomed_baseline.py

# 3. (Optional) Retrieve embeddings later using:
python examples/get_closest_mappings.py
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datastew.embedding import Vectorizer
from datastew.repository import PostgreSQLRepository

# --------------------------------------------------------------------
# 1) Connect to PostgreSQL and Initialize Schema
# --------------------------------------------------------------------
POSTGRES_USER = "user"
POSTGRES_PASSWORD = "password"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DB = "testdb"

# Use the psycopg3 driver
connection_string = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(connection_string)

# Initialize pgvector extension and tables (run once)
PostgreSQLRepository.setup_database(engine)

SessionLocal = sessionmaker(bind=engine, autoflush=False)

# Use OpenAI embeddings if you have an API key for higher-quality results:
# vectorizer = Vectorizer("text-embedding-3-small", api_key="your_openai_api_key")
vectorizer = Vectorizer()

# --------------------------------------------------------------------
# 2) Execute within a Session Context
# --------------------------------------------------------------------
with SessionLocal() as session:
    repository = PostgreSQLRepository(session=session, vectorizer=vectorizer)

    # --------------------------------------------------------------------
    # 3) Define a terminology namespace
    # --------------------------------------------------------------------
    terminology = repository.add_terminology(name="snomed CT", short_name="SNOMED")

    # --------------------------------------------------------------------
    # 4) Define example medical concepts
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
    # 5) Convert to datastew concepts + mappings
    # --------------------------------------------------------------------
    for concept_id, label in concept_texts.items():
        # Construct a globally unique string identifier
        identifier = f"SNOMED:{concept_id}"
        concept = repository.add_concept(
            terminology_id=terminology.id, pref_label=label, concept_identifier=identifier
        )
        repository.add_mapping(concept_id=concept.id, text=label)

    # Commit the transaction to persist the changes
    session.commit()

print(f"Stored {len(concept_texts)} concepts and mappings in PostgreSQL.")
print("You can now query them using `repository.get_closest_mappings()`.")

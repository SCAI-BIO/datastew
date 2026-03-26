"""
Example: Import a terminology from the Ontology Lookup Service (OLS) into PostgreSQL

This example demonstrates how to:
1. Start a local PostgreSQL instance via Docker.
2. Initialize the database schema and inject a session.
3. Initialize an embedding model for generating vector embeddings.
4. Import a terminology (for example, SNOMED CT) from the OLS service
   directly into the PostgreSQL database.

Before running this example:
- Ensure a PostgreSQL instance is running locally (for example, via Docker):

    docker run -d \
      --name datastew-postgres \
      -e POSTGRES_USER=user \
      -e POSTGRES_PASSWORD=password \
      -e POSTGRES_DB=testdb \
      -p 5432:5432 \
      pgvector/pgvector:pg17
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datastew.embedding import Vectorizer
from datastew.integrations.ols import OlsClient
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
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@" f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(connection_string)

# Initialize pgvector extension and tables (run once)
PostgreSQLRepository.setup_database(engine)

SessionLocal = sessionmaker(bind=engine, autoflush=False)
vectorizer = Vectorizer()

# --------------------------------------------------------------------
# 2) Execute within a Session Context
# --------------------------------------------------------------------
with SessionLocal() as session:
    repository = PostgreSQLRepository(session=session, vectorizer=vectorizer)

    # --------------------------------------------------------------------
    # 3) Import terminology from OLS
    # --------------------------------------------------------------------
    # The first argument is the embedding model.
    # The second argument is the OLS identifier of the terminology.
    task = OlsClient(vectorizer=vectorizer, ontology_id="snomed")

    # This method fetches, embeds, and uploads the terminology to PostgreSQL.
    task.process_to_repository(repository)

    # Commit the transaction to persist the changes
    session.commit()

print("Terminology import completed successfully.")

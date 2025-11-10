"""
Example: Import a terminology from the Ontology Lookup Service (OLS) into PostgreSQL

This example demonstrates how to:
1. Connect to a PostgreSQL database using the datastew repository.
2. Initialize an embedding model for generating vector embeddings.
3. Import a terminology (for example, SNOMED CT) from the OLS service
   directly into the PostgreSQL database.

Before running this example:
- Ensure a PostgreSQL instance is running locally (for example, via Docker):

    docker run -d \
      --name datastew-postgres \
      -e POSTGRES_USER=user \
      -e POSTGRES_PASSWORD=password \
      -e POSTGRES_DB=testdb \
      -p 5432:5432 \
      postgres:15
"""

from datastew.embedding import Vectorizer
from datastew.process.ols import OLSTerminologyImportTask
from datastew.repository import PostgreSQLRepository

# --------------------------------------------------------------------
# 1) Connect to PostgreSQL
# --------------------------------------------------------------------
POSTGRES_USER = "user"
POSTGRES_PASSWORD = "password"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_DB = "testdb"

connection_string = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@" f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
vectorizer = Vectorizer()
repository = PostgreSQLRepository(connection_string, vectorizer)

# --------------------------------------------------------------------
# 3) Import terminology from OLS
# --------------------------------------------------------------------
# The first argument is the embedding model.
# The second argument is the OLS display name of the terminology.
# The third argument is the short identifier used internally.
task = OLSTerminologyImportTask(vectorizer, "SNOMED CT", "snomed")

# This method fetches, embeds, and uploads the terminology to PostgreSQL.
task.process_to_repository(repository)

print("Terminology import completed successfully.")

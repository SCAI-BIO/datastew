from datastew.embedding import Vectorizer
from datastew.process.ols import OLSTerminologyImportTask
from datastew.repository import WeaviateRepository

# Use a local running weaviate instance on localhost:8080
repository = WeaviateRepository(mode='remote', path='localhost', port=8080)
embedding_model = Vectorizer()

task = OLSTerminologyImportTask(embedding_model, "SNOMED CT", "snomed")
task.process_to_weaviate(repository)
print("done")
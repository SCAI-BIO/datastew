from datastew.embedding import MPNetAdapter
from datastew.process.ols import OLSTerminologyImportTask
from datastew.repository import WeaviateRepository

# Use a local running weaviate instance on localhost:8080
repository = WeaviateRepository(mode='remote', path='localhost', port=8080)
embedding_model = MPNetAdapter()

task = OLSTerminologyImportTask(embedding_model, "SNONMED CT", "snomed")
task.process_to_weaviate(repository)
print("done")
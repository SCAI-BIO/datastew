from datastew.embedding import MPNetAdapter
from datastew.process.ols import OLSTerminologyImportTask
from datastew.repository.sqllite import SQLLiteRepository

repository = SQLLiteRepository(name="snomed")
embedding_model = MPNetAdapter()

task = OLSTerminologyImportTask(repository, embedding_model, "SNONMED CT", "snomed")
task.process_to_weaviate()
print("done")
from index.embedding import MPNetAdapter
from index.process.ols import OLSTerminologyImportTask
from index.repository.sqllite import SQLLiteRepository

repository = SQLLiteRepository()
embedding_model = MPNetAdapter()

task = OLSTerminologyImportTask(repository, embedding_model, "SNONMED CT", "snomed")
task.process()
print("done")
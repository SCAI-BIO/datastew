from datastew.embedding import MPNetAdapter
from datastew.process.ols import OLSTerminologyImportTask
from datastew.repository.sqllite import SQLLiteRepository

repository = SQLLiteRepository()
embedding_model = MPNetAdapter()

task = OLSTerminologyImportTask(repository, embedding_model, "SNOMED CT", "snomed")
task.process()
print("done")
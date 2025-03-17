from datastew.embedding import MPNetAdapter
from datastew.process.jsonl_adapter import WeaviateJsonlConverter

json_converter = WeaviateJsonlConverter("resources/results")
embedding_model = MPNetAdapter()

json_converter.from_ohdsi("resources/CONCEPT.csv", embedding_model)
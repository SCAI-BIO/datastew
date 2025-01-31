from datastew.process.json_adapter import WeaviateJsonConverter
from datastew.embedding import MPNetAdapter

json_converter = WeaviateJsonConverter("resources/results")
embedding_model = MPNetAdapter()

json_converter.from_ohdsi("resources/CONCEPT.csv", embedding_model)
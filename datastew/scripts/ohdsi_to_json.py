from datastew.embedding import Vectorizer
from datastew.process.jsonl_adapter import WeaviateJsonlConverter

json_converter = WeaviateJsonlConverter("resources/results")
embedding_model = Vectorizer()

json_converter.from_ohdsi("resources/CONCEPT.csv", embedding_model)
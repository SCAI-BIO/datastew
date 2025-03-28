from datastew.embedding import Vectorizer
from datastew.process.jsonl_adapter import WeaviateJsonlConverter

json_converter = WeaviateJsonlConverter("resources/results")
vectorizer = Vectorizer()

json_converter.from_ohdsi("resources/CONCEPT.csv", vectorizer)
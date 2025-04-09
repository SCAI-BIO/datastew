from datastew.process.jsonl_adapter import WeaviateJsonlConverter

jsonl_converter = WeaviateJsonlConverter("resources/results")

jsonl_converter.from_ohdsi("resources/CONCEPT.csv")
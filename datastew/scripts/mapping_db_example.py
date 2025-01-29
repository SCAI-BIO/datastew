from datastew.repository import WeaviateRepository
from datastew.repository.model import Terminology, Concept, Mapping
from datastew.embedding import MPNetAdapter

# This script demonstrates how to retrieve the closest text embeddings and their similarities for a given text

# 1) Initialize the repository and embedding model

repository = WeaviateRepository(mode='disk', path='localhost', port=8080)
embedding_model = MPNetAdapter()
# embedding_model = GPT4Adapter(key="your_key") # Use this line for higher accuracy if you have an OpenAI API key

# 2) Create a baseline of data to map to in the initialized repository. Text gets attached to any unique concept of an
# existing or custom vocabulary or terminology namespace in the form of a mapping object containing the text, embedding,
# and the name of sentence embedder used. Multiple Mapping objects with textually different but semantically equal
# descriptions can point to the same Concept.

terminology = Terminology("snomed CT", "SNOMED")

text1 = "Diabetes mellitus (disorder)"
concept1 = Concept(terminology, text1, "Concept ID: 11893007")
mapping1 = Mapping(concept1, text1, embedding_model.get_embedding(text1), embedding_model.get_model_name())

text2 = "Hypertension (disorder)"
concept2 = Concept(terminology, text2, "Concept ID: 73211009")
mapping2 = Mapping(concept2, text2, embedding_model.get_embedding(text2), embedding_model.get_model_name())

repository.store_all([terminology, concept1, mapping1, concept2, mapping2])

# 3) Retrieve the closest mappings and their similarities for a given text

text_to_map = "Sugar sickness" # Semantically similar to "Diabetes mellitus (disorder)"
embedding = embedding_model.get_embedding(text_to_map)
results = repository.get_closest_mappings_with_similarities(embedding, limit=2)

# 4) print the mappings and their similarities
for result in results:
    print(result)




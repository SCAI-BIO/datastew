from datastew.embedding import Vectorizer
from datastew.repository import WeaviateRepository
from datastew.repository.model import Concept, Mapping, Terminology

# This script demonstrates how to retrieve the closest text embeddings and their similarities for a given text

# 1) Initialize the repository and embedding model

repository = WeaviateRepository(mode="remote", path="localhost", port=8080)
vectorizer = Vectorizer()
# vectorizer = Vectorizer("text-embedding-ada-002", key="your_key") # Use this line for higher accuracy if you have an OpenAI API key

# 2) Create a baseline of data to map to in the initialized repository. Text gets attached to any unique concept of an
# existing or custom vocabulary or terminology namespace in the form of a mapping object containing the text, embedding,
# and the name of sentence embedder used. Multiple Mapping objects with textually different but semantically equal
# descriptions can point to the same Concept.

terminology = Terminology("snomed CT", "SNOMED")

text1 = "Diabetes mellitus (disorder)"
concept1 = Concept(terminology, text1, "Concept ID: 11893007")
mapping1 = Mapping(concept1, text1, vectorizer.get_embedding(text1), vectorizer.model_name)

text2 = "Hypertension (disorder)"
concept2 = Concept(terminology, text2, "Concept ID: 73211009")
mapping2 = Mapping(concept2, text2, vectorizer.get_embedding(text2), vectorizer.model_name)

repository.store_all([terminology, concept1, mapping1, concept2, mapping2])

# 3) Retrieve the closest mappings and their similarities for a given text

text_to_map = "Sugar sickness"  # Semantically similar to "Diabetes mellitus (disorder)"
embedding = vectorizer.get_embedding(text_to_map)
results = repository.get_closest_mappings(embedding, similarities=True, limit=2)

# 4) print the mappings and their similarities
for result in results:
    print(result)

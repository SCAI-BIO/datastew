from datastew.repository.model import Terminology, Concept, Mapping
from datastew.embedding import MPNetAdapter
from datastew.repository.sqllite import SQLLiteRepository

repository = SQLLiteRepository(mode="memory")
embedding_model = MPNetAdapter()

terminology = Terminology("snomed CT", "SNOMED")

text1 = "Diabetes mellitus (disorder)"
concept1 = Concept(terminology, text1, "Concept ID: 11893007")
mapping1 = Mapping(concept1, text1, embedding_model.get_embedding(text1), embedding_model.get_model_name())

text2 = "Hypertension (disorder)"
concept2 = Concept(terminology, text2, "Concept ID: 73211009")
mapping2 = Mapping(concept2, text2, embedding_model.get_embedding(text2), embedding_model.get_model_name())

repository.store_all([terminology, concept1, mapping1, concept2, mapping2])

text_to_map = "Sugar sickness"
embedding = embedding_model.get_embedding(text_to_map)
mappings, similarities = repository.get_closest_mappings(embedding, limit=2)
for mapping, similarity in zip(mappings, similarities):
    print(f"Similarity: {similarity} -> {mapping}")

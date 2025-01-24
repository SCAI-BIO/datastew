from datastew import MPNetAdapter, Concept, Mapping, Terminology
from datastew.process.json_adapter import WeaviateJsonConverter
from datastew.repository import WeaviateRepository

repository = WeaviateRepository(mode='memory', path='localhost', port=8080)

converter = WeaviateJsonConverter(dest_dir="export")

# Compute and store some exemplary Mappings
embedding_model = MPNetAdapter()

terminology = Terminology("snomed CT", "SNOMED")

text1 = "Diabetes mellitus (disorder)"
concept1 = Concept(terminology, text1, "Concept ID: 11893007")
mapping1 = Mapping(concept1, text1, embedding_model.get_embedding(text1), embedding_model.get_model_name())

text2 = "Hypertension (disorder)"
concept2 = Concept(terminology, text2, "Concept ID: 73211009")
mapping2 = Mapping(concept2, text2, embedding_model.get_embedding(text2), embedding_model.get_model_name())

text3 = "Asthma"
concept3 = Concept(terminology, text3, "Concept ID: 195967001")
mapping3 = Mapping(concept3, text3, embedding_model.get_embedding(text3), embedding_model.get_model_name())

text4 = "Heart attack"
concept4 = Concept(terminology, text4, "Concept ID: 22298006")
mapping4 = Mapping(concept4, text4, embedding_model.get_embedding(text4), embedding_model.get_model_name())

text5 = "Common cold"
concept5 = Concept(terminology, text5, "Concept ID: 13260007")
mapping5 = Mapping(concept5, text5, embedding_model.get_embedding(text5), embedding_model.get_model_name())

text6 = "Stroke"
concept6 = Concept(terminology, text6, "Concept ID: 422504002")
mapping6 = Mapping(concept6, text6, embedding_model.get_embedding(text6), embedding_model.get_model_name())

text7 = "Migraine"
concept7 = Concept(terminology, text7, "Concept ID: 386098009")
mapping7 = Mapping(concept7, text7, embedding_model.get_embedding(text7), embedding_model.get_model_name())

text8 = "Influenza"
concept8 = Concept(terminology, text8, "Concept ID: 57386000")
mapping8 = Mapping(concept8, text8, embedding_model.get_embedding(text8), embedding_model.get_model_name())

text9 = "Osteoarthritis"
concept9 = Concept(terminology, text9, "Concept ID: 399206004")
mapping9 = Mapping(concept9, text9, embedding_model.get_embedding(text9), embedding_model.get_model_name())

text10 = "Depression"
concept10 = Concept(terminology, text10, "Concept ID: 386584008")
mapping10 = Mapping(concept10, text10, embedding_model.get_embedding(text10), embedding_model.get_model_name())

repository.store_all([terminology, concept1, mapping1, concept2, mapping2, concept3, mapping3, concept4, mapping4,
                      concept5, mapping5, concept6, mapping6, concept7, mapping7, concept8, mapping8,
                      concept9, mapping9, concept10, mapping10])

converter.from_repository(repository)

repository.close()

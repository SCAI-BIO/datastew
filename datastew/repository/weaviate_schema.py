from weaviate.classes.config import (Configure, DataType, Property,
                                     ReferenceProperty)

terminology_schema = {
    "class": "Terminology",
    "description": "A terminology entry",
    "properties": [Property(name="name", data_type=DataType.TEXT)],
}

concept_schema = {
    "class": "Concept",
    "description": "A concept entry",
    "properties": [
        Property(name="conceptID", data_type=DataType.TEXT),
        Property(name="prefLabel", data_type=DataType.TEXT),
    ],
    "references": [
        ReferenceProperty(name="hasTerminology", target_collection="Terminology")
    ],
}

mapping_schema_user_vectors = {
    "class": "Mapping",
    "description": "A mapping entry",
    "properties": [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="hasSentenceEmbedder", data_type=DataType.TEXT),
    ],
    "references": [ReferenceProperty(name="hasConcept", target_collection="Concept")],
}

mapping_schema_preconfigured_embeddings = {
    "class": "Mapping",
    "description": "A mapping entry",
    "properties": [
        Property(name="text", data_type=DataType.TEXT),
    ],
    "references": [ReferenceProperty(name="hasConcept", target_collection="Concept")],
    "vectorizer_config": [
        Configure.NamedVectors.text2vec_huggingface(
            name="sentence_transformers_all_mpnet_base_v2",
            source_properties=["text"],
            model="sentence-transformers/all-mpnet-base-v2",
        )
    ],
}

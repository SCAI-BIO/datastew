from weaviate.classes.config import Property, DataType, ReferenceProperty

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

mapping_schema = {
    "class": "Mapping",
    "description": "A mapping entry",
    "properties": [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="hasSentenceEmbedder", data_type=DataType.TEXT),
    ],
    "references": [ReferenceProperty(name="hasConcept", target_collection="Concept")],
}

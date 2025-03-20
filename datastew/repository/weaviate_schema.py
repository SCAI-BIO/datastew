from typing import List, Optional

from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty
from weaviate.collections.classes.config_named_vectors import _NamedVectorConfigCreate


class WeaviateSchema:
    def __init__(
        self,
        name: str,
        description: str,
        properties: Optional[List[Property]] = None,
        references: Optional[List[ReferenceProperty]] = None,
        vectorizer_config: Optional[List[_NamedVectorConfigCreate]] = None,
    ):
        self.class_name = name
        self.description = description
        self.properties = properties
        self.references = references
        self.vectorizer_config = vectorizer_config
        self.schema = {
            "class": self.class_name,
            "description": self.description,
            "properties": self.properties,
            "references": self.references,
            "vectorizer_config": self.vectorizer_config,
        }


class TerminologySchema(WeaviateSchema):
    def __init__(
        self,
        name: str = "Terminology",
        description: str = "A terminology entry",
        properties: List[Property] = [Property(name="name", data_type=DataType.TEXT)],
    ):
        super().__init__(name, description, properties)


class ConceptSchema(WeaviateSchema):
    def __init__(
        self,
        name: str = "Concept",
        description: str = "A concept entry",
        properties: List[Property] = [
            Property(name="conceptID", data_type=DataType.TEXT),
            Property(name="prefLabel", data_type=DataType.TEXT),
        ],
        references: List[ReferenceProperty] = [
            ReferenceProperty(name="hasTerminology", target_collection="Terminology")
        ],
    ):
        super().__init__(name, description, properties, references)


class MappingSchema(WeaviateSchema):
    def __init__(
        self,
        name: str = "Mapping",
        description: str = "A mapping entry",
        properties: List[Property] = [
            Property(name="text", data_type=DataType.TEXT),
            Property(name="hasSentenceEmbedder", data_type=DataType.TEXT),
        ],
        references: List[ReferenceProperty] = [
            ReferenceProperty(name="hasConcept", target_collection="Concept")
        ],
        vectorizer_config: Optional[List[_NamedVectorConfigCreate]] = None,
    ):
        super().__init__(name, description, properties, references, vectorizer_config)


terminology_schema = TerminologySchema()
concept_schema = ConceptSchema()
mapping_schema_user_vectors = MappingSchema()
mapping_schema_preconfigured_embeddings = MappingSchema(
    properties=[Property(name="text", data_type=DataType.TEXT)],
    vectorizer_config=[
        Configure.NamedVectors.text2vec_huggingface(
            name="sentence_transformers_all_mpnet_base_v2",
            source_properties=["text"],
            model="sentence-transformers/all-mpnet-base-v2",
        )
    ],
)

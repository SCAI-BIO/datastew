import logging
import shutil
import socket

from typing import List, Tuple, Union, Optional

import weaviate
from weaviate.util import generate_uuid5
from weaviate.classes.query import Filter, QueryReference, MetadataQuery

from datastew.embedding import EmbeddingModel
from datastew.process.parsing import DataDictionarySource
from datastew.repository import Concept, Mapping, Terminology
from datastew.repository.base import BaseRepository
from datastew.repository.weaviate_schema import concept_schema, mapping_schema, terminology_schema


class WeaviateRepository(BaseRepository):
    logger = logging.getLogger(__name__)

    def __init__(self, mode="memory", path=None, port=80, http_port=8079, grpc_port=50050):
        self.mode = mode
        try:
            if mode == "memory":
                # Check if there is an existing instance of Weaviate client for the default ports
                if self._is_port_in_use(http_port) and self._is_port_in_use(grpc_port):
                    self.client.close()
                    self.client = weaviate.connect_to_embedded(persistence_data_path="db")
                else:
                    self.client = weaviate.connect_to_embedded(persistence_data_path="db")
            elif mode == "disk":
                if path is None:
                    raise ValueError("Path must be provided for disk mode.")
                # Check if there is an existing instance of Weaviate client for the default ports
                if self._is_port_in_use(http_port) and self._is_port_in_use(grpc_port):
                    self.client.close()
                    self.client = weaviate.connect_to_embedded(persistence_data_path=path)
                else:
                    self.client = weaviate.connect_to_embedded(persistence_data_path=path)
            elif mode == "remote":
                if path is None:
                    raise ValueError("Remote URL must be provided for remote mode.")
                self.client = weaviate.connect_to_local(host=path, port=port)
            else:
                raise ValueError(
                    f"Repository mode {mode} is not defined. Use either memory, disk or remote."
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {e}")

        try:
            self._create_schema_if_not_exists(terminology_schema)
            self._create_schema_if_not_exists(concept_schema)
            self._create_schema_if_not_exists(mapping_schema)
        except Exception as e:
            raise RuntimeError(f"Failed to create schema: {e}")

    def _create_schema_if_not_exists(self, schema):
        references = None
        class_name = schema["class"]
        try:
            if not self.client.collections.exists(class_name):
                description = schema["description"]
                properties = schema["properties"]
                if "references" in schema:
                    references = schema["references"]
                self.client.collections.create(
                    name=class_name,
                    description=description,
                    properties=properties,
                    references=references,
                )
            else:
                self.logger.info(f"Schema for {class_name} already exists. Skipping.")
        except Exception as e:
            raise RuntimeError(f"Failed to check/create schema for {class_name}: {e}")

    def import_data_dictionary(self, data_dictionary: DataDictionarySource, terminology_name: str, embedding_model: Optional[EmbeddingModel] = None):
        try:
            model_object_instances: List[Union[Terminology, Concept, Mapping]] = []
            data_frame = data_dictionary.to_dataframe()
            descriptions = data_frame["description"].tolist()
            if embedding_model is None:
                embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
            else:
                embedding_model_name = embedding_model.get_model_name()
            variable_to_embedding = data_dictionary.get_embeddings(embedding_model)
            terminology = Terminology(terminology_name, terminology_name)
            model_object_instances.append(terminology)
            for variable, description in zip(variable_to_embedding.keys(), descriptions):
                concept_id = f"{terminology_name}:{variable}"
                concept = Concept(
                    terminology=terminology,
                    pref_label=variable,
                    concept_identifier=concept_id
                )
                mapping = Mapping(
                    concept=concept,
                    text=description,
                    embedding=variable_to_embedding[variable],
                    sentence_embedder=embedding_model_name
                )
                model_object_instances.append(concept)
                model_object_instances.append(mapping)
            self.store_all(model_object_instances)
        except Exception as e:
            raise RuntimeError(f"Failed to import data dictionary source: {e}")
    
    def store_all(self, model_object_instances: List[Union[Terminology, Concept, Mapping]]):
        for instance in model_object_instances:
            self.store(instance)

    def get_all_sentence_embedders(self) -> List[str]:
        sentence_embedders = set()
        try:
            mapping = self.client.collections.get("Mapping")
            response = mapping.query.fetch_objects()
            for o in response.objects:
                sentence_embedders.add(o.properties["hasSentenceEmbedder"])
        except Exception as e:
            raise RuntimeError(f"Failed to fetch sentence embedders: {e}")
        return list(sentence_embedders)

    def get_concept(self, concept_id: str) -> Concept:
        try:
            if not self._concept_exists(concept_id):
                raise RuntimeError(f"Concept {concept_id} does not exists")
            concept_collection = self.client.collections.get("Concept")
            response = concept_collection.query.fetch_objects(
                filters=Filter.by_property("conceptID").equal(concept_id),
                return_references=QueryReference(link_on="hasTerminology"),
            )

            concept_data = response.objects[0]
            if concept_data.references:
                terminology_data = concept_data.references["hasTerminology"].objects[0]
                terminology_name = str(terminology_data.properties["name"])
                terminology_id = str(terminology_data.uuid)
                terminology = Terminology(terminology_name, terminology_id)

            id = str(concept_data.uuid)
            concept_name = str(concept_data.properties["prefLabel"])
            concept = Concept(terminology, concept_name, concept_id, id)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch concept {concept_id}: {e}")
        return concept

    def get_all_concepts(self) -> List[Concept]:
        concepts = []
        try:
            concept_collection = self.client.collections.get("Concept")
            response = concept_collection.query.fetch_objects(
                return_references=QueryReference(link_on="hasTerminology")
            )
            for o in response.objects:
                if o.references:
                    terminology_data = o.references["hasTerminology"].objects[0]
                    terminology_name = str(terminology_data.properties["name"])
                    terminology_id = str(terminology_data.uuid)
                    terminology = Terminology(name=terminology_name, id=terminology_id)
                concept = Concept(
                    concept_identifier=str(o.properties["conceptID"]),
                    pref_label=str(o.properties["prefLabel"]),
                    terminology=terminology,
                    id=str(o.uuid),
                )
                concepts.append(concept)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch concepts: {e}")
        return concepts

    def get_terminology(self, terminology_name: str) -> Terminology:
        try:
            if not self._terminology_exists(terminology_name):
                raise RuntimeError(f"Terminology {terminology_name} does not exists")
            terminology_collection = self.client.collections.get("Terminology")
            response = terminology_collection.query.fetch_objects(
                filters=Filter.by_property("name").equal(terminology_name),
            )

            terminology_data = response.objects[0]
            terminology_id = str(terminology_data.uuid)
            terminology = Terminology(terminology_name, terminology_id)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch terminology {terminology_name}: {e}")
        return terminology

    def get_all_terminologies(self) -> List[Terminology]:
        terminologies = []
        try:
            terminology_collection = self.client.collections.get("Terminology")
            response = terminology_collection.query.fetch_objects()

            for o in response.objects:
                terminology = Terminology(
                    name=str(o.properties["name"]), id=str(o.uuid)
                )
                terminologies.append(terminology)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch terminologies: {e}")
        return terminologies

    def get_mappings(self, terminology_name: Optional[str] = None, limit=1000) -> List[Mapping]:
        mappings = []
        try:
            mapping_collection = self.client.collections.get("Mapping")
            if not terminology_name:
                response = mapping_collection.query.fetch_objects(
                    return_references=QueryReference(
                        link_on="hasConcept",
                        return_references=QueryReference(link_on="hasTerminology"),
                    ),
                    limit=limit,
                )
            else:
                if not self._terminology_exists(terminology_name):
                    raise RuntimeError(
                        f"Terminology {terminology_name} does not exists"
                    )
                response = mapping_collection.query.fetch_objects(
                    filters=Filter.by_ref(link_on="hasConcept").by_ref(link_on="hasTerminology").by_property("name").equal(terminology_name),
                    return_references=QueryReference(
                        link_on="hasConcept",
                        return_references=QueryReference(link_on="hasTerminology"),
                    ),
                    limit=limit,
                )

            for o in response.objects:
                if o.references:
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references["hasTerminology"].objects[0]
                    terminology = Terminology(
                        name=str(terminology_data.properties["name"]),
                        id=str(terminology_data.uuid),
                    )
                    concept = Concept(
                        concept_identifier=str(concept_data.properties["conceptID"]),
                        pref_label=str(concept_data.properties["prefLabel"]),
                        terminology=terminology,
                        id=str(concept_data.uuid),
                    )
                mapping = Mapping(
                    text=str(o.properties["text"]),
                    concept=concept,
                    embedding=o.vector,
                    sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch mappings: {e}")
        return mappings

    def get_closest_mappings(self, embedding, limit=5) -> List[Mapping]:
        mappings = []
        try:
            mapping_collection = self.client.collections.get("Mapping")
            response = mapping_collection.query.near_vector(
                near_vector=embedding,
                limit=limit,
                return_references=QueryReference(
                    link_on="hasConcept",
                    return_references=QueryReference(link_on="hasTerminology"),
                ),
            )
            for o in response.objects:
                if o.references:
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references["hasTerminology"].objects[0]
                    terminology = Terminology(
                        name=str(terminology_data.properties["name"]),
                        id=str(terminology_data.uuid),
                    )
                    concept = Concept(
                        terminology=terminology,
                        pref_label=str(concept_data.properties["prefLabel"]),
                        concept_identifier=str(concept_data.properties["conceptID"]),
                        id=str(concept_data.uuid),
                    )
                mapping = Mapping(
                    concept=concept,
                    text=str(o.properties["text"]),
                    embedding=o.vector,
                    sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch closest mappings: {e}")
        return mappings

    def get_closest_mappings_with_similarities(
        self, embedding, limit=5
    ) -> List[Tuple[Mapping, float]]:
        mappings_with_similarities = []
        try:
            mapping_collection = self.client.collections.get("Mapping")
            response = mapping_collection.query.near_vector(
                near_vector=embedding,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
                return_references=QueryReference(
                    link_on="hasConcept",
                    return_references=QueryReference(link_on="hasTerminology"),
                ),
            )

            for o in response.objects:
                if o.metadata.distance:
                    similarity = 1 - o.metadata.distance
                if o.references:
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references["hasTerminology"].objects[0]
                    terminology = Terminology(
                        name=str(terminology_data.properties["name"]),
                        id=str(terminology_data.uuid),
                    )
                    concept = Concept(
                        terminology=terminology,
                        pref_label=str(concept_data.properties["prefLabel"]),
                        concept_identifier=str(concept_data.properties["conceptID"]),
                        id=str(concept_data.uuid),
                    )
                mapping = Mapping(
                    concept=concept,
                    text=str(o.properties["text"]),
                    embedding=o.vector,
                    sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                )
                mappings_with_similarities.append((mapping, similarity))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch closest mappings with similarities: {e}"
            )
        return mappings_with_similarities

    def get_terminology_and_model_specific_closest_mappings(self, embedding, terminology_name: str, sentence_embedder_name: str, limit: int = 5) -> List[Mapping]:
        mappings = []
        try:
            if not self._terminology_exists(terminology_name):
                raise RuntimeError(f"Terminology {terminology_name} does not exists")
            if not self._sentence_embedder_exists(sentence_embedder_name):
                raise RuntimeError(
                    f"Sentence Embedder {sentence_embedder_name} does not exists"
                )
            mapping_collection = self.client.collections.get("Mapping")
            response = mapping_collection.query.near_vector(
                near_vector=embedding,
                filters=Filter.by_ref("hasConcept").by_ref("hasTerminology").by_property("name").equal(terminology_name) &
                Filter.by_property("hasSentenceEmbedder").equal(sentence_embedder_name),
                return_references=QueryReference(link_on="hasConcept", return_references=QueryReference(link_on="hasTerminology")),
                limit=limit,
            )
            for o in response.objects:
                if o.references:
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references["hasTerminology"].objects[0]
                    terminology = Terminology(
                        name=str(terminology_data.properties["name"]),
                        id=str(terminology_data.uuid),
                    )
                    concept = Concept(
                        terminology=terminology,
                        pref_label=str(concept_data.properties["prefLabel"]),
                        concept_identifier=str(concept_data.properties["conceptID"]),
                        id=str(concept_data.uuid),
                    )
                mapping = Mapping(
                    text=str(o.properties["text"]),
                    concept=concept,
                    embedding=o.vector,
                    sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch the closest mappings for terminology {terminology_name} and model {sentence_embedder_name}: {e}"
            )
        return mappings

    def get_terminology_and_model_specific_closest_mappings_with_similarities(self, embedding, terminology_name: str, sentence_embedder_name: str, limit: int = 5) -> List[Tuple[Mapping, float]]:
        mappings_with_similarities = []
        try:
            if not self._terminology_exists(terminology_name):
                raise RuntimeError(f"Terminology {terminology_name} does not exists")
            if not self._sentence_embedder_exists(sentence_embedder_name):
                raise RuntimeError(
                    f"Sentence Embedder {sentence_embedder_name} does not exists"
                )
            mapping_collection = self.client.collections.get("Mapping")
            response = mapping_collection.query.near_vector(
                near_vector=embedding,
                filters=Filter.by_ref("hasConcept").by_ref("hasTerminology").by_property("name").equal(terminology_name) &
                Filter.by_property("hasSentenceEmbedder").equal(sentence_embedder_name),
                return_references=QueryReference(
                    link_on="hasConcept",
                    return_references=QueryReference(link_on="hasTerminology"),
                ),
                return_metadata=MetadataQuery(distance=True),
                limit=limit,
            )
            for o in response.objects:
                if o.metadata.distance:
                    similarity = 1 - o.metadata.distance
                if o.references:
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references["hasTerminology"].objects[0]
                    terminology = Terminology(
                        name=str(terminology_data.properties["name"]),
                        id=str(terminology_data.uuid),
                    )
                    concept = Concept(
                        terminology=terminology,
                        pref_label=str(concept_data.properties["prefLabel"]),
                        concept_identifier=str(concept_data.properties["conceptID"]),
                        id=str(concept_data.uuid),
                    )
                mapping = Mapping(
                    text=str(o.properties["text"]),
                    concept=concept,
                    embedding=o.vector,
                    sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                )
                mappings_with_similarities.append((mapping, similarity))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch the closest mappings for terminology {terminology_name} and model {sentence_embedder_name}: {e}"
            )
        return mappings_with_similarities

    def close(self):
        self.client.close()

    def shut_down(self):
        if self.mode == "memory":
            shutil.rmtree("db")

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        try:
            if isinstance(model_object_instance, Terminology):
                if not self._terminology_exists(model_object_instance.name):
                    properties = {"name": model_object_instance.name}
                    terminology_collection = self.client.collections.get("Terminology")
                    terminology_collection.data.insert(
                        properties=properties, uuid=generate_uuid5(properties)
                    )
                else:
                    self.logger.info(
                        f"Terminology with name {model_object_instance.name} already exists. Skipping."
                    )
            elif isinstance(model_object_instance, Concept):
                if not self._concept_exists(model_object_instance.concept_identifier):
                    # recursion: create terminology if not existing
                    if not self._terminology_exists(
                        model_object_instance.terminology.name
                    ):
                        self.store(model_object_instance.terminology)
                    properties = {
                        "conceptID": model_object_instance.concept_identifier,
                        "prefLabel": model_object_instance.pref_label,
                    }
                    concept_uuid = generate_uuid5(properties)
                    terminology = self.get_terminology(
                        model_object_instance.terminology.name
                    )
                    concept_collection = self.client.collections.get("Concept")
                    concept_collection.data.insert(
                        properties=properties,
                        uuid=concept_uuid,
                    )
                    concept_collection.data.reference_add(
                        from_uuid=concept_uuid,
                        from_property="hasTerminology",
                        to=str(terminology.id),
                    )
                else:
                    self.logger.info(f"Concept with identifier {model_object_instance.concept_identifier} already exists. Skipping.")
            elif isinstance(model_object_instance, Mapping):
                if not self._mapping_exists(model_object_instance.embedding):
                    if not self._concept_exists(
                        model_object_instance.concept.concept_identifier
                    ):
                        self.store(model_object_instance.concept)
                    properties = {
                        "text": model_object_instance.text,
                        "hasSentenceEmbedder": model_object_instance.sentence_embedder,
                    }
                    mapping_uuid = generate_uuid5(properties)
                    concept = self.get_concept(
                        model_object_instance.concept.concept_identifier
                    )
                    mapping_collection = self.client.collections.get("Mapping")
                    mapping_collection.data.insert(
                        properties=properties,
                        uuid=mapping_uuid,
                        vector=model_object_instance.embedding,
                    )
                    mapping_collection.data.reference_add(
                        from_uuid=mapping_uuid,
                        from_property="hasConcept",
                        to=str(concept.id),
                    )
                else:
                    self.logger.info("Mapping with same embedding already exists. Skipping.")
            else:
                raise ValueError("Unsupported model object instance type.")

        except Exception as e:
            raise RuntimeError(f"Failed to store object in Weaviate: {e}")

    def _sentence_embedder_exists(self, name: str) -> bool:
        try:
            mapping = self.client.collections.get("Mapping")
            response = mapping.query.fetch_objects(
                filters=Filter.by_property("hasSentenceEmbedder").equal(name)
            )
            if response.objects is not None:
                return len(response.objects) > 0
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to check if sentence embedder exists: {e}")

    def _terminology_exists(self, name: str) -> bool:
        try:
            terminology = self.client.collections.get("Terminology")
            response = terminology.query.fetch_objects(
                filters=Filter.by_property("name").equal(name)
            )
            if response.objects is not None:
                return len(response.objects) > 0
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to check if terminology exists: {e}")

    def _concept_exists(self, concept_id: str) -> bool:
        try:
            concept = self.client.collections.get("Concept")
            response = concept.query.fetch_objects(
                filters=Filter.by_property("conceptID").equal(concept_id)
            )
            if response.objects is not None:
                return len(response.objects) > 0
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to check if concept exists: {e}")

    def _mapping_exists(self, embedding) -> bool:
        try:
            mapping = self.client.collections.get("Mapping")
            response = mapping.query.near_vector(
                near_vector=embedding,
                distance=float(0),  # Ensure distance is explicitly casted to float
            )
            if response.objects is not None:
                return len(response.objects) > 0
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to check if mapping exists: {e}")
        
    def _is_port_in_use(self, port) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

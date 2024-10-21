import logging
import shutil
import uuid as uuid
from typing import List, Tuple, Union, Optional

import weaviate
from weaviate.classes.query import Filter

from datastew.repository import Concept, Mapping, Terminology
from datastew.repository.base import BaseRepository
from datastew.repository.weaviate_schema import (
    concept_schema,
    mapping_schema,
    terminology_schema,
)


class WeaviateRepository(BaseRepository):
    logger = logging.getLogger(__name__)

    def __init__(self, mode="memory", path=None):
        self.mode = mode
        try:
            if mode == "memory":
                self.client = weaviate.connect_to_embedded(persistence_data_path="db")
            elif mode == "disk":
                if path is None:
                    raise ValueError("Path must be provided for disk mode.")
                self.client = weaviate.connect_to_embedded(persistence_data_path="db")
            elif mode == "remote":
                if path is None:
                    raise ValueError("Remote URL must be provided for remote mode.")
                self.client = weaviate.connect_to_custom(
                    http_host=path,
                    http_port=80,
                    http_secure=False,
                    grpc_host=path,
                    grpc_port=50051,
                    grpc_secure=False,
                )
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

    def store_all(self, model_object_instances):
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
            concept = self.client.collections.get("Concept")
            response = concept.query.fetch_objects(
                filters=Filter.by_property("conceptID").equal(concept_id)
            )

            concept_data = response.objects[0]
            terminology_data = concept_data["hasTerminology"][0]
            terminology_name = terminology_data["name"]
            terminology_id = terminology_data["_additional"]["id"]
            terminology = Terminology(terminology_name, terminology_id)
            id = concept_data["_additional"]["id"]
            concept_name = result["data"]["Get"]["Concept"][0]["prefLabel"]
            concept = Concept(terminology, concept_name, concept_id, id)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch concept {concept_id}: {e}")
        return concept


    # TODO: Migrate it to v4
    def get_all_concepts(self) -> List[Concept]:
        concepts = []
        try:
            result = (
                self.client.query.get(
                    "Concept",
                    [
                        "conceptID",
                        "prefLabel",
                        "_additional { id }",
                        "hasTerminology { ... on Terminology { _additional { id } name } }",
                    ],
                )
                .with_additional("vector")
                .do()
            )
            for item in result["data"]["Get"]["Concept"]:
                terminology_data = item["hasTerminology"][
                    0
                ]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"],
                )
                concept = Concept(
                    concept_identifier=item["conceptID"],
                    pref_label=item["prefLabel"],
                    terminology=terminology,
                    id=item["_additional"]["id"],
                )
                concepts.append(concept)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch concepts: {e}")
        return concepts

    # TODO: Migrate it to v4
    def get_terminology(self, terminology_name: str) -> Terminology:
        try:
            if not self._terminology_exists(terminology_name):
                raise RuntimeError(f"Terminology {terminology_name} does not exists")
            result = (
                self.client.query.get("Terminology", ["name", "_additional { id }"])
                .with_where(
                    {"path": "name", "operator": "Equal", "valueText": terminology_name}
                )
                .do()
            )
            terminology_data = result["data"]["Get"]["Terminology"][0]
            terminology_id = terminology_data["_additional"]["id"]
            terminology = Terminology(terminology_name, terminology_id)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch terminology {terminology_name}: {e}")
        return terminology
    
    # TODO: Migrate it to v4
    def get_all_terminologies(self) -> List[Terminology]:
        terminologies = []
        try:
            result = self.client.query.get(
                "Terminology", ["name", "_additional { id }"]
            ).do()
            for item in result["data"]["Get"]["Terminology"]:
                terminology = Terminology(
                    name=item["name"], id=item["_additional"]["id"]
                )
                terminologies.append(terminology)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch terminologies: {e}")
        return terminologies

    def get_mappings(
        self, terminology_name: Optional[str] = None, limit=1000
    ) -> List[Mapping]:
        mappings = []
        try:
            if not terminology_name:
                result = (
                    self.client.query.get(
                        "Mapping",
                        [
                            "text",
                            "hasSentenceEmbedder",
                            "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }",
                        ],
                    )
                    .with_additional("vector")
                    .with_limit(limit)
                    .do()
                )
            else:
                if not self._terminology_exists(terminology_name):
                    raise RuntimeError(
                        f"Terminology {terminology_name} does not exists"
                    )
                result = (
                    self.client.query.get(
                        "Mapping",
                        [
                            "text",
                            "hasSentenceEmbedder",
                            "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }",
                        ],
                    )
                    .with_where(
                        {
                            "path": [
                                "hasConcept",
                                "Concept",
                                "hasTerminology",
                                "Terminology",
                                "name",
                            ],
                            "operator": "Equal",
                            "valueText": terminology_name,
                        }
                    )
                    .with_additional("vector")
                    .with_limit(limit)
                    .do()
                )
            for item in result["data"]["Get"]["Mapping"]:
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][
                    0
                ]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"],
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"],
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector,
                    sentence_embedder=item["hasSentenceEmbedder"],
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch mappings: {e}")
        return mappings

    # TODO: Migrate it to v4
    def get_closest_mappings(self, embedding, limit=5) -> List[Mapping]:
        mappings = []
        try:
            result = (
                self.client.query.get(
                    "Mapping",
                    [
                        "text",
                        "_additional { distance }",
                        "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }",
                        "hasSentenceEmbedder",
                    ],
                )
                .with_additional("vector")
                .with_near_vector({"vector": embedding})
                .with_limit(limit)
                .do()
            )
            for item in result["data"]["Get"]["Mapping"]:
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][
                    0
                ]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"],
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"],
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector,
                    sentence_embedder=item["hasSentenceEmbedder"],
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch closest mappings: {e}")
        return mappings
    
    # TODO: Migrate it to v4
    def get_closest_mappings_with_similarities(
        self, embedding, limit=5
    ) -> List[Tuple[Mapping, float]]:
        mappings_with_similarities = []
        try:
            result = (
                self.client.query.get(
                    "Mapping",
                    [
                        "text",
                        "_additional { distance }",
                        "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }",
                        "hasSentenceEmbedder",
                    ],
                )
                .with_additional("vector")
                .with_near_vector({"vector": embedding})
                .with_limit(limit)
                .do()
            )
            for item in result["data"]["Get"]["Mapping"]:
                similarity = 1 - item["_additional"]["distance"]
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][
                    0
                ]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"],
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"],
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector,
                    sentence_embedder=item["hasSentenceEmbedder"],
                )
                mappings_with_similarities.append((mapping, similarity))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch closest mappings with similarities: {e}"
            )
        return mappings_with_similarities

    # TODO: Migrate it to v4
    def get_terminology_and_model_specific_closest_mappings(
        self,
        embedding,
        terminology_name: str,
        sentence_embedder_name: str,
        limit: int = 5,
    ) -> List[Mapping]:
        mappings = []
        try:
            if not self._terminology_exists(terminology_name):
                raise RuntimeError(f"Terminology {terminology_name} does not exists")
            if not self._sentence_embedder_exists(sentence_embedder_name):
                raise RuntimeError(
                    f"Sentence Embedder {sentence_embedder_name} does not exists"
                )
            result = (
                self.client.query.get(
                    "Mapping",
                    [
                        "text",
                        "_additional { distance }",
                        "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }",
                        "hasSentenceEmbedder",
                    ],
                )
                .with_where(
                    {
                        "operator": "And",
                        "operands": [
                            {
                                "path": ["hasSentenceEmbedder"],
                                "operator": "Equal",
                                "valueText": sentence_embedder_name,
                            },
                            {
                                "path": [
                                    "hasConcept",
                                    "Concept",
                                    "hasTerminology",
                                    "Terminology",
                                    "name",
                                ],
                                "operator": "Equal",
                                "valueText": terminology_name,
                            },
                        ],
                    }
                )
                .with_additional("vector")
                .with_near_vector({"vector": embedding})
                .with_limit(limit)
                .do()
            )
            for item in result["data"]["Get"]["Mapping"]:
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][0]
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"],
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"],
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector,
                    sentence_embedder=item["hasSentenceEmbedder"],
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch the closest mappings for terminology {terminology_name} and model {sentence_embedder_name}: {e}"
            )
        return mappings

    # TODO: Migrate it to v4
    def get_terminology_and_model_specific_closest_mappings_with_similarities(
        self,
        embedding,
        terminology_name: str,
        sentence_embedder_name: str,
        limit: int = 5,
    ) -> List[Tuple[Mapping, float]]:
        mappings_with_similarities = []
        try:
            if not self._terminology_exists(terminology_name):
                raise RuntimeError(f"Terminology {terminology_name} does not exists")
            if not self._sentence_embedder_exists(sentence_embedder_name):
                raise RuntimeError(
                    f"Sentence Embedder {sentence_embedder_name} does not exists"
                )
            result = (
                self.client.query.get(
                    "Mapping",
                    [
                        "text",
                        "_additional { distance }",
                        "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }",
                        "hasSentenceEmbedder",
                    ],
                )
                .with_where(
                    {
                        "operator": "And",
                        "operands": [
                            {
                                "path": ["hasSentenceEmbedder"],
                                "operator": "Equal",
                                "valueText": sentence_embedder_name,
                            },
                            {
                                "path": [
                                    "hasConcept",
                                    "Concept",
                                    "hasTerminology",
                                    "Terminology",
                                    "name",
                                ],
                                "operator": "Equal",
                                "valueText": terminology_name,
                            },
                        ],
                    }
                )
                .with_additional("vector")
                .with_near_vector({"vector": embedding})
                .with_limit(limit)
                .do()
            )
            for item in result["data"]["Get"]["Mapping"]:
                similarity = 1 - item["_additional"]["distance"]
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][0]
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"],
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"],
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector,
                    sentence_embedder=item["hasSentenceEmbedder"],
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
        random_uuid = uuid.uuid4()
        model_object_instance.id = random_uuid
        try:
            if isinstance(model_object_instance, Terminology):
                if not self._terminology_exists(model_object_instance.name):
                    properties = {"name": model_object_instance.name}
                    terminology = self.client.collections.get("Terminology")
                    terminology.data.insert(properties=properties, uuid=random_uuid)
                else:
                    self.logger.info(
                        f"Terminology with name {model_object_instance.name} already exists. Skipping."
                    )
            elif isinstance(model_object_instance, Concept):
                model_object_instance.uuid = random_uuid
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
                    concept = self.client.collections.get("Concept")
                    concept.data.insert(
                        properties=properties,
                        uuid=random_uuid,
                        references={
                            "hasTerminology": model_object_instance.terminology.id
                        },
                    )
                else:
                    self.logger.info(
                        f"Concept with identifier {model_object_instance.concept_identifier} "
                        f"already exists. Skipping."
                    )
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
                    mapping = self.client.collections.get("Mapping")
                    mapping.data.insert(
                        properties=properties,
                        uuid=random_uuid,
                        vector=model_object_instance.embedding,
                        references={"hasConcept": model_object_instance.concept.uuid},
                    )
                else:
                    self.logger.info(
                        "Mapping with same embedding already exists. Skipping."
                    )
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

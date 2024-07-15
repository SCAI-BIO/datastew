import logging
import shutil
from typing import List, Union, Tuple
import uuid as uuid
import weaviate
from weaviate.embedded import EmbeddedOptions

from datastew.repository import Mapping, Terminology, Concept
from datastew.repository.base import BaseRepository
from datastew.repository.weaviate_schema import terminology_schema, concept_schema, mapping_schema


class WeaviateRepository(BaseRepository):
    logger = logging.getLogger(__name__)

    def __init__(self, mode="memory", path=None):
        self.mode = mode
        try:
            if mode == "memory":
                self.client = weaviate.Client(embedded_options=EmbeddedOptions(
                    persistence_data_path="db"
                ))
            elif mode == "disk":
                if path is None:
                    raise ValueError("Path must be provided for disk mode.")
                self.client = weaviate.Client(embedded_options=EmbeddedOptions(
                    persistence_data_path=path
                ))
            elif mode == "remote":
                if path is None:
                    raise ValueError("Remote URL must be provided for remote mode.")
                self.client = weaviate.Client(
                    url=path
                )
            else:
                raise ValueError(f'Repository mode {mode} is not defined. Use either memory, disk or remote.')
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {e}")

        try:
            self._create_schema_if_not_exists(terminology_schema)
            self._create_schema_if_not_exists(concept_schema)
            self._create_schema_if_not_exists(mapping_schema)
        except Exception as e:
            raise RuntimeError(f"Failed to create schema: {e}")

    def _create_schema_if_not_exists(self, schema):
        class_name = schema['class']
        try:
            if not self.client.schema.exists(class_name):
                self.client.schema.create_class(schema)
            else:
                self.logger.info(f"Schema for {class_name} already exists. Skipping.")
        except Exception as e:
            raise RuntimeError(f"Failed to check/create schema for {class_name}: {e}")

    def store_all(self, model_object_instances):
        for instance in model_object_instances:
            self.store(instance)

    def get_all_concepts(self) -> List[Concept]:
        concepts = []
        try:
            result = self.client.query.get(
                "Concept",
                ["conceptID", "prefLabel", "hasTerminology { ... on Terminology { _additional { id } name } }"]
            ).with_additional("vector").do()
            for item in result['data']['Get']['Concept']:
                terminology_data = item["hasTerminology"][0]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"]
                )
                concept = Concept(
                    concept_identifier=item["conceptID"],
                    pref_label=item["prefLabel"],
                    terminology=terminology,
                )
                concepts.append(concept)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch concepts: {e}")
        return concepts

    def get_all_terminologies(self) -> List[Terminology]:
        terminologies = []
        try:
            result = self.client.query.get("Terminology", ["name", "_additional { id }"]).do()
            for item in result['data']['Get']['Terminology']:
                terminology = Terminology(
                    name=item["name"],
                    id=item["_additional"]["id"]
                )
                terminologies.append(terminology)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch terminologies: {e}")
        return terminologies

    def get_all_mappings(self, limit=1000) -> List[Mapping]:
        mappings = []
        try:
            result = self.client.query.get(
                "Mapping",
                ["text",
                 "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }"]
            ).with_additional("vector").with_limit(limit).do()
            for item in result['data']['Get']['Mapping']:
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][0]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"]
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"]
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch mappings: {e}")
        return mappings

    def get_closest_mappings(self, embedding, limit=5) -> List[Mapping]:
        mappings = []
        try:
            result = self.client.query.get(
                "Mapping",
                ["text", "_additional { distance }",
                 "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }"]
            ).with_additional("vector").with_near_vector({"vector": embedding}).with_limit(limit).do()
            for item in result['data']['Get']['Mapping']:
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][0]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"]
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"]
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch closest mappings: {e}")
        return mappings

    def get_closest_mappings_with_similarities(self, embedding, limit=5) -> List[Tuple[Mapping, float]]:
        mappings_with_similarities = []
        try:
            result = self.client.query.get(
                "Mapping",
                ["text", "_additional { distance }",
                 "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }"]
            ).with_additional("vector").with_near_vector({"vector": embedding}).with_limit(limit).do()
            for item in result['data']['Get']['Mapping']:
                similarity = 1 - item["_additional"]["distance"]
                embedding_vector = item["_additional"]["vector"]
                concept_data = item["hasConcept"][0]  # Assuming it has only one concept
                terminology_data = concept_data["hasTerminology"][0]  # Assuming it has only one terminology
                terminology = Terminology(
                    name=terminology_data["name"],
                    id=terminology_data["_additional"]["id"]
                )
                concept = Concept(
                    concept_identifier=concept_data["conceptID"],
                    pref_label=concept_data["prefLabel"],
                    terminology=terminology,
                    id=concept_data["_additional"]["id"]
                )
                mapping = Mapping(
                    text=item["text"],
                    concept=concept,
                    embedding=embedding_vector
                )
                mappings_with_similarities.append((mapping, similarity))
        except Exception as e:
            raise RuntimeError(f"Failed to fetch closest mappings with similarities: {e}")
        return mappings_with_similarities

    def shut_down(self):
        if self.mode == "memory":
            shutil.rmtree("db")

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        random_uuid = uuid.uuid4()
        model_object_instance.id = random_uuid
        try:
            if isinstance(model_object_instance, Terminology):
                if not self._terminology_exists(model_object_instance.name):
                    properties = {
                        "name": model_object_instance.name
                    }
                    self.client.data_object.create(
                        class_name="Terminology",
                        data_object=properties,
                        uuid=random_uuid
                    )
                else:
                    self.logger.info(f'Terminology with name {model_object_instance.name} already exists. Skipping.')
            elif isinstance(model_object_instance, Concept):
                model_object_instance.uuid = random_uuid
                if not self._concept_exists(model_object_instance.concept_identifier):
                    properties = {
                        "conceptID": model_object_instance.concept_identifier,
                        "prefLabel": model_object_instance.pref_label,
                    }
                    self.client.data_object.create(
                        class_name="Concept",
                        data_object=properties,
                        uuid=random_uuid
                    )
                    self.client.data_object.reference.add(
                        from_class_name="Concept",
                        from_uuid=random_uuid,
                        from_property_name="hasTerminology",
                        to_class_name="Terminology",
                        to_uuid=model_object_instance.terminology.id,
                    )
                else:
                    self.logger.info(f'Concept with identifier {model_object_instance.concept_identifier} '
                                     f'already exists. Skipping.')
            elif isinstance(model_object_instance, Mapping):
                if not self._mapping_exists(model_object_instance.embedding):
                    properties = {
                        "text": model_object_instance.text,
                    }
                    self.client.data_object.create(
                        class_name="Mapping",
                        data_object=properties,
                        uuid=random_uuid,
                        vector=model_object_instance.embedding
                    )
                    self.client.data_object.reference.add(
                        from_class_name="Mapping",
                        from_uuid=random_uuid,
                        from_property_name="hasConcept",
                        to_class_name="Concept",
                        to_uuid=model_object_instance.concept.uuid,
                    )
                else:
                    self.logger.info(f'Mapping with same embedding already exists. Skipping.')
            else:
                raise ValueError("Unsupported model object instance type.")

        except Exception as e:
            raise RuntimeError(f"Failed to store object in Weaviate: {e}")

    def _terminology_exists(self, name: str) -> bool:
        try:
            result = self.client.query.get("Terminology", ["name"]).with_where({
                "path": ["name"],
                "operator": "Equal",
                "valueText": name
            }).do()
            return len(result['data']['Get']['Terminology']) > 0
        except Exception as e:
            raise RuntimeError(f"Failed to check if terminology exists: {e}")

    def _concept_exists(self, concept_id: str) -> bool:
        try:
            result = self.client.query.get("Concept", ["conceptID"]).with_where({
                "path": ["conceptID"],
                "operator": "Equal",
                "valueText": concept_id
            }).do()
            return len(result['data']['Get']['Concept']) > 0
        except Exception as e:
            raise RuntimeError(f"Failed to check if concept exists: {e}")

    def _mapping_exists(self, embedding) -> bool:
        try:
            result = self.client.query.get("Mapping", ["_additional { vector }"]).with_near_vector({
                "vector": embedding,
                "distance": float(0)  # Ensure distance is explicitly casted to float
            }).do()
            return len(result['data']['Get']['Mapping']) > 0
        except Exception as e:
            raise RuntimeError(f"Failed to check if mapping exists: {e}")

import shutil
from typing import List, Union

import uuid as uuid
import weaviate

from weaviate.embedded import EmbeddedOptions

from datastew import BaseRepository
from datastew.repository import Mapping, Terminology, Concept
from datastew.repository.weaviate_schema import terminology_schema, concept_schema, mapping_schema


class WeaviateRepository(BaseRepository):

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

        self.client.schema.create_class(terminology_schema)
        self.client.schema.create_class(concept_schema)
        self.client.schema.create_class(mapping_schema)

    def store_all(self, model_object_instances):
        for instance in model_object_instances:
            self.store(instance)

    def get_all_concepts(self) -> List[Concept]:
        concepts = []
        try:
            result = self.client.query.get("Concept", ["conceptID", "prefLabel", "hasTerminology"]).do()
            for item in result['data']['Get']['Concept']:
                concept = Concept(
                    concept_id=uuid.UUID(item["conceptID"]),
                    pref_label=item["prefLabel"],
                    terminology=Terminology(id=uuid.UUID(item["hasTerminology"][0]['uuid']))  # Assuming it has only one terminology
                )
                concepts.append(concept)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch concepts: {e}")
        return concepts

    def get_all_terminologies(self) -> List[Terminology]:
        terminologies = []
        try:
            result = self.client.query.get("Terminology", ["name"]).do()
            for item in result['data']['Get']['Terminology']:
                terminology = Terminology(
                    name=item["name"],
                    id=uuid.UUID(item["_additional"]["id"])
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
                ["text", "hasConcept { ... on Concept { _additional { id } conceptID prefLabel hasTerminology { ... on Terminology { _additional { id } name } } } }"]
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
            result = self.client.query.get("Mapping", ["text", "_additional { distance }"]).with_near_vector({"vector": embedding}).with_limit(limit).do()
            for item in result['data']['Get']['Mapping']:
                mapping = Mapping(
                    text=item["text"],
                    distance=item["_additional"]["distance"]
                )
                mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch closest mappings: {e}")
        return mappings

    def shut_down(self):
        if self.mode == "memory":
            shutil.rmtree("db")
        self.client.close()

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        random_uuid = uuid.uuid4()
        model_object_instance.id = random_uuid
        try:
            if isinstance(model_object_instance, Terminology):
                properties = {
                    "name": model_object_instance.name
                }
                self.client.data_object.create(
                    class_name="Terminology",
                    data_object=properties,
                    uuid=random_uuid
                )
            elif isinstance(model_object_instance, Concept):
                model_object_instance.uuid = random_uuid
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
            elif isinstance(model_object_instance, Mapping):
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
                raise ValueError("Unsupported model object instance type.")

        except Exception as e:
            raise RuntimeError(f"Failed to store object in Weaviate: {e}")

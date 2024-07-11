import shutil
from typing import List, Union

import uuid as uuid
import weaviate

from weaviate.classes.config import Property, DataType, ReferenceProperty

from datastew import BaseRepository
from datastew.repository import Mapping, Terminology, Concept
from datastew.repository.weaviate_schema import terminology_schema, concept_schema, mapping_schema


class WeaviateRepository(BaseRepository):

    def __init__(self, mode="memory", path=None):
        self.mode = mode
        try:
            if mode == "memory":
                self.client = weaviate.connect_to_embedded(persistence_data_path="db")
            elif mode == "disk":
                if path is None:
                    raise ValueError("Path must be provided for disk mode.")
                self.client = weaviate.connect_to_embedded(persistence_data_path=path)
            elif mode == "remote":
                if path is None:
                    raise ValueError("Remote URL must be provided for remote mode.")
                self.client = weaviate.Client(path)
            else:
                raise ValueError(f'Repository mode {mode} is not defined. Use either memory, disk or remote.')
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {e}")
        self.default_collection = self.client.collections.get("default")

        # Add schemas to Weaviate
        self.terminologies = self.client.collections.create(
            name="Terminology",
            properties=[
                Property(name="name", data_type=DataType.TEXT),
            ]
        )
        self.concepts = self.client.collections.create(
            name="Concept",
            properties=[
                Property(name="conceptID", data_type=DataType.TEXT),
                Property(name="prefLabel", data_type=DataType.TEXT),
            ],
            references=[
                ReferenceProperty(
                    name="hasTerminology",
                    target_collection="Terminology"
                )
            ]
        )
        self.mappings = self.client.collections.create(
            name="Mapping",
            properties=[
                Property(name="text", data_type=DataType.TEXT),
            ],
            references=[
                ReferenceProperty(
                    name="hasConcept",
                    target_collection="Concept"
                )
            ]
        )

    def store_all(self, model_object_instances):
        for instance in model_object_instances:
            self.store(instance)

    def get_all_concepts(self) -> List[Concept]:
        pass

    def get_all_terminologies(self) -> List[Terminology]:
        pass

    def get_all_mappings(self, limit=1000) -> List[Mapping]:
        collections = self.client.collections
        result = self.mappings.query.fetch_objects(limit=limit)
        # TODO why are the references not shown?
        pass

    def get_closest_mappings(self, embedding, limit=5):
        pass

    def shut_down(self):
        if self.mode == "memory":
            shutil.rmtree("db")
        self.client.close()

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        random_uuid = uuid.uuid4()
        model_object_instance.concept_id = random_uuid
        try:
            if isinstance(model_object_instance, Terminology):
                properties = {
                    "name": model_object_instance.name
                }
                self.terminologies.data.insert(properties=properties,
                                               uuid=random_uuid)
            elif isinstance(model_object_instance, Concept):
                properties = {
                    "conceptID": model_object_instance.concept_id,
                    "prefLabel": model_object_instance.pref_label,
                }
                references = {"hasTerminology": model_object_instance.terminology.id}
                self.concepts.data.insert(properties=properties,
                                          references=references,
                                          uuid=random_uuid)
            elif isinstance(model_object_instance, Mapping):
                properties = {
                    "text": model_object_instance.text,
                }
                references = {"hasConcept": model_object_instance.concept.concept_id}
                self.mappings.data.insert(properties=properties,
                                          vector=model_object_instance.embedding,
                                          references=references,
                                          uuid=random_uuid)
            else:
                raise ValueError("Unsupported model object instance type.")

        except Exception as e:
            raise RuntimeError(f"Failed to store object in Weaviate: {e}")

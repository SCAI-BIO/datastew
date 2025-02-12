import json
import logging
import shutil
import socket
import warnings
from typing import List, Literal, Optional, Union

import weaviate
from weaviate.classes.query import Filter, MetadataQuery, QueryReference
from weaviate.util import generate_uuid5

from datastew.embedding import EmbeddingModel
from datastew.process.parsing import DataDictionarySource
from datastew.repository import Concept, Mapping, Terminology
from datastew.repository.base import BaseRepository
from datastew.repository.model import MappingResult
from datastew.repository.pagination import Page
from datastew.repository.weaviate_schema import (
    concept_schema, mapping_schema_preconfigured_embeddings,
    mapping_schema_user_vectors, terminology_schema)


class WeaviateRepository(BaseRepository):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        bring_vectors: bool = True,
        huggingface_key: Optional[str] = None,
        mode: str = "memory",
        path: str = "db",
        port: int = 80,
        http_port: int = 8079,
        grpc_port: int = 50050,
    ):
        """Initialize the WeaviateRepository instance, connecting to either a local or remote Weaviate instance and
        setting up the appropriate schemas based on the specific options.

        :param bring_vectors: Specifies whether to use custom vectors provided by the user (True) or pre-configured
            embeddings (False). Defaults to True.
        :param huggingface_key: API key for Hugging Face if using pre-configured embeddings. Required if `bring_vectors`
            is False. Defaults to None.
        :param mode: Defines the connection mode for the repository. Can be either "disk" (local instance), "memory"
            (in-memory, using the same logic as "disk"), or "remote" (remote Weaviate instance). Defaults to "memory".
        :param path: The path for the local disk connection, used only in "disk" mode. Defaults to "db".
        :param port: The port number for remote Weaviate connection, used only in "remote" mode. Defaults to 80.
        :param http_port: The HTTP port for the local connection in "disk" mode. Defaults to 8079.
        :param grpc_port: The gRPC port for the local connection in "disk" mode. Defaults to 50050.

        :raises ValueError: If the `huggingface_key` is not provided when `bring_vectors` is False or if an invalid
            `mode` is specified.
        :raises RuntimeError: If there is a failure in creating the schema or connecting to Weaviate.
        """
        self.bring_vectors = bring_vectors
        self.mode = mode
        if not self.bring_vectors:
            if huggingface_key:
                self.headers = {"X-HuggingFace-Api-Key": huggingface_key}
            else:
                raise ValueError(
                    "A HuggingFace API key is required for generating vectors."
                )
        if self.mode == "disk" or self.mode == "memory":
            self._connect_to_disk(path, http_port, grpc_port)
        elif self.mode == "remote":
            self._connect_to_remote(path, port)
        else:
            raise ValueError(
                f"Repository mode {mode} is not defined. Use either disk or remote."
            )

        try:
            self._create_schema_if_not_exists(terminology_schema)
            self._create_schema_if_not_exists(concept_schema)
            if self.bring_vectors:
                self._create_schema_if_not_exists(mapping_schema_user_vectors)
            else:
                self._create_schema_if_not_exists(
                    mapping_schema_preconfigured_embeddings
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create schema: {e}")

    def import_data_dictionary(
        self,
        data_dictionary: DataDictionarySource,
        terminology_name: str,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """Imports a data dictionary into the Weaviate repository by converting the dictionary into a list of model
        instances (Terminology, Concept, and Mapping). Each variable in the data dictionary is mapped to a concept and
        a mapping. If `bring_vectors` is True, embeddings are generated using the provided or default embedding model
        for each variable's description. Otherwise pre-configured HuggingFace model(s) will be used.

        :param data_dictionary: The source data dictionary to be imported.
        :param terminology_name: The name assigned to the terminology being imported.
        :param embedding_model: An optional embedding model to be used for generating embeddings for the variables'
            descriptions. If None, a default model "sentence-transformers/all-mpnet-base-v2" will be used.

        :raises RuntimeError: If there is an error in importing the data dictionary or generating the embeddings.
        """
        try:
            # Initialize an empty list to store the model instances
            model_object_instances: List[Union[Terminology, Concept, Mapping]] = []

            # Convert the data dictionary to a dataframe for easier manipulation.
            data_frame = data_dictionary.to_dataframe()

            # Extract variables and descriptions from the dataframe
            variables = data_frame["variable"].tolist()
            descriptions = data_frame["description"].tolist()

            # Create a Terminology object and append it to the list of instances
            terminology = Terminology(terminology_name, terminology_name)
            model_object_instances.append(terminology)

            if self.bring_vectors:
                # If vectors are being used, select the embedding model (default or provided.)
                if embedding_model is None:
                    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
                else:
                    embedding_model_name = embedding_model.get_model_name()

                # Generate embeddings for each variable using the provided embedding model
                variable_to_embedding = data_dictionary.get_embeddings(embedding_model)

            # Create Concepts and Mappings for each variable with generated embeddings
            for variable, description in zip(variables, descriptions):
                concept_id = f"{terminology_name}:{variable}"
                concept = Concept(
                    terminology=terminology,
                    pref_label=variable,
                    concept_identifier=concept_id,
                )

                # If the user bring vectors, create the Mapping with the embedding
                if self.bring_vectors:
                    mapping = Mapping(
                        concept=concept,
                        text=description,
                        embedding=variable_to_embedding[variable],
                        sentence_embedder=embedding_model_name,
                    )
                else:
                    mapping = Mapping(
                        concept=concept,
                        text=description,
                    )

                # Add the created Concept and Mapping to the instances list
                model_object_instances.append(concept)
                model_object_instances.append(mapping)

            # Store all the created instances in Weaviate
            self.store_all(model_object_instances)
        except Exception as e:
            raise RuntimeError(f"Failed to import data dictionary source: {e}")

    def store_all(
        self, model_object_instances: List[Union[Terminology, Concept, Mapping]]
    ):
        """Stores a list of model objects (Terminology, Concept, Mapping) in the Weaviate database.

        :param model_object_instances: A list of model instances (Terminology, Concept, or Mapping) to be stored.
        """
        for instance in model_object_instances:
            self.store(instance)

    def get_iterator(self, collection: Literal["Concept", "Mapping", "Terminology"]):
        if collection == "Concept":
            return_references = QueryReference(link_on="hasTerminology")
        elif collection == "Mapping":
            return_references = QueryReference(link_on="hasConcept")
        elif collection == "Terminology":
            return_references = None
        else:
            raise ValueError(f"Collection {collection} is not supported.")
        return self.client.collections.get(collection).iterator(
            include_vector=True, return_references=return_references
        )

    def get_all_sentence_embedders(self) -> List[str]:
        """Retrieves the names of all sentence embedders used in the "Mapping" collection. If `self.bring_vectors` is
        True, it fetches the names directly from the objects in the collection. If `self.bring_vectors` is False, it
        retrieves the names from the vector keys in the embeddings returned by an object in the collection.

        :raises RuntimeError: If there is an issue fetching sentence embedders or vector configurations.
        :return: A list of sentence embedder names.
        """
        sentence_embedders = set()
        mapping_collection = self.client.collections.get("Mapping")
        try:
            if self.bring_vectors:
                # Fetch sentence embedders from the existing "Mapping" objects
                response = mapping_collection.query.fetch_objects()

                for o in response.objects:
                    sentence_embedders.add(o.properties.get("hasSentenceEmbedder"))
            else:
                # Fetch sentence embedders from the vector keys in the object response
                # Limiting to 1 object since we're only interest in the vector configuration
                response = mapping_collection.query.fetch_objects(
                    limit=1, include_vector=True
                )
                for o in response.objects:
                    sentence_embedders.add(o.vector.keys())
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

    def get_concepts(
        self, limit: int, offset: int, terminology_name: Optional[str] = None
    ) -> Page[Concept]:
        try:
            concept_collection = self.client.collections.get("Concept")

            total_count = (
                self.client.collections.get(concept_schema["class"])
                .aggregate.over_all(total_count=True)
                .total_count
            )

            # filter by terminology if set, otherwise return concepts for all terminologies
            if terminology_name is not None:
                if not self._terminology_exists(terminology_name):
                    raise ValueError(
                        f"Terminology '{terminology_name}' not found in available terminologies."
                    )
                filters = (
                    Filter.by_ref("hasTerminology")
                    .by_property("name")
                    .equal(terminology_name)
                )
            else:
                filters = None

            response = concept_collection.query.fetch_objects(
                limit=limit,
                offset=offset,
                filters=filters,
                return_references=QueryReference(link_on="hasTerminology"),
            )

            concepts = []
            for concept_data in response.objects:
                if concept_data.references:
                    terminology_data = concept_data.references[
                        "hasTerminology"
                    ].objects[0]
                    terminology_name = str(terminology_data.properties["name"])
                    terminology_id = str(terminology_data.uuid)
                    terminology = Terminology(terminology_name, terminology_id)

                id = str(concept_data.uuid)
                concept_name = str(concept_data.properties["prefLabel"])
                concept_id = str(concept_data.properties["conceptID"])

                concept = Concept(terminology, concept_name, concept_id, id)
                concepts.append(concept)

        except Exception as e:
            raise RuntimeError(f"Failed to fetch concepts: {e}")
        return Page[Concept](
            items=concepts, limit=limit, offset=offset, total_count=total_count
        )

    def get_all_concepts(self) -> List[Concept]:
        # will be infeasible to load the whole database into memory, we use pagination instead
        warnings.warn(
            "get_all_concepts is deprecated and will be removed in a future release. Use get_concepts instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    # TODO: Implement the function utilizing pre-configured vectorizers
    def get_mappings(
        self, terminology_name: Optional[str] = None, limit=1000, offset=0
    ) -> Page[Mapping]:
        mappings = []
        # filter by terminology if set, otherwise return concepts for all terminologies
        if terminology_name is not None:
            if not self._terminology_exists(terminology_name):
                raise ValueError(
                    f"Terminology '{terminology_name}' not found in available terminologies."
                )

        try:
            mapping_collection = self.client.collections.get("Mapping")
            if not terminology_name:
                response = mapping_collection.query.fetch_objects(
                    return_references=QueryReference(
                        link_on="hasConcept",
                        return_references=QueryReference(link_on="hasTerminology"),
                    ),
                    limit=limit,
                    offset=offset,
                    include_vector=True,
                )
            else:
                if not self._terminology_exists(terminology_name):
                    raise RuntimeError(
                        f"Terminology {terminology_name} does not exists"
                    )
                response = mapping_collection.query.fetch_objects(
                    filters=Filter.by_ref(link_on="hasConcept")
                    .by_ref(link_on="hasTerminology")
                    .by_property("name")
                    .equal(terminology_name),
                    return_references=QueryReference(
                        link_on="hasConcept",
                        return_references=QueryReference(link_on="hasTerminology"),
                    ),
                    limit=limit,
                    offset=offset,
                    include_vector=True,
                )

            for o in response.objects:
                if o.references:
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references[
                        "hasTerminology"
                    ].objects[0]
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
                    id=str(o.uuid),
                    text=str(o.properties["text"]),
                    concept=concept,
                    embedding=o.vector,
                    sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                )
                mappings.append(mapping)

            total_count = (
                self.client.collections.get(mapping_schema_user_vectors["class"])
                .aggregate.over_all(total_count=True)
                .total_count
            )

        except Exception as e:
            raise RuntimeError(f"Failed to fetch mappings: {e}")
        return Page[Mapping](
            items=mappings, limit=limit, offset=offset, total_count=total_count
        )

    # TODO: Implement the function utilizing pre-configured vectorizers
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
                    terminology_data = concept_data.references[
                        "hasTerminology"
                    ].objects[0]
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

    # TODO: Implement the function utilizing pre-configured vectorizers
    def get_closest_mappings_with_similarities(
        self, embedding, limit=5
    ) -> List[MappingResult]:
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
                    terminology_data = concept_data.references[
                        "hasTerminology"
                    ].objects[0]
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
                mappings_with_similarities.append(MappingResult(mapping, similarity))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch closest mappings with similarities: {e}"
            )
        return mappings_with_similarities

    # TODO: Implement the function utilizing pre-configured vectorizers
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
            mapping_collection = self.client.collections.get("Mapping")
            response = mapping_collection.query.near_vector(
                near_vector=embedding,
                filters=Filter.by_ref("hasConcept")
                .by_ref("hasTerminology")
                .by_property("name")
                .equal(terminology_name)
                & Filter.by_property("hasSentenceEmbedder").equal(
                    sentence_embedder_name
                ),
                return_references=QueryReference(
                    link_on="hasConcept",
                    return_references=QueryReference(link_on="hasTerminology"),
                ),
                limit=limit,
            )
            for o in response.objects:
                if o.references:
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references[
                        "hasTerminology"
                    ].objects[0]
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

    # TODO: Implement the function utilizing pre-configured vectorizers
    def get_terminology_and_model_specific_closest_mappings_with_similarities(
        self,
        embedding,
        terminology_name: str,
        sentence_embedder_name: str,
        limit: int = 5,
    ) -> List[MappingResult]:
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
                filters=Filter.by_ref("hasConcept")
                .by_ref("hasTerminology")
                .by_property("name")
                .equal(terminology_name)
                & Filter.by_property("hasSentenceEmbedder").equal(
                    sentence_embedder_name
                ),
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
                    terminology_data = concept_data.references[
                        "hasTerminology"
                    ].objects[0]
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
                mappings_with_similarities.append(MappingResult(mapping, similarity))
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch the closest mappings for terminology {terminology_name} and model {sentence_embedder_name}: {e}"
            )
        return mappings_with_similarities

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        try:
            if isinstance(model_object_instance, Terminology):
                if not self._terminology_exists(str(model_object_instance.name)):
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
                if not self._concept_exists(
                    str(model_object_instance.concept_identifier)
                ):
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
                    self.logger.info(
                        f"Concept with identifier {model_object_instance.concept_identifier} already exists. Skipping."
                    )
            elif isinstance(model_object_instance, Mapping):
                if not self._mapping_exists(model_object_instance):
                    if not self._concept_exists(
                        model_object_instance.concept.concept_identifier
                    ):
                        self.store(model_object_instance.concept)
                    if self.bring_vectors:
                        properties = {
                            "text": model_object_instance.text,
                            "hasSentenceEmbedder": model_object_instance.sentence_embedder,
                        }
                    else:
                        properties = {"text": model_object_instance.text}
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
                    self.logger.info(
                        "Mapping with same embedding already exists. Skipping."
                    )
            else:
                raise ValueError("Unsupported model object instance type.")

        except Exception as e:
            raise RuntimeError(f"Failed to store object in Weaviate: {e}")

    def import_json(self, input_path: str):
        return None

    def close(self):
        self.client.close()

    def shut_down(self):
        if self.mode == "memory":
            shutil.rmtree("db")

    def _sentence_embedder_exists(self, name: str) -> bool:
        try:
            mapping = self.client.collections.get("Mapping")
            if self.bring_vectors:
                response = mapping.query.fetch_objects(
                    filters=Filter.by_property("hasSentenceEmbedder").equal(name)
                )
                if response.objects is not None:
                    return len(response.objects) > 0
                else:
                    return False
            else:
                sentence_embedders = set(self.get_all_sentence_embedders())
                return name in sentence_embedders
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

    def _mapping_exists(self, mapping: Mapping) -> bool:
        """Check if an exact Mapping object already exists in the Weaviate database based on its `text`, associated
        `concept`, and optional vector matching.

        :param mapping: The Mapping object to check for existence.
        :raises RuntimeError: If an error occurs while querying the database or fetching objects.
        :return: `True` if a Mapping object with the same text, concept, and optionally the same embedding(s) already
            exists, otherwise `False`.
        """
        try:
            # Check if the concept exists first (because every mapping should have a related concept)
            concept_exists = self._concept_exists(mapping.concept.concept_identifier)
            if not concept_exists:
                return False

            # Prepare filters for fetching mappings with the same text and concept
            mapping_collection = self.client.collections.get("Mapping")
            filters = Filter.by_property("text").equal(str(mapping.text))
            filters = filters & Filter.by_ref("hasConcept").by_property(
                "conceptID"
            ).equal(mapping.concept.concept_identifier)
            filters = filters & Filter.by_ref("hasConcept").by_property(
                "prefLabel"
            ).equal(mapping.concept.pref_label)

            # Fetch mappings based on text and concept
            response = mapping_collection.query.fetch_objects(filters=filters)

            # Check if any mappings are returned with the same text and concept
            if response.objects:
                # If `self.bring_vectors` is True, compare embeddings
                if self.bring_vectors:
                    for obj in response.objects:
                        if obj.vector == mapping.embedding:
                            return True
                else:
                    # If `self.bring_vectors` is False, compare the named vectors (if applicable)
                    for obj in response.objects:
                        if mapping.embedding is not None:
                            for vector_name in obj.vector.keys():
                                if mapping.embedding == obj.vector[vector_name]:
                                    return True
            return False  # No matching Mapping found in the collection

        except Exception as e:
            raise RuntimeError(f"Failed to check if mapping exists: {e}")

    def import_from_json(self, json_path: str, object_type: str):
        """
        Imports data from a JSON file and stores it in the Weaviate database.

        Parameters:
        - json_path: Path to the JSON file.
        - object_type: The type of objects to import ("terminology", "concept", "mapping").

        Returns:
        - None
        """
        try:
            with open(json_path, "r") as file:
                data = json.load(file)

            if not isinstance(data, list):
                data = [data]

            collection = self.client.collections.get(object_type.capitalize())

            with collection.batch.dynamic() as batch:
                for item in data:
                    try:
                        object_id = item["id"]
                        properties = item["properties"]
                        vector = item.get("vector", {}).get("default")
                        references = item.get("references")

                        batch.add_object(
                            uuid=object_id,
                            properties=properties,
                            vector=vector,
                            references=references,
                        )
                    except KeyError as e:
                        print(f"Skipping object due to missing key: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at path: {json_path}")
        except ValueError as e:
            raise ValueError(f"Error in loading JSON file: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")

        return None

    def _create_schema_if_not_exists(self, schema):
        """Creates a new schema in Weaviate if it does not already exist. The schema is defined in the provided `schema`
        dictionary. The schema should define a class along with its properties and possible references. If the schema
        already exists, the method logs a message and skips the creation process.

        :param schema: A dictionary defining the schema to be created. It should include the following keys:
            - "class": The name of the class for the schema (e.g., "Terminology", "Concept", or "Mapping")
            - "description": A description of the class.
            - "properties": A list of `Property` objects defining the properties for the schema. Each `Property` includes:
                - `name`: The name of the property.
                - `data_type`: The data type of the property (e.g., `DataType.TEXT`).
            - "references" (optional): A list of `ReferenceProperty` objects, defining relationships to other classes.
            - "vectorizer_config" (optional): A configuration for vectorization (only used when `bring_vectors` is
                False), typically a list of `Configure` objects like `Configure.NamedVectors.text2vec_huggingface`.

        :raises RuntimeError: If there is an issue checking for or creating the schema in Weaviate, such as connection
            error.
        """
        references = None
        vectorizer_config = None
        class_name = schema["class"]
        try:
            if not self.client.collections.exists(class_name):
                description = schema["description"]
                properties = schema["properties"]
                if "references" in schema:
                    references = schema["references"]
                if "vectorizer_config" in schema and not self.bring_vectors:
                    vectorizer_config = schema["vectorizer_config"]
                self.client.collections.create(
                    name=class_name,
                    description=description,
                    properties=properties,
                    references=references,
                    vectorizer_config=vectorizer_config,
                )
            else:
                self.logger.info(f"Schema for {class_name} already exists. Skipping.")
        except Exception as e:
            raise RuntimeError(f"Failed to check/create schema for {class_name}: {e}")

    @staticmethod
    def _is_port_in_use(port) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _connect_to_disk(self, path: str, http_port: int, grpc_port: int):
        try:
            if path is None:
                raise ValueError("Path must be provided for disk mode.")
            if self._is_port_in_use(http_port) and self._is_port_in_use(grpc_port):
                if self.client.is_connected():
                    self.client.close()
                if self.bring_vectors:
                    self.client = weaviate.connect_to_local(
                        port=http_port, grpc_port=grpc_port
                    )
                else:
                    self.client = weaviate.connect_to_local(
                        port=http_port, grpc_port=grpc_port, headers=self.headers
                    )
            else:
                if self.bring_vectors:
                    self.client = weaviate.connect_to_embedded(
                        persistence_data_path=path
                    )
                else:
                    self.client = weaviate.connect_to_embedded(
                        persistence_data_path=path, headers=self.headers
                    )
        except Exception as e:
            raise ConnectionError(f"Failed to initalize Weaviate client: {e}")

    def _connect_to_remote(self, path: str, port: int):
        try:
            if path is None:
                raise ValueError("Remote URL must be provided for remote mode.")
            if self.bring_vectors:
                self.client = weaviate.connect_to_local(host=path, port=port)
            else:
                self.client = weaviate.connect_to_local(
                    host=path, port=port, headers=self.headers
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initalize Weaviate client: {e}")

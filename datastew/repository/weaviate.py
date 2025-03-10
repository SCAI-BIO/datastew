import json
import logging
import shutil
import socket
import warnings
from typing import List, Literal, Optional, Sequence, Union

import weaviate
from weaviate import WeaviateClient
from weaviate.classes.query import Filter, MetadataQuery, QueryReference
from weaviate.util import generate_uuid5

from datastew.embedding import EmbeddingModel, MPNetAdapter
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
        use_weaviate_vectorizer: bool = False,
        huggingface_key: Optional[str] = None,
        mode: str = "memory",
        path: str = "db",
        port: int = 80,
        http_port: int = 8079,
        grpc_port: int = 50050,
    ):
        """Initialize the WeaviateRepository instance, connecting to either a local or remote Weaviate instance and
        setting up the appropriate schemas based on the specific options.

        :param use_weaviate_vectorizer: Specifies whether to use pre-configured embeddings (True) or custom vectors
            provided by the user (False). Defaults to False.
        :param huggingface_key: API key for Hugging Face if using pre-configured embeddings. Required if `use_weaviate_vectorizer`
            is True. Defaults to None.
        :param mode: Defines the connection mode for the repository. Can be either "memory" (in-memory), or "remote"
            (remote Weaviate instance). Defaults to "memory".
        :param path: The path for the local disk connection, used only in "memory" mode. Defaults to "db".
        :param port: The port number for remote Weaviate connection, used only in "remote" mode. Defaults to 80.
        :param http_port: The HTTP port for the local connection in "memory" mode. Defaults to 8079.
        :param grpc_port: The gRPC port for the local connection in "memory" mode. Defaults to 50050.

        :raises ValueError: If the `huggingface_key` is not provided when `use_weaviate_vectorizer` is True or if an
            invalid `mode` is specified.
        :raises RuntimeError: If there is a failure in creating the schema or connecting to Weaviate.
        """
        self.use_weaviate_vectorizer = use_weaviate_vectorizer
        self.mode = mode
        self.client: Optional[WeaviateClient] = None
        self.headers = None
        if self.use_weaviate_vectorizer:
            if huggingface_key:
                self.headers = {"X-HuggingFace-Api-Key": huggingface_key}
            else:
                raise ValueError(
                    "A HuggingFace API key is required for generating vectors."
                )
        if self.mode == "memory":
            self._connect_to_memory(path, http_port, grpc_port)
        elif self.mode == "remote":
            self._connect_to_remote(path, port)
        else:
            raise ValueError(
                f"Repository mode {mode} is not defined. Use either disk or remote."
            )

        try:
            self._create_schema_if_not_exists(terminology_schema)
            self._create_schema_if_not_exists(concept_schema)
            if not self.use_weaviate_vectorizer:
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
        a mapping. If `use_weaviate_vectorizer` is False, embeddings are generated using the provided or default
        embedding model for each variable's description. Otherwise pre-configured HuggingFace model(s) will be used.

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

            if not self.use_weaviate_vectorizer:
                # If vectors are being used, select the embedding model (default or provided.)
                if embedding_model is None:
                    embedding_model = MPNetAdapter()

                embedding_model_name = embedding_model.model_name

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
                if not self.use_weaviate_vectorizer:
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
        """Retrieves the names of all sentence embedders used in the "Mapping" collection. If
        `self.use_weaviate_vectorizer` is False, it fetches the names directly from the objects in the collection. If
        `self.use_weaviate_vectorizer` is True, it retrieves the names from the vector keys in the embeddings returned
        by an object in the collection.

        :raises ValueError: If the client is not initialized.
        :raises RuntimeError: If there is an issue fetching sentence embedders or vector configurations.
        :return: A list of sentence embedder names.
        """
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
        sentence_embedders = set()
        mapping_collection = self.client.collections.get("Mapping")
        try:
            if not self.use_weaviate_vectorizer:
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
                    sentence_embedders.update(o.vector.keys())
        except Exception as e:
            raise RuntimeError(f"Failed to fetch sentence embedders: {e}")
        return list(sentence_embedders)

    def get_concept(self, concept_id: str) -> Concept:
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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

    def get_mappings(
        self,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit=1000,
        offset=0,
    ) -> Page[Mapping]:
        """Fetches a list of mappings from the Weaviate client, with optional filters for terminology and sentence
        embedder. The function can limit and offset the number of results returned

        :param terminology_name: The name of the terminology to filter the mappings, defaults to None
        :param sentence_embedder: The name of the sentence embedder to filter the mappings. Required if
            `use_weaviate_vectorizer` is `True`, defaults to None
        :param limit: The maximum number of mappings to return, defaults to 1000
        :param offset: The number of mappings to skip before returning results, defaults to 0
        :raises ValueError: If the client is not initialized or is invalid.
        :raises ValueError: If the terminology is not found.
        :raises ValueError: If the sentence embedder is not found.
        :raises ValueError: If `sentence_embedder` is `None` and `use_weaviate_vectorizer` is `True`.
        :raises RuntimeError: If the fetch operation fails.
        :return: A page object containing a list of Mapping objects, along with pagination details.
        """
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")

        mappings = []  # List to store fetched mappings
        filters = None  # List to store filters for query
        target_vector = True  # Whether to include vectors in the response

        # Apply terminology filter if specified
        if terminology_name:
            if not self._terminology_exists(terminology_name):
                raise ValueError(
                    f"Terminology '{terminology_name}' not found in available terminologies."
                )
            if filters:
                filters.append(
                    Filter.by_ref("hasConcept")
                    .by_ref("hasTerminology")
                    .by_property("name")
                    .equal(terminology_name)
                )
            else:
                filters = [
                    Filter.by_ref("hasConcept")
                    .by_ref("hasTerminology")
                    .by_property("name")
                    .equal(terminology_name)
                ]

        # Apply sentence embedder filter if specified
        if sentence_embedder:
            if not self._sentence_embedder_exists(sentence_embedder):
                raise ValueError(
                    f"Vectorizer '{sentence_embedder}' not found in available vectorizers."
                )
            # Add filter for sentence embedder only if vectors should be included
            if not self.use_weaviate_vectorizer:
                if filters:
                    filters.append(
                        Filter.by_property("hasSentenceEmbedder").equal(
                            sentence_embedder
                        )
                    )
                else:
                    filters = [
                        Filter.by_property("hasSentenceEmbedder").equal(
                            sentence_embedder
                        )
                    ]
            else:
                target_vector = sentence_embedder
        else:
            if self.use_weaviate_vectorizer:
                raise ValueError(
                    "Sentence embedder cannot be `None` while `self.use_weaviate_vectorizer` is `True`."
                )

        try:
            # Get the mapping collection from the Weaviate client
            mapping_collection = self.client.collections.get("Mapping")

            # Fetch objects based on the filters, limit, and offset
            response = mapping_collection.query.fetch_objects(
                filters=Filter.all_of(filters) if filters else None,
                limit=limit,
                offset=offset,
                include_vector=target_vector,
                return_references=QueryReference(
                    link_on="hasConcept",
                    return_references=QueryReference(link_on="hasTerminology"),
                ),
            )

            # Process the response objects into Mapping objects
            for o in response.objects:
                if o.references:
                    # Extract concept and terminology data
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

                # Create a Mapping object and add to the mappings list
                if not self.use_weaviate_vectorizer:
                    mapping = Mapping(
                        id=str(o.uuid),
                        text=str(o.properties["text"]),
                        concept=concept,
                        embedding=o.vector,
                        sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                    )
                else:
                    mapping = Mapping(
                        id=str(o.uuid),
                        text=str(o.properties["text"]),
                        concept=concept,
                        embedding=o.vector,
                    )
                mappings.append(mapping)

            # Fetch the total count of mappings for pagination
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

    def get_closest_mappings(
        self,
        embedding: Sequence[float],
        similarities: bool = False,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit=5,
    ) -> Union[List[Mapping], List[MappingResult]]:
        """Fetches the closest mappings based on an embedding vector, with optional filters for terminology and sentence
        embedder.

        :param embedding: The embedding vector to find the closest mappings.
        :param similarities: Whether to include similarity scores in the result, defaults to `False`.
        :param terminology_name: The name of the terminology to filter the mappings, defaults to None.
        :param sentence_embedder: The name of the sentence embedder to filter the mappings. Required if
            `use_weaviate_vectorizer` is `True`, defaults to None.
        :param limit: The Maximum number of closest mappings to return, defaults to 5
        :raises ValueError: If the client is not initialized.
        :raises ValueError: If terminology does not exist.
        :raises ValueError: If sentence embedder does not exist.
        :raises ValueError: If sentence embedder is not set while `use_weaviate_vectorizer` is `True`.
        :raises RuntimeError: If the fetch operation fails
        :return: A list of Mapping or MappingResult objects based on whether similarity scores are included.
        """
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")

        mappings = []  # List to store fetched mappings
        filters = None
        target_vector = None

        # Apply terminology filter if specified
        if terminology_name:
            if not self._terminology_exists(terminology_name):
                raise ValueError(
                    f"Terminology '{terminology_name}' not found in available terminologies."
                )
            if filters:
                filters.append(
                    Filter.by_ref("hasConcept")
                    .by_ref("hasTerminology")
                    .by_property("name")
                    .equal(terminology_name)
                )
            else:
                filters = [
                    Filter.by_ref("hasConcept")
                    .by_ref("hasTerminology")
                    .by_property("name")
                    .equal(terminology_name)
                ]

        # Apply sentence embedder filter if specified
        if sentence_embedder:
            if not self._sentence_embedder_exists(sentence_embedder):
                raise ValueError(
                    f"Vectorizer '{sentence_embedder}' not found in available vectorizers."
                )
            # Add filter for sentence embedder only if vectors should be included
            if not self.use_weaviate_vectorizer:
                if filters:
                    filters.append(
                        Filter.by_property("hasSentenceEmbedder").equal(
                            sentence_embedder
                        )
                    )
                else:
                    filters = [
                        Filter.by_property("hasSentenceEmbedder").equal(
                            sentence_embedder
                        )
                    ]
            else:
                target_vector = sentence_embedder
        else:
            if self.use_weaviate_vectorizer:
                raise ValueError(
                    "Sentence embedder cannot be `None` while `self.use_weaviate_vectorizer` is `True`."
                )
        try:
            # Get the mapping collection from the Weaviate client
            mapping_collection = self.client.collections.get("Mapping")

            # Perform the vector search to fetch the closest mappings
            response = mapping_collection.query.near_vector(
                near_vector=embedding,
                limit=limit,
                return_metadata=MetadataQuery(distance=True) if similarities else None,
                return_references=QueryReference(
                    link_on="hasConcept",
                    return_references=QueryReference(link_on="hasTerminology"),
                ),
                target_vector=target_vector,
                filters=Filter.all_of(filters) if filters else None,
            )

            # Process the response objects into Mapping or MappingResult objects
            for o in response.objects:
                # Calculate similarity based on distance if similarities are requested
                if o.metadata.distance:
                    similarity = 1 - o.metadata.distance
                if o.references:
                    # Extract concept and terminology data
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

                # Create Mapping objects and append them to the list
                if not self.use_weaviate_vectorizer:
                    mapping = Mapping(
                        concept=concept,
                        text=str(o.properties["text"]),
                        embedding=o.vector,
                        sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                    )
                else:
                    mapping = Mapping(
                        concept=concept,
                        text=str(o.properties["text"]),
                        embedding=o.vector,
                    )

                # Append MappingResult if similarities is True, else just Mapping
                if similarities:
                    mappings.append(MappingResult(mapping, similarity))
                else:
                    mappings.append(mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch closest mappings: {e}")
        return mappings

    def get_closest_mappings_with_similarities(
        self,
        embedding: Sequence[float],
        sentence_embedder: Optional[str] = None,
        limit=5,
    ) -> List[MappingResult]:
        """Fetches the closest mappings based on an embedding vector and includes similarity scores for each mapping.

        :param embedding: The embedding vector to find the closest mappings.
        :param sentence_embedder: The name of the sentence embedder to filter the mappings. Required if
            `use_weaviate_vectorizer` is `True`, defaults to None.
        :param limit: The maximum number of closest mappings to return, defaults to 5.
        :return: A list of MappingResult objects, each containing a mapping and its similarity score.
        """
        # Emit a deprecation warning as this function is deprecated
        warnings.warn(
            "get_closest_mappings_with_similarities is deprecated and will be removed in a future release. Use get_closest_mappings instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_closest_mappings(
            embedding=embedding,
            similarities=True,
            sentence_embedder=sentence_embedder,
            limit=limit,
        )

    def get_terminology_and_model_specific_closest_mappings(
        self,
        embedding: Sequence[float],
        terminology_name: str,
        sentence_embedder_name: str,
        limit: int = 5,
    ) -> List[Mapping]:
        """Fetches the closest mappings for a given terminology and sentence embedder model.

        This function is deprecated and will be removed in a future release. It is recommended to use
        `get_closest_mappings` instead.

        :param embedding: The embedding vector to find the closest mappings.
        :param terminology_name: The name of the terminology to filter the mappings.
        :param sentence_embedder_name: The name of the sentence embedder to filter the mappings.
        :param limit: The maximum number of closest mappings to return, defaults to 5
        :return: A list of the closest Mapping objects that match the specific filters.
        """
        # Emit a deprecation warning as this function is deprecated
        warnings.warn(
            "get_terminology_and_model_specific_closest_mappings is deprecated and will be removed in a future release. Use get_closest_mappings instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_closest_mappings(
            embedding=embedding,
            terminology_name=terminology_name,
            sentence_embedder=sentence_embedder_name,
            limit=limit,
        )

    def get_terminology_and_model_specific_closest_mappings_with_similarities(
        self,
        embedding: Sequence[float],
        terminology_name: str,
        sentence_embedder_name: str,
        limit: int = 5,
    ) -> List[MappingResult]:
        """Fetches the closest mappings for a given terminology and sentence embedder model, includes similarity scores.

        This function is deprecated and will be removed in a future release. It is recommended to use
        `get_closest_mappings` instead.

        :param embedding: The embedding vector to find the closest mappings.
        :param terminology_name: The name of the terminology to filter the mappings.
        :param sentence_embedder_name: The name of the sentence embedder to filter the mappings.
        :param limit: The maximum number closest mapping to return, defaults to 5
        :return: A list of MappingResults objects, each containing a mapping and its similarity score.
        """
        # Emit a deprecation warning as this function is deprecated
        warnings.warn(
            "get_terminology_and_model_specific_closest_mappings_with_similarities is deprecated and will be removed in a future release. Use get_closest_mappings instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_closest_mappings(
            embedding=embedding,
            similarities=True,
            terminology_name=terminology_name,
            sentence_embedder=sentence_embedder_name,
            limit=limit,
        )

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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
                    if not self.use_weaviate_vectorizer:
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

    def import_from_json(self, json_path: str, object_type: str, chunk_size: int = 100):
        """
        Imports data from a JSON file and stores it in the Weaviate database.

        :param json_path: Path to the JSON file.
        :param object_type: The type of objects to import ("terminology", "concept", "mapping").
        :param chunk_size: The number of items to process in each batch.
        """
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
        try:
            collection = self.client.collections.get(object_type.capitalize())
            chunk = []

            # Open and load the entire JSON file
            with open(json_path, "r") as file:
                data = json.load(file)  # Load the entire file into memory

            # Ensure the data is in a list
            if not isinstance(data, list):
                data = [data]

            # Iterate over the data in chunks
            for item in data:
                chunk.append(item)
                if len(chunk) >= chunk_size:
                    with collection.batch.dynamic() as batch:
                        for item in chunk:
                            object_id = item["id"]
                            properties = item["properties"]
                            vector = (
                                None
                                if self.use_weaviate_vectorizer
                                else item.get("vector", {}).get("default")
                            )
                            references = item.get("references")

                            batch.add_object(
                                uuid=object_id,
                                properties=properties,
                                vector=vector,
                                references=references,
                            )
                    chunk = []

            # Process the remaining items in the last chunk
            if chunk:
                with collection.batch.dynamic() as batch:
                    for item in chunk:
                        object_id = item["id"]
                        properties = item["properties"]
                        vector = (
                            None
                            if self.use_weaviate_vectorizer
                            else item.get("vector", {}).get("default")
                        )
                        references = item.get("references")
                        batch.add_object(
                            uuid=object_id,
                            properties=properties,
                            vector=vector,
                            references=references,
                        )

        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at path: {json_path}")
        except ValueError as e:
            raise ValueError(f"Error in loading JSON file: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred: {e}")

    def close(self):
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
        self.client.close()

    def shut_down(self):
        if self.mode == "memory":
            shutil.rmtree("db")
        else:
            self.close()

    def _sentence_embedder_exists(self, name: str) -> bool:
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
        try:
            mapping = self.client.collections.get("Mapping")
            if not self.use_weaviate_vectorizer:
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
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
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
            if not self.client:
                raise ValueError("Client is not initialized or is invalid.")
            # Check if the concept exists first
            concept_exists = self._concept_exists(mapping.concept.concept_identifier)
            if not concept_exists:
                return False

            mapping_collection = self.client.collections.get("Mapping")

            # Fetch mappings based on text and concept
            response = mapping_collection.query.fetch_objects(
                filters=(
                    Filter.all_of(
                        [
                            Filter.by_property("text").equal(str(mapping.text)),
                            Filter.by_ref("hasConcept")
                            .by_property("conceptID")
                            .equal(mapping.concept.concept_identifier),
                        ]
                    )
                )
            )
            if response.objects is not None:
                return len(response.objects) > 0
            return False

        except Exception as e:
            raise RuntimeError(f"Failed to check if mapping exists: {e}")

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
            - "vectorizer_config" (optional): A configuration for vectorization (only used when
                `use_weaviate_vectorizer` is True), typically a list of `Configure` objects like
                `Configure.NamedVectors.text2vec_huggingface`.

        :raises RuntimeError: If there is an issue checking for or creating the schema in Weaviate, such as connection
            error.
        """
        if not self.client:
            raise ValueError("Client is not initialized or is invalid.")
        references = None
        vectorizer_config = None
        class_name = schema["class"]
        try:
            if not self.client.collections.exists(class_name):
                description = schema["description"]
                properties = schema["properties"]
                if "references" in schema:
                    references = schema["references"]
                if "vectorizer_config" in schema and self.use_weaviate_vectorizer:
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

    def _connect_to_memory(self, path: str, http_port: int, grpc_port: int):
        try:
            if path is None:
                raise ValueError("Path must be provided for disk mode.")
            if self._is_port_in_use(http_port) and self._is_port_in_use(grpc_port):
                if self.client:
                    self.client.close()
                self.client = weaviate.connect_to_local(
                    port=http_port, grpc_port=grpc_port, headers=self.headers
                )
            else:
                self.client = weaviate.connect_to_embedded(
                    persistence_data_path=path, headers=self.headers
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {e}")

    def _connect_to_remote(self, path: str, port: int):
        try:
            if path is None:
                raise ValueError("Remote URL must be provided for remote mode.")
            self.client = weaviate.connect_to_local(
                host=path, port=port, headers=self.headers
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {e}")

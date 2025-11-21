import json
import logging
import socket
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import weaviate
from typing_extensions import deprecated
from weaviate import WeaviateClient
from weaviate.classes.query import Filter, MetadataQuery, QueryReference
from weaviate.collections import Collection
from weaviate.util import generate_uuid5

from datastew.embedding import Vectorizer
from datastew.exceptions import ObjectStorageError
from datastew.repository import Concept, Mapping, Terminology
from datastew.repository.base import BaseRepository
from datastew.repository.model import MappingResult
from datastew.repository.pagination import Page
from datastew.repository.weaviate_schema import (
    ConceptSchema,
    MappingSchema,
    TerminologySchema,
    concept_schema,
    mapping_schema_preconfigured_embeddings,
    mapping_schema_user_vectors,
    terminology_schema,
)


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
        terminology_schema: TerminologySchema = terminology_schema,
        concept_schema: ConceptSchema = concept_schema,
        mapping_schema: MappingSchema = mapping_schema_user_vectors,
        vectorizer: Vectorizer = Vectorizer(),
    ):
        """Initialize the WeaviateRepository instance, connecting to either a local or remote Weaviate instance and
        setting up the appropriate schemas based on the specific options.

        :param use_weaviate_vectorizer: Specifies whether to use pre-configured embeddings (True) or custom vectors
            provided by the user (False). Defaults to False.
        :param huggingface_key: API key for Hugging Face if using pre-configured embeddings. Required if
            `use_weaviate_vectorizer` is True. Defaults to None.
        :param mode: Defines the connection mode for the repository. Can be either "memory" (in-memory), or "remote"
            (remote Weaviate instance). Defaults to "memory".
        :param path: The path for the local disk connection, used only in "memory" mode. Defaults to "db".
        :param port: The port number for remote Weaviate connection, used only in "remote" mode. Defaults to 80.
        :param http_port: The HTTP port for the local connection in "memory" mode. Defaults to 8079.
        :param grpc_port: The gRPC port for the local connection in "memory" mode. Defaults to 50050.
        :param terminology_schema: Terminology schema to use for the repository. Defaults to pre-configured
            `terminology_schema`.
        :param concept_schema: Concept schema to use for the repository. Defaults to pre-configured `concept_schema`.
        :param mapping_schema: Mapping schema to use for the repository. Defaults to pre-configured
            `mapping_schema_user_vectors`. If `use_weaviate_vectorizer` is set to True and `mapping_schema` does not
            include a `vectorizer_config` key, the schema will default to pre-configured
            `mapping_schema_preconfigured_embeddings`.
        :raises ValueError: If the `huggingface_key` is not provided when `use_weaviate_vectorizer` is True or if an
            invalid `mode` is specified.
        :raises RuntimeError: If there is a failure in creating the schema or connecting to Weaviate.
        """
        super().__init__(vectorizer)
        self.use_weaviate_vectorizer = use_weaviate_vectorizer
        self.mode = mode
        self.terminology_schema = terminology_schema
        self.concept_schema = concept_schema
        self.mapping_schema = mapping_schema
        self.client: WeaviateClient
        self.headers = None
        if self.use_weaviate_vectorizer:
            if huggingface_key:
                self.headers = {"X-HuggingFace-Api-Key": huggingface_key}
        if self.mode == "memory":
            self.client = self._connect_to_memory(path, http_port, grpc_port)
        elif self.mode == "remote":
            self.client = self._connect_to_remote(path, port)
        else:
            raise ValueError(f"Repository mode {mode} is not defined. Use either disk or remote.")

        try:
            self._create_schema_if_not_exists(self.terminology_schema.schema)
            self._create_schema_if_not_exists(self.concept_schema.schema)
            if self.use_weaviate_vectorizer and not self.mapping_schema.schema["vectorizer_config"]:
                self.logger.warning(
                    "Provided mapping schema lacks `vectorizer_config` even though"
                    "`use_weaviate_vectorizer` is set to True. Defaulting to"
                    f"{mapping_schema_preconfigured_embeddings.schema}"
                )
                self.mapping_schema = mapping_schema_preconfigured_embeddings
            self._create_schema_if_not_exists(self.mapping_schema.schema)

        except Exception as e:
            raise RuntimeError(f"Failed to create schema: {e}")

    def store_all(self, model_object_instances: List[Union[Terminology, Concept, Mapping]]):
        """Stores a list of model objects (Terminology, Concept, Mapping) in the Weaviate database.

        :param model_object_instances: A list of model instances (Terminology, Concept, or Mapping) to be stored.
        """
        for instance in model_object_instances:
            self.store(instance)

    def get_iterator(self, collection: Literal["Concept", "Mapping", "Terminology"]):
        """Retrieves an iterator for the specified collection from the client.

        :param collection: The collection type to retrieve the iterator for.
        :raises ValueError: If the specified collection is not supported.
        :return: An iterator for the specified collection from the client's collection,
            with vectors and references as needed.
        """
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

        :raises RuntimeError: If there is an issue fetching sentence embedders or vector configurations.
        :return: A list of sentence embedder names.
        """
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
                response = mapping_collection.query.fetch_objects(limit=1, include_vector=True)
                for o in response.objects:
                    sentence_embedders.update(o.vector.keys())
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

    def get_concepts(self, limit: int = 100, offset: int = 0, terminology_name: Optional[str] = None) -> Page[Concept]:
        try:
            concept_collection = self.client.collections.get("Concept")

            total_count = (
                self.client.collections.get(self.concept_schema.schema["class"])
                .aggregate.over_all(total_count=True)
                .total_count
            )
            if total_count is None:
                raise ValueError("Total count of concepts could not be retrieved.")
            # filter by terminology if set, otherwise return concepts for all terminologies
            if terminology_name is not None:
                if not self._terminology_exists(terminology_name):
                    raise ValueError(f"Terminology '{terminology_name}' not found in available terminologies.")
                filters = Filter.by_ref("hasTerminology").by_property("name").equal(terminology_name)
            else:
                filters = None

            response = concept_collection.query.fetch_objects(
                limit=limit, offset=offset, filters=filters, return_references=QueryReference(link_on="hasTerminology")
            )

            concepts = []
            for concept_data in response.objects:
                if concept_data.references:
                    terminology_data = concept_data.references["hasTerminology"].objects[0]
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
        return Page[Concept](items=concepts, limit=limit, offset=offset, total_count=total_count)

    def get_terminology(self, terminology_name: str) -> Terminology:
        try:
            if not self._terminology_exists(terminology_name):
                raise RuntimeError(f"Terminology {terminology_name} does not exists")
            terminology_collection = self.client.collections.get("Terminology")
            response = terminology_collection.query.fetch_objects(
                filters=Filter.by_property("name").equal(terminology_name)
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
                terminology = Terminology(name=str(o.properties["name"]), id=str(o.uuid))
                terminologies.append(terminology)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch terminologies: {e}")
        return terminologies

    def get_mappings(
        self,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Page[Mapping]:
        """Fetches a list of mappings from the Weaviate client, with optional filters for terminology and sentence
        embedder. The function can limit and offset the number of results returned

        :param terminology_name: The name of the terminology to filter the mappings, defaults to None
        :param sentence_embedder: The name of the sentence embedder to filter the mappings. Required if
            `use_weaviate_vectorizer` is `True`, defaults to None
        :param limit: The maximum number of mappings to return, defaults to 1000
        :param offset: The number of mappings to skip before returning results, defaults to 0
        :raises ValueError: If the terminology is not found.
        :raises ValueError: If the sentence embedder is not found.
        :raises ValueError: If `sentence_embedder` is `None` and `use_weaviate_vectorizer` is `True`.
        :raises RuntimeError: If the fetch operation fails.
        :return: A page object containing a list of Mapping objects, along with pagination details.
        """
        mappings = []  # List to store fetched mappings
        filters = None  # List to store filters for query
        target_vector = True  # Whether to include vectors in the response

        # Apply terminology filter if specified
        if terminology_name:
            if not self._terminology_exists(terminology_name):
                raise ValueError(f"Terminology '{terminology_name}' not found in available terminologies.")
            if filters:
                filters.append(
                    Filter.by_ref("hasConcept").by_ref("hasTerminology").by_property("name").equal(terminology_name)
                )
            else:
                filters = [
                    Filter.by_ref("hasConcept").by_ref("hasTerminology").by_property("name").equal(terminology_name)
                ]

        # Apply sentence embedder filter if specified
        if sentence_embedder:
            if not self._sentence_embedder_exists(sentence_embedder):
                raise ValueError(f"Vectorizer '{sentence_embedder}' not found in available vectorizers.")
            # Add filter for sentence embedder only if vectors should be included
            if not self.use_weaviate_vectorizer:
                if filters:
                    filters.append(Filter.by_property("hasSentenceEmbedder").equal(sentence_embedder))
                else:
                    filters = [Filter.by_property("hasSentenceEmbedder").equal(sentence_embedder)]
            else:
                target_vector = sentence_embedder
        else:
            if self.use_weaviate_vectorizer:
                raise ValueError("Sentence embedder cannot be `None` while `self.use_weaviate_vectorizer` is `True`.")

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
                    link_on="hasConcept", return_references=QueryReference(link_on="hasTerminology")
                ),
            )

            # Process the response objects into Mapping objects
            for o in response.objects:
                if o.references:
                    # Extract concept and terminology data
                    concept_data = o.references["hasConcept"].objects[0]
                    terminology_data = concept_data.references["hasTerminology"].objects[0]
                    terminology = Terminology(
                        name=str(terminology_data.properties["name"]), id=str(terminology_data.uuid)
                    )
                    concept = Concept(
                        concept_identifier=str(concept_data.properties["conceptID"]),
                        pref_label=str(concept_data.properties["prefLabel"]),
                        terminology=terminology,
                        id=str(concept_data.uuid),
                    )

                id = str(o.uuid)
                text = str(o.properties["text"])
                # o.vector is a dictionary with varying key
                embedding = next(iter(o.vector.values()), None)

                # Create a Mapping object and add to the mappings list
                if not self.use_weaviate_vectorizer:
                    mapping = Mapping(
                        id=id,
                        text=text,
                        concept=concept,
                        embedding=embedding,
                        sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                    )
                else:
                    mapping = Mapping(id=id, text=text, concept=concept, embedding=embedding)
                mappings.append(mapping)

            # Fetch the total count of mappings for pagination
            total_count = (
                self.client.collections.get(self.mapping_schema.schema["class"])
                .aggregate.over_all(total_count=True)
                .total_count
            )
            if total_count is None:
                raise ValueError("Total count of mappings could not be retrieved.")

        except Exception as e:
            raise RuntimeError(f"Failed to fetch mappings: {e}")
        return Page[Mapping](items=mappings, limit=limit, offset=offset, total_count=total_count)

    def get_closest_mappings(
        self,
        embedding: Sequence[float],
        similarities: bool = True,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit=5,
    ) -> Union[List[Mapping], List[MappingResult]]:
        """Fetches the closest mappings based on an embedding vector, with optional filters for terminology and sentence
        embedder.

        :param embedding: The embedding vector to find the closest mappings.
        :param similarities: Whether to include similarity scores in the result, defaults to `True`.
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
        mappings = []  # List to store fetched mappings
        filters = None
        target_vector = None

        # Apply terminology filter if specified
        if terminology_name:
            if not self._terminology_exists(terminology_name):
                raise ValueError(f"Terminology '{terminology_name}' not found in available terminologies.")
            if filters:
                filters.append(
                    Filter.by_ref("hasConcept").by_ref("hasTerminology").by_property("name").equal(terminology_name)
                )
            else:
                filters = [
                    Filter.by_ref("hasConcept").by_ref("hasTerminology").by_property("name").equal(terminology_name)
                ]

        # Apply sentence embedder filter if specified
        if sentence_embedder:
            if not self._sentence_embedder_exists(sentence_embedder):
                raise ValueError(f"Vectorizer '{sentence_embedder}' not found in available vectorizers.")
            # Add filter for sentence embedder only if vectors should be included
            if not self.use_weaviate_vectorizer:
                if filters:
                    filters.append(Filter.by_property("hasSentenceEmbedder").equal(sentence_embedder))
                else:
                    filters = [Filter.by_property("hasSentenceEmbedder").equal(sentence_embedder)]
            else:
                target_vector = sentence_embedder
        else:
            if self.use_weaviate_vectorizer:
                raise ValueError("Sentence embedder cannot be `None` while `self.use_weaviate_vectorizer` is `True`.")
        try:
            # Get the mapping collection from the Weaviate client
            mapping_collection = self.client.collections.get("Mapping")

            # Perform the vector search to fetch the closest mappings
            response = mapping_collection.query.near_vector(
                near_vector=embedding,
                limit=limit,
                return_metadata=MetadataQuery(distance=True) if similarities else None,
                return_references=QueryReference(
                    link_on="hasConcept", return_references=QueryReference(link_on="hasTerminology")
                ),
                target_vector=target_vector,
                filters=Filter.all_of(filters) if filters else None,
            )

            # Process the response objects into Mapping or MappingResult objects
            for o in response.objects:
                # Calculate similarity based on distance if similarities are requested
                if similarities:
                    dist = getattr(getattr(o, "metadata", None), "distance", None)
                    if dist is not None:
                        similarity = 1.0 - dist

                # Get references with a defensive approach
                concept = None
                if getattr(o, "references", None):
                    has_concept = o.references.get("hasConcept")
                    if has_concept and getattr(has_concept, "objects", None):
                        concept_data = has_concept.objects[0]
                        term_ref = getattr(concept_data, "references", None)
                        has_term = term_ref.get("hasTerminology") if term_ref else None
                        if has_term and getattr(has_term, "objects", None):
                            terminology_data = has_term.objects[0]
                            terminology = Terminology(
                                name=str(terminology_data.properties["name"]), id=str(terminology_data.uuid)
                            )
                            concept = Concept(
                                terminology=terminology,
                                pref_label=str(concept_data.properties["prefLabel"]),
                                concept_identifier=str(concept_data.properties["conceptID"]),
                                id=str(concept_data.uuid),
                            )

                text = str(o.properties["text"])
                # o.vector is a dictionary with varying key
                obj_embedding = next(iter(getattr(o, "vector", {}).values()), None)

                # Create Mapping objects and append them to the list
                if not self.use_weaviate_vectorizer:
                    mapping = Mapping(
                        concept=concept,
                        text=text,
                        embedding=obj_embedding,
                        sentence_embedder=str(o.properties["hasSentenceEmbedder"]),
                    )
                else:
                    mapping = Mapping(concept=concept, text=text, embedding=obj_embedding)

                # Append MappingResult if similarities is True, else just Mapping
                mappings.append(MappingResult(mapping, similarity) if similarities else mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch closest mappings: {e}")
        return mappings

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        try:
            if isinstance(model_object_instance, Terminology):
                properties = {"name": model_object_instance.name}
                terminology_uuid = generate_uuid5(properties)
                if not self._uuid_exists("Terminology", terminology_uuid):
                    terminology_collection = self.client.collections.get("Terminology")
                    terminology_collection.data.insert(properties=properties, uuid=terminology_uuid)
                else:
                    self.logger.info(
                        f"[SKIPPED] Terminology '{model_object_instance.name}' already exists with UUID {terminology_uuid}"
                    )
            elif isinstance(model_object_instance, Concept):
                properties = {
                    "conceptID": model_object_instance.concept_identifier,
                    "prefLabel": model_object_instance.pref_label,
                }
                concept_uuid = generate_uuid5(properties)
                if not self._uuid_exists("Concept", concept_uuid):
                    # recursion: create terminology if not existing
                    terminology_properties = {"name": model_object_instance.terminology.name}
                    terminology_uuid = generate_uuid5(terminology_properties)
                    if not self._uuid_exists("Terminology", terminology_uuid):
                        self.store(model_object_instance.terminology)
                    concept_collection = self.client.collections.get("Concept")
                    concept_collection.data.insert(properties=properties, uuid=concept_uuid)
                    concept_collection.data.reference_add(
                        from_uuid=concept_uuid,
                        from_property="hasTerminology",
                        to=str(terminology_uuid),
                    )
                else:
                    self.logger.info(
                        f"[SKIPPED] Concept '{model_object_instance.concept_identifier}' already exists with UUID {concept_uuid}."
                    )
            elif isinstance(model_object_instance, Mapping):
                if not self.use_weaviate_vectorizer:
                    properties = {
                        "text": model_object_instance.text,
                        "hasSentenceEmbedder": model_object_instance.sentence_embedder,
                    }
                else:
                    properties = {"text": model_object_instance.text}
                mapping_uuid = generate_uuid5(properties)
                if not self._uuid_exists("Mapping", mapping_uuid):
                    concept_properties = {
                        "conceptID": model_object_instance.concept.concept_identifier,
                        "prefLabel": model_object_instance.concept.pref_label,
                    }
                    concept_uuid = generate_uuid5(concept_properties)
                    if not self._uuid_exists("Concept", concept_uuid):
                        self.store(model_object_instance.concept)

                    mapping_collection = self.client.collections.get("Mapping")
                    mapping_collection.data.insert(
                        properties=properties, uuid=mapping_uuid, vector=model_object_instance.embedding
                    )
                    mapping_collection.data.reference_add(
                        from_uuid=mapping_uuid, from_property="hasConcept", to=str(concept_uuid)
                    )
                else:
                    self.logger.info(
                        f"[SKIPPED] Mapping for text '{model_object_instance.text}' already exists with UUID {mapping_uuid}."
                    )
            else:
                raise ValueError("Unsupported model object instance type.")

        except Exception as e:
            raise ObjectStorageError("Failed to store object in the database.", e)

    def import_from_jsonl(
        self, jsonl_path: str, object_type: Literal["terminology", "concept", "mapping"], chunk_size: int = 100
    ):
        """Imports data from a JSONL file and stores it in the Weaviate database.

        :param jsonl_path: Path to the JSONL file.
        :param object_type: Literal specifying the object type, must be "terminology", "concept", or "mapping".
        :param chunk_size: The number of items to process in each batch, defaults to 100.
        :raises ValueError: If 'id' or 'properties' is missing in a JSON object.
        :raises ValueError: If there is an invalid JSON object.
        :raises RuntimeError: If an unexpected error occurs during import.
        """
        try:
            collection = self.client.collections.get(object_type.capitalize())
            chunk = []

            with open(jsonl_path, "r") as file:
                for idx, line in enumerate(file):
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {idx}: {e}")

                    # Validate essential fields
                    if "id" not in item or "properties" not in item:
                        raise ValueError(f"Missing 'id' or 'properties' on line {idx}")
                    chunk.append(item)
                    if len(chunk) >= chunk_size:
                        self._process_batch(chunk, collection)
                        chunk = []
                if chunk:
                    self._process_batch(chunk, collection)
        except ValueError:
            raise
        except FileNotFoundError as e:
            raise RuntimeError(f"File not found: {e}")
        except IOError as e:
            raise RuntimeError(f"File reading error: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during import: {e}")

    @deprecated("close is deprecated and will be removed in a future release, use shut_down instead")
    def close(self):
        self.shut_down()

    def shut_down(self):
        self.client.close()

    def clear_all(self):
        """Deletes all data and schema classes (Mapping, Concept, Terminology) and re-creates them."""
        for schema in [self.mapping_schema, self.concept_schema, self.terminology_schema]:
            class_name = schema.schema["class"]
            if self.client.collections.exists(class_name):
                self.client.collections.delete(class_name)

        # Re-create in order to statisfy dependencies
        self._create_schema_if_not_exists(self.terminology_schema.schema)
        self._create_schema_if_not_exists(self.concept_schema.schema)
        if self.use_weaviate_vectorizer and not self.mapping_schema.schema["vectorizer_config"]:
            self.logger.warning(
                "Provided mapping schema lacks `vectorizer_config` even though"
                "`use_weaviate_vectorizer` is set to True. Defaulting to"
                f"{mapping_schema_preconfigured_embeddings.schema}"
            )
            self.mapping_schema = mapping_schema_preconfigured_embeddings
        self._create_schema_if_not_exists(self.mapping_schema.schema)

    def _sentence_embedder_exists(self, name: str) -> bool:
        try:
            mapping = self.client.collections.get("Mapping")
            if not self.use_weaviate_vectorizer:
                response = mapping.query.fetch_objects(filters=Filter.by_property("hasSentenceEmbedder").equal(name))
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
            response = terminology.query.fetch_objects(filters=Filter.by_property("name").equal(name))
            if response.objects is not None:
                return len(response.objects) > 0
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to check if terminology exists: {e}")

    def _concept_exists(self, concept_id: str) -> bool:
        try:
            concept = self.client.collections.get("Concept")
            response = concept.query.fetch_objects(filters=Filter.by_property("conceptID").equal(concept_id))
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
            # Check if the concept exists first
            concept_properties = {
                "conceptID": mapping.concept.concept_identifier,
                "prefLabel": mapping.concept.pref_label,
            }
            concept_uuid = generate_uuid5(concept_properties)
            concept_exists = self._uuid_exists("Concept", concept_uuid)
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

    def _uuid_exists(self, collection_name: str, uuid: str) -> bool:
        """Check if an object with the specified UUID exists in the given Weaviate collection.

        :param collection_name: The name of the Weaviate collection to search in.
        :param uuid: The UUID of the object to check for existence.
        :raises ValueError: If the client is not initialized.
        :raises RuntimeError: If an error occurs during the query.
        :return: True if the object with the given UUID exists, False otherwise.
        """
        try:
            collection = self.client.collections.get(collection_name)
            obj = collection.query.fetch_object_by_id(uuid)
            return obj is not None
        except Exception as e:
            raise RuntimeError(f"Failed to check if UUID exists in {collection_name}: {e}")

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

    def _connect_to_memory(self, path: str, http_port: int, grpc_port: int) -> WeaviateClient:
        try:
            if path is None:
                raise ValueError("Path must be provided for disk mode.")
            if self._is_port_in_use(http_port) and self._is_port_in_use(grpc_port):
                return weaviate.connect_to_local(port=http_port, grpc_port=grpc_port, headers=self.headers)
            else:
                return weaviate.connect_to_embedded(
                    persistence_data_path=path,
                    headers=self.headers,
                    environment_variables={"ENABLE_MODULES": "text2vec-ollama"},
                )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {e}")

    def _connect_to_remote(self, path: str, port: int) -> WeaviateClient:
        try:
            if path is None:
                raise ValueError("Remote URL must be provided for remote mode.")
            return weaviate.connect_to_local(host=path, port=port, headers=self.headers)
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Weaviate client: {e}")

    def _process_batch(self, chunk: List[Dict[str, Any]], collection: Collection):
        """Processes a batch of items and adds them to the Weaviate collection.

        :param chunk: List of items to process.
        :param collection: Weaviate collection instance.
        """
        vectors = None

        if collection.name == "Mapping" and not self.use_weaviate_vectorizer:
            if "vector" not in chunk[0]:
                texts = [item["properties"]["text"] for item in chunk]
                vectors = self.vectorizer.get_embeddings(texts)

        with collection.batch.dynamic() as batch:
            for i, item in enumerate(chunk):
                object_id = item.get("id")
                properties = item.get("properties", {})
                references = item.get("references")

                if self.use_weaviate_vectorizer:
                    vector = None
                elif vectors is not None:
                    vector = vectors[i]
                    item["sentence_embedder"] = self.vectorizer.model_name
                else:
                    vector = item.get("vector", {}).get("default")

                batch.add_object(uuid=object_id, properties=properties, vector=vector, references=references)

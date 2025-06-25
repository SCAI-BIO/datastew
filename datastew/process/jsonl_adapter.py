import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from weaviate.util import generate_uuid5

from datastew.embedding import Vectorizer
from datastew.repository import WeaviateRepository
from datastew.repository.base import BaseRepository
from datastew.repository.model import Concept, Mapping, Terminology
from datastew.repository.postgresql import PostgreSQLRepository
from datastew.repository.sqllite import SQLLiteRepository
from datastew.repository.weaviate_schema import (
    concept_schema,
    mapping_schema_user_vectors,
    terminology_schema,
)


class BaseJsonlConverter(ABC):
    def __init__(self, dest_dir: str, buffer_size: int = 1000):
        """Initialize the converter.

        :param dest_dir: Destination directory for the exported JSONL files.
        :param buffer_size: Number of records to buffer before writing to disk, defaults to 1000
        """
        self.dest_dir = dest_dir
        self._buffer = []
        self._buffer_size = buffer_size
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Ensures the output directory exists."""
        os.makedirs(self.dest_dir, exist_ok=True)

    def _get_file_path(self, collection: str) -> str:
        """
        Returns the file path for a specific collection.

        :param collection: The collection name (e.g., "terminology", "concept", "mapping").
        :return: The full file path.
        """
        return os.path.join(self.dest_dir, f"{collection}.jsonl")

    def _write_to_jsonl(self, file_path: str, data):
        """
        Writes data to a JSONL file for the specified collection using a buffer.

        :param file_path: The file path for the collection.
        :param data: The data to write (individual JSON objects).
        :return: None
        """
        # Add the data to the buffer
        self._buffer.append(data)

        # Check if the buffer size is reached
        if len(self._buffer) >= self._buffer_size:
            self._flush_to_file(file_path)

    def _flush_to_file(self, file_path: str):
        """
        Writes the buffered data to the file and clears the buffer.

        :param file_path: The file path for the collection.
        :return: None
        """
        if not self._buffer:
            return

        with open(file_path, "a", encoding="utf-8") as file:
            for entry in self._buffer:
                if isinstance(entry.get("embedding"), np.ndarray):
                    entry["embedding"] = entry["embedding"].tolist()
                file.write(json.dumps(entry) + "\n")

        self._buffer.clear()

    @abstractmethod
    def from_repository(self, repository: BaseRepository):
        pass

    @abstractmethod
    def from_ohdsi(self, src: str, vectorizer: Vectorizer = Vectorizer(), include_vectors: bool = True):
        pass

    @abstractmethod
    def _object_to_dict(self, obj: Any) -> Dict[str, Any]:
        pass


class SQLJsonlConverter(BaseJsonlConverter):
    def __init__(self, dest_dir: str, buffer_size: int = 1000):
        super().__init__(dest_dir=dest_dir, buffer_size=buffer_size)

    def from_repository(self, repository: Union[PostgreSQLRepository, SQLLiteRepository]):
        """Export all records from a PostgreSQLRepository to JSONL files

        :param repository: Active database repository instace.
        """
        session = repository.session

        # Export Terminologies
        terminology_file_path = self._get_file_path("terminology")
        for t in tqdm(session.query(Terminology).all(), desc="Exporting Terminologies"):
            terminology = self._object_to_dict(t)
            self._write_to_jsonl(terminology_file_path, terminology)
        self._flush_to_file(terminology_file_path)

        # Export Concepts
        concept_file_path = self._get_file_path("concept")
        for c in tqdm(session.query(Concept).all(), desc="Exporting Concepts"):
            concept = self._object_to_dict(c)
            self._write_to_jsonl(concept_file_path, concept)
        self._flush_to_file(concept_file_path)

        # Export Mappings
        mapping_file_path = self._get_file_path("mapping")
        for m in tqdm(session.query(Mapping).all(), desc="Exporting Mappings"):
            mapping = self._object_to_dict(m)
            self._write_to_jsonl(mapping_file_path, mapping)
        self._flush_to_file(mapping_file_path)

    def from_ohdsi(self, src: str, vectorizer: Vectorizer = Vectorizer(), include_vectors: bool = True):
        """
        Converts data from OHDSI to SQL-compatible JSONL format.

        :param src: Path to the OHDSI CONCEPT.csv file.
        :param vectorizer: Vectorizer to use for text embeddings.
        :param include_vectors: Whether to include vector data in mappings.
        """
        if not os.path.exists(src):
            raise FileNotFoundError(f"OHDSI concept file '{src}' does not exist or is not a file.")

        terminology_file_path = self._get_file_path("terminology")
        concept_file_path = self._get_file_path("concept")
        mapping_file_path = self._get_file_path("mapping")

        # Write single OHDSI terminology entry
        self._write_to_jsonl(terminology_file_path, {"id": "OHDSI", "name": "OHDSI"})
        self._flush_to_file(terminology_file_path)

        for chunk in tqdm(
            pd.read_csv(
                src,
                delimiter="\t",
                usecols=["concept_name", "concept_id"],
                chunksize=10000,
                dtype={"concept_id": str, "concept_name": str},
            ),
            desc="Processing OHDSI concepts",
        ):
            concept_names = chunk["concept_name"].tolist()
            concept_ids = chunk["concept_id"].tolist()

            if include_vectors:
                embeddings = vectorizer.get_embeddings(concept_names)

            for i in range(len(concept_names)):
                concept_identifier = f"OHDSI:{concept_ids[i]}"
                label = concept_names[i]

                # Concept JSONL (no uuid)
                self._write_to_jsonl(
                    concept_file_path,
                    {
                        "concept_identifier": concept_identifier,
                        "pref_label": label,
                        "terminology_id": "OHDSI",
                    },
                )

                # Mapping JSONL
                mapping = {
                    "concept_identifier": concept_identifier,
                    "text": label,
                }
                if include_vectors:
                    mapping["sentence_embedder"] = vectorizer.model_name
                    mapping["embedding"] = embeddings[i]

                self._write_to_jsonl(mapping_file_path, mapping)

            self._flush_to_file(concept_file_path)
            self._flush_to_file(mapping_file_path)

    def _object_to_dict(self, obj: Union[Terminology, Concept, Mapping]) -> Dict[str, Any]:
        if isinstance(obj, Terminology):
            return {
                "id": obj.id,
                "name": obj.name,
            }
        elif isinstance(obj, Concept):
            return {
                "concept_identifier": obj.concept_identifier,
                "pref_label": obj.pref_label,
                "terminology_id": obj.terminology.id,
            }
        elif isinstance(obj, Mapping):
            return {
                "concept_identifier": obj.concept.concept_identifier,
                "text": obj.text,
                "embedding": obj.embedding,
                "sentence_embedder": obj.sentence_embedder,
            }
        else:
            raise TypeError(f"Unsupported object type: {type(obj)}")


class WeaviateJsonlConverter(BaseJsonlConverter):
    def __init__(
        self,
        dest_dir: str,
        terminology_schema: dict = terminology_schema.schema,
        concept_schema: dict = concept_schema.schema,
        mapping_schema: dict = mapping_schema_user_vectors.schema,
        buffer_size: int = 1000,
    ):
        super().__init__(dest_dir=dest_dir, buffer_size=buffer_size)
        self.terminology_schema = terminology_schema
        self.concept_schema = concept_schema
        self.mapping_schema = mapping_schema

    def from_repository(self, repository: WeaviateRepository):
        """
        Converts data from a WeaviateRepository to our JSONL format.

        :param repository: WeaviateRepository
        """
        # Process terminology first
        terminology_file_path = self._get_file_path("terminology")
        for terminology in repository.get_iterator(self.terminology_schema["class"]):
            self._write_to_jsonl(terminology_file_path, self._object_to_dict(terminology))
        self._flush_to_file(terminology_file_path)

        # Process concept next
        concept_file_path = self._get_file_path("concept")
        for concept in repository.get_iterator(self.concept_schema["class"]):
            self._write_to_jsonl(concept_file_path, self._object_to_dict(concept))
        self._flush_to_file(concept_file_path)

        # Process mapping last
        mapping_file_path = self._get_file_path("mapping")
        for mapping in repository.get_iterator(self.mapping_schema["class"]):
            self._write_to_jsonl(mapping_file_path, self._object_to_dict(mapping))
        self._flush_to_file(mapping_file_path)

    def from_ohdsi(self, src: str, vectorizer: Vectorizer = Vectorizer(), include_vectors: bool = True):
        """
        Converts data from OHDSI to our JSONL format.

        :param src: The file path to the OHDSI CONCEPT.csv file.
        :param vectorizer: Vectorizer model to be utilized for vector generation, defaults to Vectorizer.
        :param include_vectors: Whether to include vectors in JSONL file, defaults to True.
        """

        if not os.path.exists(src):
            raise FileNotFoundError(f"OHDSI concept file '{src}' does not exist or is not a file.")

        terminology_file_path = self._get_file_path("terminology")
        concept_file_path = self._get_file_path("concept")
        mapping_file_path = self._get_file_path("mapping")

        # Create a single OHDSI terminology entry with a fixed
        terminology_properties = {"name": "OHDSI"}
        terminology_id = generate_uuid5(terminology_properties)
        ohdsi_terminology = {
            "class": "Terminology",
            "id": terminology_id,
            "properties": terminology_properties,
        }

        self._write_to_jsonl(terminology_file_path, ohdsi_terminology)
        self._flush_to_file(terminology_file_path)

        # Process concepts one at a time
        for chunk in tqdm(
            pd.read_csv(
                src,
                delimiter="\t",
                usecols=["concept_name", "concept_id"],
                chunksize=10000,
                dtype={"concept_id": str, "concept_name": str},
            ),
            desc="Processing OHDSI concepts",
        ):
            concepts = []
            mappings = []

            concept_names = chunk["concept_name"].astype(str).tolist()
            concept_ids = chunk["concept_id"].astype(str).tolist()

            if include_vectors:
                # Compute batch embeddings
                embeddings = vectorizer.get_embeddings(concept_names)

            for i in range(len(concept_names)):
                concept_properties = {"conceptID": str(concept_ids[i]), "prefLabel": concept_names[i]}
                concept_uuid = generate_uuid5(concept_properties)

                # Concept JSON
                concepts.append(
                    {
                        "class": self.concept_schema["class"],
                        "id": concept_uuid,
                        "properties": concept_properties,
                        "references": {"hasTerminology": terminology_id},
                    }
                )

                # Mapping JSON
                mapping_uuid = generate_uuid5({"text": concept_names[i]})
                if include_vectors:
                    mappings.append(
                        {
                            "class": self.mapping_schema["class"],
                            "id": mapping_uuid,
                            "properties": {
                                "text": concept_names[i],
                                "hasSentenceEmbedder": vectorizer.model_name,
                            },
                            "references": {"hasConcept": concept_uuid},
                            "vector": {"default": embeddings[i]},
                        }
                    )
                else:
                    mappings.append(
                        {
                            "class": self.mapping_schema["class"],
                            "id": mapping_uuid,
                            "properties": {
                                "text": concept_names[i],
                            },
                            "references": {"hasConcept": concept_uuid},
                        }
                    )

            # Write results in batch
            for concept_data in concepts:
                self._write_to_jsonl(concept_file_path, concept_data)
            self._flush_to_file(concept_file_path)
            for mapping_data in mappings:
                self._write_to_jsonl(mapping_file_path, mapping_data)
            self._flush_to_file(mapping_file_path)

    def _object_to_dict(self, obj) -> Dict[str, Any]:
        """Conver a Weaviate object to a schema-compliant JSON dictionary.

        :param obj: Weaviate object instance.
        :return: Formatted dictionary containing class, id, properties, vector, and references.
        """
        if obj.references is not None:
            # FIXME: This is a hack to get the UUID of the referenced object. Replace as soon as weaviate devs offer an
            #  actual solution for this.
            vals = [value.objects for _, value in obj.references.items()]
            uuid = [str(obj.uuid) for sublist in vals for obj in sublist][0]
            references = {key: uuid for key, _ in obj.references.items()}
        else:
            references = {}

        return {
            "class": obj.collection,
            "id": str(obj.uuid),
            "properties": obj.properties,
            "vector": obj.vector,
            "references": references,
        }

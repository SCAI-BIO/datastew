import json
import os
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from datastew.embedding import Vectorizer
from datastew.repository import PostgreSQLRepository, SQLLiteRepository
from datastew.repository.model import Concept, Mapping, Terminology


class SQLJsonlConverter:
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

            concepts = []
            mappings = []

            concept_names = chunk["concept_name"].astype(str).tolist()
            concept_ids = chunk["concept_id"].astype(str).tolist()

            if include_vectors:
                embeddings = vectorizer.get_embeddings(concept_names)

            for i in range(len(concept_names)):
                concept_identifier = f"OHDSI:{concept_ids[i]}"
                label = concept_names[i]

                # Concept JSON
                concepts.append(
                    {
                        "concept_identifier": concept_identifier,
                        "pref_label": label,
                        "terminology_id": "OHDSI",
                    }
                )

                # Mapping JSON
                mapping = {
                    "concept_identifier": concept_identifier,
                    "text": label,
                }
                if include_vectors:
                    mapping["sentence_embedder"] = vectorizer.model_name
                    mapping["embedding"] = embeddings[i]

                mappings.append(mapping)

            # Write results in batch
            for concept_data in concepts:
                self._write_to_jsonl(concept_file_path, concept_data)
            self._flush_to_file(concept_file_path)
            for mapping_data in mappings:
                self._write_to_jsonl(mapping_file_path, mapping_data)
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

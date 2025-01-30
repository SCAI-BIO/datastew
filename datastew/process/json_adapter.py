import json
import os
from typing import Dict, Sequence, Union

import pandas as pd
from tqdm import tqdm
from weaviate.util import generate_uuid5

from datastew.embedding import EmbeddingModel, MPNetAdapter
from datastew.repository import WeaviateRepository
from datastew.repository.weaviate_schema import (concept_schema,
                                                 mapping_schema,
                                                 terminology_schema)


class WeaviateJsonConverter(object):
    """
    Converts data to our JSON format for Weaviate schema.
    """

    def __init__(
        self,
        dest_dir: str,
        schema_terminology: dict = terminology_schema,
        schema_concept: dict = concept_schema,
        schema_mapping: dict = mapping_schema,
        buffer_size: int = 1000,
    ):
        self.dest_dir = dest_dir
        self.terminology_schema = schema_terminology
        self.concept_schema = schema_concept
        self.mapping_schema = schema_mapping
        self._buffers = {
            "terminology": [],
            "concept": [],
            "mapping": [],
        }
        self._buffer_size = buffer_size
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """
        Ensures the output directory exists.

        :return: None
        """
        os.makedirs(self.dest_dir, exist_ok=True)

    def _get_file_path(self, collection: str) -> str:
        """
        Returns the file path for a specific collection.

        :param collection: The collection name (e.g., "terminology", "concept", "mapping").
        :return: The full file path.
        """
        return os.path.join(self.dest_dir, f"{collection}.json")

    def _write_to_json(self, file_path: str, collection: str, data):
        """
        Writes data to a JSON file for the specified collection using a buffer.

        :param file_path: The file path for the collection.
        :param collection: The name of the collection (e.g., "terminology", "concept", "mapping").
        :param data: The data to write (individual JSON objects).
        :return: None
        """
        # Add the data to the buffer
        self._buffers[collection].append(data)

        # Check if the buffer size is reached
        if len(self._buffers[collection]) >= self._buffer_size:
            self._flush_to_file(file_path, collection)

    def _flush_to_file(self, file_path: str, collection: str):
        """
        Writes the buffered data to the file and clears the buffer.

        :param file_path: The file path for the collection.
        :param collection: The name of the collection (e.g., "terminology", "concept", "mapping").
        :return: None
        """
        if not self._buffers[collection]:
            return

        with open(file_path, "a", encoding="utf-8") as file:
            for entry in self._buffers[collection]:
                file.write(json.dumps(entry) + "\n")

        self._buffers[collection].clear()

    def from_repository(self, repository: WeaviateRepository) -> None:
        """
        Converts data from a WeaviateRepository to our JSON format.

        :param repository: WeaviateRepository
        :return: None
        """
        # Process terminology first
        terminology_file_path = self._get_file_path("terminology")
        for terminology in repository.get_iterator(self.terminology_schema["class"]):
            self._write_to_json(
                terminology_file_path,
                "terminology",
                self._weaviate_object_to_dict(terminology),
            )
        self._flush_to_file(terminology_file_path, "terminology")

        # Process concept next
        concept_file_path = self._get_file_path("concept")
        for concept in repository.get_iterator(self.concept_schema["class"]):
            self._write_to_json(
                concept_file_path, "concept", self._weaviate_object_to_dict(concept)
            )
        self._flush_to_file(concept_file_path, "concept")

        # Process mapping last
        mapping_file_path = self._get_file_path("mapping")
        for mapping in repository.get_iterator(self.mapping_schema["class"]):
            self._write_to_json(
                mapping_file_path, "mapping", self._weaviate_object_to_dict(mapping)
            )
        self._flush_to_file(mapping_file_path, "mapping")

    def from_ohdsi(self, src: str):
        """
        Converts data from OHDSI to our JSON format.

        :param src: The file path to the OHDSI CONCEPT.csv file.
        """

        if not os.path.exists(src):
            raise FileNotFoundError(
                f"OHDSI concept file '{src}' does not exist or is not a file."
            )

        terminology_file_path = self._get_file_path("terminology")
        concept_file_path = self._get_file_path("concept")
        mapping_file_path = self._get_file_path("mapping")

        # Initialize Embedding Model once
        embedding_model = MPNetAdapter()

        # Create a single OHDSI terminology entry with a fixed
        terminology_properties = {"name": "OHDSI"}
        terminology_id = generate_uuid5(terminology_properties)
        ohdsi_terminology = {
            "class": self.terminology_schema["class"],
            "id": terminology_id,
            "properties": terminology_properties,
        }

        self._write_to_json(terminology_file_path, "terminology", ohdsi_terminology)
        self._flush_to_file(terminology_file_path, "terminology")

        # Process concepts one at a time
        for chunk in tqdm(
            pd.read_csv(
                src,
                delimiter="\t",
                chunksize=1000,
                usecols=["concept_name", "concept_id"],
            ),
            desc="Processing OHDSI concepts",
        ):
            for _, row in chunk.iterrows():
                concept_data = self._ohdsi_row_to_concept(row, terminology_id)
                self._write_to_json(concept_file_path, "concept", concept_data)

                # Compute embedding **one at a time** (low memory usage)
                mapping_data = self._ohdsi_concept_to_mappings(
                    row["concept_name"], concept_data["id"], embedding_model
                )
                self._write_to_json(mapping_file_path, "mapping", mapping_data)

        self._flush_to_file(concept_file_path, "concept")
        self._flush_to_file(mapping_file_path, "mapping")

    def _ohdsi_row_to_concept(
        self, row: pd.Series, terminology_id: str
    ) -> Dict[str, Union[str, Dict[str, str]]]:
        """Converts an OHDSI row into a concept dictionary formatted for Weaviate.

        :param row: A Pandas Series representing a row for OHDSI CONCEPT.csv.
        :param terminology_id: UUID of the OHDSI terminology.
        :return: A dictionary formatted for Weaviate.
        """
        properties = {
            "conceptID": str(row["concept_id"]),
            "prefLabel": row["concept_name"],
        }
        concept_id = generate_uuid5(properties)
        return {
            "class": self.concept_schema["class"],
            "id": concept_id,
            "properties": properties,
            "references": {
                "hasTerminology": terminology_id,
            },
        }

    def _ohdsi_concept_to_mappings(
        self, pref_label: str, concept_id: str, embedding_model: EmbeddingModel
    ) -> Dict[str, Union[str, Dict[str, str], Dict[str, Sequence[float]]]]:
        """Generates an embedding **one at a time** to avoid excessive memory usage.

        :param pref_label: The preferred label of the concept.
        :param concept_id: The UUID of the concept.
        :param embedding_model: An instance of EmbeddingModel (e.g., MPNetAdapter).
        :return: A dictionary formatted Weaviate.
        """
        embedding = embedding_model.get_embedding(pref_label)  # Single text embedding
        properties = {
            "text": pref_label,
            "hasSentenceEmbedder": embedding_model.get_model_name(),
        }
        mapping_id = generate_uuid5(properties)
        return {
            "class": self.mapping_schema["class"],
            "id": mapping_id,
            "properties": properties,
            "references": {
                "hasConcept": concept_id,
            },
            "vector": {
                "default": embedding,  # Directly store the embedding for the concept
            },
        }

    @staticmethod
    def _weaviate_object_to_dict(weaviate_object):

        if weaviate_object.references is not None:
            # FIXME: This is a hack to get the UUID of the referenced object. Replace as soon as weaviate devs offer an
            #  actual solution for this.
            vals = [value.objects for _, value in weaviate_object.references.items()]
            uuid = [str(obj.uuid) for sublist in vals for obj in sublist][0]
            references = {key: uuid for key, _ in weaviate_object.references.items()}
        else:
            references = {}

        return {
            "class": weaviate_object.collection,
            "id": str(weaviate_object.uuid),
            "properties": weaviate_object.properties,
            "vector": weaviate_object.vector,
            "references": references,
        }

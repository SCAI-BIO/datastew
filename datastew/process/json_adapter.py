import json
import os

import pandas as pd
from tqdm import tqdm
from weaviate.util import generate_uuid5

from datastew.embedding import EmbeddingModel
from datastew.repository import WeaviateRepository
from datastew.repository.weaviate_schema import (concept_schema,
                                                 mapping_schema_user_vectors,
                                                 terminology_schema)


class WeaviateJsonConverter(object):
    """
    Converts data to our JSON format for Weaviate schema.
    """

    def __init__(
        self,
        dest_dir: str,
        terminology_schema: dict = terminology_schema,
        concept_schema: dict = concept_schema,
        mapping_schema: dict = mapping_schema_user_vectors,
        buffer_size: int = 1000,
    ):
        self.dest_dir = dest_dir
        self.terminology_schema = terminology_schema
        self.concept_schema = concept_schema
        self.mapping_schema_user_vectors = mapping_schema
        self._buffer = []
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

    def _write_to_json(self, file_path: str, data):
        """
        Writes data to a JSON file for the specified collection using a buffer.

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
                file.write(json.dumps(entry) + "\n")

        self._buffer.clear()

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
                self._weaviate_object_to_dict(terminology),
            )
        self._flush_to_file(terminology_file_path)

        # Process concept next
        concept_file_path = self._get_file_path("concept")
        for concept in repository.get_iterator(self.concept_schema["class"]):
            self._write_to_json(
                concept_file_path, self._weaviate_object_to_dict(concept)
            )
        self._flush_to_file(concept_file_path)

        # Process mapping last
        mapping_file_path = self._get_file_path("mapping")
        for mapping in repository.get_iterator(
            self.mapping_schema_user_vectors["class"]
        ):
            self._write_to_json(
                mapping_file_path, self._weaviate_object_to_dict(mapping)
            )
        self._flush_to_file(mapping_file_path)

    def from_ohdsi(self, src: str, embedding_model: EmbeddingModel):
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

        # Create a single OHDSI terminology entry with a fixed
        terminology_properties = {"name": "OHDSI"}
        terminology_id = generate_uuid5(terminology_properties)
        ohdsi_terminology = {
            "class": self.terminology_schema["class"],
            "id": terminology_id,
            "properties": terminology_properties,
        }

        self._write_to_json(terminology_file_path, ohdsi_terminology)
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

            # Compute batch embeddings
            embeddings = embedding_model.get_embeddings(concept_names)

            for i in range(len(concept_names)):
                concept_properties = {
                    "conceptID": str(concept_ids[i]),
                    "prefLabel": concept_names[i],
                }
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
                mappings.append(
                    {
                        "class": self.mapping_schema_user_vectors["class"],
                        "id": mapping_uuid,
                        "properties": {
                            "text": concept_names[i],
                            "hasSentenceEmbedder": embedding_model.get_model_name(),
                        },
                        "references": {"hasConcept": concept_uuid},
                        "vector": {"default": embeddings[i]},
                    }
                )

            # Write results in batch
            for concept_data in concepts:
                self._write_to_json(concept_file_path, concept_data)
            self._flush_to_file(concept_file_path)
            for mapping_data in mappings:
                self._write_to_json(mapping_file_path, mapping_data)
            self._flush_to_file(mapping_file_path)

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

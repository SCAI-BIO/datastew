import json
import os

from datastew.repository import WeaviateRepository
from datastew.repository.weaviate_schema import terminology_schema, concept_schema, mapping_schema


class WeaviateJsonConverter(object):
    """
    Converts data to our JSON format for Weaviate schema.
    """

    def __init__(self, dest_dir: str,
                 schema_terminology: dict = terminology_schema,
                 schema_concept: dict = concept_schema,
                 schema_mapping: dict = mapping_schema,
                 buffer_size: int = 1000):
        self.dest_dir = dest_dir
        self.terminology_schema = schema_terminology
        self.concept_schema = schema_concept
        self.mapping_schema = schema_mapping
        self._buffer = []
        self._buffer_size = buffer_size
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """
        Ensures the output directory exists.

        :return: None
        """
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

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

        with open(file_path, 'a') as file:
            for entry in self._buffer:
                file.write(json.dumps(entry) + '\n')

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
            self._write_to_json(terminology_file_path, self._weaviate_object_to_dict(terminology))
        self._flush_to_file(terminology_file_path)

        # Process concept next
        concept_file_path = self._get_file_path("concept")
        for concept in repository.get_iterator(self.concept_schema["class"]):
            self._write_to_json(concept_file_path, self._weaviate_object_to_dict(concept))
        self._flush_to_file(concept_file_path)

        # Process mapping last
        mapping_file_path = self._get_file_path("mapping")
        for mapping in repository.get_iterator(self.mapping_schema["class"]):
            self._write_to_json(mapping_file_path, self._weaviate_object_to_dict(mapping))
        self._flush_to_file(mapping_file_path)

    def from_ohdsi(self):
        """
        Converts data from OHDSI to our JSON format.

        :return: None
        """
        raise NotImplementedError("Not implemented yet.")

    @staticmethod
    def _weaviate_object_to_dict(weaviate_object):

        if weaviate_object.references is not None:
            # FIXME: This is a hack to get the UUID of the referenced object. Replace as soon as weaviate devs offer an
            #  actual solution for this.
            vals = [value.objects for key, value in weaviate_object.references.items()]
            uuid = [str(obj.uuid) for sublist in vals for obj in sublist][0]
            references = {key: uuid for key, value in weaviate_object.references.items()}
        else:
            references = {}

        return {
            "class": weaviate_object.collection,
            "id": str(weaviate_object.uuid),
            "properties": weaviate_object.properties,
            "vector": weaviate_object.vector,
            "references": references
        }

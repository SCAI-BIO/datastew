import json
import os

from datastew.repository import WeaviateRepository
from datastew.repository.weaviate_schema import terminology_schema, concept_schema, mapping_schema


class WeaviateJsonConverter(object):
    """
    Converts data to our JSON format for Weaviate schema.
    """

    def __init__(self, dest_path: str,
                 schema_terminology: dict = terminology_schema,
                 schema_concept: dict = concept_schema,
                 schema_mapping: dict = mapping_schema,
                 buffer_size: int = 1000):
        self.dest_path = dest_path
        self.terminology_schema = schema_terminology
        self.concept_schema = schema_concept
        self.mapping_schema = schema_mapping
        self.output_file_path = dest_path
        self._buffer = []
        self._buffer_size = buffer_size
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """
        Ensures the directory and file exist. Creates them if they do not.

        :return: None
        """
        directory = os.path.dirname(self.output_file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Create an empty file if it doesn't exist
        if not os.path.exists(self.output_file_path):
            with open(self.output_file_path, 'w') as file:
                pass

    def _write_to_json(self, data):
        """
        Writes data to a JSON file in a performant manner using a buffer.

        :param data: The data to write (individual JSON objects).
        :return: None
        """
        # Add the data to the buffer
        self._buffer.append(data)

        # Check if the buffer size is reached
        if len(self._buffer) >= self._buffer_size:
            self._flush_to_file()

    def _flush_to_file(self):
        """
        Writes the buffered data to the file and clears the buffer.

        :return: None
        """
        if not self._buffer:
            return

        with open(self.output_file_path, 'a') as file:
            # Write each JSON object in the buffer as a new line
            for entry in self._buffer:
                file.write(json.dumps(entry) + '\n')

        # Clear the buffer
        self._buffer.clear()


    def from_repository(self, repository: WeaviateRepository) -> None:
        """
        Converts data from a WeaviateRepository to our JSON format.

        :param repository: WeaviateRepository

        :return: None
        """
        for concept in repository.get_iterator(self.concept_schema["class"]):
            self._write_to_json(self._weaviate_object_to_dict(concept))
        for mapping in repository.get_iterator(self.mapping_schema["class"]):
            self._write_to_json(self._weaviate_object_to_dict(mapping))
        self._flush_to_file()

    def from_ohdsi(self):
        """
        Converts data from OHDSI to our JSON format.

        :return: None
        """
        # use schema to construct the objects
        raise NotImplementedError("Not implemented yet.")

    def _weaviate_object_to_dict(self, object):
        return {
            "class": object.collection,
            "id": str(object.uuid),
            "properties": object.properties,
            "vector": object.vector
        }

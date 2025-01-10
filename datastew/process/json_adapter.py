import json

from datastew import Terminology, Concept, Mapping
from datastew.repository import WeaviateRepository
from datastew.repository.weaviate_schema import terminology_schema, concept_schema, mapping_schema


class WeaviateJsonConverter(object):
    """
    Converts data to our JSON format for Weaviate schema.
    """

    def __init__(self, dest_path: str,
                 schema_terminology: dict = terminology_schema,
                 schema_concept: dict = concept_schema,
                 schema_mapping: dict = mapping_schema):
        self.dest_path = dest_path
        self.terminology_schema = schema_terminology
        self.concept_schema = schema_concept
        self.mapping_schema = schema_mapping

    def from_repository(self, repository: WeaviateRepository, terminology_name: str = None, limit=1000) -> None:
        """
        Converts data from a WeaviateRepository to our JSON format.

        :param repository: WeaviateRepository
        :param terminology_name: The name of the terminology to filter.
        :param limit: page size

        :return: None
        """
        current_offset = 0
        while repository.get_concepts(terminology_name=terminology_name, limit=limit,
                                      offset=current_offset).has_next_page():
            concepts = repository.get_concepts(limit, offset)
            self._write_to_json(concepts.items[0].terminology, concepts, [])
            offset = offset + limit
        # reset offset
        current_offset = 0
        while repository.get_mappings(terminology_name=terminology_name, limit=limit,
                                      offset=current_offset).has_next_page():
            mappings = repository.get_concepts(limit, offset)
            self._write_to_json(mappings.items[0].concept.terminology.name, mappings, [])
            offset = offset + limit

    def from_ohdsi(self):
        """
        Converts data from OHDSI to our JSON format.

        :return: None
        """
        # use schema to construct the objects
        raise NotImplementedError("Not implemented yet.")

    def from_object(self, object):
        """
        Writes single weaviate objects to JSON files. Appends to existing files.

        :return: None
        """
        if isinstance(object, Terminology):
            self._write_to_json(object, [], [])
        elif isinstance(object, Concept):
            self._write_to_json([], object, [])
        elif isinstance(object, Mapping):
            self._write_to_json([], [], object)
        else:
            raise ValueError("Object is not a Terminology, Concept or Mapping.")

    def _write_to_json(self, terminology: Terminology, concepts, mappings):
        """
        Writes the data to JSON files.
        """
        terminology_name = terminology.name
        with open(f"{self.dest_path}/{terminology_name}_terminology.json", "w") as f:
            f.write(json.dumps(terminology, indent=2))
        with open(f"{self.dest_path}/{terminology_name}_concepts.json", "w") as f:
            f.write(json.dumps(concepts, indent=2))
        with open(f"{self.dest_path}/{terminology_name}_mappings.json", "w") as f:
            f.write(json.dumps(mappings, indent=2))

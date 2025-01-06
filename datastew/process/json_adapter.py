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

    def from_repository(self, repository: WeaviateRepository, terminology_name: str = None) -> None:
        """
        Converts data from a WeaviateRepository to our JSON format.

        :param repository: WeaviateRepository

        :return: None
        """
        # TODO: This needs to work for specific terminologies, we need those endpoints in the repository
        # TODO: Also: implement paging in the repository
        # retrieve all terminologies, concepts and mappings
        terminologies = repository.get_all_terminologies()
        concepts = repository.get_all_concepts()
        mappings = repository.get_mappings(limit =100)
        # store the data in the JSON format
        for terminology in terminologies:
            # get the concepts and mappings for the terminology
            terminology_name = terminology.name
            terminology_concepts = [concept for concept in concepts if concept.hasTerminology == terminology_name]
            terminology_mappings = [mapping for mapping in mappings if
                                    mapping.hasConcept.hasTerminology == terminology_name]
            self._write_to_json(terminology, terminology_concepts, terminology_mappings)

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

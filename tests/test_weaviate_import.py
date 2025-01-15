import os
import json
from unittest import TestCase

from datastew import Terminology, Concept, Mapping, MPNetAdapter
from datastew.repository import WeaviateRepository
from datastew.process.json_adapter import WeaviateJsonConverter


class TestWeaviateRepositoryImport(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = WeaviateRepository()
        cls.import_dir = "test_import"

        # Create a test directory for import
        os.makedirs(cls.import_dir, exist_ok=True)

        # Write sample JSON files for testing
        cls.terminology_data = [{"class": "Terminology", "id": "b4389093-d778-5ed0-97ac-ac132299369a", "properties": {"name": "snomed CT"}, "vector": {}}]
        cls.concept_data = [
            {"class": "Concept", "id": "10152406-0c9b-5bbc-a94d-c3d6dd0158fc", "properties": {"conceptID": "Concept ID: 386098009", "prefLabel": "Migraine"}, "vector": {}},
            {"class": "Concept", "id": "15f4419f-20aa-5746-91e0-0b894f925597", "properties": {"conceptID": "Concept ID: 57386000", "prefLabel": "Influenza"}, "vector": {}}
        ]
        cls.mapping_data = [{"class": "Mapping", "id": "50ff1cf8-45c6-568f-a496-61e6c9623f58", "properties": {"source": "snomed CT", "target": "ICD10"}, "vector": {'default:'[1,1,1]}}]

        with open(f"{cls.import_dir}/terminology.json", "w") as file:
            json.dump(cls.terminology_data, file)
        with open(f"{cls.import_dir}/concept.json", "w") as file:
            json.dump(cls.concept_data, file)
        with open(f"{cls.import_dir}/mapping.json", "w") as file:
            json.dump(cls.mapping_data, file)

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up created files and directory
        os.remove(f"{cls.import_dir}/terminology.json")
        os.remove(f"{cls.import_dir}/concept.json")
        os.remove(f"{cls.import_dir}/mapping.json")
        os.rmdir(cls.import_dir)

    def test_json_import_terminology(self):
        self.repository.import_from_json(f"{self.import_dir}/terminology.json", "terminology")
        terminology = self.repository.fetch_all(Terminology)
        self.assertEqual(len(terminology), 1)
        self.assertEqual(terminology[0].properties["name"], "snomed CT")

    def test_json_import_concept(self):
        self.repository.import_from_json(f"{self.import_dir}/concept.json", "concept")
        concepts = self.repository.fetch_all(Concept)
        self.assertEqual(len(concepts), 2)
        self.assertIn("Migraine", [concept.properties["prefLabel"] for concept in concepts])
        self.assertIn("Influenza", [concept.properties["prefLabel"] for concept in concepts])

    def test_json_import_mapping(self):
        self.repository.import_from_json(f"{self.import_dir}/mapping.json", "mapping")
        mappings = self.repository.fetch_all(Mapping)
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0].properties["source"], "snomed CT")
        self.assertEqual(mappings[0].properties["target"], "ICD10")

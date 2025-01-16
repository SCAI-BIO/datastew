import os
import json
import shutil
from unittest import TestCase

from datastew import Terminology, Concept, Mapping
from datastew.repository import WeaviateRepository


class TestWeaviateRepositoryImport(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = WeaviateRepository()
        cls.import_dir = "test_import"

        # Create a test directory for import
        os.makedirs(cls.import_dir, exist_ok=True)

        # Write sample JSON files for testing
        cls.terminology_data = [
            {"class": "Terminology", "id": "94331523-fa7e-5871-9375-8f559d6035dd",
             "properties": {"name": "import_test"}, "vector": {}, "references": {}}
        ]
        cls.concept_data = [
            {"class": "Concept", "id": "064cb594-41cd-561d-b5a8-2bf226006f09",
             "properties": {"conceptID": "import_test:G", "prefLabel": "G"}, "vector": {},
             "references": {"hasTerminology": "94331523-fa7e-5871-9375-8f559d6035dd"}},
        ]
        cls.mapping_data = [
            {"class": "Mapping", "id": "0423d64c-fa89-54c3-b92c-4738a630d7d6",
             "properties": {"text": "pancreas", "hasSentenceEmbedder": "sentence-transformers/all-mpnet-base-v2"},
             "vector": {
                 "default": [0.048744574189186096, -0.0035385489463806152]},
             "references": {"hasConcept": "064cb594-41cd-561d-b5a8-2bf226006f09"}}
        ]

        with open(f"{cls.import_dir}/terminology.json", "w") as file:
            json.dump(cls.terminology_data, file)
        with open(f"{cls.import_dir}/concept.json", "w") as file:
            json.dump(cls.concept_data, file)
        with open(f"{cls.import_dir}/mapping.json", "w") as file:
            json.dump(cls.mapping_data, file)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.repository.close()
        shutil.rmtree(os.path.join(os.getcwd(), "db"))

    @classmethod
    def tearDownClass(cls) -> None:
        # Clean up created files and directory
        os.remove(f"{cls.import_dir}/terminology.json")
        os.remove(f"{cls.import_dir}/concept.json")
        os.remove(f"{cls.import_dir}/mapping.json")
        os.rmdir(cls.import_dir)

    def test_json_import_terminology(self):
        self.repository.import_from_json(f"{self.import_dir}/terminology.json", "terminology")
        terminology = self.repository.get_all_terminologies()
        self.assertEqual(len(terminology), 1)
        self.assertEqual(terminology[0].name, "import_test")

    def test_json_import_concept(self):
        self.repository.import_from_json(f"{self.import_dir}/concept.json", "concept")
        concepts = self.repository.get_concepts(limit=5, offset=0).items
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].pref_label, "G")
        # check for correct referencing
        self.assertEqual(concepts[0].terminology.name, "import_test")

    def test_json_import_mapping(self):
        self.repository.import_from_json(f"{self.import_dir}/mapping.json", "mapping")
        mappings = self.repository.get_mappings(limit=5, offset=0).items
        self.assertEqual(len(mappings), 1)
        self.assertEqual(mappings[0].text, "pancreas")
        # check for correct referencing
        self.assertEqual(mappings[0].concept.pref_label, "G")

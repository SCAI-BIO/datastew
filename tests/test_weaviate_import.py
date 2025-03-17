import json
import os
import shutil
import tempfile
from typing import Any, Dict, List
from unittest import TestCase

from datastew.repository import WeaviateRepository


class TestWeaviateRepositoryImport(TestCase):

    def setUp(self) -> None:
        self.repository = WeaviateRepository()
        self.temp_dir = tempfile.mkdtemp()

        # Sample data for JSONL files
        self.data_files = {
            "terminology": [
                {
                    "class": "Terminology",
                    "id": "94331523-fa7e-5871-9375-8f559d6035dd",
                    "properties": {"name": "import_test"},
                    "vector": {},
                    "references": {},
                }
            ],
            "concept": [
                {
                    "class": "Concept",
                    "id": "064cb594-41cd-561d-b5a8-2bf226006f09",
                    "properties": {"conceptID": "import_test:G", "prefLabel": "G"},
                    "vector": {},
                    "references": {
                        "hasTerminology": "94331523-fa7e-5871-9375-8f559d6035dd"
                    },
                },
                {
                    "class": "Concept",
                    "id": "12345678-41cd-561d-b5a8-2bf226006f09",
                    "properties": {"conceptID": "import_test:H", "prefLabel": "H"},
                    "vector": {},
                    "references": {
                        "hasTerminology": "94331523-fa7e-5871-9375-8f559d6035dd"
                    },
                },
            ],
            "mapping": [
                {
                    "class": "Mapping",
                    "id": "0423d64c-fa89-54c3-b92c-4738a630d7d6",
                    "properties": {
                        "text": "pancreas",
                        "hasSentenceEmbedder": "sentence-transformers/all-mpnet-base-v2",
                    },
                    "vector": {
                        "default": [0.048744574189186096, -0.0035385489463806152]
                    },
                    "references": {
                        "hasConcept": "064cb594-41cd-561d-b5a8-2bf226006f09"
                    },
                },
                {
                    "class": "Mapping",
                    "id": "12345678-fa89-54c3-b92c-4738a630d7d6",
                    "properties": {
                        "text": "liver",
                        "hasSentenceEmbedder": "sentence-transformers/all-mpnet-base-v2",
                    },
                    "vector": {"default": [0.1, -0.2]},
                    "references": {
                        "hasConcept": "12345678-41cd-561d-b5a8-2bf226006f09"
                    },
                },
            ],
        }

        # Write data to JSONL files
        for key, data in self.data_files.items():
            self.write_jsonl(os.path.join(self.temp_dir, f"{key}.jsonl"), data)

    @staticmethod
    def write_jsonl(file_path: str, data: List[Dict[str, Any]]):
        """Write data to a JSONL file."""
        with open(file_path, "w") as file:
            for obj in data:
                json.dump(obj, file)
                file.write("\n")

    def import_data(self, data_types: List[str]):
        """Helper method to import multiple data types."""
        for data_type in data_types:
            file_path = os.path.join(self.temp_dir, f"{data_type}.jsonl")
            self.repository.import_from_jsonl(file_path, data_type)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.repository.close()
        shutil.rmtree(os.path.join(os.getcwd(), "db"), ignore_errors=True)

    def test_import_terminology(self):
        self.import_data(["terminology"])
        terminology = self.repository.get_all_terminologies()

        self.assertEqual(len(terminology), 1)
        with self.subTest("Terminology ID"):
            self.assertEqual(terminology[0].id, "94331523-fa7e-5871-9375-8f559d6035dd")
        with self.subTest("Terminology Name"):
            self.assertEqual(terminology[0].name, "import_test")

    def test_import_concepts(self):
        self.import_data(["terminology", "concept"])
        concepts = self.repository.get_concepts(limit=5, offset=0).items

        self.assertEqual(len(concepts), 2)

        for concept in concepts:
            with self.subTest(f"Concept ID: {concept.id}"):
                self.assertIn(
                    concept.id,
                    [
                        "064cb594-41cd-561d-b5a8-2bf226006f09",
                        "12345678-41cd-561d-b5a8-2bf226006f09",
                    ],
                )
            with self.subTest(f"Concept Properties: {concept.pref_label}"):
                self.assertIn(concept.pref_label, ["G", "H"])
                self.assertIn(
                    concept.concept_identifier, ["import_test:G", "import_test:H"]
                )
            with self.subTest(f"Terminology Reference for {concept.pref_label}"):
                self.assertEqual(concept.terminology.name, "import_test")

    def test_import_mappings(self):
        self.import_data(["terminology", "concept", "mapping"])
        mappings = self.repository.get_mappings(limit=10, offset=0).items

        self.assertEqual(len(mappings), 2)

        for mapping in mappings:
            with self.subTest(f"Mapping ID: {mapping.id}"):
                self.assertIn(
                    mapping.id,
                    [
                        "0423d64c-fa89-54c3-b92c-4738a630d7d6",
                        "12345678-fa89-54c3-b92c-4738a630d7d6",
                    ],
                )
            with self.subTest(f"Mapping Text for {mapping.text}"):
                self.assertIn(mapping.text, ["pancreas", "liver"])
            with self.subTest(f"Sentence Embedder for {mapping.text}"):
                self.assertEqual(
                    mapping.sentence_embedder, "sentence-transformers/all-mpnet-base-v2"
                )
            with self.subTest(f"Vector Length for {mapping.text}"):
                self.assertEqual(len(mapping.embedding["default"]), 2)
            with self.subTest(f"Concept Reference for Mapping {mapping.text}"):
                expected_label = "G" if mapping.text == "pancreas" else "H"
                self.assertEqual(mapping.concept.pref_label, expected_label)

    def test_import_invalid_jsonl(self):
        invalid_file = os.path.join(self.temp_dir, "invalid.jsonl")
        with open(invalid_file, "w") as file:
            file.write("{ invalid jsonl }")

        with self.assertRaises(ValueError):
            self.repository.import_from_jsonl(invalid_file, "terminology")

    def test_import_missing_id(self):
        file_path = os.path.join(self.temp_dir, "missing_id.jsonl")
        self.write_jsonl(
            file_path, [{"class": "Terminology", "properties": {"name": "missing_id"}}]
        )

        with self.assertRaises(KeyError):
            self.repository.import_from_jsonl(file_path, "terminology")

    def test_import_empty_file(self):
        empty_file = os.path.join(self.temp_dir, "empty.jsonl")
        open(empty_file, "w").close()

        self.repository.import_from_jsonl(empty_file, "terminology")
        terminology = self.repository.get_all_terminologies()
        self.assertEqual(len(terminology), 0)

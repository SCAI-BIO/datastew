import json
import os
import shutil
import tempfile
import unittest

import pandas as pd
from weaviate.util import generate_uuid5

from datastew.embedding import MPNetAdapter
from datastew.process.json_adapter import WeaviateJsonConverter


class MockMPNetAdapter(MPNetAdapter):
    """Mock embedding model to return fixed values for testing."""

    def get_embedding(self, text: str):
        return [0.1, 0.2, 0.3]  # Fixed vector for testing

    def get_model_name(self):
        return "MockMPNet"


class TestWeaviateJsonConverter(unittest.TestCase):
    """Unit tests for WeaviateJsonConverter"""

    def setUp(self):
        """Creates a temporary directory and mock OHDSI CONCEPT.csv file"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_concept_file = os.path.join(self.temp_dir, "CONCEPT.csv")

        # Sample OHDSI data
        data = pd.DataFrame(
            {
                "concept_id": [1, 2, 3],
                "concept_name": ["Hypertension", "Diabetes", "Asthma"],
                "domain_id": ["Condition", "Condition", "Condition"],
                "vocabulary_id": ["OHDSI", "OHDSI", "OHDSI"],
                "concept_class_id": [
                    "Clinical Finding",
                    "Clinical Finding",
                    "Clinical Finding",
                ],
                "standard_concept": ["S", "S", "S"],
                "concept_code": ["C001", "C002", "C003"],
                "valid_start_date": ["2000-01-01", "2000-01-01", "2000-01-01"],
                "valid_end_data": ["2099-12-31", "2099-12-31", "2099-12-31"],
                "invalid_reason": ["", "", ""],
            }
        )

        # Save mock data as tab-separated CSV
        data.to_csv(self.mock_concept_file, sep="\t", index=False)

        # Initialize WeaviateJsonConverter
        self.converter = WeaviateJsonConverter(dest_dir=self.temp_dir)

        # Replace the embedding model with a mock
        self.converter._ohdsi_concept_to_mappings = (
            lambda pref_label, concept_id, embedding_model: {
                "class": self.converter.mapping_schema["class"],
                "id": generate_uuid5({"text": pref_label}),
                "properties": {
                    "text": pref_label,
                    "hasSentenceEmbedder": "MockMPNet",
                },
                "references": {"hasConcept": concept_id},
                "vector": {"default": [0.1, 0.2, 0.3]},
            }
        )

    def tearDown(self):
        """Removes the temporary directory after tests"""
        shutil.rmtree(self.temp_dir)

    def test_from_ohdsi_creates_expected_files(self):
        """Tests that from_ohdsi creates the expected JSON files"""

        # Run the OHDSI to JSON conversion
        self.converter.from_ohdsi(self.mock_concept_file)

        # Expected output files
        terminology_file = os.path.join(self.temp_dir, "terminology.json")
        concept_file = os.path.join(self.temp_dir, "concept.json")
        mapping_file = os.path.join(self.temp_dir, "mapping.json")

        # Check if all files are created
        self.assertTrue(
            os.path.exists(terminology_file), "Terminology file was not created."
        )
        self.assertTrue(os.path.exists(concept_file), "Concept file was not created.")
        self.assertTrue(os.path.exists(mapping_file), "Mapping file was not created.")

    def test_terminology_file_content(self):
        """Tests the correctness of the terminology.json file"""
        self.converter.from_ohdsi(self.mock_concept_file)

        terminology_file = os.path.join(self.temp_dir, "terminology.json")
        with open(terminology_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        self.assertEqual(
            len(data), 1, "Terminology file should contain only one entry."
        )
        self.assertEqual(
            data[0]["properties"]["name"], "OHDSI", "Incorrect terminology name."
        )

    def test_concept_file_content(self):
        """Tests the correctness of the concept.json file"""
        self.converter.from_ohdsi(self.mock_concept_file)

        concept_file = os.path.join(self.temp_dir, "concept.json")
        with open(concept_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        self.assertEqual(len(data), 3, "Concept file should contain 3 concepts.")
        expected_concepts = ["Hypertension", "Diabetes", "Asthma"]
        extracted_labels = [entry["properties"]["prefLabel"] for entry in data]

        self.assertCountEqual(
            extracted_labels,
            expected_concepts,
            "Concept labels do not match expected values.",
        )

    def test_mapping_file_content(self):
        """Tests the correctness of the mapping.json file"""
        self.converter.from_ohdsi(self.mock_concept_file)

        mapping_file = os.path.join(self.temp_dir, "mapping.json")
        with open(mapping_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        self.assertEqual(len(data), 3, "Mapping file should contain 3 entries.")

        # Verify that the embeddings are correct
        for entry in data:
            self.assertEqual(
                entry["vector"]["default"],
                [0.1, 0.2, 0.3],
                "Embedding vector is incorrect.",
            )

    def test_invalid_file_raises_error(self):
        """Tests that a missing file raises a FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.converter.from_ohdsi(os.path.join(self.temp_dir, "missing_file.csv"))

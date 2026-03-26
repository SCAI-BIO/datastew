import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from datastew.embedding import Vectorizer
from datastew.io.adapters import JsonlAdapter
from datastew.repository.model import Concept, Mapping, Terminology


class MockVectorizer(Vectorizer):
    """Mock vectorizer model to return numpy arrays to test conversion logic."""

    def __init__(self):
        self.model_name = "mock-model"

    def get_embeddings(self, texts):
        return [np.array([0.1, 0.2, 0.3]) for _ in texts]


class TestJsonlAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_concept_file = os.path.join(self.temp_dir, "CONCEPT.csv")

        # Sample OHDSI data
        data = pd.DataFrame(
            {
                "concept_id": [1, 2, 3],
                "concept_name": ["Hypertension", "Diabetes", "Asthma"],
                "domain_id": ["Condition", "Condition", "Condition"],
                "vocabulary_id": ["OHDSI"] * 3,
                "concept_class_id": ["Clinical Finding"] * 3,
                "standard_concept": ["S"] * 3,
                "concept_code": ["C001", "C002", "C003"],
                "valid_start_date": ["2000-01-01"] * 3,
                "valid_end_data": ["2099-12-31"] * 3,
                "invalid_reason": [""] * 3,
            }
        )
        data.to_csv(self.mock_concept_file, sep="\t", index=False)

        # Initialize with a tiny buffer size (2) to force flush testing
        self.converter = JsonlAdapter(dest_dir=self.temp_dir, buffer_size=2)
        self.mock_vectorizer_model = MockVectorizer()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_from_ohdsi_with_vectors(self):
        """Tests that from_ohdsi creates files and populates embeddings properly."""
        self.converter.from_ohdsi(self.mock_concept_file, self.mock_vectorizer_model, include_vectors=True)

        mapping_file = os.path.join(self.temp_dir, "mapping.jsonl")
        self.assertTrue(os.path.exists(mapping_file))

        with open(mapping_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        self.assertEqual(len(data), 3)
        # Verify the numpy array was successfully converted to a standard list
        self.assertEqual(data[0]["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(data[0]["vectorizer"], "mock-model")

    def test_from_ohdsi_without_vectors(self):
        """Tests the execution path where embeddigns are skipped."""
        self.converter.from_ohdsi(self.mock_concept_file, self.mock_vectorizer_model, include_vectors=False)

        mapping_file = os.path.join(self.temp_dir, "mapping.jsonl")
        with open(mapping_file, "r") as f:
            data = [json.loads(line) for line in f.readlines()]

        self.assertEqual(len(data), 3)
        self.assertNotIn("embedding", data[0])
        self.assertNotIn("vectorizer", data[0])

    def test_invalid_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            self.converter.from_ohdsi(os.path.join(self.temp_dir, "missing-file.csv"), self.mock_vectorizer_model)

    def test_from_repository(self):
        """Tests extracting ORM objects and saving them to JSONL."""
        mock_repo = MagicMock()
        mock_session = mock_repo.session

        # Setup mock Terminology
        term = Terminology(name="Test Term", short_name="TT")

        # Setup mock Concept
        concept = Concept(concept_identifier="TT:1", pref_label="Test Concept")
        concept.terminology = term

        # Setup mock Mapping
        mapping = Mapping(text="Test Desc", embedding=[0.5, 0.5], vectorizer="test-vec")
        mapping.concept = concept

        # Configure the mock session to return our specific objects when required
        def mock_query(model):
            query_mock = MagicMock()
            if model == Terminology:
                query_mock.all.return_value = [term]
            elif model == Concept:
                query_mock.all.return_value = [concept]
            elif model == Mapping:
                query_mock.all.return_value = [mapping]
            return query_mock

        mock_session.query.side_effect = mock_query
        self.converter.from_repository(mock_repo)

        with open(os.path.join(self.temp_dir, "terminology.jsonl"), "r") as f:
            t_data = json.loads(f.readline())
            self.assertEqual(t_data["short_name"], "TT")

        with open(os.path.join(self.temp_dir, "concept.jsonl"), "r") as f:
            c_data = json.loads(f.readline())
            self.assertEqual(c_data["terminology_short_name"], "TT")
            self.assertEqual(c_data["pref_label"], "Test Concept")

        with open(os.path.join(self.temp_dir, "mapping.jsonl"), "r") as f:
            m_data = json.loads(f.readline())
            self.assertEqual(m_data["concept_identifier"], "TT:1")
            self.assertEqual(m_data["embedding"], [0.5, 0.5])

    def test_object_to_dict_unsupported_type(self):
        """Verifies TypeError is raised for invalid ORM objects."""

        class UnknownModel:
            pass

        with self.assertRaisesRegex(TypeError, "Unsupported object type"):
            self.converter._object_to_dict(UnknownModel())  # type: ignore


if __name__ == "__main__":
    unittest.main()

import json
import os
import random
import shutil
import tempfile
import unittest
from typing import Any, Literal
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datastew.io.importer import Importer
from datastew.io.source import DataDictionarySource
from datastew.repository import PostgreSQLRepository


class TestImporter(unittest.TestCase):
    def setUp(self):
        postgres_url = os.getenv("TEST_POSTGRES_URI", "postgresql://testuser:testpass@localhost/testdb")
        if postgres_url.startswith("postgresql://"):
            postgres_url = postgres_url.replace("postgresql://", "postgresql+psycopg://", 1)

        self.TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

        self.engine = create_engine(postgres_url)
        PostgreSQLRepository.setup_database(self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False)
        self.session = self.SessionLocal()

        self.repository = PostgreSQLRepository(session=self.session)
        self.repository.clear_all()
        self.session.commit()

        self.importer = Importer(self.repository)
        self.temp_dir = tempfile.mkdtemp()

        # Sample data for JSONL files
        self.data_files = {
            "terminology": [{"name": "import_test", "short_name": "import_test"}],
            "concept": [
                {"concept_identifier": "import_test:G", "pref_label": "G", "terminology_short_name": "import_test"},
                {"concept_identifier": "import_test:H", "pref_label": "H", "terminology_short_name": "import_test"},
            ],
            "mapping": [
                {
                    "text": "pancreas",
                    "concept_identifier": "import_test:G",
                    "embedding": [random.uniform(-1, 1) for _ in range(768)],
                    "vectorizer": "sentence-transformers/all-mpnet-base-v2",
                },
                {
                    "text": "liver",
                    "concept_identifier": "import_test:H",
                    "embedding": [random.uniform(-1, 1) for _ in range(768)],
                    "vectorizer": "sentence-transformers/all-mpnet-base-v2",
                },
            ],
        }

        # Write data to JSONL files
        for key, data in self.data_files.items():
            self.write_jsonl(os.path.join(self.temp_dir, f"{key}.jsonl"), data)

    @staticmethod
    def write_jsonl(file_path: str, data: list[dict[str, Any]]):
        """Write data to a JSONL file."""
        with open(file_path, "w", encoding="utf-8") as file:
            for obj in data:
                file.write(json.dumps(obj) + "\n")

    def import_data(self, data_types: list[Literal["terminology", "concept", "mapping"]]):
        """Helper method to import multiple data types."""
        for data_type in data_types:
            file_path = os.path.join(self.temp_dir, f"{data_type}.jsonl")
            self.importer.import_from_jsonl(file_path, data_type)
        self.session.commit()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.session.close()
        self.engine.dispose()

    def test_import_terminology(self):
        self.import_data(["terminology"])
        terminology = self.repository.get_all_terminologies()

        self.assertEqual(len(terminology), 1)
        with self.subTest("Terminology Short Name"):
            self.assertEqual(terminology[0].short_name, "import_test")
        with self.subTest("Terminology Name"):
            self.assertEqual(terminology[0].name, "import_test")
        with self.subTest("Terminology ID is Integer"):
            self.assertIsInstance(terminology[0].id, int)

    def test_import_concepts(self):
        self.import_data(["terminology", "concept"])
        concepts = self.repository.get_concepts(limit=5, offset=0).items

        self.assertEqual(len(concepts), 2)

        for concept in concepts:
            with self.subTest(f"Concept Properties: {concept.pref_label}"):
                self.assertIn(concept.pref_label, ["G", "H"])
                self.assertIn(concept.concept_identifier, ["import_test:G", "import_test:H"])
            with self.subTest(f"Terminology Reference for {concept.pref_label}"):
                self.assertEqual(concept.terminology.name, "import_test")

    def test_import_mappings(self):
        self.import_data(["terminology", "concept", "mapping"])
        mappings = self.repository.get_mappings(limit=10, offset=0).items

        self.assertEqual(len(mappings), 2)

        for mapping in mappings:
            with self.subTest(f"Mapping Text for {mapping.text}"):
                self.assertIn(mapping.text, ["pancreas", "liver"])
            with self.subTest(f"Vectorizer for {mapping.text}"):
                self.assertEqual(mapping.vectorizer, "sentence-transformers/all-mpnet-base-v2")
            with self.subTest(f"Vector Length for {mapping.text}"):
                embedding = mapping.embedding
                self.assertIsNotNone(embedding)
                assert embedding is not None
                self.assertEqual(len(embedding), 768)
            with self.subTest(f"Concept Reference for Mapping {mapping.text}"):
                expected_label = "G" if mapping.text == "pancreas" else "H"
                self.assertEqual(mapping.concept.pref_label, expected_label)

    def test_import_invalid_jsonl(self):
        invalid_file = os.path.join(self.temp_dir, "invalid.jsonl")
        with open(invalid_file, "w") as file:
            file.write("{ invalid jsonl }")

        with self.assertRaises(ValueError):
            self.importer.import_from_jsonl(invalid_file, "terminology")

    def test_import_invalid_object_type(self):
        """Verify that an unsupported object type raises a ValueError."""
        file_path = os.path.join(self.temp_dir, "terminology.jsonl")
        with self.assertRaisesRegex(ValueError, "Unsupported object_type: unknown"):
            self.importer.import_from_jsonl(file_path, "unknown")  # type: ignore

    def test_import_missing_required_field(self):
        file_path = os.path.join(self.temp_dir, "missing_required.jsonl")
        # Missing 'short_name'
        self.write_jsonl(file_path, [{"name": "missing_id"}])

        with self.assertRaises(ValueError):
            self.importer.import_from_jsonl(file_path, "terminology")

    def test_import_mapping_generate_embeddings_success(self):
        """Verify that mappings can be imported and embedded on the fly."""
        self.import_data(["terminology", "concept"])

        # Create a mapping file WITHOUT embeddings
        file_path = os.path.join(self.temp_dir, "mapping_no_emb.jsonl")
        data = [{"text": "kidney", "concept_identifier": "import_test:G"}]
        self.write_jsonl(file_path, data)

        # Mock the vectorizer to return a fake embedding
        mock_vectorizer = Mock()
        mock_vectorizer.model_name = "fake-model"
        mock_vectorizer.get_embeddings.return_value = [[0.5] * 768]

        # Temporarily replace the importer's vectorizer
        original_vectorizer = self.importer.vectorizer
        self.importer.vectorizer = mock_vectorizer

        try:
            self.importer.import_from_jsonl(file_path, "mapping", generate_embeddings=True)
            self.session.commit()
            mappings = self.repository.get_mappings().items
            self.assertEqual(len(mappings), 1)
            self.assertEqual(mappings[0].vectorizer, "fake-model")
            self.assertEqual(mappings[0].text, "kidney")
            mock_vectorizer.get_embeddings.assert_called_once_with(["kidney"])
        finally:
            self.importer.vectorizer = original_vectorizer

    @patch("datastew.io.importer.Terminology")
    def test_import_data_dictionary_runtime_error(self, mock_terminology):
        """Verify that exceptions during data dictionary imports are wrapped in RuntimeErrors."""
        # Force an exception when creating the Terminology object
        mock_terminology.side_effect = Exception("Database locked")

        path = os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv")
        source = DataDictionarySource(path, "VAR_1", "DESC")

        with self.assertRaisesRegex(RuntimeError, "Failed to import data dictionary source: Database locked"):
            self.importer.import_data_dictionary(source, terminology_name="fail_test", short_name="FAIL")

    def test_import_empty_file(self):
        empty_file = os.path.join(self.temp_dir, "empty.jsonl")
        open(empty_file, "w").close()

        self.importer.import_from_jsonl(empty_file, "terminology")
        self.session.commit()
        terminology = self.repository.get_all_terminologies()
        self.assertEqual(len(terminology), 0)

    def test_import_data_dictionary(self):
        path = os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv")
        source = DataDictionarySource(path, "VAR_1", "DESC")

        self.importer.import_data_dictionary(source, terminology_name="import_test", short_name="IMPORT")
        self.session.commit()

        terminology = self.repository.get_terminology_by_name("import_test")
        self.assertEqual("import_test", terminology.name)

        mappings = self.repository.get_mappings(terminology_name="import_test").items
        texts = [m.text for m in mappings]
        df = source.to_dataframe()

        for row in df.itertuples(index=False):
            cid = f"import_test:{row.variable}"
            concept = self.repository.get_concept_by_identifier(cid)
            self.assertEqual(concept.pref_label, row.variable)
            self.assertIn(row.description, texts)


if __name__ == "__main__":
    unittest.main()

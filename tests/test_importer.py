import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path
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
        self.jsonl_path = os.path.join(self.temp_dir, "records.jsonl")
        self.terminology: dict[str, object] = {"id": 1, "name": "import_test", "short_name": "IMPORT"}
        self.records: list[dict[str, object]] = [
            {
                "terminology": self.terminology,
                "concept": {"id": 101, "concept_identifier": "import_test:G", "pref_label": "G"},
                "mappings": [
                    {
                        "text": "pancreas",
                        "embedding": [random.uniform(-1, 1) for _ in range(768)],
                        "vectorizer": "sentence-transformers/all-mpnet-base-v2",
                    }
                ],
            },
            {
                "terminology": self.terminology,
                "concept": {"id": 102, "concept_identifier": "import_test:H", "pref_label": "H"},
                "mappings": [
                    {
                        "text": "liver",
                        "embedding": [random.uniform(-1, 1) for _ in range(768)],
                        "vectorizer": "sentence-transformers/all-mpnet-base-v2",
                    }
                ],
            },
        ]
        self.write_jsonl(self.jsonl_path, self.records)

    @staticmethod
    def write_jsonl(file_path: str, data: list[dict[str, object]]) -> None:
        """Write one self-contained import record per JSONL line.

        :param file_path: Destination JSONL path.
        :param data: Records containing terminology, concept, and mappings.
        :return: None.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            for record in data:
                file.write(json.dumps(record) + "\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.session.close()
        self.engine.dispose()

    def test_import_terminology(self) -> None:
        result = self.importer.import_from_jsonl(self.jsonl_path)
        terminology = self.repository.get_all_terminologies()

        self.assertEqual(result.terminologies, 1)
        self.assertEqual(len(terminology), 1)
        self.assertEqual(terminology[0].id, 1)
        self.assertEqual(terminology[0].name, "import_test")
        self.assertEqual(terminology[0].short_name, "IMPORT")

    def test_import_concepts(self) -> None:
        """Verify predefined concept IDs and terminology relationships are preserved.

        :return: None.
        """
        result = self.importer.import_from_jsonl(self.jsonl_path)
        concepts = self.repository.get_concepts(limit=10).items

        self.assertEqual(result.concepts, 2)
        self.assertEqual({concept.id for concept in concepts}, {101, 102})
        self.assertEqual({concept.concept_identifier for concept in concepts}, {"import_test:G", "import_test:H"})
        self.assertTrue(all(concept.terminology_id == 1 for concept in concepts))

    def test_import_mappings(self) -> None:
        """Verify mapping embeddings and concept references are stored with COPY.

        :return: None.
        """
        result = self.importer.import_from_jsonl(self.jsonl_path)
        mappings = self.repository.get_mappings(limit=10).items

        self.assertEqual(result.mappings, 2)
        self.assertEqual(result.generated_embeddings, 0)
        self.assertEqual({mapping.text for mapping in mappings}, {"pancreas", "liver"})
        for mapping in mappings:
            with self.subTest(mapping=mapping.text):
                embedding = mapping.embedding
                self.assertIsNotNone(embedding)
                assert embedding is not None
                self.assertEqual(len(embedding), 768)
                self.assertEqual(mapping.vectorizer, "sentence-transformers/all-mpnet-base-v2")
                expected_label = "G" if mapping.text == "pancreas" else "H"
                self.assertEqual(mapping.concept.pref_label, expected_label)

    def test_import_multiple_terminologies(self) -> None:
        """Verify one JSONL import can create more than one terminology.

        :return: None.
        """
        additional: dict[str, object] = {
            "terminology": {"id": 2, "name": "another_terminology", "short_name": "OTHER"},
            "concept": {"id": 103, "concept_identifier": "OTHER:1", "pref_label": "Third concept"},
            "mappings": [],
        }
        self.write_jsonl(self.jsonl_path, [*self.records, additional])

        result = self.importer.import_from_jsonl(self.jsonl_path, batch_size=2)

        self.assertEqual(result.terminologies, 2)
        self.assertEqual(result.concepts, 3)
        self.assertEqual(result.mappings, 2)
        self.assertEqual(len(self.repository.get_all_terminologies()), 2)
        concept = self.repository.get_concept_by_identifier("OTHER:1")
        self.assertEqual(concept.terminology_id, 2)

    def test_import_multiple_jsonl_files(self) -> None:
        """Verify sharded files can share one terminology without duplicate inserts.

        :return: None.
        """
        first_path = os.path.join(self.temp_dir, "part_1.jsonl")
        second_path = os.path.join(self.temp_dir, "part_2.jsonl")
        self.write_jsonl(first_path, self.records[:1])
        self.write_jsonl(second_path, self.records[1:])

        result = self.importer.import_from_jsonl([Path(first_path), Path(second_path)], batch_size=1)

        self.assertEqual(result.files, 2)
        self.assertEqual(result.terminologies, 1)
        self.assertEqual(result.concepts, 2)
        self.assertEqual(len(self.repository.get_all_terminologies()), 1)

    def test_import_invalid_jsonl(self) -> None:
        """Verify malformed JSONL input raises an error.

        :return: None.
        """
        invalid_file = os.path.join(self.temp_dir, "invalid.jsonl")
        with open(invalid_file, "w", encoding="utf-8") as file:
            file.write("{ invalid jsonl }\n")

        with self.assertRaises(ValueError):
            self.importer.import_from_jsonl(invalid_file)

    def test_import_missing_required_field(self) -> None:
        """Verify that malformed terminology metadata is rejected.

        :return: None.
        """
        invalid_file = os.path.join(self.temp_dir, "missing_required.jsonl")
        record: dict[str, object] = {
            "terminology": {"id": 1, "name": "import_test"},
            "concept": {"id": 101, "concept_identifier": "import_test:G", "pref_label": "G"},
            "mappings": [],
        }
        self.write_jsonl(invalid_file, [record])

        with self.assertRaisesRegex(ValueError, "terminology.short_name"):
            self.importer.import_from_jsonl(invalid_file)

    def test_import_conflicting_terminology(self) -> None:
        """Verify inconsistent terminology definitions are rejected.

        :return: None.
        """
        changed_record: dict[str, object] = {
            **self.records[1],
            "terminology": {"id": 1, "name": "wrong_name", "short_name": "IMPORT"},
        }
        self.write_jsonl(self.jsonl_path, [self.records[0], changed_record])

        with self.assertRaisesRegex(ValueError, "Conflicting metadata"):
            self.importer.import_from_jsonl(self.jsonl_path)

    def test_import_mapping_generate_embeddings_success(self) -> None:
        """Verify missing mapping embeddings are generated during import.

        This preserves the mock-vectorizer approach used by the original test.

        :return: None.
        """
        record: dict[str, object] = {
            "terminology": self.terminology,
            "concept": {"id": 101, "concept_identifier": "import_test:G", "pref_label": "G"},
            "mappings": [{"text": "kidney", "embedding": None, "vectorizer": None}],
        }
        self.write_jsonl(self.jsonl_path, [record])

        mock_vectorizer = Mock()
        mock_vectorizer.model_name = "fake-model"
        mock_vectorizer.get_embeddings.return_value = [[0.5] * 768]

        # The bulk importer reads the vectorizer from the repository, not Importer.vectorizer.
        original_vectorizer = self.repository.vectorizer
        self.repository.vectorizer = mock_vectorizer
        try:
            result = self.importer.import_from_jsonl(self.jsonl_path, generate_embeddings=True)
            mappings = self.repository.get_mappings().items

            self.assertEqual(result.generated_embeddings, 1)
            self.assertEqual(len(mappings), 1)
            self.assertEqual(mappings[0].vectorizer, "fake-model")
            self.assertEqual(mappings[0].text, "kidney")
            mock_vectorizer.get_embeddings.assert_called_once_with(["kidney"])
        finally:
            self.repository.vectorizer = original_vectorizer

    def test_import_missing_embedding_without_generation(self) -> None:
        """Verify incomplete mapping data is rejected when generation is disabled.

        :return: None.
        """
        record: dict[str, object] = {
            "terminology": self.terminology,
            "concept": {"id": 101, "concept_identifier": "import_test:G", "pref_label": "G"},
            "mappings": [{"text": "kidney", "embedding": None, "vectorizer": None}],
        }
        self.write_jsonl(self.jsonl_path, [record])

        with self.assertRaisesRegex(ValueError, "Missing embedding"):
            self.importer.import_from_jsonl(self.jsonl_path)

        self.assertEqual(self.repository.get_concepts().total_count, 0)
        self.assertEqual(len(self.repository.get_all_terminologies()), 0)

    @patch("datastew.io.importer.Terminology")
    def test_import_data_dictionary_runtime_error(self, mock_terminology: Mock) -> None:
        """Verify data-dictionary failures are wrapped in a RuntimeError.

        :param mock_terminology: Patched terminology constructor used to force an error.
        :return: None.
        """
        mock_terminology.side_effect = Exception("Database locked")
        path = os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv")
        source = DataDictionarySource(path, "VAR_1", "DESC")

        with self.assertRaisesRegex(RuntimeError, "Failed to import data dictionary source: Database locked"):
            self.importer.import_data_dictionary(source, terminology_name="fail_test", short_name="FAIL")

    def test_import_empty_file(self) -> None:
        """Verify empty JSONL files do not create records.

        :return: None.
        """
        empty_file = os.path.join(self.temp_dir, "empty.jsonl")
        open(empty_file, "w").close()

        result = self.importer.import_from_jsonl(empty_file)

        self.assertEqual(result.terminologies, 0)
        self.assertEqual(result.concepts, 0)
        self.assertEqual(result.mappings, 0)
        self.assertEqual(len(self.repository.get_all_terminologies()), 0)

    def test_import_data_dictionary(self) -> None:
        """Verify an actual data dictionary is imported into PostgreSQL.

        :return: None.
        """
        path = os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv")
        source = DataDictionarySource(path, "VAR_1", "DESC")

        self.importer.import_data_dictionary(source, terminology_name="import_test", short_name="IMPORT")
        self.session.commit()

        terminology = self.repository.get_terminology_by_name("import_test")
        self.assertEqual("import_test", terminology.name)

        mappings = self.repository.get_mappings(terminology_name="import_test").items
        texts = [mapping.text for mapping in mappings]
        df = source.to_dataframe()

        for row in df.itertuples(index=False):
            concept_identifier = f"import_test:{row.variable}"
            concept = self.repository.get_concept_by_identifier(concept_identifier)
            self.assertEqual(concept.pref_label, row.variable)
            self.assertIn(row.description, texts)


if __name__ == "__main__":
    unittest.main()

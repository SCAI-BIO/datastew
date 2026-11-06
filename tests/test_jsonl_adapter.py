import gzip
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from datastew.embedding import Vectorizer
from datastew.io.adapters import JsonlAdapter
from datastew.io.bulk import ImportObject
from datastew.repository.model import EMBEDDING_DIMENSIONS, Concept, Mapping, Terminology


class MockVectorizer(Vectorizer):
    def __init__(self) -> None:
        self.model_name = "mock-model"

    def get_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Return a compatible embedding for each input text.

        :param texts: Texts to embed.
        :return: NumPy arrays matching DataStew's configured embedding dimensions.
        """
        values = [0.1, 0.2, 0.3] + [0.0] * (EMBEDDING_DIMENSIONS - 3)
        return [np.array(values, dtype=float) for _ in texts]


class TestJsonlAdapter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_concept_file = os.path.join(self.temp_dir, "CONCEPT.csv")

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

        self.converter = JsonlAdapter(dest_dir=self.temp_dir, buffer_size=2)
        self.mock_vectorizer_model = MockVectorizer()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _read_jsonl(paths: list[Path]) -> list[ImportObject]:
        """Read exported JSONL shards, including gzip-compressed files.

        :param paths: Paths returned by an adapter export method.
        :return: Exported import records in file order.
        """
        records: list[ImportObject] = []
        for path in paths:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as handle:
                records.extend(cast(ImportObject, json.loads(line)) for line in handle)
        return records

    def test_from_ohdsi_with_vectors(self) -> None:
        """Verify OHDSI exports embed concept labels and serialize NumPy arrays."""
        paths = self.converter.from_ohdsi(
            self.mock_concept_file, self.mock_vectorizer_model, include_vectors=True, records_per_file=2
        )
        records = self._read_jsonl(paths)

        self.assertEqual([path.name for path in paths], ["ohdsi_00001.jsonl", "ohdsi_00002.jsonl"])
        self.assertEqual(len(records), 3)
        self.assertEqual([record["concept"]["id"] for record in records], [1, 2, 3])
        self.assertEqual(
            [record["concept"]["concept_identifier"] for record in records],
            ["OHDSI:1", "OHDSI:2", "OHDSI:3"],
        )
        self.assertEqual(
            [record["concept"]["pref_label"] for record in records],
            ["Hypertension", "Diabetes", "Asthma"],
        )

        for record in records:
            with self.subTest(concept=record["concept"]["id"]):
                terminology = record["terminology"]
                self.assertEqual(terminology["id"], 1)
                self.assertEqual(terminology["short_name"], "OHDSI")
                self.assertEqual(terminology["name"], "Observational Health Data Sciences and Informatics")
                self.assertEqual(len(record["mappings"]), 1)
                mapping = record["mappings"][0]
                self.assertEqual(mapping["text"], record["concept"]["pref_label"])
                self.assertEqual(mapping["vectorizer"], "mock-model")
                embedding = mapping["embedding"]
                self.assertIsNotNone(embedding)
                assert embedding is not None
                self.assertEqual(len(embedding), EMBEDDING_DIMENSIONS)
                self.assertEqual(embedding[:3], [0.1, 0.2, 0.3])

    def test_from_ohdsi_without_vectors(self) -> None:
        """Verify that embeddings and model names are null when generation is deferred.

        :return: None.
        """
        paths = self.converter.from_ohdsi(
            self.mock_concept_file,
            self.mock_vectorizer_model,
            include_vectors=False,
        )
        records = self._read_jsonl(paths)

        self.assertEqual(len(paths), 1)
        self.assertEqual(len(records), 3)
        for record in records:
            mapping = record["mappings"][0]
            self.assertIsNone(mapping["embedding"])
            self.assertIsNone(mapping["vectorizer"])
            self.assertEqual(mapping["text"], record["concept"]["pref_label"])

    def test_from_ohdsi_custom_ids(self) -> None:
        """Verify configurable terminology IDs and sequential concept IDs.

        :return: None.
        """
        paths = self.converter.from_ohdsi(
            self.mock_concept_file,
            include_vectors=False,
            terminology_id=7,
            first_concept_id=100,
        )
        records = self._read_jsonl(paths)

        self.assertEqual([record["concept"]["id"] for record in records], [100, 101, 102])
        self.assertTrue(all(record["terminology"]["id"] == 7 for record in records))
        self.assertEqual(records[0]["concept"]["concept_identifier"], "OHDSI:1")

    def test_from_ohdsi_compressed_shards(self) -> None:
        """Verify compressed shards are individually readable and respect record limits.

        :return: None.
        """
        paths = self.converter.from_ohdsi(
            self.mock_concept_file,
            include_vectors=False,
            records_per_file=2,
            compress=True,
        )

        self.assertEqual([path.name for path in paths], ["ohdsi_00001.jsonl.gz", "ohdsi_00002.jsonl.gz"])
        self.assertEqual([len(self._read_jsonl([path])) for path in paths], [2, 1])
        self.assertEqual(len(self._read_jsonl(paths)), 3)

    def test_invalid_file_raises_error(self) -> None:
        """Verify nonexistent OHDSI sources raise FileNotFoundError.

        :return: None.
        """
        with self.assertRaises(FileNotFoundError):
            self.converter.from_ohdsi(os.path.join(self.temp_dir, "missing-file.csv"), self.mock_vectorizer_model)

    def test_invalid_source_columns(self) -> None:
        """Verify missing OHDSI source columns are rejected.

        :return: None.
        """
        invalid_path = Path(self.temp_dir) / "invalid_columns.csv"
        invalid_path.write_text("concept_id\twrong_column\n1\tDiabetes\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "requires 'concept_id' and 'concept_name'"):
            self.converter.from_ohdsi(invalid_path, include_vectors=False)

    def test_invalid_source_removes_partial_shards(self) -> None:
        """Verify a failed export removes any shards created during that export.

        :return: None.
        """
        invalid_path = Path(self.temp_dir) / "invalid_rows.csv"
        invalid_path.write_text(
            "concept_id\tconcept_name\n1\tFirst\n2\tSecond\n3\t\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Missing OHDSI concept_name"):
            self.converter.from_ohdsi(invalid_path, include_vectors=False, records_per_file=1)

        self.assertEqual(list(Path(self.temp_dir).glob("ohdsi_*.jsonl")), [])

    def test_from_ohdsi_rejects_existing_shards(self) -> None:
        """Verify existing output files are not overwritten by repeated exports.

        :return: None.
        """
        paths = self.converter.from_ohdsi(self.mock_concept_file, include_vectors=False)
        original = paths[0].read_bytes()

        with self.assertRaises(FileExistsError):
            self.converter.from_ohdsi(self.mock_concept_file, include_vectors=False)

        self.assertEqual(paths[0].read_bytes(), original)

    def test_from_repository(self) -> None:
        """Verify repository export combines real ORM objects into a single record.

        The repository session remains a MagicMock, as in the original tests.

        :return: None.
        """
        mock_repo = MagicMock()
        term = Terminology(id=5, name="Test Term", short_name="TT")
        concept = Concept(id=42, terminology_id=5, concept_identifier="TT:1", pref_label="Test Concept")
        concept.terminology = term
        mapping = Mapping(
            id=70,
            concept_id=42,
            text="Test Desc",
            embedding=[0.5] * EMBEDDING_DIMENSIONS,
            vectorizer="test-vec",
        )
        concept.mappings = [mapping]
        mock_repo.session.scalar.return_value = None
        mock_repo.session.scalars.return_value = [concept]

        paths = self.converter.from_repository(mock_repo)
        records = self._read_jsonl(paths)

        self.assertEqual([path.name for path in paths], ["repository_00001.jsonl"])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["terminology"], {"id": 5, "name": "Test Term", "short_name": "TT"})
        self.assertEqual(
            records[0]["concept"],
            {"id": 42, "concept_identifier": "TT:1", "pref_label": "Test Concept"},
        )
        self.assertEqual(len(records[0]["mappings"]), 1)
        self.assertEqual(records[0]["mappings"][0]["text"], "Test Desc")
        self.assertEqual(records[0]["mappings"][0]["vectorizer"], "test-vec")
        self.assertEqual(records[0]["mappings"][0]["embedding"], [0.5] * EMBEDDING_DIMENSIONS)
        self.assertNotIn("id", records[0]["mappings"][0])

    def test_from_repository_rejects_empty_terminology(self) -> None:
        """Verify a terminology without concepts is not silently omitted.

        :return: None.
        """
        mock_repo = MagicMock()
        mock_repo.session.scalar.return_value = 5

        with self.assertRaisesRegex(ValueError, "has no concepts"):
            self.converter.from_repository(mock_repo)

        self.assertEqual(list(Path(self.temp_dir).glob("repository_*.jsonl")), [])


if __name__ == "__main__":
    unittest.main()

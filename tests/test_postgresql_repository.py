import os
import unittest

from datastew.embedding import Vectorizer
from datastew.process.jsonl_adapter import SQLJsonlConverter
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import MappingResult


class TestPostgreSQLRepository(unittest.TestCase):
    """Tests for the PostgreSQL repository backend."""

    POSTGRES_TEST_URL = os.getenv("TEST_POSTGRES_URI", "postgresql://testuser:testpass@localhost/testdb")
    TEST_CONCEPTS = [
        ("Diabetes mellitus (disorder)", "Concept ID: 11893007", "v1"),
        ("Hypertension (disorder)", "Concept ID: 73211009", "v2"),
        ("Asthma", "Concept ID: 195967001", "v1"),
        ("Heart attack", "Concept ID: 22298006", "v2"),
        ("Complex General Surgical Oncology", "Concept ID: 45756764", "v1"),
        ("Cancer", "Concept ID: 45877275", "v1"),
        ("Common cold", "Concept ID: 13260007", "v1"),
        ("Stroke", "Concept ID: 422504002", "v2"),
        ("Migraine", "Concept ID: 386098009", "v1"),
        ("Influenza", "Concept ID: 57386000", "v2"),
        ("Osteoarthritis", "Concept ID: 399206004", "v1"),
    ]

    @classmethod
    def setUpClass(cls):
        """Setup for vectorizers, paths, and PostgreSQL repository."""
        cls.TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        cls.vectorizer1 = Vectorizer("sentence-transformers/all-mpnet-base-v2")
        cls.vectorizer2 = Vectorizer("FremyCompany/BioLORD-2023")
        cls.model_name1 = cls.vectorizer1.model_name
        cls.model_name2 = cls.vectorizer2.model_name
        cls.test_text = "The flu"

        cls.repo_args = (cls.POSTGRES_TEST_URL, cls.vectorizer1)
        cls.repository = PostgreSQLRepository(*cls.repo_args)
        cls.jsonl_converter = SQLJsonlConverter(dest_dir="test_export")

    @classmethod
    def tearDownClass(cls):
        """Ensure repository is properly closed"""
        if hasattr(cls, "repository") and cls.repository:
            cls.repository.shut_down()

    def setUp(self):
        """Reset the repository state before each test."""
        self._reset_repository()

    def _reset_repository(self):
        """Clear the repo and re-import base data"""
        self.repository.clear_all()

        term1 = self.repository.add_terminology("snomed CT", "SNOMED")
        term2 = self.repository.add_terminology("NCI Thesaurus OBO Edition", "NCIT")

        for i, (label, cid, vkey) in enumerate(self.TEST_CONCEPTS):
            term = term1 if i < 6 else term2
            vectorizer = self.vectorizer1 if vkey == "v1" else self.vectorizer2

            concept = self.repository.add_concept(terminology_id=term.id, pref_label=label, concept_identifier=cid)

            embedding = vectorizer.get_embedding(label)
            self.repository.add_mapping(
                concept_id=concept.id, text=label, embedding=embedding, sentence_embedder=vectorizer.model_name
            )

    def test_terminology_retrieval(self):
        terminology = self.repository.get_terminology_by_name("snomed CT")
        terminologies = self.repository.get_all_terminologies()
        names = [t.name for t in terminologies]

        self.assertEqual(terminology.name, "snomed CT")
        self.assertIn("NCI Thesaurus OBO Edition", names)
        self.assertIn("snomed CT", names)

    def test_concept_retrieval(self):
        concepts = self.repository.get_concepts()
        concept = self.repository.get_concept_by_identifier("Concept ID: 11893007")

        self.assertEqual(concept.concept_identifier, "Concept ID: 11893007")
        self.assertEqual(concept.pref_label, "Diabetes mellitus (disorder)")
        self.assertEqual(concept.terminology.name, "snomed CT")
        self.assertEqual(len(concepts.items), 11)

    def test_mapping_retrieval(self):
        mappings = self.repository.get_mappings(limit=5).items
        self.assertEqual(len(mappings), 5)

    def test_sentence_embedders(self):
        embedders = self.repository.get_all_sentence_embedders()
        self.assertIn(self.model_name1, embedders)
        self.assertIn(self.model_name2, embedders)

    def test_closest_mappings(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        assert embedding is not None
        closest = self.repository.get_closest_mappings(embedding)

        self.assertEqual(len(closest), 5)

        result = closest[0]
        assert isinstance(result, MappingResult)

        self.assertEqual(result.mapping.text, "Common cold")
        self.assertEqual(result.mapping.sentence_embedder, self.model_name1)

    def test_terminology_and_model_specific_mappings(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        closest = self.repository.get_closest_mappings(
            embedding, terminology_name="snomed CT", sentence_embedder=self.model_name1
        )
        self.assertEqual(len(closest), 4)

        result = closest[0]
        assert isinstance(result, MappingResult)

        self.assertEqual(result.mapping.text, "Asthma")

    def test_closest_mappings_with_similarities(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        assert embedding is not None
        closest = self.repository.get_closest_mappings(embedding)

        result = closest[0]
        assert isinstance(result, MappingResult)

        self.assertAlmostEqual(result.similarity, 0.6747197, places=3)

    def test_closest_mapping_with_similarity_for_identical_entry(self):
        embedding = self.vectorizer1.get_embedding("Cancer")
        assert embedding is not None
        closest = self.repository.get_closest_mappings(embedding)

        result = closest[0]
        assert isinstance(result, MappingResult)

        self.assertEqual(result.mapping.text, "Cancer")
        self.assertAlmostEqual(result.similarity, 1.0, places=3)

    def test_terminology_and_model_specific_mappings_with_similarities(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        assert embedding is not None
        closest = self.repository.get_closest_mappings(embedding, True, "snomed CT", self.model_name1)

        self.assertEqual(len(closest), 4)

        result = closest[0]
        assert isinstance(result, MappingResult)

        self.assertEqual(result.mapping.text, "Asthma")
        self.assertAlmostEqual(result.similarity, 0.3947341, places=3)

    def test_repository_restart(self):
        repo_class = type(self.repository)
        args = getattr(self, "repo_args", ())
        repo = repo_class(*args)

        mappings = repo.get_mappings(limit=5).items
        self.assertEqual(len(mappings), 5)

        concepts = repo.get_concepts()
        self.assertEqual(len(concepts.items), 11)

    def test_jsonl_export(self):
        converter = self.jsonl_converter
        converter.from_repository(self.repository)
        # assert that the dest dir
        self.assertTrue(converter.dest_dir)
        # assert that the files is not empty
        with open(converter.dest_dir + "/terminology.jsonl", "r") as file:
            self.assertTrue(file.read())
        with open(converter.dest_dir + "/concept.jsonl", "r") as file:
            self.assertTrue(file.read())
        with open(converter.dest_dir + "/mapping.jsonl", "r") as file:
            self.assertTrue(file.read())
        # assert that the file contains the expected data
        with open(converter.dest_dir + "/terminology.jsonl", "r") as file:
            self.assertIn("snomed CT", file.read())
        with open(converter.dest_dir + "/concept.jsonl", "r") as file:
            self.assertIn("Diabetes mellitus (disorder)", file.read())
        with open(converter.dest_dir + "/mapping.jsonl", "r") as file:
            self.assertIn("Diabetes mellitus (disorder)", file.read())
        # remove the created dir and files
        os.remove(converter.dest_dir + "/terminology.jsonl")
        os.remove(converter.dest_dir + "/concept.jsonl")
        os.remove(converter.dest_dir + "/mapping.jsonl")
        # remove the created dir
        os.rmdir(converter.dest_dir)

import os
import unittest
from typing import List, Tuple

from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.repository import Concept, Mapping, Terminology
from datastew.repository.base import BaseRepository


class BaseRepositoryTestSetup(unittest.TestCase):
    """Base class for setting up test data and shared tests for all repository backends."""
    __test__ = False
    repository: BaseRepository

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
        """Shared setup for vectorizers and paths."""
        cls.TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        cls.vectorizer1 = Vectorizer("sentence-transformers/all-mpnet-base-v2")
        cls.vectorizer2 = Vectorizer("FremyCompany/BioLORD-2023")
        cls.model_name1 = cls.vectorizer1.model_name
        cls.model_name2 = cls.vectorizer2.model_name
        cls.test_text = "The flu"

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

        self.terminology1 = Terminology("snomed CT", "SNOMED")
        self.terminology2 = Terminology("NCI Thesaurus OBO Edition", "NCIT")

        self.concepts_mappings: List[Tuple[Concept, Mapping]] = []
        for i, (label, cid, vkey) in enumerate(self.TEST_CONCEPTS):
            terminology = self.terminology1 if i < 6 else self.terminology2
            vectorizer = self.vectorizer1 if vkey == "v1" else self.vectorizer2
            concept, mapping = self._create_concept_and_mapping(terminology, label, cid, vectorizer)
            self.concepts_mappings.append((concept, mapping))

        # Store fresh objects in repo
        self.repository.store_all(
            [self.terminology1, self.terminology2]
            + [c for c, _ in self.concepts_mappings]
            + [m for _, m in self.concepts_mappings]
        )

    def _create_concept_and_mapping(self, terminology: Terminology, label: str, cid: str, vectorizer: Vectorizer):
        concept = Concept(terminology, label, cid)
        mapping = Mapping(
            concept,
            label,
            vectorizer.get_embedding(label),
            vectorizer.model_name,
        )
        return concept, mapping

    # Shared test logic (repository must be set in the subclass)
    def test_terminology_retrieval(self):
        terminology = self.repository.get_terminology("snomed CT")
        terminologies = self.repository.get_all_terminologies()
        names = [t.name for t in terminologies]
        self.assertEqual(terminology.name, "snomed CT")
        self.assertIn("NCI Thesaurus OBO Edition", names)
        self.assertIn("snomed CT", names)

    def test_concept_retrieval(self):
        concepts = self.repository.get_concepts()
        concept = self.repository.get_concept("Concept ID: 11893007")
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
        closest = self.repository.get_closest_mappings(embedding)
        self.assertEqual(len(closest), 5)
        self.assertEqual(closest[0].mapping.text, "Common cold")
        self.assertEqual(closest[0].mapping.sentence_embedder, self.model_name1)

    def test_terminology_and_model_specific_mappings(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        mappings = self.repository.get_closest_mappings(
            embedding, terminology_name="snomed CT", sentence_embedder=self.model_name1
        )
        self.assertEqual(len(mappings), 4)
        self.assertEqual(mappings[0].mapping.text, "Asthma")

    def test_closest_mappings_with_similarities(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        mappings = self.repository.get_closest_mappings(embedding)
        self.assertAlmostEqual(mappings[0].similarity, 0.6747197, 3)

    def test_closest_mapping_with_similarity_for_identical_entry(self):
        embedding = self.vectorizer1.get_embedding("Cancer")
        mappings = self.repository.get_closest_mappings(embedding)
        self.assertEqual(mappings[0].mapping.text, "Cancer")
        self.assertAlmostEqual(mappings[0].similarity, 1.0, 3)

    def test_terminology_and_model_specific_mappings_with_similarities(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        mappings = self.repository.get_closest_mappings(embedding, True, "snomed CT", self.model_name1)
        self.assertEqual(len(mappings), 4)
        self.assertEqual(mappings[0].mapping.text, "Asthma")
        self.assertAlmostEqual(mappings[0].similarity, 0.3947341, 3)

    def test_import_data_dictionary(self):
        path = os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv")
        source = DataDictionarySource(path, "VAR_1", "DESC")
        self.repository.import_data_dictionary(source, terminology_name="import_test")

        terminology = self.repository.get_terminology("import_test")
        self.assertEqual("import_test", terminology.name)

        mappings = self.repository.get_mappings("import_test").items
        texts = [m.text for m in mappings]
        df = source.to_dataframe()
        for row in df.itertuples(index=False):
            cid = f"import_test:{row.variable}"
            concept = self.repository.get_concept(cid)
            self.assertEqual(concept.pref_label, row.variable)
            self.assertIn(row.description, texts)

    def test_repository_restart(self):
        # Re-instantiate the repository class using optional repo_args
        repo_class = type(self.repository)
        args = getattr(self, "repo_args", ())
        repo = repo_class(*args)

        repo.store_all(
            [self.terminology1, self.terminology2]
            + [c for c, _ in self.concepts_mappings]
            + [m for _, m in self.concepts_mappings]
        )

        mappings = repo.get_mappings(limit=5).items
        self.assertEqual(len(mappings), 5)

        concepts = repo.get_concepts()
        self.assertEqual(len(concepts.items), 11)

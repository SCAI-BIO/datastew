import os
import unittest

from datastew.embedding import Vectorizer
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
                concept_id=concept.id, text=label, embedding=embedding, vectorizer=vectorizer.model_name
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
        self.assertEqual(concepts.total_count, 11)

    def test_mapping_retrieval(self):
        mappings = self.repository.get_mappings(limit=5).items
        self.assertEqual(len(mappings), 5)

    def test_get_mappings_empty_result(self):
        """Verify behavior when get_mappings finds nothing (e.g., bad filter)."""
        page = self.repository.get_mappings(terminology_name="NonExistent Terminology")
        self.assertEqual(page.total_count, 0)
        self.assertEqual(len(page.items), 0)

    def test_vectorizers(self):
        embedders = self.repository.get_all_vectorizers()
        self.assertIn(self.model_name1, embedders)
        self.assertIn(self.model_name2, embedders)

    def test_repository_restart(self):
        """Verify data persists across repository instantiations."""
        repo_class = type(self.repository)
        args = getattr(self, "repo_args", ())
        repo = repo_class(*args)

        mappings = repo.get_mappings(limit=5).items
        self.assertEqual(len(mappings), 5)

        concepts = repo.get_concepts()
        self.assertEqual(concepts.total_count, 11)

    def test_closest_mappings(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        assert embedding is not None
        closest = self.repository.get_closest_mappings(embedding)

        self.assertEqual(len(closest), 5)

        result = closest[0]
        assert isinstance(result, MappingResult)

        self.assertEqual(result.mapping.text, "Common cold")
        self.assertEqual(result.mapping.vectorizer, self.model_name1)

    def test_terminology_and_model_specific_mappings(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        closest = self.repository.get_closest_mappings(
            embedding, terminology_name="snomed CT", vectorizer=self.model_name1
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

    def test_edit_operations(self):
        """Test editing functionality for terminologies, concepts, and mappings."""
        # Terminology
        term = self.repository.get_terminology_by_name("snomed CT")
        updated_term = self.repository.edit_terminology(term.id, name="SNOMED Clinical Terms")
        self.assertEqual(updated_term.name, "SNOMED Clinical Terms")

        # Concept
        concept = self.repository.get_concept_by_identifier("Concept ID: 11893007")
        updated_concept = self.repository.edit_concept(concept.id, pref_label="Diabetes Type 2")
        self.assertEqual(updated_concept.pref_label, "Diabetes Type 2")

        # Mapping
        mappings = self.repository.get_mappings(limit=1).items
        mapping = mappings[0]
        updated_mapping = self.repository.edit_mapping(mapping.id, text="Updated text")
        self.assertEqual(updated_mapping.text, "Updated text")

    def test_delete_operations(self):
        """Test cascading deletes. Deleting a mapping shouldn't delete the concept, but deleting a concept deletes its mappings."""
        concept = self.repository.get_concept_by_identifier("Concept ID: 11893007")
        mappings = self.repository.get_mappings().items
        mapping_to_delete = [m for m in mappings if m.concept_id == concept.id][0]

        # Delete Mapping
        self.repository.delete_mapping(mapping_to_delete.id)
        with self.assertRaisesRegex(ValueError, "No Mapping found with ID"):
            self.repository.get_mapping(mapping_to_delete.id)

        # Concept should still exist
        self.repository.get_concept(concept.id)

        # Delete Concept
        self.repository.delete_concept(concept.id)
        with self.assertRaisesRegex(ValueError, "No Concept found with ID"):
            self.repository.get_concept(concept.id)

        # Delete Terminology
        term = self.repository.get_terminology_by_name("NCI Thesaurus OBO Edition")
        self.repository.delete_terminology(term.id)
        with self.assertRaisesRegex(ValueError, "No Terminology found with name"):
            self.repository.get_terminology_by_name("NCI Thesaurus OBO Edition")

    def test_missing_records(self):
        """Test ValueError triggers for non-existent records."""
        with self.assertRaisesRegex(ValueError, "No Terminology found with ID"):
            self.repository.get_terminology(99999)

        with self.assertRaisesRegex(ValueError, "No Concept found with identifier"):
            self.repository.get_concept_by_identifier("INVALID_ID")

        with self.assertRaisesRegex(ValueError, "No Mapping found with ID"):
            self.repository.get_mapping(99999)

    def test_add_mapping_missing_vectorizer(self):
        """Verify adding a mapping without an embedding raises an error if no default vectorizer is configured."""
        # Temporarily remove the default vectorizer
        original_vectorizer = self.repository.vectorizer
        self.repository.vectorizer = None  # type: ignore

        try:
            concept = self.repository.get_concept_by_identifier("Concept ID: 11893007")
            with self.assertRaisesRegex(ValueError, "Both embedding and vectorizer must be provided"):
                self.repository.add_mapping(concept_id=concept.id, text="Will Fail")
        finally:
            self.repository.vectorizer = original_vectorizer


if __name__ == "__main__":
    unittest.main()

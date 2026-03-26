import os
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from datastew.embedding import Vectorizer
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import MappingResult


class TestPostgreSQLRepository(unittest.TestCase):
    """Tests for the PostgreSQL repository backend using synchronous psycopg3."""

    # Ensure the driver is specified for psycopg3
    _URL = os.getenv("TEST_POSTGRES_URI", "postgresql://testuser:testpass@localhost/testdb")
    POSTGRES_TEST_URL = (
        _URL.replace("postgresql://", "postgresql+psycopg://", 1) if _URL.startswith("postgresql://") else _URL
    )

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
        """Setup for engine, schema initialization, and vectorizers."""
        cls.TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        cls.vectorizer1 = Vectorizer("sentence-transformers/all-mpnet-base-v2")
        cls.vectorizer2 = Vectorizer("FremyCompany/BioLORD-2023")
        cls.model_name1 = cls.vectorizer1.model_name
        cls.model_name2 = cls.vectorizer2.model_name
        cls.test_text = "The flu"

        cls.engine = create_engine(cls.POSTGRES_TEST_URL)
        PostgreSQLRepository.setup_database(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine, autoflush=False)

    @classmethod
    def tearDownClass(cls):
        """Dispose of the engine resources."""
        if hasattr(cls, "engine"):
            cls.engine.dispose()

    def setUp(self):
        """Create a fresh session and repository instance for every test."""
        self.session: Session = self.SessionLocal()
        self.repository = PostgreSQLRepository(session=self.session, vectorizer=self.vectorizer1)
        self._reset_repository()

    def tearDown(self):
        """Close the session after each test."""
        self.session.close()

    def _reset_repository(self):
        """Clear the repo and re-import base data."""
        self.repository.clear_all()
        self.session.commit()

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

        self.session.commit()

    def test_terminology_retrieval(self):
        terminology = self.repository.get_terminology_by_name("snomed CT")
        terminologies = self.repository.get_all_terminologies()
        names = [t.name for t in terminologies]

        self.assertEqual(terminology.name, "snomed CT")
        self.assertIn("NCI Thesaurus OBO Edition", names)
        self.assertIn("snomed CT", names)

    def test_concept_retrieval(self):
        concepts_page = self.repository.get_concepts()
        concept = self.repository.get_concept_by_identifier("Concept ID: 11893007")

        self.assertEqual(concept.concept_identifier, "Concept ID: 11893007")
        self.assertEqual(concept.pref_label, "Diabetes mellitus (disorder)")
        self.assertEqual(concept.terminology.name, "snomed CT")
        self.assertEqual(concepts_page.total_count, 11)

    def test_mapping_retrieval(self):
        mappings_page = self.repository.get_mappings(limit=5)
        self.assertEqual(len(mappings_page.items), 5)

    def test_get_mappings_empty_result(self):
        """Verify behavior when get_mappings finds nothing."""
        page = self.repository.get_mappings(terminology_name="NonExistent Terminology")
        self.assertEqual(page.total_count, 0)
        self.assertEqual(len(page.items), 0)

    def test_vectorizers(self):
        embedders = self.repository.get_all_vectorizers()
        self.assertIn(self.model_name1, embedders)
        self.assertIn(self.model_name2, embedders)

    def test_repository_restart(self):
        """Verify data persists across repository instantiations."""
        new_repo = PostgreSQLRepository(session=self.session, vectorizer=self.vectorizer1)

        mappings_page = new_repo.get_mappings(limit=5)
        self.assertEqual(len(mappings_page.items), 5)

        concepts_page = new_repo.get_concepts()
        self.assertEqual(concepts_page.total_count, 11)

    def test_closest_mappings(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        assert embedding is not None
        page = self.repository.get_closest_mappings(embedding, limit=5)

        self.assertEqual(len(page.items), 5)
        self.assertEqual(page.total_count, 11)  # Total mappings in DB

        result = page.items[0]
        assert isinstance(result, MappingResult)
        self.assertEqual(result.mapping.text, "Common cold")

    def test_closest_mappings_pagination(self):
        """Verify limit and offset work for similarity searches."""
        embedding = self.vectorizer1.get_embedding(self.test_text)
        assert embedding is not None

        page1 = self.repository.get_closest_mappings(embedding, limit=1, offset=0)
        page2 = self.repository.get_closest_mappings(embedding, limit=1, offset=1)

        self.assertEqual(len(page1.items), 1)
        self.assertEqual(len(page2.items), 1)

        res1 = page1.items[0]
        res2 = page2.items[0]

        assert isinstance(res1, MappingResult)
        assert isinstance(res2, MappingResult)

        self.assertNotEqual(res1.mapping.id, res2.mapping.id)

    def test_terminology_and_model_specific_mappings(self):
        embedding = self.vectorizer1.get_embedding(self.test_text)
        page = self.repository.get_closest_mappings(
            embedding, terminology_name="snomed CT", vectorizer=self.model_name1
        )
        # 6 snomed concepts, 4 use v1 (MPNet)
        self.assertEqual(len(page.items), 4)

        result = page.items[0]
        assert isinstance(result, MappingResult)
        self.assertEqual(result.mapping.text, "Asthma")

    def test_closest_mapping_with_similarity_for_identical_entry(self):
        embedding = self.vectorizer1.get_embedding("Cancer")
        assert embedding is not None
        page = self.repository.get_closest_mappings(embedding)

        result = page.items[0]
        assert isinstance(result, MappingResult)
        self.assertEqual(result.mapping.text, "Cancer")
        self.assertAlmostEqual(result.similarity, 1.0, places=3)

    def test_edit_operations(self):
        term = self.repository.get_terminology_by_name("snomed CT")
        self.repository.edit_terminology(term.id, name="SNOMED Clinical Terms")
        self.session.commit()
        self.assertEqual(self.repository.get_terminology(term.id).name, "SNOMED Clinical Terms")

        concept = self.repository.get_concept_by_identifier("Concept ID: 11893007")
        self.repository.edit_concept(concept.id, pref_label="Diabetes Type 2")
        self.session.commit()
        self.assertEqual(self.repository.get_concept(concept.id).pref_label, "Diabetes Type 2")

    def test_delete_operations(self):
        concept = self.repository.get_concept_by_identifier("Concept ID: 11893007")
        self.repository.delete_concept(concept.id)
        self.session.commit()

        with self.assertRaisesRegex(ValueError, "No Concept found with ID"):
            self.repository.get_concept(concept.id)

        term = self.repository.get_terminology_by_name("NCI Thesaurus OBO Edition")
        self.repository.delete_terminology(term.id)
        self.session.commit()

        with self.assertRaisesRegex(ValueError, "No Terminology found with name"):
            self.repository.get_terminology_by_name("NCI Thesaurus OBO Edition")

    def test_missing_records(self):
        with self.assertRaisesRegex(ValueError, "No Terminology found with ID"):
            self.repository.get_terminology(99999)

        with self.assertRaisesRegex(ValueError, "No Concept found with identifier"):
            self.repository.get_concept_by_identifier("INVALID_ID")


if __name__ == "__main__":
    unittest.main()

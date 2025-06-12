import os
import unittest

from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.repository.postgresql import PostgreSQLRepository
from tests.base_repository_setup import BaseRepositoryTestSetup


class TestPostgreSQLRepository(unittest.TestCase, BaseRepositoryTestSetup):

    POSTGRES_TEST_URL = os.getenv("TEST_POSTGRES_URI", "postgresql://testuser:testpass@localhost/testdb")

    @classmethod
    def setUpClass(cls) -> None:
        BaseRepositoryTestSetup.setUpClass()  # Explicitly initialize base setup
        cls.TEST_DIR_PATH = BaseRepositoryTestSetup.TEST_DIR_PATH
        cls.vectorizer1 = BaseRepositoryTestSetup.vectorizer1
        cls.vectorizer2 = BaseRepositoryTestSetup.vectorizer2
        cls.model_name1 = BaseRepositoryTestSetup.model_name1
        cls.model_name2 = BaseRepositoryTestSetup.model_name2
        cls.terminology1 = BaseRepositoryTestSetup.terminology1
        cls.terminology2 = BaseRepositoryTestSetup.terminology2
        cls.concepts_mappings = BaseRepositoryTestSetup.concepts_mappings
        cls.test_text = BaseRepositoryTestSetup.test_text
        cls.repository = PostgreSQLRepository(
            cls.POSTGRES_TEST_URL, Vectorizer("sentence-transformers/all-mpnet-base-v2")
        )
        cls.repository.store_all(
            [cls.terminology1, cls.terminology2]
            + [c for c, _ in cls.concepts_mappings]
            + [m for _, m in cls.concepts_mappings]
        )

    @classmethod
    def tearDownCase(cls):
        cls.repository.shut_down()

    def test_terminology_retrieval(self):
        """Test retrieval of individual and all terminologies."""
        terminology = self.repository.get_terminology("snomed CT")
        terminologies = self.repository.get_all_terminologies()
        terminology_names = [t.name for t in terminologies]
        self.assertEqual(terminology.name, "snomed CT")
        self.assertEqual(len(terminologies), 2)
        self.assertIn("NCI Thesaurus OBO Edition", terminology_names)
        self.assertIn("snomed CT", terminology_names)

    def test_concept_retrieval(self):
        """Test retrieval of individual and all concepts."""
        concepts = self.repository.get_concepts()
        concept = self.repository.get_concept("Concept ID: 11893007")
        self.assertEqual(concept.concept_identifier, "Concept ID: 11893007")
        self.assertEqual(concept.pref_label, "Diabetes mellitus (disorder)")
        self.assertEqual(concept.terminology.name, "snomed CT")
        self.assertEqual(len(concepts.items), 11)

    def test_mapping_retrieval(self):
        """Test storing and retrieving mappings from the repository."""
        mappings = self.repository.get_mappings(limit=5).items
        self.assertEqual(len(mappings), 5)

    def test_sentence_embedders(self):
        """Test retrieval of sentence embedders from the repository."""
        sentence_embedders = self.repository.get_all_sentence_embedders()
        self.assertEqual(len(sentence_embedders), 2)
        self.assertIn(self.model_name1, sentence_embedders)
        self.assertIn(self.model_name2, sentence_embedders)

    def test_closest_mappings(self):
        """Test retrieval of the closest mappings based on a test embedding."""
        test_embedding = self.vectorizer1.get_embedding(self.test_text)
        closest_mappings = self.repository.get_closest_mappings(test_embedding)
        self.assertEqual(len(closest_mappings), 5)
        self.assertEqual(closest_mappings[0].mapping.text, "Common cold")
        self.assertEqual(closest_mappings[0].mapping.sentence_embedder, self.model_name1)

    def test_terminology_and_model_specific_mappings(self):
        """Test retrieval of mappings filtered by terminology and model."""
        test_embedding = self.vectorizer1.get_embedding(self.test_text)
        specific_mappings = self.repository.get_closest_mappings(
            test_embedding, terminology_name="snomed CT", sentence_embedder=self.model_name1
        )
        self.assertEqual(len(specific_mappings), 4)
        self.assertEqual(specific_mappings[0].mapping.text, "Asthma")
        self.assertEqual(specific_mappings[0].mapping.concept.terminology.name, "snomed CT")
        self.assertEqual(specific_mappings[0].mapping.sentence_embedder, self.model_name1)

    def test_closest_mappings_with_similarities(self):
        """Test retrieval of closest mappings with similarity scores."""
        test_embedding = self.vectorizer1.get_embedding(self.test_text)
        closest_mappings_with_similarities = self.repository.get_closest_mappings(test_embedding)
        self.assertEqual(len(closest_mappings_with_similarities), 5)
        self.assertEqual(closest_mappings_with_similarities[0].mapping.text, "Common cold")
        self.assertEqual(
            closest_mappings_with_similarities[0].mapping.sentence_embedder,
            self.model_name1,
        )
        self.assertAlmostEqual(closest_mappings_with_similarities[0].similarity, 0.6747197, 3)

    def test_closest_mapping_with_similarity_for_indetical_entry(self):
        """Test retrieval of closest mapping for an identical entry"""
        test_embedding = self.vectorizer1.get_embedding("Cancer")
        closest_mappings_with_similarities = self.repository.get_closest_mappings(test_embedding)
        self.assertEqual(closest_mappings_with_similarities[0].mapping.text, "Cancer")
        self.assertAlmostEqual(closest_mappings_with_similarities[0].similarity, 1.0, 3)

    def test_terminology_and_model_specific_mappings_with_similarities(self):
        """Test retrieval of terminology and model-specific mappings with similarity scores."""
        test_embedding = self.vectorizer1.get_embedding(self.test_text)
        specific_mappings_with_similarities = self.repository.get_closest_mappings(
            test_embedding, terminology_name="snomed CT", sentence_embedder=self.model_name1
        )
        self.assertEqual(len(specific_mappings_with_similarities), 4)
        self.assertEqual(specific_mappings_with_similarities[0].mapping.text, "Asthma")
        self.assertEqual(
            specific_mappings_with_similarities[0].mapping.concept.terminology.name,
            "snomed CT",
        )
        self.assertEqual(
            specific_mappings_with_similarities[0].mapping.sentence_embedder,
            self.model_name1,
        )
        self.assertAlmostEqual(specific_mappings_with_similarities[0].similarity, 0.3947341, 3)

    def test_import_data_dictionary(self):
        """Test importing a data dictionary."""
        data_dictionary_source = DataDictionarySource(
            os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv"),
            "VAR_1",
            "DESC",
        )
        self.repository.import_data_dictionary(data_dictionary_source, terminology_name="import_test")
        terminology = self.repository.get_terminology("import_test")
        self.assertEqual("import_test", terminology.name)

        mappings = self.repository.get_mappings("import_test").items
        mapping_texts = [mapping.text for mapping in mappings]
        data_frame = data_dictionary_source.to_dataframe()
        for row in data_frame.index:
            variable = data_frame.loc[row, "variable"]
            description = data_frame.loc[row, "description"]
            concept = self.repository.get_concept(f"import_test:{variable}")
            self.assertEqual(concept.terminology.name, "import_test")
            self.assertEqual(concept.pref_label, variable)
            self.assertEqual(f"import_test:{variable}", concept.concept_identifier)
            self.assertIn(description, mapping_texts)
            for mapping in mappings:
                if mapping.text == description:
                    self.assertEqual(mapping.concept.concept_identifier, f"import_test:{variable}")
                    self.assertEqual(
                        mapping.sentence_embedder,
                        "sentence-transformers/all-mpnet-base-v2",
                    )

    def test_repository_restart(self):
        """Test the repository restart functionality to ensure no data is lost or corrupted."""
        # Re-initialize repository
        repository = PostgreSQLRepository(
            self.POSTGRES_TEST_URL, Vectorizer("sentence-transformers/all-mpnet-base-v2")
        )

        # Try storing the same data again (should not create duplicates)
        repository.store_all(
            [self.terminology1, self.terminology2]
            + [item[0] for item in self.concepts_mappings]
            + [item[1] for item in self.concepts_mappings]
        )

        # Check if mappings and concepts are intact
        mappings = repository.get_mappings(limit=5).items
        self.assertEqual(len(mappings), 5)

        concepts = repository.get_concepts()
        self.assertEqual(len(concepts.items), 22)

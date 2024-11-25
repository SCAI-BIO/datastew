import os
import unittest
from unittest import TestCase

from datastew import MPNetAdapter
from datastew.process.parsing import DataDictionarySource
from datastew.repository import Terminology, Concept, Mapping
from datastew.repository.weaviate import WeaviateRepository

class TestWeaviateRepository(TestCase):

    TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    @classmethod
    def setUpClass(cls):
        """Set up reusable components for the tests."""
        cls.repository = WeaviateRepository(mode="disk", path="db")
        cls.embedding_model1 = MPNetAdapter()
        cls.embedding_model2 = MPNetAdapter("FremyCompany/BioLORD-2023")
        cls.model_name1 = cls.embedding_model1.get_model_name()
        cls.model_name2 = cls.embedding_model2.get_model_name()

        # Terminologies
        cls.terminology1 = Terminology("snomed CT", "SNOMED")
        cls.terminology2 = Terminology("NCI Thesaurus OBO Edition", "NCIT")

        # Concepts and mappings
        cls.concepts_mappings = [
            cls._create_mapping(cls.terminology1, "Diabetes mellitus (disorder)", "Concept ID: 11893007", cls.embedding_model1),
            cls._create_mapping(cls.terminology1, "Hypertension (disorder)", "Concept ID: 73211009", cls.embedding_model2),
            cls._create_mapping(cls.terminology1, "Asthma", "Concept ID: 195967001", cls.embedding_model1),
            cls._create_mapping(cls.terminology1, "Heart attack", "Concept ID: 22298006", cls.embedding_model2),
            cls._create_mapping(cls.terminology2, "Common cold", "Concept ID: 13260007", cls.embedding_model1),
            cls._create_mapping(cls.terminology2, "Stroke", "Concept ID: 422504002", cls.embedding_model2),
            cls._create_mapping(cls.terminology2, "Migraine", "Concept ID: 386098009", cls.embedding_model1),
            cls._create_mapping(cls.terminology2, "Influenza", "Concept ID: 57386000", cls.embedding_model2),
            cls._create_mapping(cls.terminology2, "Osteoarthritis", "Concept ID: 399206004", cls.embedding_model1),
        ]

        cls.test_text = "The flu"

        # Store terminologies, concepts, and mappings in the repository
        cls.repository.store_all([cls.terminology1, cls.terminology2] + [item[0] for item in cls.concepts_mappings] + [item[1] for item in cls.concepts_mappings])

    @staticmethod
    def _create_mapping(terminology, text, concept_id, embedding_model):
        """Helper function to create a concept and mapping."""
        concept = Concept(terminology, text, concept_id)
        mapping = Mapping(concept, text, embedding_model.get_embedding(text), embedding_model.get_model_name())
        return concept, mapping

    def test_store_and_retrieve_mappings(self):
        """Test storing and retrieving mappings from the repository."""
        mappings = self.repository.get_mappings(limit=5)
        self.assertEqual(len(mappings), 5)

    def test_concept_retrieval(self):
        """Test retrieval of individual and all concepts."""
        concepts = self.repository.get_all_concepts()
        concept = self.repository.get_concept("Concept ID: 11893007")
        self.assertEqual(concept.concept_identifier, "Concept ID: 11893007")
        self.assertEqual(concept.pref_label, "Diabetes mellitus (disorder)")
        self.assertEqual(concept.terminology.name, "snomed CT")
        self.assertEqual(len(concepts), 9)

    def test_terminology_retrieval(self):
        """Test retrieval of individual and all terminologies."""
        terminology = self.repository.get_terminology("snomed CT")
        terminologies = self.repository.get_all_terminologies()
        terminology_names = [t.name for t in terminologies]
        self.assertEqual(terminology.name, "snomed CT")
        self.assertEqual(len(terminologies), 3)
        self.assertIn("NCI Thesaurus OBO Edition", terminology_names)
        self.assertIn("snomed CT", terminology_names)

    def test_sentence_embedders(self):
        """Test retrieval of sentence embedders from the repository."""
        sentence_embedders = self.repository.get_all_sentence_embedders()
        self.assertEqual(len(sentence_embedders), 2)
        self.assertIn(self.model_name1, sentence_embedders)
        self.assertIn(self.model_name2, sentence_embedders)

    def test_closest_mappings(self):
        """Test retrieval of the closest mappings based on a test embedding."""
        test_embedding = self.embedding_model1.get_embedding(self.test_text)
        closest_mappings = self.repository.get_closest_mappings(test_embedding)
        self.assertEqual(len(closest_mappings), 5)
        self.assertEqual(closest_mappings[0].text, "Common cold")
        self.assertEqual(closest_mappings[0].sentence_embedder, self.model_name1)

    def test_terminology_and_model_specific_mappings(self):
        """Test retrieval of mappings filtered by terminology and model."""
        test_embedding = self.embedding_model1.get_embedding(self.test_text)
        specific_mappings = self.repository.get_terminology_and_model_specific_closest_mappings(test_embedding, "snomed CT", self.model_name1)
        self.assertEqual(len(specific_mappings), 2)
        self.assertEqual(specific_mappings[0].text, "Asthma")
        self.assertEqual(specific_mappings[0].concept.terminology.name, "snomed CT")
        self.assertEqual(specific_mappings[0].sentence_embedder, self.model_name1)

    def test_closest_mappings_with_similarities(self):
        """Test retrieval of closest mappings with similarity scores."""
        test_embedding = self.embedding_model1.get_embedding(self.test_text)
        closest_mappings_with_similarities = self.repository.get_closest_mappings_with_similarities(test_embedding)
        self.assertEqual(len(closest_mappings_with_similarities), 5)
        self.assertEqual(closest_mappings_with_similarities[0][0].text, "Common cold")
        self.assertEqual(closest_mappings_with_similarities[0][0].sentence_embedder, self.model_name1)
        self.assertAlmostEqual(closest_mappings_with_similarities[0][1], 0.6747197, 3)

    def test_terminology_and_model_specific_mappings_with_similarities(self):
        """Test retrieval of terminology and model-specific mappings with similarity scores."""
        test_embedding = self.embedding_model1.get_embedding(self.test_text)
        specific_mappings_with_similarities = self.repository.get_terminology_and_model_specific_closest_mappings_with_similarities(test_embedding, "snomed CT", self.model_name1)
        self.assertEqual(len(specific_mappings_with_similarities), 2)
        self.assertEqual(specific_mappings_with_similarities[0][0].text, "Asthma")
        self.assertEqual(specific_mappings_with_similarities[0][0].concept.terminology.name, "snomed CT")
        self.assertEqual(specific_mappings_with_similarities[0][0].sentence_embedder, self.model_name1)
        self.assertAlmostEqual(specific_mappings_with_similarities[0][1], 0.3947341, 3)

    def test_import_data_dictionary(self):
        """Test importing a data dictionary."""
        data_dictionary_source = DataDictionarySource(os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv"), "VAR_1", "DESC")
        self.repository.import_data_dictionary(data_dictionary_source, terminology_name="import_test")
        terminology = self.repository.get_terminology("import_test")
        self.assertEqual("import_test", terminology.name)

        mappings = self.repository.get_mappings("import_test")
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
                    self.assertEqual(mapping.sentence_embedder, "sentence-transformers/all-mpnet-base-v2")

    def test_repository_restart(self):
        """Test the repository restart functionality to ensure no data is lost or corrupted."""
        # Re-initialize repository
        repository = WeaviateRepository(mode="disk", path="db")
        
        # Try storing the same data again (should not create duplicates)
        repository.store_all([self.terminology1, self.terminology2] + [item[0] for item in self.concepts_mappings] + [item[1] for item in self.concepts_mappings])
        
        # Check if mappings and concepts are intact
        mappings = repository.get_mappings(limit=5)
        self.assertEqual(len(mappings), 5)

        concepts = repository.get_all_concepts()
        self.assertEqual(len(concepts), 20)
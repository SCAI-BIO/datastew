import os
import unittest

from datastew.process.parsing import DataDictionarySource
from datastew.repository.model import Concept, Mapping, Terminology
from datastew.repository.sqllite import SQLLiteRepository


class TestGetClosestEmbedding(unittest.TestCase):

    TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    def setUp(self):
        self.repository = SQLLiteRepository(mode="memory")

    def tearDown(self):
        self.repository.shut_down()

    def test_get_terminology_specific_mappings(self):
        terminology1 = Terminology(name="Terminology 1", id="1")
        terminology2 = Terminology(name="Terminology 2", id="2")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        concept1 = Concept(terminology=terminology1, pref_label="Concept 1", concept_identifier="1")
        concept2 = Concept(terminology=terminology2, pref_label="Concept 2", concept_identifier="2")
        mapping_1 = Mapping(concept=concept1, text="Text 1", embedding=[0.1, 0.2, 0.3], sentence_embedder=model_name)
        mapping_2 = Mapping(concept=concept2, text="Text 2", embedding=[0.2, 0.3, 0.4], sentence_embedder=model_name)
        mapping_3 = Mapping(concept=concept1, text="Text 3", embedding=[1.2, 2.3, 3.4], sentence_embedder=model_name)
        self.repository.store_all([terminology1, terminology2, concept1, concept2, mapping_1, mapping_2, mapping_3])
        filtered_mappings1 = self.repository.get_mappings(terminology_name="Terminology 1")
        filtered_mappings2 = self.repository.get_mappings(terminology_name="Terminology 2")
        self.assertEqual(len(filtered_mappings1), 2)
        self.assertEqual(len(filtered_mappings2), 1)

    def test_get_closest_mappings(self):
        terminology = Terminology(name="Terminology 1", id="1")
        model_name = "sentence-transformers/all-mpnet-base-v2"
        concept = Concept(terminology=terminology, pref_label="Concept 1", concept_identifier="1")
        mapping_1 = Mapping(concept=concept, text="Text 1", embedding=[0.1, 0.2, 0.3], sentence_embedder=model_name)
        mapping_2 = Mapping(concept=concept, text="Text 2", embedding=[0.2, 0.3, 0.4], sentence_embedder=model_name)
        mapping_3 = Mapping(concept=concept, text="Text 3", embedding=[1.2, 2.3, 3.4], sentence_embedder=model_name)
        self.repository.store_all([terminology, concept, mapping_1, mapping_2, mapping_3])
        sample_embedding = [0.2, 0.4, 0.35]
        closest_mappings, distances = self.repository.get_closest_mappings(sample_embedding, limit=3)
        self.assertEqual(len(closest_mappings), 3)
        self.assertEqual(mapping_2.text, closest_mappings[0].text)

    def test_get_all_sentence_embedders(self):
        terminology = Terminology(name="Terminology 1", id="1")
        model_name_1 = "sentence-transformers/all-mpnet-base-v2"
        model_name_2 = "text-embedding-ada-002"
        concept = Concept(terminology=terminology, pref_label="Concept 1", concept_identifier="1")
        mapping_1 = Mapping(concept=concept, text="Text 1", embedding=[0.1, 0.2, 0.3], sentence_embedder=model_name_1)
        mapping_2 = Mapping(concept=concept, text="Text 1", embedding=[0.1, 0.2, 0.3], sentence_embedder=model_name_2)
        self.repository.store_all([terminology, concept, mapping_1, mapping_2])
        sentence_embedders = self.repository.get_all_sentence_embedders()
        self.assertEqual(len(sentence_embedders), 2)
        self.assertEqual(sentence_embedders[0], "sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(sentence_embedders[1], "text-embedding-ada-002")

    def test_import_data_dictionary(self):
        data_dictionary_source = DataDictionarySource(os.path.join(self.TEST_DIR_PATH, "resources", "test_data_dict.csv"), "VAR_1", "DESC")
        self.repository.import_data_dictionary(data_dictionary_source, terminology_name="import_test")
        terminologies = [terminology.name for terminology in self.repository.get_all_terminologies()]
        concept_identifiers = [concept.concept_identifier for concept in self.repository.get_all_concepts()]
        self.assertIn("import_test", terminologies)

        data_frame = data_dictionary_source.to_dataframe()
        for row in data_frame.index:
            variable = data_frame.loc[row, "variable"]
            description = data_frame.loc[row, "description"]
            self.assertIn(f"import_test:{variable}", concept_identifiers)
            for mapping in self.repository.get_mappings("import_test"):
                if mapping.text == description:
                    self.assertEqual(mapping.concept_identifier, f"import_test:{variable}")
                    self.assertEqual(mapping.sentence_embedder, "sentence-transformers/all-mpnet-base-v2")

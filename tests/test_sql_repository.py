import unittest

from datastew.repository.model import Terminology, Concept, Mapping
from datastew.repository.sqllite import SQLLiteRepository


class TestGetClosestEmbedding(unittest.TestCase):

    def setUp(self):
        self.repository = SQLLiteRepository(mode="memory")

    def tearDown(self):
        self.repository.shut_down()

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

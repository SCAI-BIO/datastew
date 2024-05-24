import unittest

from datastew.repository.model import Terminology, Concept, Mapping
from datastew.embedding import MPNetAdapter
from datastew.repository.sqllite import SQLLiteRepository


class TestGetClosestEmbedding(unittest.TestCase):

    def setUp(self):
        self.repository = SQLLiteRepository(mode="memory")
        self.embedding_model = MPNetAdapter()

    def tearDown(self):
        self.repository.shut_down()

    def test_mapping_storage_and_closest_retrieval(self):
        # preset knowledge
        terminology = Terminology("test", "test")
        concept1 = Concept(terminology, "cat", "TEST:1")
        concept1_description = "The cat is sitting on the mat."
        mapping1 = Mapping(concept1, concept1_description, self.embedding_model.get_embedding(concept1_description))
        concept2 = Concept(terminology, "sunrise", "TEST:2")
        concept2_description = "The sun rises in the east."
        mapping2 = Mapping(concept2, concept2_description, self.embedding_model.get_embedding(concept2_description))
        concept3 = Concept(terminology, "dog", "TEST:3")
        concept3_description = "A loyal companion to humans."
        mapping3 = Mapping(concept3, concept3_description, self.embedding_model.get_embedding(concept3_description))
        self.repository.store_all([terminology, concept1, mapping1, concept2, mapping2, concept3, mapping3])
        # test new mappings
        text1 = "A furry feline rests on the rug."
        text1_embedding = self.embedding_model.get_embedding(text1)
        text2 = "Dawn breaks over the horizon."
        text2_embedding = self.embedding_model.get_embedding(text2)
        text3 = "A faithful friend."
        text3_embedding = self.embedding_model.get_embedding(text3)
        mappings1, similarities1 = self.repository.get_closest_mappings(text1_embedding, limit=3)
        mappings2, similarities2 = self.repository.get_closest_mappings(text2_embedding, limit=3)
        mappings3, similarities3 = self.repository.get_closest_mappings(text3_embedding, limit=3)
        self.assertEqual(len(mappings1), 3)
        self.assertEqual(len(mappings2), 3)
        self.assertEqual(len(mappings3), 3)
        self.assertEqual(concept1_description, mappings1[0].text)
        self.assertEqual(concept2_description, mappings2[0].text)
        self.assertEqual(concept3_description, mappings3[0].text)


import unittest

from datastew.embedding import Vectorizer
from datastew.repository.model import Concept, Mapping, Terminology
from datastew.repository.sqllite import SQLLiteRepository


class TestGetClosestEmbedding(unittest.TestCase):

    def setUp(self):
        self.repository = SQLLiteRepository(mode="memory")
        self.vectorizer = Vectorizer()

    def tearDown(self):
        self.repository.shut_down()

    def test_mapping_storage_and_closest_retrieval(self):
        # preset knowledge
        terminology = Terminology("test", "test")
        concept1 = Concept(terminology, "cat", "TEST:1")
        concept1_description = "The cat is sitting on the mat."
        sentence_embedder = "test"
        mapping1 = Mapping(
            concept1,
            concept1_description,
            list(self.vectorizer.get_embedding(concept1_description)),
            sentence_embedder=sentence_embedder,
        )
        concept2 = Concept(terminology, "sunrise", "TEST:2")
        concept2_description = "The sun rises in the east."
        mapping2 = Mapping(
            concept2,
            concept2_description,
            list(self.vectorizer.get_embedding(concept2_description)),
            sentence_embedder=sentence_embedder,
        )
        concept3 = Concept(terminology, "dog", "TEST:3")
        concept3_description = "A loyal companion to humans."
        mapping3 = Mapping(
            concept3,
            concept3_description,
            list(self.vectorizer.get_embedding(concept3_description)),
            sentence_embedder=sentence_embedder,
        )
        self.repository.store_all([terminology, concept1, mapping1, concept2, mapping2, concept3, mapping3])
        # test new mappings
        text1 = "A furry feline rests on the rug."
        text1_embedding = self.vectorizer.get_embedding(text1)
        text2 = "Dawn breaks over the horizon."
        text2_embedding = self.vectorizer.get_embedding(text2)
        text3 = "A faithful friend."
        text3_embedding = self.vectorizer.get_embedding(text3)
        mappings1 = self.repository.get_closest_mappings(list(text1_embedding), limit=3)
        mappings2 = self.repository.get_closest_mappings(list(text2_embedding), limit=3)
        mappings3 = self.repository.get_closest_mappings(list(text3_embedding), limit=3)
        self.assertEqual(len(mappings1), 3)
        self.assertEqual(len(mappings2), 3)
        self.assertEqual(len(mappings3), 3)
        self.assertEqual(concept1_description, mappings1[0].mapping.text)
        self.assertEqual(concept2_description, mappings2[0].mapping.text)
        self.assertEqual(concept3_description, mappings3[0].mapping.text)

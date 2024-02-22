import unittest

from index.db.model import Terminology, Concept, Mapping
from index.embedding import MPNetAdapter
from index.repository.sqllite import SQLLiteRepository


class TestGetClosestEmbedding(unittest.TestCase):

    def setUp(self):
        self.repository = SQLLiteRepository(mode="memory")
        self.embedding_model = MPNetAdapter()

    def tearDown(self):
        self.repository.shut_down()

    def test_mapping_storage_and_closest_retrieval(self):
        # preset knowledge
        terminology = Terminology("test", "test")
        concept1 = Concept(terminology, "depression", "TEST:1")
        concept1_description = "A heavy fog obscures joy, suffocating hope in an endless struggle."
        mapping1 = Mapping(concept1, concept1_description, self.embedding_model.get_embedding(concept1_description))
        concept2 = Concept(terminology, "euphoria", "TEST:2")
        concept2_description = "An intense state of joy and elation, engulfing the senses."
        mapping2 = Mapping(concept2, concept2_description, self.embedding_model.get_embedding(concept2_description))
        self.repository.store_all([terminology, concept1, mapping1, concept2, mapping2])
        # test new mappings
        text1 = "Trapped in glass, distant from life's vibrancy, feeling isolated and disconnected."
        text1_embedding = self.embedding_model.get_embedding(text1)
        text2 = "A profound feeling of happiness, exuberance, and boundless positivity."
        text2_embedding = self.embedding_model.get_embedding(text2)
        mappings1, distances1 = self.repository.get_closest_mappings(text1_embedding, limit=2)
        mappings2, distances2 = self.repository.get_closest_mappings(text2_embedding, limit=2)
        self.assertEqual(len(mappings1), 2)
        self.assertEqual(len(mappings1), 2)
        self.assertEqual(concept1_description, mappings1[0].text)
        self.assertEqual(concept2_description, mappings2[0].text)



import unittest

from index.db.model import Terminology, Concept, Mapping
from index.repository.sqllite import SQLLiteRepository


class TestGetClosestEmbedding(unittest.TestCase):

    def setUp(self):
        self.repository = SQLLiteRepository(mode="memory")

    def tearDown(self):
        self.repository.shut_down()

    def test_get_closest_mappings(self):
        terminology = Terminology(name="Terminology 1", id="1")
        concept = Concept(terminology=terminology, name="Concept 1", id="1")
        mapping_1 = Mapping(concept=concept, text="Text 1", embedding=[0.1, 0.2, 0.3])
        mapping_2 = Mapping(concept=concept, text="Text 2", embedding=[0.2, 0.3, 0.4])
        mapping_3 = Mapping(concept=concept, text="Text 3", embedding=[1.2, 2.3, 3.4])
        self.repository.store_all([terminology, concept, mapping_1, mapping_2, mapping_3])
        sample_embedding = [0.15, 0.25, 0.35]
        closest_mappings, distances = self.repository.get_closest_mappings(sample_embedding, limit=3)
        self.assertEqual(len(closest_mappings), 3)
        self.assertEqual(mapping_1, closest_mappings[0])
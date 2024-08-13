import unittest
from unittest import TestCase

from datastew import MPNetAdapter
from datastew.repository import Terminology, Concept, Mapping, SentenceEmbedder
from datastew.repository.weaviate import WeaviateRepository


class Test(TestCase):
    @unittest.skip("currently broken on github workflows")
    def test_repository(self):

        repository = WeaviateRepository(mode="disk", path="db")

        embedding_model = MPNetAdapter()

        terminology = Terminology("snomed CT", "SNOMED")
        sentence_embedder = SentenceEmbedder("sentence-transformers/all-mpnet-base-v2")

        text1 = "Diabetes mellitus (disorder)"
        concept1 = Concept(terminology, text1, "Concept ID: 11893007")
        mapping1 = Mapping(concept1, text1, embedding_model.get_embedding(text1), sentence_embedder)

        text2 = "Hypertension (disorder)"
        concept2 = Concept(terminology, text2, "Concept ID: 73211009")
        mapping2 = Mapping(concept2, text2, embedding_model.get_embedding(text2), sentence_embedder)

        text3 = "Asthma"
        concept3 = Concept(terminology, text3, "Concept ID: 195967001")
        mapping3 = Mapping(concept3, text3, embedding_model.get_embedding(text3), sentence_embedder)

        text4 = "Heart attack"
        concept4 = Concept(terminology, text4, "Concept ID: 22298006")
        mapping4 = Mapping(concept4, text4, embedding_model.get_embedding(text4), sentence_embedder)

        text5 = "Common cold"
        concept5 = Concept(terminology, text5, "Concept ID: 13260007")
        mapping5 = Mapping(concept5, text5, embedding_model.get_embedding(text5), sentence_embedder)

        text6 = "Stroke"
        concept6 = Concept(terminology, text6, "Concept ID: 422504002")
        mapping6 = Mapping(concept6, text6, embedding_model.get_embedding(text6), sentence_embedder)

        text7 = "Migraine"
        concept7 = Concept(terminology, text7, "Concept ID: 386098009")
        mapping7 = Mapping(concept7, text7, embedding_model.get_embedding(text7), sentence_embedder)

        text8 = "Influenza"
        concept8 = Concept(terminology, text8, "Concept ID: 57386000")
        mapping8 = Mapping(concept8, text8, embedding_model.get_embedding(text8), sentence_embedder)

        text9 = "Osteoarthritis"
        concept9 = Concept(terminology, text9, "Concept ID: 399206004")
        mapping9 = Mapping(concept9, text9, embedding_model.get_embedding(text9), sentence_embedder)

        text10 = "The flu"

        repository.store_all([
            terminology, concept1, mapping1, concept2, mapping2, concept3, mapping3,
            concept4, mapping4, concept5, mapping5, concept6, mapping6, concept7, mapping7,
            concept8, mapping8, concept9, mapping9
        ])

        mappings = repository.get_all_mappings(limit=5)
        self.assertEqual(len(mappings), 5)

        concepts = repository.get_all_concepts()
        self.assertEqual(len(concepts), 9)

        terminologies = repository.get_all_terminologies()
        self.assertEqual(len(terminologies), 1)

        sentence_embedders = repository.get_all_sentence_embedders()
        self.assertEqual(len(sentence_embedders), 1)
        self.assertEqual(sentence_embedders[0].name, "sentence-transformers/all-mpnet-base-v2")

        test_embedding = embedding_model.get_embedding(text10)

        closest_mappings = repository.get_closest_mappings(test_embedding)
        self.assertEqual(len(closest_mappings), 5)
        self.assertEqual(closest_mappings[0].text, "Influenza")

        closest_mappings_with_similarities = repository.get_closest_mappings_with_similarities(test_embedding)
        self.assertEqual(len(closest_mappings_with_similarities), 5)
        self.assertEqual(closest_mappings_with_similarities[0][0].text, "Influenza")
        self.assertEqual(closest_mappings_with_similarities[0][1], 0.86187172)

        # check if it crashed (due to schema re-creation) after restart
        repository = WeaviateRepository(mode="disk", path="db")

        # try to store all again (should not create new entries since they already exist)
        repository.store_all([
            terminology, concept1, mapping1, concept2, mapping2, concept3, mapping3,
            concept4, mapping4, concept5, mapping5, concept6, mapping6, concept7, mapping7,
            concept8, mapping8, concept9, mapping9
        ])


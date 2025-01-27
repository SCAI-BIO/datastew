import unittest
from time import time

import numpy as np

from datastew.embedding import MPNetAdapter, TextEmbedding


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.mpnet_adapter = MPNetAdapter(model_name="sentence-transformers/all-mpnet-base-v2")

    def test_mpnet_adapter_get_embedding(self):
        text = "This is a test sentence."
        embedding = self.mpnet_adapter.get_embedding(text)
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(len(embedding), 768)

    def test_mpnet_adapter_get_embeddings(self):
        messages = ["This is message 1.", "This is message 2."]
        embeddings = self.mpnet_adapter.get_embeddings(messages)
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), len(messages))
        self.assertEqual(len(embeddings[0]), 768)

    def test_text_embedding(self):
        text = "This is a test sentence."
        embedding = [0.1, 0.2, 0.3, 0.4]
        text_embedding = TextEmbedding(text, embedding)
        self.assertEqual(text_embedding.text, text)
        self.assertEqual(text_embedding.embedding, embedding)

    def test_sanitization(self):
        text1 = " Test"
        text2 = "test "
        embedding1 = self.mpnet_adapter.get_embedding(text1)
        embedding2 = self.mpnet_adapter.get_embedding(text2)
        self.assertListEqual(embedding1, embedding2)
    
    def test_caching_get_embedding(self):
        text = "This is a test sentence."
        self.mpnet_adapter._cache.clear()

        # Measure time for the first call
        start_time = time()
        embedding1 = self.mpnet_adapter.get_embedding(text)
        first_call_time = time() - start_time

        # Measure time for the second call
        start_time = time()
        embedding2 = self.mpnet_adapter.get_embedding(text)
        second_call_time = time() - start_time

        self.assertLess(second_call_time, first_call_time)
        self.assertListEqual(embedding1, embedding2)

    def test_caching_get_embeddings(self):
        messages = ["This is message 1.", "This is message 2."]
        self.mpnet_adapter._cache.clear()

        start_time = time()
        embeddings1 = self.mpnet_adapter.get_embeddings(messages)
        first_call_time = time() - start_time

        start_time = time()
        embeddings2 = self.mpnet_adapter.get_embeddings(messages)
        second_call_time = time() - start_time

        self.assertLess(second_call_time, first_call_time)

        for emb1, emb2 in zip(embeddings1, embeddings2):
            self.assertListEqual(emb1, emb2)





import unittest
from time import time
from typing import Sequence

from datastew.embedding import MPNetAdapter, TextEmbedding


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.mpnet_adapter = MPNetAdapter(
            model_name="sentence-transformers/all-mpnet-base-v2", cache=True
        )

    def test_mpnet_adapter_get_embedding(self):
        text = "This is a test sentence."
        embedding = self.mpnet_adapter.get_embedding(text)
        self.assertIsInstance(embedding, Sequence)
        self.assertEqual(len(embedding), 768)

    def test_mpnet_adapter_get_embeddings(self):
        messages = ["This is message 1.", "This is message 2."]
        embeddings = self.mpnet_adapter.get_embeddings(messages)
        self.assertIsInstance(embeddings, Sequence)
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
        self.assertSequenceEqual(embedding1, embedding2)

    def test_caching_get_embedding(self):
        text = "This is a test sentence."
        if self.mpnet_adapter._cache:
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
        self.assertSequenceEqual(embedding1, embedding2)

    def test_caching_get_embeddings(self):
        messages = ["This is message 1.", "This is message 2."]
        if self.mpnet_adapter._cache:
            self.mpnet_adapter._cache.clear()

        start_time = time()
        embeddings1 = self.mpnet_adapter.get_embeddings(messages)
        first_call_time = time() - start_time

        start_time = time()
        embeddings2 = self.mpnet_adapter.get_embeddings(messages)
        second_call_time = time() - start_time

        self.assertLess(second_call_time, first_call_time)

        for emb1, emb2 in zip(embeddings1, embeddings2):
            self.assertSequenceEqual(emb1, emb2)

    def test_cache_vs_no_cache_performance(self):
        messages = ["This is message 1.", "This is message 2."]

        adapter_with_cache = MPNetAdapter(cache=True)
        if adapter_with_cache._cache:
            adapter_with_cache._cache.clear()

        start_time = time()
        adapter_with_cache.get_embeddings(messages)
        first_call_time_with_cache = time() - start_time

        adapter_without_cache = MPNetAdapter()

        start_time = time()
        adapter_without_cache.get_embeddings(messages)
        first_call_time_without_cache = time() - start_time

        self.assertLess(first_call_time_without_cache, first_call_time_with_cache)

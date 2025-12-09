import unittest
from time import time
from typing import Sequence

from datastew.embedding import HuggingFaceAdapter


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.hugging_face_adapter = HuggingFaceAdapter(cache=True)

    def test_hugging_face_adapter_get_embedding(self):
        text = "This is a test sentence."
        embedding = self.hugging_face_adapter.get_embedding(text)
        self.assertIsInstance(embedding, Sequence)

    def test_hugging_face_adapter_get_embeddings(self):
        messages = [f"This is message {i}." for i in range(20)]
        embeddings = self.hugging_face_adapter.get_embeddings(messages)
        self.assertIsInstance(embeddings, Sequence)
        self.assertEqual(len(embeddings), len(messages))

    def test_sanitization(self):
        text1 = " Test"
        text2 = "test "
        embedding1 = self.hugging_face_adapter.get_embedding(text1)
        embedding2 = self.hugging_face_adapter.get_embedding(text2)
        self.assertSequenceEqual(embedding1, embedding2)

    def test_caching_get_embedding(self):
        text = "This is a test sentence."
        if self.hugging_face_adapter._cache:
            self.hugging_face_adapter._cache.clear()

        # Measure time for the first call
        start_time = time()
        embedding1 = self.hugging_face_adapter.get_embedding(text)
        first_call_time = time() - start_time

        # Measure time for the second call
        start_time = time()
        embedding2 = self.hugging_face_adapter.get_embedding(text)
        second_call_time = time() - start_time

        self.assertLess(second_call_time, first_call_time)
        self.assertSequenceEqual(embedding1, embedding2)

    def test_caching_get_embeddings(self):
        messages = [f"This is message {i}." for i in range(20)]
        if self.hugging_face_adapter._cache:
            self.hugging_face_adapter._cache.clear()

        start_time = time()
        embeddings1 = self.hugging_face_adapter.get_embeddings(messages)
        first_call_time = time() - start_time

        start_time = time()
        embeddings2 = self.hugging_face_adapter.get_embeddings(messages)
        second_call_time = time() - start_time

        self.assertLess(second_call_time, first_call_time)

        for emb1, emb2 in zip(embeddings1, embeddings2):
            self.assertSequenceEqual(emb1, emb2)

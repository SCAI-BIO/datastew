import unittest
from typing import Sequence
from unittest.mock import patch

from datastew.embedding.base import _GLOBAL_CACHES, _GLOBAL_LOCKS
from datastew.embedding.hugging_face import HuggingFaceAdapter


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        _GLOBAL_CACHES.clear()
        _GLOBAL_LOCKS.clear()
        self.adapter = HuggingFaceAdapter(cache=True)

    def test_hugging_face_adapter_get_embedding(self):
        text = "This is a test sentence."
        embedding = self.adapter.get_embedding(text)
        self.assertIsInstance(embedding, Sequence)

    def test_hugging_face_adapter_get_embeddings(self):
        messages = [f"This is message {i}." for i in range(20)]
        embeddings = self.adapter.get_embeddings(messages)
        self.assertIsInstance(embeddings, Sequence)
        self.assertEqual(len(embeddings), len(messages))

    def test_sanitization(self):
        text1 = " Test"
        text2 = "test "
        embedding1 = self.adapter.get_embedding(text1)
        embedding2 = self.adapter.get_embedding(text2)
        self.assertSequenceEqual(embedding1, embedding2)

    def test_caching_get_embedding_deterministic(self):
        text = "deterministic test sentence"

        with patch.object(self.adapter.model, "encode", wraps=self.adapter.model.encode) as spy_encode:
            emb1 = self.adapter.get_embedding(text)
            spy_encode.assert_called_once()

            emb2 = self.adapter.get_embedding(text)
            self.assertEqual(spy_encode.call_count, 1)
            self.assertSequenceEqual(emb1, emb2)

    def test_caching_get_embeddings_batch(self):
        messages = [f"Batch message {i}." for i in range(5)]

        with patch.object(self.adapter.model, "encode", wraps=self.adapter.model.encode) as spy_encode:
            embeddings1 = self.adapter.get_embeddings(messages)
            self.assertEqual(spy_encode.call_count, 1)
            embeddings2 = self.adapter.get_embeddings(messages)
            self.assertEqual(spy_encode.call_count, 1)

            for emb1, emb2 in zip(embeddings1, embeddings2):
                self.assertSequenceEqual(emb1, emb2)

    def test_partial_cache_hit_batch(self):
        messages = ["A", "B", "C"]
        # Cache "A" manually
        self.adapter._add_batch_to_cache([self.adapter._sanitize("A")], [[0.1, 0.2]])

        with patch.object(self.adapter.model, "encode", wraps=self.adapter.model.encode) as spy_encode:
            embeddings = self.adapter.get_embeddings(messages)
            # encode should only be called once, and only for ["b", "c"]
            spy_encode.assert_called_once_with(["b", "c"], show_progress_bar=True)
            self.assertEqual(len(embeddings), 3)
            self.assertEqual(embeddings[0], [0.1, 0.2])

    def test_exception_handling_aborts_silence(self):
        messages = ["Fail 1", "Fail 2"]

        with patch.object(self.adapter.model, "encode", side_effect=Exception("API Down")):
            with self.assertRaises(Exception) as context:
                self.adapter.get_embeddings(messages)

            self.assertTrue("API Down" in str(context.exception))

    @patch("datastew.embedding.hugging_face.SentenceTransformer")
    def test_huggingface_model_loaded_once(self, mock_transformer):
        """Verify LRUCache singleton behavior and maxsize enforcement"""
        HuggingFaceAdapter._model_cache.clear()

        adapter1 = HuggingFaceAdapter(model_name="sentence-transformers/all-MiniLM-L6-v2")
        adapter2 = HuggingFaceAdapter(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.assertIs(adapter1.model, adapter2.model)
        self.assertEqual(len(HuggingFaceAdapter._model_cache), 1)
        self.assertEqual(mock_transformer.call_count, 1)
        HuggingFaceAdapter(model_name="model_A")
        HuggingFaceAdapter(model_name="model_B")
        HuggingFaceAdapter(model_name="model_C")
        self.assertEqual(len(HuggingFaceAdapter._model_cache), 3)
        self.assertNotIn("sentence-transformers/all-MiniLM-L6-v2", HuggingFaceAdapter._model_cache)
        self.assertEqual(mock_transformer.call_count, 4)


if __name__ == "__main__":
    unittest.main()

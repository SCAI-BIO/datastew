import unittest
from typing import Sequence
from unittest.mock import patch

from datastew.embedding.base import _GLOBAL_CACHES, _GLOBAL_LOCKS, EmbeddingModel


class DummyAdapter(EmbeddingModel):
    """A minimal implementation of EmbeddingModel to strictly test base class logic."""

    def _generate_embedding(self, text: str) -> Sequence[float]:
        """Generate a dummy embedding vector based on text length."""
        val = float(len(text))
        return [val, val]

    def _generate_embeddings(self, messages: list[str]) -> Sequence[Sequence[float]]:
        """Generate a batch of dummy embedding vectors based on text lengths."""
        return [[float(len(msg)), float(len(msg))] for msg in messages]


class TestEmbeddingBase(unittest.TestCase):
    def setUp(self):
        """Reset global caches and initialize a fresh DummyAdapter before each test."""
        _GLOBAL_CACHES.clear()
        _GLOBAL_LOCKS.clear()
        self.adapter = DummyAdapter(model_name="dummy-model", cache=True)

    def test_sanitization(self):
        """Verify that text is correctly trimmed, lowercased, and whitespace-normalized."""
        text1 = " Test"
        text2 = "test "
        embedding1 = self.adapter.get_embedding(text1)
        embedding2 = self.adapter.get_embedding(text2)
        self.assertSequenceEqual(embedding1, embedding2)
        self.assertEqual(self.adapter._sanitize("  Multiple   Spaces  "), "multiple spaces")

    def test_caching_get_embedding_deterministic(self):
        """Ensure single embeddings are retrieved from cache on subsequent calls."""
        text = "deterministic test sentence"

        with patch.object(self.adapter, "_generate_embedding", wraps=self.adapter._generate_embedding) as spy:
            emb1 = self.adapter.get_embedding(text)
            spy.assert_called_once()
            emb2 = self.adapter.get_embedding(text)
            self.assertEqual(spy.call_count, 1)
            self.assertSequenceEqual(emb1, emb2)

    def test_caching_get_embeddings_batch(self):
        """Ensure batch embeddings are retrieved from cache on subsequent calls."""
        messages = [f"Batch message {i}." for i in range(5)]

        with patch.object(self.adapter, "_generate_embeddings", wraps=self.adapter._generate_embeddings) as spy:
            embeddings1 = self.adapter.get_embeddings(messages)
            self.assertEqual(spy.call_count, 1)

            embeddings2 = self.adapter.get_embeddings(messages)
            self.assertEqual(spy.call_count, 1)

            for emb1, emb2 in zip(embeddings1, embeddings2):
                self.assertSequenceEqual(emb1, emb2)

    def test_partial_cache_hit_batch(self):
        """Verify that batch processing only generates embeddings for uncached items."""
        messages = ["A", "B", "C"]
        self.adapter._add_batch_to_cache(["a"], [[0.1, 0.2]])

        with patch.object(self.adapter, "_generate_embeddings", return_value=[[0.3, 0.4], [0.5, 0.6]]) as spy:
            embeddings = self.adapter.get_embeddings(messages)
            spy.assert_called_once_with(["b", "c"])
            self.assertEqual(len(embeddings), 3)
            self.assertEqual(embeddings[0], [0.1, 0.2])
            self.assertEqual(embeddings[1], [0.3, 0.4])


if __name__ == "__main__":
    unittest.main()

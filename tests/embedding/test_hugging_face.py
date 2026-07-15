import unittest
from unittest.mock import patch

import numpy as np

from datastew.embedding.hugging_face import HuggingFaceAdapter


class TestHuggingFaceAdapter(unittest.TestCase):

    @patch("datastew.embedding.hugging_face.SentenceTransformer")
    def setUp(self, mock_transformer):
        """Reset the class-level model cache and initialize a mocked adapter."""
        HuggingFaceAdapter._model_cache.clear()
        self.mock_model_instance = mock_transformer.return_value
        self.adapter = HuggingFaceAdapter(cache=False)

    def test_generate_embedding(self):
        """Verify single embedding generation and NumPy to list conversion."""
        self.mock_model_instance.encode.return_value = np.array([0.1, 0.2])
        result = self.adapter._generate_embedding("test")
        self.assertEqual(result, [0.1, 0.2])
        self.mock_model_instance.encode.assert_called_once_with("test")

    def test_generate_embeddings(self):
        """Verify batch embedding generation and proper conversion of nested NumPy arrays."""
        messages = ["msg1", "msg2"]
        self.mock_model_instance.encode.return_value = np.array([[0.1], [0.2]])
        result = self.adapter._generate_embeddings(messages)
        self.assertEqual(result, [[0.1], [0.2]])
        self.mock_model_instance.encode.assert_called_once_with(messages, show_progress_bar=True)

    def test_exception_handling_aborts_silence(self):
        """Ensure that exceptions from the underlying model are logged and reraised."""
        messages = ["Fail 1", "Fail 2"]
        self.mock_model_instance.encode.side_effect = Exception("Model Down")
        with self.assertRaisesRegex(Exception, "Model Down"):
            self.adapter._generate_embeddings(messages)

    @patch("datastew.embedding.hugging_face.SentenceTransformer")
    def test_huggingface_model_loaded_once(self, mock_transformer):
        """Verify LRUCache singleton behavior and maxsize enforcement."""
        HuggingFaceAdapter._model_cache.clear()
        adapter1 = HuggingFaceAdapter(model_name="sentence-transformers/all-MiniLM-L6-v2")
        adapter2 = HuggingFaceAdapter(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.assertIs(adapter1.model, adapter2.model)
        self.assertEqual(len(HuggingFaceAdapter._model_cache), 1)
        self.assertEqual(mock_transformer.call_count, 1)

        # Force eviction
        HuggingFaceAdapter(model_name="model_A")
        HuggingFaceAdapter(model_name="model_B")
        HuggingFaceAdapter(model_name="model_C")
        self.assertEqual(len(HuggingFaceAdapter._model_cache), 3)
        self.assertNotIn("sentence-transformers/all-MiniLM-L6-v2", HuggingFaceAdapter._model_cache)
        self.assertEqual(mock_transformer.call_count, 4)


if __name__ == "__main__":
    unittest.main()

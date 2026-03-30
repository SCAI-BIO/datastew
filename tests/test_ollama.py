import unittest
from unittest.mock import patch

from datastew.embedding.ollama import OllamaAdapter


class TestOllamaAdapter(unittest.TestCase):
    @patch("datastew.embedding.ollama.Client")
    def setUp(self, mock_client):
        self.mock_client_instance = mock_client.return_value
        self.adapter = OllamaAdapter(cache=False)

    def test_generate_embedding(self):
        self.mock_client_instance.embed.return_value = {"embeddings": [[0.3, 0.4]]}
        result = self.adapter._generate_embedding("test")
        self.assertEqual(result, [0.3, 0.4])
        self.mock_client_instance.embed.assert_called_once_with("nomic-embed-text", "test")

    def test_generate_embeddings_chunking(self):
        messages = [f"msg{i}" for i in range(1200)]
        self.mock_client_instance.embed.return_value = {"embeddings": [[0.5]]}
        self.adapter._generate_embeddings(messages)
        self.assertEqual(self.mock_client_instance.embed.call_count, 3)

import unittest
from unittest.mock import MagicMock, patch

from datastew.embedding.openai import GPT4Adapter


class TestOpenAIAdapter(unittest.TestCase):
    @patch("datastew.embedding.openai.OpenAI")
    def setUp(self, mock_openai):
        self.mock_client = mock_openai.return_value
        self.adapter = GPT4Adapter(api_key="test-key", cache=False)

    def test_generate_embedding(self):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
        self.mock_client.embeddings.create.return_value = mock_response
        result = self.adapter._generate_embedding("test")
        self.assertEqual(result, [0.1, 0.2])
        self.mock_client.embeddings.create.assert_called_once_with(input=["test"], model="text-embedding-ada-002")

    def test_generate_embeddings_chunking(self):
        messages = [f"msg{i}" for i in range(2500)]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.5])]
        self.mock_client.embeddings.create.return_value = mock_response
        self.adapter._generate_embeddings(messages)
        self.assertEqual(self.mock_client.embeddings.create.call_count, 3)

    def test_api_failure_propagation(self):
        self.mock_client.embeddings.create.side_effect = Exception("API Error")
        with self.assertRaisesRegex(Exception, "API Error"):
            self.adapter._generate_embedding("test")

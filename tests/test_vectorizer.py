import unittest

from datastew.embedding.hugging_face import HuggingFaceAdapter
from datastew.embedding.ollama import OllamaAdapter
from datastew.embedding.openai import GPT4Adapter
from datastew.embedding.vectorizer import Vectorizer


class TestVectorizer(unittest.TestCase):
    def test_initialize_hugging_face(self):
        vec = Vectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        self.assertIsInstance(vec.model, HuggingFaceAdapter)

    def test_initialize_openai_valid(self):
        vec = Vectorizer(model="text-embedding-ada-002", api_key="test_key")
        self.assertIsInstance(vec.model, GPT4Adapter)

    def test_initialize_openai_missing_key(self):
        with self.assertRaisesRegex(ValueError, "API key is required"):
            Vectorizer(model="text-embedding-ada-002")

    def test_initialize_ollama(self):
        vec = Vectorizer(model="nomic-embed-text")
        self.assertIsInstance(vec.model, OllamaAdapter)

    def test_initialize_unsupported_model(self):
        with self.assertRaisesRegex(NotImplementedError, "not supported in the registry"):
            Vectorizer(model="invalid-model-string")  # type: ignore

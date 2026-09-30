import logging
from typing import Sequence

from ollama import Client

from datastew.embedding.base import EmbeddingModel


class OllamaAdapter(EmbeddingModel):
    def __init__(
        self, model_name: str = "nomic-embed-text", host: str = "http://localhost:11434", cache: bool = False
    ):
        super().__init__(model_name, cache)
        self.client = Client(host)

    def _generate_embedding(self, text: str) -> Sequence[float]:
        try:
            return self.client.embed(self.model_name, text).get("embeddings")[0]
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            raise

    def _generate_embeddings(self, messages: list[str]) -> Sequence[Sequence[float]]:
        chunk_size = 500
        all_embeddings = []

        try:
            for i in range(0, len(messages), chunk_size):
                chunk = messages[i : i + chunk_size]
                response = self.client.embed(self.model_name, chunk)
                all_embeddings.extend(response.get("embeddings", []))
            return all_embeddings
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            raise

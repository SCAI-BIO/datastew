import logging
from typing import List, Sequence

from ollama import Client

from datastew.embedding.base import EmbeddingModel


class OllamaAdapter(EmbeddingModel):
    def __init__(
        self, model_name: str = "nomic-embed-text", host: str = "http://localhost:11434", cache: bool = False
    ):
        super().__init__(model_name, cache)
        self.client = Client(host)

    def get_embedding(self, text: str) -> Sequence[float]:
        if not text:
            logging.warning("Empty text passed to get_embedding")
            return []
        text = self.sanitize(text)

        if self._cache is not None:
            cached = self.get_from_cache(text)
            if cached:
                return cached
        try:
            embedding = self.client.embed(self.model_name, text).get("embeddings")[0]
            self.add_to_cache(text, embedding)
            return embedding
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            raise

    def get_embeddings(self, messages: List[str]) -> Sequence[Sequence[float]]:
        sanitized_messages = [self.sanitize(msg) for msg in messages]

        if self._cache is not None:
            embeddings, uncached_indices, uncached_messages = self.get_cached_embeddings(sanitized_messages)

            if uncached_messages:
                try:
                    new_embeddings = self.client.embed(self.model_name, uncached_messages).get("embeddings")
                    self.add_batch_to_cache(uncached_messages, new_embeddings)
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        embeddings[idx] = embedding
                except Exception as e:
                    logging.error(f"Failed processing messages: {e}")
                    raise

            return [emb for emb in embeddings if emb is not None]

        try:
            return self.client.embed(self.model_name, sanitized_messages).get("embeddings")
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            raise

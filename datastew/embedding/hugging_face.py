import logging
from typing import Sequence

from cachetools import LRUCache
from sentence_transformers import SentenceTransformer

from datastew.embedding.base import EmbeddingModel


class HuggingFaceAdapter(EmbeddingModel):
    _model_cache = LRUCache(maxsize=3)

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache: bool = False):
        super().__init__(model_name, cache)

        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)

        self.model = self._model_cache[model_name]

    def _generate_embedding(self, text: str) -> Sequence[float]:
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            raise

    def _generate_embeddings(self, messages: list[str]) -> Sequence[Sequence[float]]:
        try:
            return self.model.encode(messages, show_progress_bar=True).tolist()
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            raise

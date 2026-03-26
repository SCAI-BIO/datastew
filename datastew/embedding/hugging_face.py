import logging
from typing import List, Sequence

from sentence_transformers import SentenceTransformer

from datastew.embedding.base import EmbeddingModel


class HuggingFaceAdapter(EmbeddingModel):
    _model_cache = {}
    _load_count = 0

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache: bool = False):
        super().__init__(model_name, cache)

        if model_name not in self._model_cache:
            HuggingFaceAdapter._load_count += 1
            self._model_cache[model_name] = SentenceTransformer(model_name)

        self.model = self._model_cache[model_name]

    def get_embedding(self, text: str) -> Sequence[float]:
        if not text or not isinstance(text, str):
            logging.warning("Empty or invalid text passed to get_embedding")
            return []
        text = self.sanitize(text)

        if self._cache is not None:
            cached = self.get_from_cache(text)
            if cached:
                return cached

        try:
            embedding = self.model.encode(text)
            embedding = [float(x) for x in embedding]
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
                    new_embeddings = self.model.encode(uncached_messages, show_progress_bar=True)
                    flattened_embeddings = [
                        [float(element) for element in row] for row in new_embeddings if row is not None
                    ]
                    self.add_batch_to_cache(uncached_messages, flattened_embeddings)
                    for idx, embedding in zip(uncached_indices, flattened_embeddings):
                        embeddings[idx] = embedding
                except Exception as e:
                    logging.error(f"Failed processing messages: {e}")
                    raise

            return [emb for emb in embeddings if emb is not None]

        try:
            embeddings = self.model.encode(sanitized_messages, show_progress_bar=True)
            flattened_embeddings = [[float(element) for element in row] for row in embeddings]
            return flattened_embeddings
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            raise

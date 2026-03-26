import logging
from typing import List, Sequence

import openai

from datastew.embedding.base import EmbeddingModel


class GPT4Adapter(EmbeddingModel):
    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-ada-002",
        cache: bool = False,
    ):
        """Initialize the GPT-4 adapter with OpenAI API key and model name.

        :param api_key: The API key for accessing OpenAI services.
        :param model_name: The specific embedding model to use, defaults to text-embedding-ada-002.
        :param cache: Enable or disable caching, defaults to False.
        """
        super().__init__(model_name, cache)
        self.api_key = api_key
        openai.api_key = api_key

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
            response = openai.embeddings.create(input=[text], model=self.model_name)
            embedding = response.data[0].embedding
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
                    response = openai.embeddings.create(model=self.model_name, input=uncached_messages)
                    new_embeddings = [item.embedding for item in response.data]
                    self.add_batch_to_cache(uncached_messages, new_embeddings)
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        embeddings[idx] = embedding
                except Exception as e:
                    logging.error(f"Error in processing chunk: {e}")
                    raise

            return [emb for emb in embeddings if emb is not None]

        try:
            response = openai.embeddings.create(model=self.model_name, input=sanitized_messages)
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            raise

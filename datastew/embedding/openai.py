import logging
from typing import Sequence

from openai import OpenAI

from datastew.embedding.base import EmbeddingModel


class GPT4Adapter(EmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002", cache: bool = False):
        """Initialize the GPT-4 adapter with OpenAI API key and model name.

        :param api_key: The API key for accessing OpenAI services.
        :param model_name: The specific embedding model to use, defaults to text-embedding-ada-002.
        :param cache: Enable or disable caching, defaults to False.
        """
        super().__init__(model_name, cache)
        self.client = OpenAI(api_key=api_key)

    def _generate_embedding(self, text: str) -> Sequence[float]:
        try:
            response = self.client.embeddings.create(input=[text], model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            raise

    def _generate_embeddings(self, messages: list[str]) -> Sequence[Sequence[float]]:
        chunk_size = 1000
        all_embeddings = []

        try:
            for i in range(0, len(messages), chunk_size):
                chunk = messages[i : i + chunk_size]
                response = self.client.embeddings.create(model=self.model_name, input=chunk)
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            raise

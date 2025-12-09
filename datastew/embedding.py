import logging
from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Optional, Sequence, Tuple

import openai
from cachetools import LRUCache
from ollama import Client
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EmbeddingModel(ABC):
    def __init__(self, model_name: str, cache: bool = False, cache_size: int = 10000):
        """Initialize the embedding model with an optional caching mechanism.

        :param model_name: Name of the embedding model.
        :param cache: Enable or disable caching, defaults to False.
        :param cache_size: Maximum cache size when caching is enabled, defaults to 10,000.
        """
        self.model_name = model_name
        self._cache = LRUCache(maxsize=cache_size) if cache else None
        self._cache_lock = Lock() if cache else None

    @abstractmethod
    def get_embedding(self, text: str) -> Sequence[float]:
        """Retrieve the embedding vector for a single text input.

        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        pass

    @abstractmethod
    def get_embeddings(self, messages: List[str]) -> Sequence[Sequence[float]]:
        """Retrieve embeddings for a list of text messages

        :param messages: A list of text messages to embed.
        :return: A sequence of embedding vectors.
        """
        pass

    def add_to_cache(self, text: str, embedding: Sequence[float]):
        """Add a text-embedding pair to cache.

        :param text: The input text to be cached.
        :param embedding: The embedding of the input text.
        """
        if self._cache_lock and self._cache is not None:
            with self._cache_lock:
                self._cache[text] = embedding

    def get_from_cache(self, text: str) -> Optional[Sequence[float]]:
        """Retrieve an embedding from the cache.

        :param text: Cached input text.
        :return: Embedding of the cached input text or `None` if not present.
        """
        if self._cache_lock and self._cache is not None:
            with self._cache_lock:
                return self._cache.get(text, None)
        return None

    def get_cached_embeddings(self, messages: List[str]) -> Tuple[List[Sequence[float]], List[int], List[str]]:
        """Retrieve cached embeddings and identify uncached messages.

        :param messages: A list of input text messages.
        :return: A tuple containing:
            - A list of embeddings (cached where available, `None` for uncached).
            - A list of indices for uncached messages.
            - A list of uncached messages.
        """
        embeddings, uncached_indices, uncached_messages = [], [], []

        for i, msg in enumerate(messages):
            cached = self.get_from_cache(msg)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_messages.append(msg)
        return embeddings, uncached_indices, uncached_messages

    def sanitize(self, message: str) -> str:
        """Clean up the input text by trimming and converting to lowercase.

        :param message: The input text message.
        :return: Sanitized text.
        """
        return message.strip().lower().replace("\n", " ").replace("\t", " ")


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
        """Retrieve an embedding for a single text input using OpenAI API.

        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        if not text:
            logging.warning("Empty text passed to get_embedding")
            return []
        text = self.sanitize(text)

        if self._cache:
            # Check cache
            cached = self.get_from_cache(text)
            if cached:
                return cached

        # Request from OpenAI API
        try:
            response = openai.embeddings.create(input=[text], model=self.model_name)
            embedding = response.data[0].embedding
            self.add_to_cache(text, embedding)
            return embedding
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            return []

    def get_embeddings(self, messages: List[str]) -> Sequence[Sequence[float]]:
        """Retrieve embeddings for a list of text messages.

        :param messages: A list of text messages to embed.
        :return: A sequence of embedding vectors.
        """

        sanitized_messages = [self.sanitize(msg) for msg in messages]

        if self._cache:
            embeddings, uncached_indices, uncached_messages = self.get_cached_embeddings(sanitized_messages)

            if uncached_messages:
                try:
                    response = openai.embeddings.create(model=self.model_name, input=uncached_messages)
                    new_embeddings = [item.embedding for item in response.data]
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        self.add_to_cache(sanitized_messages[idx], embedding)
                        embeddings[idx] = embedding
                except Exception as e:
                    logging.error(f"Error in processing chunk: {e}")
                    return []

            return [emb for emb in embeddings if emb is not None]

        try:
            response = openai.embeddings.create(model=self.model_name, input=sanitized_messages)
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            return []


class HuggingFaceAdapter(EmbeddingModel):
    _model_cache = {}
    _load_count = 0 # For testing

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache: bool = False):
        """Initialize the Hugging Face adapter with a specified model name.

        :param model_name: The model name for sentence transformers, defaults to sentence-transformers/all-MiniLM-L6-v2.
        :param cache: Enable or disable caching, defaults to False.
        """
        super().__init__(model_name, cache)

        if model_name not in self._model_cache:
            HuggingFaceAdapter._load_count += 1
            self._model_cache[model_name] = SentenceTransformer(model_name)

        self.model = self._model_cache[model_name]

    def get_embedding(self, text: str) -> Sequence[float]:
        """Retrieve an embedding for a single text input using MPnet.

        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        if not text or not isinstance(text, str):
            logging.warning("Empty or invalid text passed to get_embedding")
            return []
        text = self.sanitize(text)

        # Check cache
        if self._cache:
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
            return []

    def get_embeddings(self, messages: List[str]) -> Sequence[Sequence[float]]:
        """Retrieve embeddings for a list of text messages using MPNet.

        :param messages: A list of text messages to embed.
        :return: A sequence of embedding vectors.
        """
        sanitized_messages = [self.sanitize(msg) for msg in messages]
        if self._cache:
            embeddings, uncached_indices, uncached_messages = self.get_cached_embeddings(sanitized_messages)

            if uncached_messages:
                try:
                    new_embeddings = self.model.encode(uncached_messages, show_progress_bar=True)
                    flattened_embeddings = [
                        [float(element) for element in row] for row in new_embeddings if row is not None
                    ]
                    for idx, embedding in zip(uncached_indices, flattened_embeddings):
                        self.add_to_cache(sanitized_messages[idx], embedding)
                        embeddings[idx] = embedding
                except Exception as e:
                    logging.error(f"Failed processing messages: {e}")
                    return []

            return [emb for emb in embeddings if emb is not None]

        try:
            embeddings = self.model.encode(sanitized_messages, show_progress_bar=True)
            flattened_embeddings = [[float(element) for element in row] for row in embeddings]
            return flattened_embeddings
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            return []


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
        if self._cache:
            cached = self.get_from_cache(text)
            if cached:
                return cached
        try:
            embedding = self.client.embed(self.model_name, text).get("embeddings")[0]
            self.add_to_cache(text, embedding)
            return embedding
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            return []

    def get_embeddings(self, messages: List[str]) -> Sequence[Sequence[float]]:
        sanitized_messages = [self.sanitize(msg) for msg in messages]

        if self._cache:
            embeddings, uncached_indices, uncached_messages = self.get_cached_embeddings(sanitized_messages)

            if uncached_messages:
                try:
                    new_embeddings = self.client.embed(self.model_name, uncached_messages).get("embeddings")
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        self.add_to_cache(sanitized_messages[idx], embedding)
                        embeddings[idx] = embedding
                except Exception as e:
                    logging.error(f"Failed processing messages: {e}")
                    return []

            return [emb for emb in embeddings if emb is not None]

        try:
            return self.client.embed(self.model_name, sanitized_messages).get("embeddings")
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            return []


class Vectorizer:
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
        host: str = "http://localhost:11434",
        cache: bool = False,
    ):
        """Initializes the Vectorizer with the specified model and settings.

        :param model: The model to use for generating embeddings, defaults to sentence-transformers/all-MiniLM-L6-v2.
        :param api_key: The API key for GPT-based models, defaults to None.
        :param host: The host URL for locally hosted Ollama models. defaults to http://localhost:11434.
        :param cache: Whether to enable caching for embeddings, defaults to False.
        """
        self.model = self.initialize_model(model, api_key, host, cache)
        self.model_name = self.model.model_name

    def initialize_model(self, model: str, api_key: Optional[str], host: str, cache: bool):
        if model == "sentence-transformers/all-MiniLM-L6-v2":
            return HuggingFaceAdapter(model, cache)
        elif model == "sentence-transformers/all-mpnet-base-v2":
            return HuggingFaceAdapter(model, cache)
        elif model == "FremyCompany/BioLORD-2023":
            return HuggingFaceAdapter(model, cache)
        elif model == "text-embedding-ada-002" and api_key:
            return GPT4Adapter(api_key, model, cache)
        elif model == "text-embedding-3-large" and api_key:
            return GPT4Adapter(api_key, model, cache)
        elif model == "text-embedding-3-small" and api_key:
            return GPT4Adapter(api_key, model, cache)
        elif model == "nomic-embed-text":
            return OllamaAdapter(model, host, cache)
        else:
            raise NotImplementedError(f"The model '{model}' is not supported.")

    def get_embedding(self, text: str):
        return self.model.get_embedding(text)

    def get_embeddings(self, messages: List[str]):
        return self.model.get_embeddings(messages)

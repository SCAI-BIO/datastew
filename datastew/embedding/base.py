import logging
import re
from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Optional, Sequence, Tuple

from cachetools import LRUCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_GLOBAL_CACHES = {}
_GLOBAL_LOCKS = {}
_INIT_LOCK = Lock()

_WHITESPACE_RE = re.compile(r"\s+")


class EmbeddingModel(ABC):
    def __init__(self, model_name: str, cache: bool = False, cache_size: int = 10000):
        """Initialize the embedding model with an optional caching mechanism.

        :param model_name: Name of the embedding model.
        :param cache: Enable or disable caching, defaults to False.
        :param cache_size: Maximum cache size when caching is enabled, defaults to 10,000.
        """
        self.model_name = model_name
        self.use_cache = cache

        if self.use_cache:
            with _INIT_LOCK:
                if model_name not in _GLOBAL_CACHES:
                    _GLOBAL_CACHES[model_name] = LRUCache(maxsize=cache_size)
                    _GLOBAL_LOCKS[model_name] = Lock()

            self._cache = _GLOBAL_CACHES[model_name]
            self._cache_lock = _GLOBAL_LOCKS[model_name]

        else:
            self._cache = None
            self._cache_lock = None

    def get_embedding(self, text: str) -> Sequence[float]:
        """Retrieve the embedding vector for a single text input.

        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        if not text or not isinstance(text, str):
            logging.warning("Empty or invalid text passed to get_embedding")
            return []

        text = self._sanitize(text)

        if self._cache is not None:
            cached = self._get_from_cache(text)
            if cached:
                return cached

        embedding = self._generate_embedding(text)
        self._add_to_cache(text, embedding)
        return embedding

    def get_embeddings(self, messages: List[str]) -> Sequence[Sequence[float]]:
        """Retrieve embeddings for a list of text messages

        :param messages: A list of text messages to embed.
        :return: A sequence of embedding vectors.
        """
        sanitized_messages = [self._sanitize(msg) for msg in messages if msg and isinstance(msg, str)]

        if self._cache is not None:
            embeddings, uncached_indices, uncached_messages = self._get_cached_embeddings(sanitized_messages)

            if uncached_messages:
                new_embeddings = self._generate_embeddings(uncached_messages)
                self._add_batch_to_cache(uncached_messages, new_embeddings)
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding

            return [emb for emb in embeddings if emb is not None]

        return self._generate_embeddings(sanitized_messages)

    @abstractmethod
    def _generate_embedding(self, text: str) -> Sequence[float]:
        """Generate an embedding vector for a single sanitized text input

        This is an abstract method that must be implemented by subclasses to interface
        with the specific model's API or inference engine.

        :param text: The sanitized input text to embed.
        :return: A sequence of floats respresenting the generated embedding.
        """
        pass

    @abstractmethod
    def _generate_embeddings(self, messages: list[str]) -> Sequence[Sequence[float]]:
        """Generate embedding vectors for a batch of sanitized text messages.

        This is an abstract method that must be implemented by subclasses to interface
        with the specific model's API or inference engine.

        :param messages: A list of sanitized text messages to embed.
        :return: A sequence of embedding vectors corresponding to the input messages.
        """
        pass

    def _sanitize(self, message: str) -> str:
        """Clean up the input text by trimming and converting to lowercase.

        :param message: The input text message.
        :return: Sanitized text.
        """
        return _WHITESPACE_RE.sub(" ", message).strip().lower()

    def _add_to_cache(self, text: str, embedding: Sequence[float]):
        """Add a text-embedding pair to cache.

        :param text: The input text to be cached.
        :param embedding: The embedding of the input text.
        """
        if self._cache_lock is not None and self._cache is not None:
            with self._cache_lock:
                self._cache[text] = embedding

    def _add_batch_to_cache(self, texts: Sequence[str], embeddings: Sequence[Sequence[float]]):
        """Add a batch of text-embedding pairs to the cache.

        Acquires the thread lock once for the entire batch update to minimize overhead.

        :param texts: A sequence of input texts to be cached.
        :param embeddings: A sequence of corresponding embedding vectors.
        """
        if self._cache_lock is not None and self._cache is not None:
            with self._cache_lock:
                for text, emb in zip(texts, embeddings):
                    self._cache[text] = emb

    def _get_from_cache(self, text: str) -> Optional[Sequence[float]]:
        """Retrieve an embedding from the cache.

        :param text: Cached input text.
        :return: Embedding of the cached input text or `None` if not present.
        """
        if self._cache_lock is not None and self._cache is not None:
            with self._cache_lock:
                return self._cache.get(text, None)
        return None

    def _get_cached_embeddings(
        self, messages: List[str]
    ) -> Tuple[List[Optional[Sequence[float]]], List[int], List[str]]:
        """Retrieve cached embeddings and identify uncached messages.

        :param messages: A list of input text messages.
        :return: A tuple containing:
            - A list of embeddings (cached where available, `None` for uncached).
            - A list of indices for uncached messages.
            - A list of uncached messages.
        """
        if self._cache_lock is None or self._cache is None:
            return [None] * len(messages), list(range(len(messages))), messages

        embeddings: List[Optional[Sequence[float]]] = []
        uncached_indices: List[int] = []
        uncached_messages: List[str] = []

        with self._cache_lock:
            for i, msg in enumerate(messages):
                cached = self._cache.get(msg, None)
                embeddings.append(cached)
                if cached is None:
                    uncached_indices.append(i)
                    uncached_messages.append(msg)

        return embeddings, uncached_indices, uncached_messages

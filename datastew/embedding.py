import concurrent.futures
import logging
from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Sequence, Tuple

import openai
from cachetools import LRUCache
from openai.error import OpenAIError
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingModel(ABC):
    def __init__(self, model_name: str, cache_size: int = 10000):
        self.model_name = model_name
        self._cache = LRUCache(maxsize=cache_size)
        self._cache_lock = Lock()
    
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
        with self._cache_lock:
            self._cache[text] = embedding

    def get_from_cache(self, text: str) -> Sequence[float]:
        """Retrieve an embedding from the cache.
        
        :param text: Cached input text.
        :return: Embedding of the cached input text.
        """
        with self._cache_lock:
            return self._cache.get(text)
        
    def get_cached_embeddings(
            self, messages: List[str]
    ) -> Tuple[List[Sequence[float]], List[int], List[str]]:
        """Retrieve cached embeddings and identify uncached messages.

        This method checks the cache for embeddings of the given messages. It returns
        three items:
        1. A list of embeddings where cached embeddings are included, and `None` is 
        used as a placeholder for uncached messages.
        2. A list of indices corresponding to the positions of uncached messages in 
        the input list.
        3. A list of the uncached messages themselves.

        :param messages: A list of input text messages to check for cached embeddings.
        :return: A tuple containing:
            - A list of embeddings (cached or `None` for uncached messages).
            - A list of indices corresponding to uncached messages.
            - A list of uncached messages.
        """
        embeddings = []
        uncached_indices = []
        uncached_messages = []

        for i, msg in enumerate(messages):
            cached = self.get_from_cache(msg)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_messages.append(msg)
        return embeddings, uncached_indices, uncached_messages
    
    def get_model_name(self) -> str:
        """Return the name of the embedding model.
        
        :return: The name of the model.
        """
        return self.model_name

    def sanitize(self, message: str) -> str:
        """Clean up the input text by trimming and converting to lowercase.
        
        :param message: The input text message.
        :return: Sanitized text.
        """
        return message.strip().lower()


class GPT4Adapter(EmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        """Initialize the GPT-4 adapter with OpenAI API key and model name.
        
        :param api_key: The API key for accessing OpenAI services.
        :param model_name: The specific embedding model to use.
        """
        super().__init__(model_name)
        self.api_key = api_key
        openai.api_key = api_key

    def get_embedding(self, text: str) -> Sequence[float]:
        """Retrieve an embedding for a single text input using OpenAI API.
        
        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        if not text or not isinstance(text, str):
            logging.warning("Empty or invalid text passed to get_embedding")
            return []
        text = self.sanitize(text.replace("\n", " "))

        # Check cache
        cached = self.get_from_cache(text)
        if cached:
            logging.info("Cache hit for single text.")
            return cached
        
        # Request from OpenAI API
        try:
            response = openai.Embedding.create(input=[text], model=self.model_name)
            embedding = response["data"][0]["embedding"]
            self.add_to_cache(text, embedding)
            return embedding
        except OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return []

    def get_embeddings(self, messages: List[str], max_length: int = 2048, num_workers: int = 4) -> Sequence[Sequence[float]]:
        """Retrieve embeddings for a list of text messages using batching and multithreading.

        :param messages: A list of text messages to embed.
        :param max_length: Maximum length of each batch of messages.
        :param num_workers: Number of threads for parallel processing.
        :return: A sequence of embedding vectors.
        """
        if max_length <= 0:
            logging.warning(f"max_length is set to {max_length}, using default value 2048")
            max_length = 2048
        
        sanitized_messages = [self.sanitize(msg) for msg in messages]
        embeddings, uncached_indices, uncached_messages = self.get_cached_embeddings(sanitized_messages)

        if uncached_messages:
            chunks = [uncached_messages[i:i + max_length] for i in range(0, len(uncached_messages), max_length)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(self._process_chunk, chunk): chunk for chunk in chunks}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        chunk_embeddings = future.result()
                        for idx, embedding in zip(uncached_indices, chunk_embeddings):
                            self.add_to_cache(sanitized_messages[idx], embedding)
                            embeddings[idx] = embedding
                    except Exception as e:
                        logging.error(f"Error in processing chunk: {e}")
                        return []
                    
        cache_hits = len(messages) - len(uncached_messages)
        logging.info(f"Processed {len(messages)} messages ({cache_hits} cache hits).")
        return [emb for emb in embeddings if emb is not None]
    
    def _process_chunk(self, chunk: List[str], retries: int = 3) -> Sequence[Sequence[float]]:
        """Process a batch of text messages to retrieve embeddings.
        
        :param chunk: A list of sanitized messages.
        :param retries: Maximum number of attempts to retrieve embeddings.
        :return: A sequence of embedding vectors.
        """
        for attempt in range(retries):
            try:
                response = openai.Embedding.create(input=chunk, model=self.model_name)
                return [item["embedding"] for item in response["data"]]
            except Exception as e:
                logging.error(f"Attempt {attempt + 1}/{retries} failed for chunk: {e}")
        logging.error(f"All attempts failed for chunk: {chunk}")
        return []


class MPNetAdapter(EmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """Initialize the MPNet adapter with a specified model name and threading settings.
        
        :param model_name: The model name for sentence transformers.
        :param num_threads: The number of CPU threads for inference.
        """
        super().__init__(model_name)
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text: str) -> Sequence[float]:
        """Retrieve an embedding for a single text input using MPnet.
        
        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        if not text or not isinstance(text, str):
            logging.warning("Empty or invalid text passed to get_embedding")
            return []
        text = self.sanitize(text.replace("\n", " "))

        # Check cache
        cached = self.get_from_cache(text)
        if cached:
            logging.info("Cache hit for single text.")
            return cached
        
        try:
            embedding = self.model.encode(text)
            embedding = [float(x) for x in embedding]
            self.add_to_cache(text, embedding)
            return embedding
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            return []

    def get_embeddings(self, messages: List[str], batch_size: int = 64) -> Sequence[Sequence[float]]:
        """Retrieve embeddings for a list of text messages using MPNet.
        
        :param messages: A list of text messages to embed.
        :param batch_size: The batch size for processing.
        :return: A sequence of embedding vectors.
        """
        sanitized_messages = [self.sanitize(msg) for msg in messages]
        embeddings, uncached_indices, uncached_messages = self.get_cached_embeddings(messages)

        if uncached_messages:
            try:
                new_embeddings = self.model.encode(uncached_messages, batch_size=batch_size, show_progress_bar=True)
                flattened_embeddings = [[float(element) for element in row] for row in new_embeddings if row is not None]
                for idx, embedding in zip(uncached_indices, flattened_embeddings):
                    self.add_to_cache(sanitized_messages[idx], embedding)
                    embeddings[idx] = embedding
            except Exception as e:
                logging.error(f"Failed processing messages: {e}")
                return []
        
        cache_hits = len(messages) - len(uncached_messages)
        logging.info(f"Processed {len(messages)} messages ({cache_hits} cache hits).")
        return [emb for emb in embeddings if emb is not None]


class TextEmbedding:
    def __init__(self, text: str, embedding: List[float]):
        self.text = text
        self.embedding = embedding

import concurrent.futures
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Sequence

import openai
import torch
from openai.error import OpenAIError
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmbeddingModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
    
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
        try:
            response = openai.Embedding.create(input=[text], model=self.model_name)
            return response["data"][0]["embedding"]
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
        chunks = [sanitized_messages[i:i + max_length] for i in range(0, len(sanitized_messages), max_length)]
        embeddings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._process_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    embeddings.extend(future.result())
                except Exception as e:
                    logging.error(f"Error in processing chunk: {e}")
        return embeddings
    
    def _process_chunk(self, chunk: List[str]) -> Sequence[Sequence[float]]:
        """Process a batch of text messages to retrieve embeddings.
        
        :param chunk: A list of sanitized messages.
        :return: A sequence of embedding vectors.
        """
        try:
            response = openai.Embedding.create(input=chunk, model=self.model_name)
            return [item["embedding"] for item in response["data"]]
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
            return []


class MPNetAdapter(EmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", num_threads: int = 4):
        """Initialize the MPNet adapter with a specified model name and threading settings.
        
        :param model_name: The model name for sentence transformers.
        :param num_threads: The number of CPU threads for inference.
        """
        super().__init__(model_name)
        device, available_threads = self._initialize_device(num_threads)
        self.model = SentenceTransformer(model_name).to(device)

        if device == "cpu":
            torch.set_num_threads(available_threads)
            logging.info(
                f"MPNet model '{model_name} initialized on CPU  with {available_threads} threads."
            )
        elif device == "cuda":
            logging.info(
                f"MPNet model '{model_name}' initialized on GPU. GPU thread management is handled automatically."
            )
            

    def get_embedding(self, text: str) -> Sequence[float]:
        """Retrieve an embedding for a single text input using MPnet.
        
        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        if not text or not isinstance(text, str):
            logging.warning("Empty or invalid text passed to get_embedding")
            return []
        text = self.sanitize(text.replace("\n", " "))
        try:
            embedding = self.model.encode(text, device=str(self.model.device))
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
        try:
            embeddings = self.model.encode(sanitized_messages, batch_size=batch_size, show_progress_bar=True)
            flattened_embeddings = [[float(element) for element in row] for row in embeddings]
            return flattened_embeddings
        except Exception as e:
            logging.error(f"Failed processing messages: {e}")
            return []
    
    def _initialize_device(self, num_threads: int) -> tuple[str, int]:
        """Determine the appropriate device (CPU or GPU) and set the thread count.
        
        :param num_threads: The requested number of threads for inference.
        :return: A tuple containing the selected device ("cuda" or "cpu") and the final number of threads to use.
        :raise RuntimeError: If CPU core count cannot be determined when no GPU is available.
        """
        if torch.cuda.is_available():
            return "cuda", num_threads # num_threads does not affect GPU operations
        
        # Fallback to CPU
        cpu_count = os.cpu_count()
        if cpu_count is None:
            raise RuntimeError("Unable to determine the number of CPU cores.")
        
        threads = min(num_threads, cpu_count)
        if num_threads > cpu_count:
            logging.warning(
                f"Requested {num_threads} threads, but only {cpu_count} CPU cores available. Using {cpu_count} threads."
            )
        return "cpu", threads


class TextEmbedding:
    def __init__(self, text: str, embedding: List[float]):
        self.text = text
        self.embedding = embedding

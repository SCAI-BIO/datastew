import logging
from abc import ABC

import numpy as np
import openai
from sentence_transformers import SentenceTransformer


class EmbeddingModel(ABC):
    def get_embedding(self, text: str) -> [float]:
        pass

    def get_embeddings(self, messages: [str]) -> [[float]]:
        pass

    def get_model_name(self) -> str:
        pass

    @staticmethod
    def sanitize(self, message: str) -> str:
        return message.strip().lower()


class GPT4Adapter(EmbeddingModel):
    def __init__(self, api_key: str, model_name: str = "text-embedding-ada-002"):
        self.api_key = api_key
        openai.api_key = api_key
        self.model_name = model_name
        logging.getLogger().setLevel(logging.INFO)

    def get_embedding(self, text: str):
        logging.info(f"Getting embedding for {text}")
        try:
            if text is None or text == "" or text is np.nan:
                logging.warning(f"Empty text passed to get_embedding")
                return None
            if isinstance(text, str):
                text = text.replace("\n", " ")
                text = self.sanitize(text)
            return openai.Embedding.create(input=[text], model=self.model_name)["data"][0]["embedding"]
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            return None

    def get_embeddings(self, messages: [str], max_length=2048):
        sanitized_messages = [self.sanitize(message) for message in messages]
        embeddings = []
        total_chunks = (len(sanitized_messages) + max_length - 1) // max_length
        current_chunk = 0
        for i in range(0, len(sanitized_messages), max_length):
            current_chunk += 1
            chunk = sanitized_messages[i:i + max_length]
            response = openai.Embedding.create(input=chunk, model=self.model_name)
            embeddings.extend([item["embedding"] for item in response["data"]])
            logging.info("Processed chunk %d/%d", current_chunk, total_chunks)
        return embeddings

    def get_model_name(self) -> str:
        return self.model_name


class MPNetAdapter(EmbeddingModel):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        logging.getLogger().setLevel(logging.INFO)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name  # For Weaviate

    def get_embedding(self, text: str):
        logging.info(f"Getting embedding for {text}")
        try:
            if text is None or text == "" or text is np.nan:
                logging.warn(f"Empty text passed to get_embedding")
                return None
            if isinstance(text, str):
                text = text.replace("\n", " ")
                text = self.sanitize(text)
            return self.model.encode(text)
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            return None

    def get_embeddings(self, messages: [str]) -> [[float]]:
        sanitized_messages = [self.sanitize(message) for message in messages]
        try:
            embeddings = self.model.encode(sanitized_messages)
        except Exception as e:
            logging.error(f"Failed for messages {sanitized_messages}")
        flattened_embeddings = [[float(element) for element in row] for row in embeddings]
        return flattened_embeddings

    def get_model_name(self) -> str:
        return self.model_name


class TextEmbedding:
    def __init__(self, text: str, embedding: [float]):
        self.text = text
        self.embedding = embedding


def get_default_embedding_model() -> EmbeddingModel:
    return MPNetAdapter()

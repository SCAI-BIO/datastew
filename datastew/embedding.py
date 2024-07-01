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

    def sanitize(self, message: str) -> str:
        return message.strip().lower()


class GPT4Adapter(EmbeddingModel):
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        logging.getLogger().setLevel(logging.INFO)

    def get_embedding(self, text: str, model="text-embedding-ada-002"):
        logging.info(f"Getting embedding for {text}")
        try:
            if text is None or text == "" or text is np.nan:
                logging.warning(f"Empty text passed to get_embedding")
                return None
            if isinstance(text, str):
                text = text.replace("\n", " ")
                text = self.sanitize(text)
            return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            return None

    def get_embeddings(self, messages: [str], model="text-embedding-ada-002", max_length=2048):
        sanitized_messages = [self.sanitize(message) for message in messages]
        embeddings = []
        total_chunks = (len(sanitized_messages) + max_length - 1) // max_length
        current_chunk = 0
        for i in range(0, len(sanitized_messages), max_length):
            current_chunk += 1
            chunk = sanitized_messages[i:i + max_length]
            response = openai.Embedding.create(input=chunk, model=model)
            embeddings.extend([item["embedding"] for item in response["data"]])
            logging.info("Processed chunk %d/%d", current_chunk, total_chunks)
        return embeddings


class MPNetAdapter(EmbeddingModel):
    def __init__(self, model="sentence-transformers/all-mpnet-base-v2"):
        logging.getLogger().setLevel(logging.INFO)
        self.mpnet_model = SentenceTransformer(model)

    def get_embedding(self, text: str):
        logging.info(f"Getting embedding for {text}")
        try:
            if text is None or text == "" or text is np.nan:
                logging.warn(f"Empty text passed to get_embedding")
                return None
            if isinstance(text, str):
                text = text.replace("\n", " ")
                text = self.sanitize(text)
            return self.mpnet_model.encode(text)
        except Exception as e:
            logging.error(f"Error getting embedding for {text}: {e}")
            return None

    def get_embeddings(self, messages: [str]) -> [[float]]:
        sanitized_messages = [self.sanitize(message) for message in messages]
        try:
            embeddings = self.mpnet_model.encode(sanitized_messages)
        except Exception as e:
            logging.error(f"Failed for messages {sanitized_messages}")
        flattened_embeddings = [[float(element) for element in row] for row in embeddings]
        return flattened_embeddings


class TextEmbedding:
    def __init__(self, text: str, embedding: [float]):
        self.text = text
        self.embedding = embedding

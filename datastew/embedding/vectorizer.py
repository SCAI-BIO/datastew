from typing import Literal, Optional, Sequence

SupportedModel = Literal[
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "FremyCompany/BioLORD-2023",
    "text-embedding-ada-002",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "nomic-embed-text",
]


class Vectorizer:
    """Factory class to initialize and route requests to the appropriate embedding model adapter.

    Acts as a unified interface for local Hugging Face models, OpenAI API models, and locally hosted Ollama models.
    """

    _MODEL_REGISTRY = {
        "sentence-transformers/all-MiniLM-L6-v2": "hugging_face",
        "sentence-transformers/all-mpnet-base-v2": "hugging_face",
        "FremyCompany/BioLORD-2023": "hugging_face",
        "text-embedding-ada-002": "openai",
        "text-embedding-3-large": "openai",
        "text-embedding-3-small": "openai",
        "nomic-embed-text": "ollama",
    }

    def __init__(
        self,
        model: SupportedModel = "sentence-transformers/all-mpnet-base-v2",
        api_key: Optional[str] = None,
        host: str = "http://localhost:11434",
        cache: bool = False,
    ):
        """Initializes the Vectorizer with the specified model and settings.

        :param model: The model to use for generating embeddings, defaults to sentence-transformers/all-mpnet-base-v2.
        :param api_key: The API key for GPT-based models, defaults to None.
        :param host: The host URL for locally hosted Ollama models, defaults to http://localhost:11434.
        :param cache: Whether to enable caching for embeddings, defaults to False.
        """
        self.model = self._initialize_model(model, api_key, host, cache)
        self.model_name = self.model.model_name

    def get_embedding(self, text: str) -> Sequence[float]:
        """Retrieve the embedding vector for a single text input.

        :param text: The input text to embed.
        :return: A sequence of floats representing the embedding.
        """
        return self.model.get_embedding(text)

    def get_embeddings(self, messages: list[str]) -> Sequence[Sequence[float]]:
        """Retrieve embedding vectors for a batch of text messages.

        :param messages: A list of text messages to embed.
        :return: A sequence of embedding vectors corresponding to the input messages.
        """
        return self.model.get_embeddings(messages)

    def _initialize_model(self, model: SupportedModel, api_key: Optional[str], host: str, cache: bool):
        """Resolve the provider from the registry and instantiate the corresponding adapter.

        :param model: The specific embedding model identifier.
        :param api_key: The API key required for OpenAI models.
        :param host: The host URL for Ollama models.
        :param cache: Flag to enable LRU caching within the adapter.
        :raises ValueError: If an OpenAI model is required but no API key is provided.
        :raises NotImplementedError: If the requested model is not found in the registry.
        :return: An initialized subclass of EmbeddingModel.
        """
        provider = self._MODEL_REGISTRY.get(model)

        if provider == "hugging_face":
            from datastew.embedding.hugging_face import HuggingFaceAdapter

            return HuggingFaceAdapter(model, cache)

        elif provider == "openai":
            if not api_key:
                raise ValueError(f"API key is required for OpenAI model '{model}'.")
            from datastew.embedding.openai import GPT4Adapter

            return GPT4Adapter(api_key, model, cache)

        elif provider == "ollama":
            from datastew.embedding.ollama import OllamaAdapter

            return OllamaAdapter(model, host, cache)

        else:
            raise NotImplementedError(f"The model '{model}' is not supported in the registry.")

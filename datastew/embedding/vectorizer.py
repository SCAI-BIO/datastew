from typing import Literal

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
    def __init__(
        self,
        model: SupportedModel = "sentence-transformers/all-mpnet-base-v2",
        api_key: str | None = None,
        host: str = "http://localhost:11434",
        cache: bool = False,
    ):
        """Initializes the Vectorizer with the specified model and settings.

        :param model: The model to use for generating embeddings, defaults to sentence-transformers/all-mpnet-base-v2.
        :param api_key: The API key for GPT-based models, defaults to None.
        :param host: The host URL for locally hosted Ollama models. defaults to http://localhost:11434.
        :param cache: Whether to enable caching for embeddings, defaults to False.
        """
        self.model = self.initialize_model(model, api_key, host, cache)
        self.model_name = self.model.model_name

    def initialize_model(self, model: SupportedModel, api_key: str | None, host: str, cache: bool):
        if model in [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "FremyCompany/BioLORD-2023",
        ]:
            from datastew.embedding.hugging_face import HuggingFaceAdapter  # noqa: PLC0415

            return HuggingFaceAdapter(model, cache)

        elif (
            model
            in [
                "text-embedding-ada-002",
                "text-embedding-3-large",
                "text-embedding-3-small",
            ]
            and api_key
        ):
            from datastew.embedding.openai import GPT4Adapter  # noqa: PLC0415

            return GPT4Adapter(api_key, model, cache)

        elif model == "nomic-embed-text":
            from datastew.embedding.ollama import OllamaAdapter  # noqa: PLC0415

            return OllamaAdapter(model, host, cache)

        else:
            raise NotImplementedError(f"The model '{model}' is not supported or missing API key.")

    def get_embedding(self, text: str):
        return self.model.get_embedding(text)

    def get_embeddings(self, messages: list[str]):
        return self.model.get_embeddings(messages)

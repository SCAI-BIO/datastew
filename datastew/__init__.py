from ._version import __version__
from .embedding import EmbeddingModel, GPT4Adapter, HuggingFaceAdapter, OllamaAdapter, Vectorizer

# Importing submodules to expose their attributes if needed
from .process import jsonl_adapter, mapping, ols, parsing
from .repository import (
    Concept,
    Mapping,
    Terminology,
    base,
    model,
    pagination,
    postgresql,
    sqllite,
    weaviate,
    weaviate_schema,
)
from .visualisation import (
    bar_chart_average_acc_two_distributions,
    enrichment_plot,
    get_plot_for_current_database_state,
    plot_embeddings,
)

__all__ = [
    "__version__",
    "jsonl_adapter",
    "mapping",
    "ols",
    "parsing",
    "base",
    "model",
    "pagination",
    "postgresql",
    "sqllite",
    "weaviate",
    "weaviate_schema",
    "EmbeddingModel",
    "GPT4Adapter",
    "HuggingFaceAdapter",
    "OllamaAdapter",
    "Vectorizer",
    "Terminology",
    "Concept",
    "Mapping",
    "bar_chart_average_acc_two_distributions",
    "enrichment_plot",
    "get_plot_for_current_database_state",
    "plot_embeddings",
]

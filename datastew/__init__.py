from ._version import __version__
from .embedding import *

# Importing submodules to expose their attributes if needed
from .process import mapping, parsing
from .repository import Concept, Mapping, Terminology, base, model, sqllite, weaviate
from .visualisation import *

__all__ = [
    "mapping",
    "parsing",
    "model",
    "base",
    "sqllite",
    "weaviate",
    "DataDictionarySource",
    "HuggingFaceAdapter",
    "Terminology",
    "Concept",
    "Mapping",
    "get_default_embedding_model"
]

from abc import ABC, abstractmethod
from typing import List

from datastew.repository.model import Mapping, Concept, Terminology, SentenceEmbedder


class BaseRepository(ABC):

    @abstractmethod
    def store(self, model_object_instance):
        """Store a single model object instance."""
        pass

    @abstractmethod
    def store_all(self, model_object_instances):
        """Store multiple model object instances."""
        pass

    @abstractmethod
    def get_all_concepts(self) -> List[Concept]:
        """ Retrieve all concepts from the database."""
        pass

    @abstractmethod
    def get_all_terminologies(self) -> List[Terminology]:
        """ Retrieve all terminologies from the database."""
        pass

    @abstractmethod
    def get_all_mappings(self, limit=1000) -> List[Mapping]:
        """Get all embeddings up to a limit"""
        pass

    @abstractmethod
    def get_all_sentence_embedders(self) -> List[SentenceEmbedder]:
        pass

    @abstractmethod
    def get_closest_mappings(self, embedding, limit=5):
        """Get the closest mappings based on embedding."""
        pass

    @abstractmethod
    def shut_down(self):
        """Shut down the repository."""
        pass

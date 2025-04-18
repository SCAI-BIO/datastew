from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from datastew.process.parsing import DataDictionarySource
from datastew.repository.model import Concept, Mapping, MappingResult, Terminology
from datastew.repository.pagination import Page


class BaseRepository(ABC):

    @abstractmethod
    def import_data_dictionary(self, data_dictionary: DataDictionarySource, terminology_name: str):
        """Store a data dictionary"""

    @abstractmethod
    def store(self, model_object_instance):
        """Store a single model object instance."""
        pass

    @abstractmethod
    def store_all(self, model_object_instances):
        """Store multiple model object instances."""
        pass

    @abstractmethod
    def get_concepts(self) -> List[Concept]:
        """Retrieve all concepts from the database."""
        pass

    @abstractmethod
    def get_all_terminologies(self) -> List[Terminology]:
        """Retrieve all terminologies from the database."""
        pass

    @abstractmethod
    def get_mappings(
        self,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Page[Mapping]:
        """Get all embeddings up to a limit"""
        pass

    @abstractmethod
    def get_all_sentence_embedders(self) -> List[str]:
        pass

    @abstractmethod
    def get_closest_mappings(
        self,
        embedding: Sequence[float],
        similarities: bool = False,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit=5,
    ) -> Union[List[Mapping], List[MappingResult]]:
        """Get the closest mappings based on embedding."""
        pass

    @abstractmethod
    def shut_down(self):
        """Shut down the repository."""
        pass

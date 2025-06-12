import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.repository.model import Concept, Mapping, MappingResult, Terminology
from datastew.repository.pagination import Page

logger = logging.getLogger(__name__)


class BaseRepository(ABC):
    def __init__(self, vectorizer: Vectorizer = Vectorizer()):
        self.vectorizer = vectorizer

    @abstractmethod
    def store(self, model_object_instance):
        """Store a single model object instance."""
        pass

    @abstractmethod
    def store_all(self, model_object_instances):
        """Store multiple model object instances."""
        pass

    @abstractmethod
    def get_concept(self, concept_id: str) -> Concept:
        """Retrieve a Concept by ID from the database."""
        pass

    @abstractmethod
    def get_concepts(self) -> Page[Concept]:
        """Retrieve all concepts from the database."""
        pass

    @abstractmethod
    def get_terminology(self, terminology_name: str) -> Terminology:
        """Retrieve a Terminology by name from the database."""
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

    @abstractmethod
    def clear_all(self):
        """Clear all entries in the database."""
        pass

    def import_data_dictionary(self, data_dictionary: DataDictionarySource, terminology_name: str):
        """Imports a data dictionary, generating concepts and embeddings, and stores them in the database.

        :param data_dictionary: Source of variable descriptions and metadata.
        :param terminology_name: Name of the terminology being imported.
        :raises RuntimeError: If the import or transformation fails.
        """
        try:
            objects = self._parse_data_dictionary(data_dictionary, terminology_name)
            self.store_all(objects)
        except Exception as e:
            logger.exception("Failed to import data dictionary.")
            raise RuntimeError(f"Failed to import data dictionary source: {e}")

    def _parse_data_dictionary(
        self, data_dictionary: DataDictionarySource, terminology_name: str
    ) -> List[Union[Concept, Mapping, Terminology]]:
        df = data_dictionary.to_dataframe()
        descriptions = df["description"].tolist()
        vectorizer_name = self.vectorizer.model_name
        variable_to_embedding = data_dictionary.get_embeddings(self.vectorizer)

        terminology = Terminology(name=terminology_name, id=terminology_name)
        objects: List[Union[Concept, Mapping, Terminology]] = [terminology]

        for variable, description in zip(variable_to_embedding.keys(), descriptions):
            concept_id = f"{terminology_name}:{variable}"
            concept = Concept(terminology=terminology, pref_label=variable, concept_identifier=concept_id)
            mapping = Mapping(
                concept=concept,
                text=description,
                embedding=variable_to_embedding[variable],
                sentence_embedder=vectorizer_name,
            )
            objects.extend([concept, mapping])

        return objects

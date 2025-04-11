from typing import List, Optional, Union

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.repository.base import BaseRepository
from datastew.repository.model import Base, Concept, Mapping, MappingResult, Terminology
from datastew.repository.pagination import Page


class SQLLiteRepository(BaseRepository):

    def __init__(self, mode: str = "memory", path: Optional[str] = None, vectorizer: Vectorizer = Vectorizer()):
        """Initializes the repository with a SQLite backend.

        :param mode: Storage mode control, defaults to "memory".
        :param path: File path to SQLite DB when mode is "disk", defaults to None.
        :param vectorizer: An instance of Vectorizer for generating embeddings, defaults to Vectorizer().
        :raises ValueError: Undefined DB mode.
        """
        if mode == "disk":
            self.engine = create_engine(f"sqlite:///{path}")
        # for tests
        elif mode == "memory":
            self.engine = create_engine("sqlite:///:memory:")
        else:
            raise ValueError(f"DB mode {mode} is not defined. Use either disk or memory.")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine, autoflush=False)
        self.session = Session()
        self.vectorizer = vectorizer

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        """Stores a single Terminology, Concept, or Mapping object in the database.

        :param model_object_instance: An instance of Terminology, Concept, or Mapping.
        :raises IOError: If the object cannot be stored (e.g., due to DB errors).
        """
        try:
            self.session.add(model_object_instance)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise IOError(e)

    def store_all(self, model_object_instances: List[Union[Terminology, Concept, Mapping]]):
        """Stores a list of Terminology, Concept, or Mapping objects in the database.

        :param model_object_instances: List of model objects to store.
        """
        self.session.add_all(model_object_instances)
        self.session.commit()

    def import_data_dictionary(self, data_dictionary: DataDictionarySource, terminology_name: str):
        """Imports a data dictionary, generating concepts and embeddings, and stores them in the database.

        :param data_dictionary: Source of variable descriptions and metadata.
        :param terminology_name: Name of the terminology being imported.
        :raises RuntimeError: If the import or transformation fails.
        """
        try:
            model_object_instances: List[Union[Terminology, Concept, Mapping]] = []
            data_frame = data_dictionary.to_dataframe()
            descriptions = data_frame["description"].tolist()
            vectorizer_name = self.vectorizer.model_name
            variable_to_embedding = data_dictionary.get_embeddings(self.vectorizer)
            terminology = Terminology(terminology_name, terminology_name)
            model_object_instances.append(terminology)
            for variable, description in zip(variable_to_embedding.keys(), descriptions):
                concept_id = f"{terminology_name}:{variable}"
                concept = Concept(terminology=terminology, pref_label=variable, concept_identifier=concept_id)
                mapping = Mapping(
                    concept=concept,
                    text=description,
                    embedding=variable_to_embedding[variable],
                    sentence_embedder=vectorizer_name,
                )
                model_object_instances.append(concept)
                model_object_instances.append(mapping)
            self.store_all(model_object_instances)
        except Exception as e:
            raise RuntimeError(f"Failed to import data dictionary source: {e}")

    def get_concepts(self) -> List[Concept]:
        """Retrieves all concepts from the database.

        :return: All stored Concept objects.
        """
        return self.session.query(Concept).all()

    def get_all_terminologies(self) -> List[Terminology]:
        """Retrieves all terminologies from the database.

        :return: All stored Terminology objects.
        """
        return self.session.query(Terminology).all()

    def get_mappings(
        self,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Page[Mapping]:
        """Retrieves a paginated list of mappings, optionally filtered by terminology name and/or sentence embedder.

        :param terminology_name: Name of the terminology to filter by, defaults to None
        :param sentence_embedder: Name of the sentence embedding model to filter by, defaults to None
        :param limit: Maximum number of results to return, defaults to 1000
        :param offset: Number of items to skip, defaults to 0
        :return: A paginated result containing mappings and metadata.
        """
        query = self.session.query(Mapping)

        if terminology_name:
            query = query.join(Concept).join(Terminology).filter(Terminology.name == terminology_name)

        if sentence_embedder:
            query = query.filter(Mapping.sentence_embedder == sentence_embedder)

        total_count = query.count()

        if total_count == 0:
            return Page(items=[], limit=limit, offset=offset, total_count=total_count)

        items = query.offset(offset).limit(limit).all()

        return Page(items=items, limit=limit, offset=offset, total_count=total_count)

    def get_all_sentence_embedders(self) -> List[str]:
        """Retrieves all distinct sentence embedder names used in the mappings.

        :return: Unique sentence embedder identifiers.
        """
        return [embedder for embedder, in self.session.query(Mapping.sentence_embedder).distinct().all()]

    def get_closest_mappings(
        self,
        embedding: List[float],
        similarities: bool = True,
        terminology_name: Optional[str] = None,
        sentence_embedder: Optional[str] = None,
        limit: int = 5,
    ) -> Union[List[Mapping], List[MappingResult]]:
        """Finds the closest mappings by cosine similarity to a given embedding, optionally filtered.

        :param embedding: The target embedding vector to compare against.
        :param similarities: If True, returns MappingResult objects with similarity scores, defaults to True.
        :param terminology_name: Filter by terminology name, defaults to None.
        :param sentence_embedder: Filter by sentence embedder name, defaults to None.
        :param limit: Maximum number of results to return, defaults to 5.
        :return: Closest mappings, with or without similarity scores.
        """
        query = self.session.query(Mapping)

        if terminology_name:
            query = query.join(Concept).join(Terminology).filter(Terminology.name == terminology_name)

        if sentence_embedder:
            query = query.filter(Mapping.sentence_embedder == sentence_embedder)

        mappings = query.all()

        if not mappings:
            return []

        all_embeddings = np.array([mapping.embedding for mapping in mappings])
        target_embedding = np.array(embedding)

        if similarities:
            denominator = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(target_embedding)
            # Substitute denominator with 1e-10 in case the one of the either norms is 0
            denominator = np.where(denominator == 0, 1e-10, denominator)
            similarity = np.dot(all_embeddings, target_embedding) / denominator
            sorted_indices = np.argsort(similarity)[::-1]
            results = [MappingResult(mapping=mappings[i], similarity=similarity[i]) for i in sorted_indices[:limit]]
            return results

        return mappings[:limit]

    def shut_down(self):
        """
        Closes the SQLAlchemy session and releases database resources.
        """
        self.session.close()

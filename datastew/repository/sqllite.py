import logging
from typing import List, Optional, Union

import numpy as np
from sqlalchemy import create_engine, func, inspect
from sqlalchemy.orm import joinedload, sessionmaker
from sqlalchemy.pool import StaticPool

from datastew.embedding import Vectorizer
from datastew.exceptions import ObjectStorageError
from datastew.repository.base import BaseRepository
from datastew.repository.model import Base, Concept, Mapping, MappingResult, Terminology
from datastew.repository.pagination import Page

logger = logging.getLogger(__name__)


class SQLLiteRepository(BaseRepository):

    def __init__(
        self,
        mode: str = "memory",
        path: Optional[str] = None,
        vectorizer: Vectorizer = Vectorizer(),
    ):
        """Initializes the repository with a SQLite backend.

        :param mode: Storage mode control, defaults to "memory".
        :param path: File path to SQLite DB when mode is "disk", defaults to None.
        :param vectorizer: An instance of Vectorizer for generating embeddings, defaults to Vectorizer().
        :raises ValueError: Undefined DB mode.
        """
        super().__init__(vectorizer)
        if mode == "disk":
            self.engine = create_engine(f"sqlite:///{path}")
        # for tests
        elif mode == "memory":
            self.engine = create_engine(
                "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        else:
            raise ValueError(f"DB mode {mode} is not defined. Use either disk or memory.")
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine, autoflush=False)
        self.session = Session()

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        """Stores a single Terminology, Concept, or Mapping object in the database.

        :param model_object_instance: An instance of Terminology, Concept, or Mapping.
        :raises ObjectStorageError: If the object cannot be stored (e.g., due to DB errors).
        """
        try:
            if self._is_duplicate(model_object_instance):
                return

            self.session.add(model_object_instance)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.exception("Failed to store object.")
            raise ObjectStorageError("Failed to store object in the database.", e)

    def store_all(self, model_object_instances: List[Union[Terminology, Concept, Mapping]]):
        """Stores a list of Terminology, Concept, or Mapping objects in the database.

        :param model_object_instances: List of model objects to store.
        """
        for obj in model_object_instances:
            try:
                self.store(obj)
            except ObjectStorageError as e:
                logger.warning(f"Skipping failed insert for {obj}: {e}")

    def get_concept(self, concept_id: str) -> Concept:
        """Retrieves a Concept by its ID.

        :param concept_id: ID of the Concept.
        :raises ValueError: If no Concept with given ID is found.
        :return: Concept object.
        """
        concept = self.session.query(Concept).filter_by(concept_identifier=concept_id).first()
        if concept is None:
            raise ValueError(f"No Concept found with ID: {concept_id}")
        return concept

    def get_concepts(self, terminology_name: Optional[str] = None, offset: int = 0, limit: int = 100) -> Page[Concept]:
        """Retrieves all concepts from the database.

        :return: All stored Concept objects.
        """
        query = self.session.query(Concept).options(joinedload(Concept.terminology))

        if terminology_name:
            query = query.join(Concept.terminology).filter(Terminology.name == terminology_name)

        total_count = query.with_entities(func.count()).scalar()
        concepts = query.offset(offset).limit(limit).all()
        return Page[Concept](items=concepts, limit=limit, offset=offset, total_count=total_count)

    def get_terminology(self, terminology_name: str) -> Terminology:
        """Retrieves a Terminology objects by its name.

        :param terminology_name: Name of the terminology.
        :raises ValueError: If no terminology with the given name is found.
        :return: Terminology object.
        """
        terminology = self.session.query(Terminology).filter_by(name=terminology_name).first()
        if terminology is None:
            raise ValueError(f"No Terminology found with name: {terminology_name}")
        return terminology

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

    def clear_all(self):
        """Deletes all Terminology, Concept, and Mapping entries from the database.

        This method is primarily intended for test environments to ensure a clean
        state before or after test execution. It performs bulk deletions in the
        correct dependency order (Mappings → Concepts → Terminologies) and commits
        the changes.
        """
        self.session.query(Mapping).delete()
        self.session.query(Concept).delete()
        self.session.query(Terminology).delete()
        self.session.commit()

    def _is_duplicate(self, model_object_instance: Union[Terminology, Concept, Mapping]) -> bool:
        """Checks whether an object with the same primary key already exists.

        :param model_object_instance: SQLAlchemy model instance.
        :return: True if a duplicate exists, False otherwise.
        """
        cls = type(model_object_instance)
        pk_attrs = inspect(cls).primary_key

        if len(pk_attrs) != 1:
            logger.warning(
                f"Duplicate check only supports single-column primary keys. Skipping check for {cls.__name__}"
            )
            return False

        pk_attr = pk_attrs[0].name
        pk_value = getattr(model_object_instance, pk_attr)
        existing = self.session.get(cls, pk_value)

        if existing:
            logger.info(f"Skipped storing existing {cls.__name__} with {pk_attr}={pk_value}")
            return True
        return False

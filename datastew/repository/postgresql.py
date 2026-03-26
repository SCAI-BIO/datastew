import logging
from collections import defaultdict
from typing import Optional, Sequence, Union

from sqlalchemy import Engine, delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, joinedload

from datastew.embedding import Vectorizer
from datastew.exceptions import ObjectStorageError
from datastew.repository.model import Base, Concept, Mapping, MappingResult, Terminology
from datastew.repository.pagination import Page

logger = logging.getLogger(__name__)


class PostgreSQLRepository:
    def __init__(self, session: Session, vectorizer: Vectorizer = Vectorizer()):
        """Initializes the repository with an injected database session.

        :param session: The active SQLAlchemy session for database operations.
        :param vectorizer: An instance of Vectorizer for generating embeddings.
        """
        self.session = session
        self.vectorizer = vectorizer

    @staticmethod
    def setup_database(engine: Engine):
        """Creates the pgvector extension and database tables.
        Must be executed once before the repository is used.
        """
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(engine)

    def store(self, objects: Union[Terminology, Concept, Mapping, list[Union[Terminology, Concept, Mapping]]]):
        """Store one or multiple database objects (Terminology, Concept, or Mapping ) in bulk. Handles conflicts by
        updating existing records based on unique constraints.

        :param objects: A single object or a list of objects to be stored.
        :raises ObjectStorageError: If the bulk insert or update operation fails.
        """
        if not isinstance(objects, list):
            objects = [objects]

        if not objects:
            return

        grouped = defaultdict(list)
        for obj in objects:
            data = obj.__dict__.copy()
            data.pop("_sa_instance_state", None)
            grouped[type(obj)].append(data)

        try:
            for model_class in [Terminology, Concept, Mapping]:
                if model_class not in grouped:
                    continue

                stmt = pg_insert(model_class).values(grouped[model_class])

                if model_class == Terminology:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["name"], set_={"short_name": stmt.excluded.short_name}
                    )
                elif model_class == Concept:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["terminology_id", "concept_identifier"],
                        set_={"pref_label": stmt.excluded.pref_label},
                    )
                elif model_class == Mapping:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["concept_id", "vectorizer", "text"],
                        set_={"embedding": stmt.excluded.embedding},
                    )

                self.session.execute(stmt)
        except Exception as e:
            logger.exception("Failed to store objects in bulk.")
            raise ObjectStorageError("Failed to store objects in the database.") from e

    def add_terminology(self, name: str, short_name: str) -> Terminology:
        """Create and store a new Terminology.

        :param name: The full name of the terminology.
        :param short_name: A shortened name or acronym for the terminology.
        :return: The stored Terminology object.
        """
        terminology = Terminology(name=name, short_name=short_name)
        self.store(terminology)
        return self.get_terminology_by_name(name)

    def get_terminology(self, id: int) -> Terminology:
        """Retrieve a Terminology by its database ID.

        :param id: The primary key ID of the terminology.
        :raises ValueError: If no terminology is found with the given ID.
        :return: The matching Terminology object.
        """
        terminology = self.session.get(Terminology, id)
        if terminology is None:
            raise ValueError(f"No Terminology found with ID: {id}")
        return terminology

    def get_terminology_by_name(self, terminology_name: str) -> Terminology:
        """Retrieve a Terminology by its full name.

        :param terminology_name: The exact name of the terminology.
        :raises ValueError: If no terminology is found with the given name.
        :return: The matching Terminology object.
        """
        stmt = select(Terminology).filter_by(name=terminology_name)
        terminology = self.session.execute(stmt).scalar_one_or_none()
        if terminology is None:
            raise ValueError(f"No Terminology found with name: {terminology_name}")
        return terminology

    def get_all_terminologies(self) -> list[Terminology]:
        """Retrieve all stored Terminologies.

        :return: A list of all Terminology object in the database.
        """
        stmt = select(Terminology)
        return list(self.session.execute(stmt).scalars().all())

    def edit_terminology(self, id: int, **kwargs) -> Terminology:
        """Update the attributes of an existing Terminology.

        :param id: The primary key ID of the terminology to update.
        :param kwargs: Key-value pairs representing the attributes to update.
        :return: The updated Terminology object.
        """
        terminology = self.get_terminology(id)
        for key, value in kwargs.items():
            if hasattr(terminology, key):
                setattr(terminology, key, value)
        return terminology

    def delete_terminology(self, id: int):
        """Delete a Terminology from the database by its ID.

        :param id: The primary key ID of the terminology to delete.
        """
        terminology = self.get_terminology(id)
        self.session.delete(terminology)

    def add_concept(self, terminology_id: int, pref_label: str, concept_identifier: str) -> Concept:
        """Create and store a new Concept linked to a specific Terminology.

        :param terminology_id: The ID of the parent Terminology.
        :param pref_label: The preferred label or standard name for the concept.
        :param concept_identifier: The unique identifier code for the concept.
        :return: The stored Concept object.
        """
        concept = Concept(terminology_id=terminology_id, pref_label=pref_label, concept_identifier=concept_identifier)
        self.store(concept)
        return self.get_concept_by_identifier(concept_identifier)

    def get_concept(self, id: int) -> Concept:
        """Retrieve a Concept by its database ID.

        :param id: The primary key ID of the concept.
        :raises ValueError: If no concept is found with the given ID.
        :return: The matching Concept object.
        """
        concept = self.session.get(Concept, id)
        if concept is None:
            raise ValueError(f"No Concept found with ID: {id}")
        return concept

    def get_concept_by_identifier(self, concept_identifier: str) -> Concept:
        """Retrieve a Concept by its unique identifier code.

        :param concept_identifier: The identifier string of the concept.
        :raises ValueError: If no concept is found with the given identifier.
        :return: The matching Concept object.
        """
        stmt = select(Concept).filter_by(concept_identifier=concept_identifier)
        concept = self.session.execute(stmt).scalar_one_or_none()
        if concept is None:
            raise ValueError(f"No Concept found with identifier: {concept_identifier}")
        return concept

    def get_concepts(self, terminology_name: Optional[str] = None, offset: int = 0, limit: int = 100) -> Page[Concept]:
        """Retrieves all concepts from the database.

        :return: All stored Concept objects.
        """
        stmt = select(Concept).options(joinedload(Concept.terminology))
        count_stmt = select(func.count()).select_from(Concept)

        if terminology_name:
            stmt = stmt.join(Concept.terminology).filter(Terminology.name == terminology_name)
            count_stmt = count_stmt.join(Concept.terminology).filter(Terminology.name == terminology_name)

        total_count = self.session.scalar(count_stmt) or 0
        concepts = list(self.session.execute(stmt.offset(offset).limit(limit)).scalars().all())
        return Page[Concept](items=concepts, limit=limit, offset=offset, total_count=total_count)

    def edit_concept(self, id: int, **kwargs) -> Concept:
        """Update the attributes of an existing Concept.

        :param id: The primary key ID of the concept to update.
        :param kwargs: Key-value pairs representing the attributes to update.
        :return: The updated Concept object.
        """
        concept = self.get_concept(id)
        for key, value in kwargs.items():
            if hasattr(concept, key):
                setattr(concept, key, value)
        return concept

    def delete_concept(self, id: int):
        """Delete a Concept from the database by its ID.

        :param id: The primary key ID of the concept to delete.
        """
        concept = self.get_concept(id)
        self.session.delete(concept)

    def add_mapping(
        self,
        concept_id: int,
        text: str,
        embedding: Optional[Sequence[float]] = None,
        vectorizer: Optional[str] = None,
    ) -> Mapping:
        """Create and store a new Mapping for a concept, generating an embedding if not provided.

        :param concept_id: The ID of the concept this mapping belongs to.
        :param text: The source text for the mapping.
        :param embedding: The vectory representation of the text. Generated if None, defaults to None.
        :param vectorizer: The name of the model used for the embedding. Generated if None, defaults to None
        :raises RuntimeError: If the mapping cannot be retrieved after storage.
        :return: The stored Mapping object.
        """
        if embedding is None or vectorizer is None:
            embedding = self.vectorizer.get_embedding(text)
            vectorizer = self.vectorizer.model_name

        mapping = Mapping(concept_id=concept_id, text=text, embedding=embedding, vectorizer=vectorizer)
        self.store(mapping)

        stmt = select(Mapping).filter_by(concept_id=concept_id, vectorizer=vectorizer, text=text)
        saved_mapping = self.session.execute(stmt).scalar_one_or_none()

        if saved_mapping is None:
            raise RuntimeError("Failed to retrieve the mapping after storing it.")

        return saved_mapping

    def get_mapping(self, id: int) -> Mapping:
        """Retrieve a Mapping by its database ID.

        :param id: The primary key ID of the mapping.
        :raises ValueError: If no mapping is found with the given ID.
        :return: The matching Mapping object.
        """
        mapping = self.session.get(Mapping, id)
        if mapping is None:
            raise ValueError(f"No Mapping found with ID: {id}")
        return mapping

    def get_mappings(
        self,
        terminology_name: Optional[str] = None,
        vectorizer: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Page[Mapping]:
        """Retrieves a paginated list of mappings, optionally filtered by terminology name and/or vectorizer.

        :param terminology_name: Name of the terminology to filter by, defaults to None
        :param vectorizer: Name of the vectorizer model to filter by, defaults to None
        :param limit: Maximum number of results to return, defaults to 1000
        :param offset: Number of items to skip, defaults to 0
        :return: A paginated result containing mappings and metadata.
        """
        stmt = select(Mapping)
        count_stmt = select(func.count()).select_from(Mapping)

        if terminology_name:
            stmt = stmt.join(Concept).join(Terminology).filter(Terminology.name == terminology_name)
            count_stmt = count_stmt.join(Concept).join(Terminology).filter(Terminology.name == terminology_name)
        if vectorizer:
            stmt = stmt.filter(Mapping.vectorizer == vectorizer)
            count_stmt = count_stmt.filter(Mapping.vectorizer == vectorizer)

        total_count = self.session.scalar(count_stmt) or 0
        if total_count == 0:
            return Page(items=[], limit=limit, offset=offset, total_count=total_count)

        items = list(self.session.execute(stmt.offset(offset).limit(limit)).scalars().all())
        return Page(items=items, limit=limit, offset=offset, total_count=total_count)

    def edit_mapping(self, id: int, **kwargs) -> Mapping:
        """Update the attributes of an existing Mapping.

        :param id: The primary key ID of the mapping to update.
        :param kwargs: Key-value pairs representing the attributes to update.
        :return: The updated Mapping object.
        """
        mapping = self.get_mapping(id)
        for key, value in kwargs.items():
            if hasattr(mapping, key):
                setattr(mapping, key, value)
        return mapping

    def delete_mapping(self, id: int):
        """Delete a Mapping from the database by its ID.

        :param id: The primary key ID of the mapping to delete.
        """
        mapping = self.get_mapping(id)
        self.session.delete(mapping)

    def get_all_vectorizers(self) -> list[str]:
        """Retrieves all distinct vectorizer names used in the mappings.

        :return: Unique vectorizer identifiers.
        """
        stmt = select(Mapping.vectorizer).distinct()
        return list(self.session.execute(stmt).scalars().all())

    def get_closest_mappings(
        self,
        embedding: Sequence[float],
        similarities: bool = True,
        terminology_name: Optional[str] = None,
        vectorizer: Optional[str] = None,
        limit: int = 5,
    ) -> Union[list[Mapping], list[MappingResult]]:
        """Finds the closest mappings by cosine similarity to a given embedding, optionally filtered.

        :param embedding: The target embedding vector to compare against.
        :param similarities: If True, returns MappingResult objects with similarity scores, defaults to True.
        :param terminology_name: Filter by terminology name, defaults to None.
        :param vectorizer: Filter by vectorizer name, defaults to None.
        :param limit: Maximum number of results to return, defaults to 5.
        :return: Closest mappings, with or without similarity scores.
        """
        stmt = select(Mapping, Mapping.embedding.cosine_distance(embedding).label("distance"))

        if terminology_name:
            stmt = stmt.join(Concept).join(Terminology).filter(Terminology.name == terminology_name)

        if vectorizer:
            stmt = stmt.filter(Mapping.vectorizer == vectorizer)

        results = self.session.execute(stmt.order_by("distance").limit(limit)).all()

        if similarities:
            return [MappingResult(mapping=row.Mapping, similarity=1 - row.distance) for row in results]
        return [row.Mapping for row in results]

    def clear_all(self):
        """Deletes all Terminology, Concept, and Mapping entries from the database.

        This method is primarily intended for test environments to ensure a clean
        state before or after test execution. It performs bulk deletions in the
        correct dependency order (Mappings → Concepts → Terminologies) and commits
        the changes.
        """
        self.session.execute(delete(Mapping))
        self.session.execute(delete(Concept))
        self.session.execute(delete(Terminology))

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Union

from sqlalchemy import create_engine, func, inspect, text
from sqlalchemy.orm import joinedload, sessionmaker

from datastew.embedding import Vectorizer
from datastew.exceptions import ObjectStorageError
from datastew.repository.base import BaseRepository
from datastew.repository.model import Base, Concept, Mapping, MappingResult, Terminology
from datastew.repository.pagination import Page

logger = logging.getLogger(__name__)


class PostgreSQLRepository(BaseRepository):
    def __init__(
        self,
        connection_string: str,
        vectorizer: Vectorizer = Vectorizer(),
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
    ):
        """Initializes the repository with a PostgreSQL backend.

        :param connection_string: Full DB URI (e.g., 'postgresql://user:pass@localhost/dbname').
        :param vectorizer: An instance of Vectorizer for generating embeddings,
            defaults to Vectorizer("FremyCompany/BioLORD-2023").
        :param pool_size: The number of connections to keep in the connection pool.
        :param max_overflow: The maximum number of connections to allow in overflow.
        :param pool_timeout: The maximum time (in seconds) to wait for a connection from
            the pool before raising an exception.
        """
        super().__init__(vectorizer)
        self.engine = create_engine(
            connection_string, pool_size=pool_size, max_overflow=max_overflow, pool_timeout=pool_timeout
        )
        self._initialize_pgvector()
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
            self.store(obj)

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
        embedding: Sequence[float],
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
        query = self.session.query(Mapping, Mapping.embedding.cosine_distance(embedding).label("distance"))

        if terminology_name:
            query = query.join(Concept).join(Terminology).filter(Terminology.name == terminology_name)

        if sentence_embedder:
            query = query.filter(Mapping.sentence_embedder == sentence_embedder)

        results = query.order_by("distance").limit(limit).all()

        if similarities:
            return [MappingResult(mapping=m, similarity=1 - d) for m, d in results]
        return [m for m, _ in results]

    def shut_down(self):
        """
        Closes the SQLAlchemy session and releases database resources.
        """
        self.session.close()
        self.engine.dispose()

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

    def import_from_jsonl(self, jsonl_path: str, object_type: str, chunk_size: int = 100):
        """Imports data from a JSONL file and stores it in the database in chunks.

        :param jsonl_path: Path to the JSONL file containing the data to be imported.
        :param object_type: The type of objects to import. Must be one of: "terminology", "concept", or "mapping"
        :param chunk_size: Number of objects to store in a single batch, defaults to 100.
        :raises ValueError: If the JSON is malformed or required fields are missing.
        :raises ValueError: If the provided `object_type` is unsupported.
        :raises RuntimeError: If the file cannot be found.
        :raises RuntimeError: If a general I/O or database error occurs during import.
        """
        buffer = []

        try:
            with open(jsonl_path, "r", encoding="utf-8") as file:
                for idx, line in enumerate(file):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {idx + 1}: {e}")

                    obj = self._deserialize_object(object_type, data)
                    buffer.append(obj)

                    if len(buffer) >= chunk_size:
                        self.store_all(buffer)
                        buffer = []

                if buffer:
                    self.store_all(buffer)

        except FileNotFoundError:
            raise RuntimeError(f"File not found: {jsonl_path}")
        except Exception as e:
            raise RuntimeError(f"Error while import {object_type}: {e}")

    def _deserialize_object(self, object_type: str, data: Dict[str, Any]) -> Union[Terminology, Concept, Mapping]:
        """Deserializes a JSON object into an SQLAlchemy model instance, resolving any required relationships.

        :param object_type: The type of object to deserialize.
        :param data: The dictionary representing the object, as loaded from a JSONL line.
        :raises ValueError: If a related object (e.g., a referenced concept or terminology) cannot be found.
        :raises ValueError: If the object_type is not one of the supported values.
        :return: An instance of the appropriate SQLAlchemy model.
        """
        if object_type == "terminology":
            return Terminology(**data)

        elif object_type == "concept":
            terminology = self.session.get(Terminology, data["terminology_id"])
            if not terminology:
                raise ValueError(f"Terminology with ID {data['terminology_id']} not found")
            return Concept(
                terminology=terminology,
                pref_label=data["pref_label"],
                concept_identifier=data["concept_identifier"],
            )

        elif object_type == "mapping":
            concept = self.session.get(Concept, data["concept_identifier"])
            if not concept:
                raise ValueError(f"Concept with ID {data['concept_identifier']} not found")

            embedding = data.get("embedding")
            sentence_embedder = data.get("sentence_embedder")

            if (embedding is None or sentence_embedder is None) and self.vectorizer:
                embedding = self.vectorizer.get_embedding(data["text"])
                sentence_embedder = self.vectorizer.model_name

            return Mapping(
                concept=concept,
                text=data["text"],
                embedding=embedding,
                sentence_embedder=sentence_embedder,
            )

        else:
            raise ValueError(f"Unsupported object_type: {object_type}")

    def _initialize_pgvector(self):
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

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

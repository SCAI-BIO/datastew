import logging
from typing import List, Optional, Sequence, Union

from sqlalchemy import create_engine, func, inspect, text
from sqlalchemy.orm import joinedload, sessionmaker

from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.repository.base import BaseRepository
from datastew.repository.model import Base, Concept, Mapping, MappingResult, Terminology
from datastew.repository.pagination import Page

logger = logging.getLogger(__name__)


class PostgreSQLRepository(BaseRepository):
    def __init__(self, connection_string: str, vectorizer: Vectorizer = Vectorizer()):
        """Initializes the repository with a PostgreSQL backend.

        :param connection_string: Full DB URI (e.g., 'postgresql://user:pass@localhost/dbname').
        :param vectorizer: An instance of Vectorizer for generating embeddings,
            defaults to Vectorizer("FremyCompany/BioLORD-2023").
        """
        self.vectorizer = vectorizer
        self.engine = create_engine(connection_string, pool_size=10, max_overflow=20, pool_timeout=30)
        self.dialect = inspect(self.engine).dialect.name
        self.initialize_pgvector()
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine, autoflush=False)
        self.session = Session()

    def initialize_pgvector(self):
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        """Stores a single Terminology, Concept, or Mapping object in the database.

        :param model_object_instance: An instance of Terminology, Concept, or Mapping.
        :raises IOError: If the object cannot be stored (e.g., due to DB errors).
        """
        try:
            if self._is_duplicate(model_object_instance):
                return

            self.session.add(model_object_instance)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            logger.exception("Failed to store object.")
            raise IOError(e)

    def store_all(self, model_object_instances: List[Union[Terminology, Concept, Mapping]]):
        """Stores a list of Terminology, Concept, or Mapping objects in the database.

        :param model_object_instances: List of model objects to store.
        """
        for obj in model_object_instances:
            try:
                self.store(obj)
            except IOError:
                logger.warning(f"Skipping failed insert for {obj}")

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

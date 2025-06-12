import json
from typing import Any, Optional, Sequence

from pgvector.sqlalchemy import Vector
from sqlalchemy import TEXT, Column, Dialect, ForeignKey, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql.type_api import TypeEngine

Base = declarative_base()


class VectorType(TypeDecorator):
    impl = TEXT
    cache_ok = True

    @property
    def comparator_factory(self):
        return Vector.comparator_factory

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        """Returns the appropriate SQLAlchemy type descriptor based on the database dialect.

        :param dialect: The SQLAlchemy Dialect in use (e.g., 'postgresql', 'sqlite').
        :return: A type descriptor suitable for the target database dialect.
        """
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(768))
        else:
            return dialect.type_descriptor(TEXT())

    def process_bind_param(self, value: Any | None, dialect: Dialect) -> Any:
        """Serializes the Python object to a format suitable for storage in the database.

        :param value: The Python value to be stored in the database (typically a list of floats).
        :param dialect: The SQLAlchemy Dialect in use.
        :return: A database-compatible representation of the value.
        """
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        else:
            return json.dumps(value)

    def process_result_value(self, value: Any | None, dialect: Dialect) -> Any | None:
        """Deserializes the database value back into a Python object

        :param value: The value fetched from the database.
        :param dialect: The SQLAlchemy Dialect in use.
        :return: The deserialized Python object (typically a list of floats).
        """
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        else:
            return json.loads(value)


class Terminology(Base):
    __tablename__ = "terminology"
    id = Column(String, primary_key=True)
    name = Column(String)

    def __init__(self, name: str, id: str):
        self.name = name
        self.id = id


class Concept(Base):
    __tablename__ = "concept"
    concept_identifier = Column(String, primary_key=True)
    pref_label = Column(String)
    terminology_id = Column(String, ForeignKey("terminology.id"))
    terminology = relationship("Terminology")
    uuid = Column(String)

    def __init__(self, terminology: Terminology, pref_label: str, concept_identifier: str, id: Optional[str] = None):
        self.terminology = terminology
        self.pref_label = pref_label
        # should be unique
        self.concept_identifier = concept_identifier
        # enforced to be unique
        self.id = id


class Mapping(Base):
    __tablename__ = "mapping"

    id = Column(Integer, primary_key=True, autoincrement=True)
    concept_identifier = Column(String, ForeignKey("concept.concept_identifier"))
    concept = relationship("Concept")
    text = Column(Text)
    embedding = Column(VectorType)
    sentence_embedder = Column(Text)

    def __init__(
        self,
        concept: Concept,
        text: str,
        embedding: Optional[Sequence[float]] = None,
        sentence_embedder: Optional[str] = None,
        id: Optional[str] = None,
    ):
        self.concept = concept
        self.text = text
        self.embedding = embedding
        self.sentence_embedder = sentence_embedder
        self.id = id

    def __str__(self):
        return (
            f"{self.concept.terminology.name} > "
            f"{self.concept.concept_identifier} : "
            f"{self.concept.pref_label} | {self.text}"
        )


class MappingResult:

    def __init__(self, mapping: Mapping, similarity: float):
        self.mapping = mapping
        self.similarity = similarity

    def __str__(self):
        return f"{self.mapping} | Similarity: {self.similarity}"

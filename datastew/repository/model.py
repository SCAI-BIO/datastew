from dataclasses import dataclass
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Terminology(Base):
    __tablename__ = "terminology"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    short_name: Mapped[str] = mapped_column(String, nullable=False, unique=True)

    concepts: Mapped[list["Concept"]] = relationship(back_populates="terminology", cascade="all, delete-orphan")


class Concept(Base):
    __tablename__ = "concept"
    __table_args__ = (UniqueConstraint("terminology_id", "concept_identifier", name="uix_term_concept"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    concept_identifier: Mapped[str] = mapped_column(String, nullable=False)
    pref_label: Mapped[str] = mapped_column(String, nullable=False)

    terminology_id: Mapped[int] = mapped_column(ForeignKey("terminology.id", ondelete="CASCADE"))
    terminology: Mapped["Terminology"] = relationship(back_populates="concepts")

    mappings: Mapped[list["Mapping"]] = relationship(back_populates="concept", cascade="all, delete-orphan")


class Mapping(Base):
    __tablename__ = "mapping"
    __table_args__ = (UniqueConstraint("concept_id", "vectorizer", "text", name="uix_mapping_concept_embedder_text"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(768), nullable=False)
    vectorizer: Mapped[str] = mapped_column(String, nullable=False)

    concept_id: Mapped[int] = mapped_column(ForeignKey("concept.id", ondelete="CASCADE"))
    concept: Mapped["Concept"] = relationship(back_populates="mappings")

    def __str__(self):
        return (
            f"{self.concept.terminology.name} > "
            f"{self.concept.concept_identifier} : "
            f"{self.concept.pref_label} | {self.text}"
        )


@dataclass
class MappingResult:
    mapping: Mapping
    similarity: float

    def __str__(self):
        return f"{self.mapping} | Similarity: {self.similarity}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": {
                "concept_identifier": self.mapping.concept.concept_identifier,
                "pref_label": self.mapping.concept.pref_label,
                "terminology": {
                    "short_name": self.mapping.concept.terminology.short_name,
                    "name": self.mapping.concept.terminology.name,
                },
            },
            "text": self.mapping.text,
            "similarity": self.similarity,
        }

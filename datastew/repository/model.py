import json

import numpy as np
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Terminology(Base):
    __tablename__ = 'terminology'
    id = Column(String, primary_key=True)
    name = Column(String)

    def __init__(self, name: str, id: str) -> object:
        self.name = name
        self.id = id


class Concept(Base):
    __tablename__ = 'concept'
    concept_identifier = Column(String, primary_key=True)
    pref_label = Column(String)
    terminology_id = Column(String, ForeignKey('terminology.id'))
    terminology = relationship("Terminology")
    uuid = Column(String)

    def __init__(self, terminology: Terminology, pref_label: str, concept_identifier: str, id: str = None) -> object:
        self.terminology = terminology
        self.pref_label = pref_label
        # should be unique
        self.concept_identifier = concept_identifier
        # enforced to be unique
        self.id = id


class Mapping(Base):
    __tablename__ = 'mapping'
    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-incrementing primary key
    concept_identifier = Column(String, ForeignKey('concept.concept_identifier'))
    concept = relationship("Concept")
    text = Column(Text)
    embedding_json = Column(Text)
    sentence_embedder = Column(Text)

    def __init__(self, concept: Concept, text: str, embedding: list, sentence_embedder: str) -> object:
        self.concept = concept
        self.text = text
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        self.embedding_json = json.dumps(embedding)  # Store embedding as JSON
        self.sentence_embedder = sentence_embedder

    def __str__(self):
        return f"{self.concept.terminology.name} > {self.concept.concept_identifier} : {self.concept.pref_label} | {self.text}"

    @property
    def embedding(self):
        return json.loads(self.embedding_json)

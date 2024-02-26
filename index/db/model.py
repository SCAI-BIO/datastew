import json

import numpy as np
from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

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
    id = Column(String, primary_key=True)
    name = Column(String)
    terminology_id = Column(String, ForeignKey('terminology.id'))
    terminology = relationship("Terminology")

    def __init__(self, terminology: Terminology, name: str, id: str) -> object:
        self.terminology = terminology
        self.name = name
        self.id = id


class Mapping(Base):
    __tablename__ = 'mapping'
    id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-incrementing primary key
    concept_id = Column(String, ForeignKey('concept.id'))
    concept = relationship("Concept")
    text = Column(Text)
    embedding_json = Column(Text)

    def __init__(self, concept: Concept, text: str, embedding: list) -> object:
        self.concept = concept
        self.text = text
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        self.embedding_json = json.dumps(embedding)  # Store embedding as JSON

    def __str__(self):
        return f"{self.concept_id} : {self.concept.name} | {self.text}"

    @property
    def embedding(self):
        return json.loads(self.embedding_json)

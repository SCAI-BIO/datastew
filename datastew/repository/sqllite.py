import random
from typing import List, Union, Optional

import numpy as np
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from datastew.repository.base import BaseRepository
from datastew.repository.model import Base, Concept, Mapping, Terminology


class SQLLiteRepository(BaseRepository):

    def __init__(self, mode="memory", path=None):
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

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        try:
            self.session.add(model_object_instance)
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise IOError(e)

    def store_all(self, model_object_instances: List[Union[Terminology, Concept, Mapping]]):
        self.session.add_all(model_object_instances)
        self.session.commit()

    def get_all_concepts(self) -> List[Concept]:
        return self.session.query(Concept).all()

    def get_all_terminologies(self) -> List[Terminology]:
        return self.session.query(Terminology).all()

    def get_mappings(self, terminology_name: Optional[str] = None, limit=1000) -> List[Mapping]:
        if not terminology_name:
            # Determine the total count of mappings in the database
            total_count = self.session.query(func.count(Mapping.id)).scalar()
            # Generate random indices for the subset of embeddings
            random_indices = random.sample(range(total_count), min(limit, total_count))
            # Query for mappings corresponding to the random indices
            mappings = self.session.query(Mapping).filter(Mapping.id.in_(random_indices)).all()
        else:
            query = (
            self.session.query(Mapping)
            .join(Concept)
            .join(Terminology)
            .filter(Terminology.name == terminology_name)
            )

            total_count = query.count()
            if total_count == 0:
                return []
            
            mappings = query.all()
            mappings = random.sample(mappings, min(limit, len(mappings))) if mappings else []
        return mappings
    
    def get_all_sentence_embedders(self) -> List[str]:
        return [embedder for embedder, in self.session.query(Mapping.sentence_embedder).distinct().all()]

    def get_closest_mappings(self, embedding: List[float], limit=5):
        mappings = self.session.query(Mapping).all()
        all_embeddings = np.array([mapping.embedding for mapping in mappings])
        similarities = np.dot(all_embeddings, np.array(embedding)) / (
                np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(np.array(embedding)))
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_mappings = [mappings[i] for i in sorted_indices[:limit]]
        sorted_similarities = [similarities[i] for i in sorted_indices[:limit]]
        return sorted_mappings, sorted_similarities

    def shut_down(self):
        self.session.close()

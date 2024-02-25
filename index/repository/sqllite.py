import numpy as np

from typing import Union, List

from sqlalchemy import create_engine, func, cast, String
from sqlalchemy.orm import sessionmaker, aliased
from index.db.model import Base, Terminology, Concept, Mapping
from index.repository.base import BaseRepository


class SQLLiteRepository(BaseRepository):

    def __init__(self, mode="disk"):
        if mode == "disk":
            self.engine = create_engine('sqlite:///index.db')
        # for tests
        elif mode == "memory":
            self.engine = create_engine('sqlite:///:memory:')
        else:
            raise ValueError(f'DB mode {mode} is not defined. Use either disk or memory.')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def store(self, model_object_instance: Union[Terminology, Concept, Mapping]):
        self.session.add(model_object_instance)
        self.session.commit()

    def store_all(self, model_object_instances: List[Union[Terminology, Concept, Mapping]]):
        self.session.add_all(model_object_instances)
        self.session.commit()

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

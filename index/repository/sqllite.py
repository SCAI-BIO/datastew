from typing import Union, List

from sqlalchemy import create_engine, func
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

    def get_closest_mappings(self, embedding, limit=5):
        # Calculate Euclidean distance for each embedding in the database
        mapping_embedding_alias = aliased(Mapping)

        # Construct the SQL query dynamically based on the length of the embedding
        distance_expression = sum(
            func.pow(func.json_extract(mapping_embedding_alias.embedding_json, f"$[{i}]") - embedding[i], 2)
            for i in range(len(embedding))
        ).label('distance')

        distances_and_mappings_query = self.session.query(
            mapping_embedding_alias,
            distance_expression
        )

        distances_and_mappings = distances_and_mappings_query.all()
        print(distances_and_mappings)
        # Sort results based on distances
        sorted_results = sorted(distances_and_mappings, key=lambda x: x[1])

        # Extract deserialized mappings and distances
        closest_mappings = [(mapping, float(distance)) for mapping, distance in sorted_results[:limit]]

        # Extract Concept objects for mappings
        concept_ids = [mapping.concept_id for mapping, _ in closest_mappings]
        concepts = self.session.query(Concept).filter(Concept.id.in_(concept_ids)).all()

        # Create mapping objects with corresponding Concept objects
        closest_mapping_objects = []
        for mapping, distance in closest_mappings:
            concept = next(concept for concept in concepts if concept.id == mapping.concept_id)
            closest_mapping_objects.append((mapping, concept, distance))

        # Extract mapping objects and distances
        result_mappings = [mapping for mapping, _, _ in closest_mapping_objects]
        distances = [distance for _, _, distance in closest_mapping_objects]

        return result_mappings, distances

    def shut_down(self):
        self.session.close()

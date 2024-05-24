import logging

import requests

from datastew.repository.model import Terminology, Concept, Mapping
from datastew.embedding import EmbeddingModel
from datastew.repository.base import BaseRepository


class OLSTerminologyImportTask:

    def __init__(self, repository: BaseRepository, embedding_model: EmbeddingModel, ontology_name: str,
                 ontology_id: str, ols_api_base_url:str = 'https://www.ebi.ac.uk/ols4/api/', page_size:int = 200):
        logging.getLogger().setLevel(logging.INFO)
        self.repository = repository
        self.embedding_model = embedding_model
        self.ontology_id = ontology_id
        self.ontology_name = ontology_name
        self.OLS_BASE_URL = ols_api_base_url
        self.page_size = page_size
        self.num_pages = self.get_number_of_pages()
        self.current_page = 0
        self.terminology = Terminology(self.ontology_name, self.ontology_id)
        self.repository.store(self.terminology)
        
    def get_number_of_pages(self):
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?size={self.page_size}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data['page']['totalPages']
        except Exception as e:
            logging.error(f"Failed to fetch concepts and descriptions from OLS: {str(e)}")

    def process(self):
        # index starts at 0
        while self.current_page < self.num_pages - 1:
            url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?page={self.current_page}&size={self.page_size}"
            logging.info(f"Processing page {self.current_page}/{self.num_pages}.")
            try:
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                terms = data["_embedded"]["terms"]
                identifiers = [term["obo_id"] for term in terms]
                labels = [term["label"] for term in terms]
                descriptions = []
                for term in terms:
                    if "description" in term and len(term["description"]) > 0:
                        descriptions.append(term["description"])
                    else:
                        descriptions.append(term["label"])
                embeddings = self.embedding_model.get_embeddings(descriptions)
                for identifier, label, description, embedding in zip(identifiers, labels, descriptions, embeddings):
                    concept = Concept(self.terminology, label, identifier)
                    mapping = Mapping(concept, description, embedding)
                    self.repository.store(concept)
                    self.repository.store(mapping)
            except Exception as e:
                logging.error(f"Failed to fetch concepts and descriptions from OLS: {str(e)}")
            finally:
                self.current_page = self.current_page + 1




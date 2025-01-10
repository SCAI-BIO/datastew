import logging

import requests

from datastew.embedding import EmbeddingModel
from datastew.process.json_adapter import WeaviateJsonConverter
from datastew.repository.base import BaseRepository
from datastew.repository.model import Concept, Mapping, Terminology


class OLSTerminologyImportTask:

    def __init__(self, embedding_model: EmbeddingModel, ontology_name: str,
                 ontology_id: str, ols_api_base_url: str = "https://www.ebi.ac.uk/ols4/api/",
                 page_size: int = 200):
        logging.getLogger().setLevel(logging.INFO)
        self.embedding_model = embedding_model
        self.ontology_id = ontology_id
        self.ontology_name = ontology_name
        self.terminology = Terminology(self.ontology_name, self.ontology_id)
        self.OLS_BASE_URL = ols_api_base_url
        self.page_size = page_size
        self.num_pages = self.get_number_of_pages()
        self.current_page = 0

    def get_number_of_pages(self):
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?size={self.page_size}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data["page"]["totalPages"]
        except Exception as e:
            logging.error(f"Failed to fetch concepts and descriptions from OLS: {str(e)}")

    def __process_page(self, page: int) -> ([Concept], [Mapping]):
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?page={page}&size={self.page_size}"
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
            model_name = self.embedding_model.get_model_name()
            concepts = []
            mappings = []
            for identifier, label, description, embedding in zip(identifiers, labels, descriptions, embeddings):
                concept = Concept(self.terminology, label, identifier)
                concepts.append(concept)
                mapping = Mapping(concept, description, embedding, model_name)
                mappings.append(mapping)
            return concepts, mappings
        except Exception as e:
            logging.error(f"Failed to fetch concepts and descriptions from OLS for page {page}: {str(e)}")

    def process_to_weaviate(self, repository: BaseRepository):
        """
        Fetches concepts and descriptions from the OLS API and stores them in a Weaviate repository.

        :param repository: The Weaviate repository to store the concepts and mappings.

        :return: None
        """
        # init terminology
        repository.store(self.terminology)
        # index starts at 0
        while self.current_page < self.num_pages - 1:
            concepts, mappings = self.__process_page(self.current_page)
            # concepts have to be stored always first before the corresponding mappings
            for idx, concept in enumerate(concepts):
                repository.store(concept)
                repository.store(mappings[idx])
            self.current_page += 1

    def process_to_weaviate_json(self, dest_path: str):
        """
        Fetches concepts and descriptions from the OLS API and stores them in a JSON file.
        """
        converter = WeaviateJsonConverter(dest_path)
        # start with the terminology
        with open(f"{dest_path}/terminology.json", "w") as f:
            f.write(self.terminology.to_json())
        # for each page
        while self.current_page < self.num_pages - 1:
            concepts, mappings = self.__process_page(self.current_page)
            with open(f"{dest_path}/concepts_{self.current_page}.json", "w") as f:
                for concept in concepts:
                    f.write(concept.to_json())
            with open(f"{dest_path}/mappings_{self.current_page}.json", "w") as f:
                for mapping in mappings:
                    f.write(mapping.to_json())
            self.current_page += 1

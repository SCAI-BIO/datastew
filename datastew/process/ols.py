import logging
import os
from typing import List, Tuple

import requests

from datastew.embedding import Vectorizer
from datastew.repository.base import BaseRepository
from datastew.repository.model import Concept, Mapping, Terminology


class OLSTerminologyImportTask:

    def __init__(
        self,
        vectorizer: Vectorizer,
        ontology_name: str,
        ontology_id: str,
        ols_api_base_url: str = "https://www.ebi.ac.uk/ols4/api/",
        page_size: int = 200,
        use_weaviate_vectorizer: bool = False,
    ):
        logging.getLogger().setLevel(logging.INFO)
        self.vectorizer = vectorizer
        self.ontology_id = ontology_id
        self.ontology_name = ontology_name
        self.terminology = Terminology(self.ontology_name, self.ontology_id)
        self.OLS_BASE_URL = ols_api_base_url
        self.page_size = page_size
        self.use_weaviate_vectorizer = use_weaviate_vectorizer
        self.num_pages = self.get_number_of_pages()
        self.current_page = 0

    def get_number_of_pages(self) -> int:
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?size={self.page_size}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data["page"]["totalPages"]
        except Exception as e:
            logging.error(f"Failed to fetch concepts and descriptions from OLS: {str(e)}")
            return 0

    def __process_page(self, page: int) -> Tuple[List[Concept], List[Mapping]]:
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
                description = term.get("description")
                if isinstance(description, list) and description:
                    descriptions.append(description[0])
                elif isinstance(description, str) and description:
                    descriptions.append(description)
                else:
                    descriptions.append(term["label"])
            if not self.use_weaviate_vectorizer:
                embeddings = self.vectorizer.get_embeddings(descriptions)
                model_name = self.vectorizer.model_name
            concepts = []
            mappings = []
            if not self.use_weaviate_vectorizer:
                for identifier, label, description, embedding in zip(identifiers, labels, descriptions, embeddings):
                    concept = Concept(self.terminology, label, identifier)
                    concepts.append(concept)
                    mapping = Mapping(concept, description, embedding, model_name)
                    mappings.append(mapping)
            else:
                for identifier, label, description in zip(identifiers, labels, descriptions):
                    concept = Concept(self.terminology, label, identifier)
                    concepts.append(concept)
                    mapping = Mapping(concept, description)
                    mappings.append(mapping)
            return concepts, mappings
        except Exception as e:
            logging.error(f"Failed to fetch concepts and descriptions from OLS for page {page}: {str(e)}")
            return [], []

    def process_to_repository(self, repository: BaseRepository):
        """
        Fetches concepts and descriptions from the OLS API and stores them in a repository.

        :param repository: The repository to store the concepts and mappings.

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

    def process_to_json(self, dest_path: str):
        """
        Fetches concepts and descriptions from the OLS API and stores them in a JSON file.
        """
        os.makedirs(dest_path, exist_ok=True)
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

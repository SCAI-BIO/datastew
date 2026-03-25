import json
import logging
import os
from typing import Sequence, Tuple

import requests

from datastew.embedding import Vectorizer
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import Concept, Mapping, Terminology


class OlsClient:

    def __init__(
        self,
        vectorizer: Vectorizer,
        ontology_id: str,
        ols_api_base_url: str = "https://www.ebi.ac.uk/ols4/api/",
        page_size: int = 200,
    ):
        """Initializes the OlsClient to fetch ontology data from the OLS API.

        :param vectorizer: The model used to convert concept descriptions into embeddings.
        :param ontology_id: The identifier of the target ontology in OLS (e.g., 'ncit', 'efo').
        :param ols_api_base_url: The base URL for the OLS API, defaults to "https://www.ebi.ac.uk/ols4/api/"
        :param page_size: The number of terms to fetch per API request, defaults to 200
        """
        logging.getLogger().setLevel(logging.INFO)
        self.vectorizer = vectorizer
        self.ontology_id = ontology_id
        self.OLS_BASE_URL = ols_api_base_url
        self.page_size = page_size

        self.ontology_name = self.get_ontology_name()
        self.ontology_short_name = self.get_ontology_short_name()
        self.num_pages = self.get_number_of_pages()
        self.current_page = 0

    def get_ontology_name(self) -> str:
        """Retrieves the full title of the ontology from the OLS API.

        :return: The full name/title of the ontology.
        """
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data["config"]["title"]
        except Exception as e:
            logging.error(f"Failed to fetch ontology name from OLS: {str(e)}")
            raise

    def get_ontology_short_name(self) -> str:
        """Retrieves the preferred prefix or short name of the ontology.

        :return: The short name of the ontology.
        """
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data["config"].get(
                "preferredPrefix", data["config"].get("prefferedPrefix", self.ontology_id.upper())
            )
        except Exception as e:
            logging.error(f"Failed to fetch ontology short name from OLS: {str(e)}")
            raise

    def get_number_of_pages(self) -> int:
        """Calculates the total number of pages required to fetch all terms based on the configured page size.

        :return: The total number of pages available for the ontology.
        """
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?size={self.page_size}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data["page"]["totalPages"]
        except Exception as e:
            logging.error(f"Failed to fetch concepts and descriptions from OLS: {str(e)}")
            raise

    def process_to_repository(self, repository: PostgreSQLRepository) -> None:
        """Fetches concepts and descriptions from the OLS API and stores them in a repository.

        :param repository: The repository to store the concepts and mappings.
        :return: None
        """
        # init terminology
        repository.store([Terminology(name=self.ontology_name, short_name=self.ontology_short_name)])
        term_db = repository.get_terminology_by_name(self.ontology_name)

        while self.current_page < self.num_pages:
            identifiers, labels, descriptions, embeddings = self._fetch_page_data(self.current_page)

            if not identifiers:
                self.current_page += 1
                continue

            concepts = []
            for identifier, label in zip(identifiers, labels):
                concepts.append(
                    Concept(
                        terminology_id=term_db.id,
                        pref_label=label,
                        concept_identifier=f"{self.ontology_short_name}:{identifier}",
                    )
                )
                repository.store(concepts)

                concept_identifiers = [c.concept_identifier for c in concepts]
                saved_concepts = (
                    repository.session.query(Concept.id, Concept.concept_identifier)
                    .filter(Concept.concept_identifier.in_(concept_identifiers), Concept.terminology_id == term_db.id)
                    .all()
                )
                concept_map = {identifier: c_id for c_id, identifier in saved_concepts}

                mappings = []
                model_name = self.vectorizer.model_name
                for identifier, description, embedding in zip(identifiers, descriptions, embeddings):
                    cid_str = f"{self.ontology_short_name}:{identifier}"
                    if cid_str in concept_map:
                        mappings.append(
                            Mapping(
                                concept_id=concept_map[cid_str],
                                text=description,
                                embedding=embedding,
                                vectorizer=model_name,
                            )
                        )
                repository.store(mappings)
                self.current_page += 1

    def process_to_json(self, dest_path: str) -> None:
        """Fetches concepts and descriptions from the OLS API and stores them in a JSON files to a specified directory.

        :param dest_path: The directory path where the resulting JSON files will be saved.
        :return: None.
        """
        os.makedirs(dest_path, exist_ok=True)
        terminology_data = {"name": self.ontology_name, "short_name": self.ontology_short_name}
        # start with the terminology
        with open(f"{dest_path}/terminology.json", "w", encoding="utf-8") as f:
            json.dump(terminology_data, f, indent=2)
        # for each page
        while self.current_page < self.num_pages:
            identifiers, labels, descriptions, embeddings = self._fetch_page_data(self.current_page)

            concepts = []
            mappings = []
            for identifier, label, desc, emb in zip(identifiers, labels, descriptions, embeddings):
                cid = f"{self.ontology_short_name}:{identifier}"
                concepts.append(
                    {
                        "concept_identifier": cid,
                        "pref_label": label,
                        "terminology_short_name": self.ontology_short_name,
                    }
                )
                mappings.append(
                    {
                        "concept_identifier": cid,
                        "text": desc,
                        "embedding": emb,
                        "vectorizer": self.vectorizer.model_name,
                    }
                )

            with open(os.path.join(dest_path, f"concepts_{self.current_page}.json"), "w", encoding="utf-8") as f:
                json.dump(concepts, f, indent=2)
            with open(os.path.join(dest_path, f"mappings_{self.current_page}.json"), "w", encoding="utf-8") as f:
                json.dump(mappings, f, indent=2)

            self.current_page += 1

    def _fetch_page_data(self, page: int) -> Tuple[list[str], list[str], list[str], Sequence[Sequence[float]]]:
        """Retrieves a single page of terms from the OLS API and computes text embeddings for their descriptions.

        :param page: The page index to fetch from the API.
        :return: A tuple containing four elements:
                - A list of concept identifiers (obo_id).
                - A list of preferred labels.
                - A list of text descriptions.
                - A sequence of computed embeddings for the descriptions.
        """
        url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?page={page}&size={self.page_size}"
        logging.info(f"Processing page {self.current_page}/{self.num_pages}.")

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            terms = data.get("_embedded", {}).get("terms", [])

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

            embeddings = self.vectorizer.get_embeddings(descriptions)
            return identifiers, labels, descriptions, embeddings
        except Exception as e:
            logging.error(f"Failed to fetch concepts and descriptions from OLS for page {page}: {str(e)}")
            raise

import json
import logging
import os
from typing import Optional, Sequence, Tuple

import requests

from datastew.embedding import Vectorizer
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import Concept, Mapping, Terminology

logger = logging.getLogger(__name__)


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
        self.vectorizer = vectorizer
        self.ontology_id = ontology_id
        self.OLS_BASE_URL = ols_api_base_url
        self.page_size = page_size

        self._ontology_name: Optional[str] = None
        self._ontology_short_name: Optional[str] = None
        self._num_pages: Optional[int] = None

    def _initialize_metadata(self) -> None:
        """Fetches ontology metadata in a single network request block."""
        if self._num_pages is not None:
            return

        # Fetch config (Name & Short Name)
        config_url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}"
        try:
            resp = requests.get(config_url)
            resp.raise_for_status()
            data = resp.json()
            self._ontology_name = data["config"]["title"]
            self._ontology_short_name = data["config"].get(
                "preferredPrefix", data["config"].get("prefferedPrefix", self.ontology_id.upper())
            )
        except Exception as e:
            logger.error(f"Failed to fetch ontology config: {e}")
            raise

        # Fetch page count
        terms_url = f"{self.OLS_BASE_URL}ontologies/{self.ontology_id}/terms?size={self.page_size}"
        try:
            resp = requests.get(terms_url)
            resp.raise_for_status()
            self._num_pages = resp.json()["page"]["totalPages"]
        except Exception as e:
            logger.error(f"Failed to fetch term pagination: {e}")
            raise

    def process_to_repository(self, repository: PostgreSQLRepository, start_page: int = 0) -> None:
        """Fetches concepts and descriptions from the OLS API and stores them in a repository.

        :param repository: The repository to store the concepts and mappings.
        :return: None
        """
        self._initialize_metadata()

        num_pages = self._num_pages
        if num_pages is None or self._ontology_name is None or self._ontology_short_name is None:
            raise RuntimeError("Ontology metadata failed to initialize properly.")

        repository.store([Terminology(name=self._ontology_name, short_name=self._ontology_short_name)])
        term_db = repository.get_terminology_by_name(self._ontology_name)

        current_page = start_page
        while current_page < num_pages:
            try:
                identifiers, labels, descriptions, embeddings = self._fetch_page_data(current_page)

                if not identifiers:
                    current_page += 1
                    continue

                unique_concepts = {}
                for _, (ident, label) in enumerate(zip(identifiers, labels)):
                    if ident not in unique_concepts:
                        unique_concepts[ident] = Concept(
                            terminology_id=term_db.id, pref_label=label, concept_identifier=ident
                        )

                repository.store(list(unique_concepts.values()))
                repository.session.flush()

                saved_concepts = (
                    repository.session.query(Concept.id, Concept.concept_identifier)
                    .filter(Concept.concept_identifier.in_(identifiers), Concept.terminology_id == term_db.id)
                    .all()
                )
                concept_map = {c_identifier: c_id for c_id, c_identifier in saved_concepts}

                unique_mappings = []
                seen_mapping_ids = set()

                for ident, desc, emb in zip(identifiers, descriptions, embeddings):
                    if ident in concept_map and ident not in seen_mapping_ids:
                        seen_mapping_ids.add(ident)
                        unique_mappings.append(
                            Mapping(
                                concept_id=concept_map[ident],
                                text=desc,
                                embedding=emb,
                                vectorizer=self.vectorizer.model_name,
                            )
                        )

                repository.store(unique_mappings)
                repository.session.commit()
                current_page += 1
            except Exception as e:
                repository.session.rollback()
                logger.error(f"Failed at page {current_page}: {e}")
                raise

    def process_to_json(self, dest_path: str, start_page: int = 0) -> None:
        """Fetches concepts and descriptions from the OLS API and stores them in a JSON files to a specified directory.

        :param dest_path: The directory path where the resulting JSON files will be saved.
        :return: None.
        """
        self._initialize_metadata()

        num_pages = self._num_pages
        if num_pages is None or self._ontology_name is None or self._ontology_short_name is None:
            raise RuntimeError("Ontology metadata failed to initialize properly.")

        os.makedirs(dest_path, exist_ok=True)
        terminology_data = {"name": self._ontology_name, "short_name": self._ontology_short_name}

        with open(os.path.join(dest_path, "terminology.json"), "w", encoding="utf-8") as f:
            json.dump(terminology_data, f, indent=2)

        current_page = start_page
        while current_page < num_pages:
            identifiers, labels, descriptions, embeddings = self._fetch_page_data(current_page)

            if not identifiers:
                current_page += 1
                continue

            concepts, mappings = [], []
            for ident, label, desc, emb in zip(identifiers, labels, descriptions, embeddings):
                concepts.append(
                    {
                        "concept_identifier": ident,
                        "pref_label": label,
                        "terminology_short_name": self._ontology_short_name,
                    }
                )
                mappings.append(
                    {
                        "concept_identifier": ident,
                        "text": desc,
                        "embedding": emb,
                        "vectorizer": self.vectorizer.model_name,
                    }
                )

            with open(os.path.join(dest_path, f"concepts_{current_page}.json"), "w", encoding="utf-8") as f:
                json.dump(concepts, f, indent=2)
            with open(os.path.join(dest_path, f"mappings_{current_page}.json"), "w", encoding="utf-8") as f:
                json.dump(mappings, f, indent=2)

            current_page += 1

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
        total = self._num_pages if self._num_pages else 0
        logger.info(f"Processing page {page}/{total}.")

        try:
            resp = requests.get(url)
            resp.raise_for_status()
            terms = resp.json().get("_embedded", {}).get("terms", [])

            identifiers = []
            labels = []
            descriptions = []

            for term in terms:
                ident = term.get("obo_id")
                if not ident:
                    continue

                identifiers.append(ident)
                labels.append(term.get("label", ""))
                desc = term.get("description")
                if isinstance(desc, list) and desc:
                    descriptions.append(desc[0])
                elif isinstance(desc, str) and desc:
                    descriptions.append(desc)
                else:
                    descriptions.append(term.get("label", ""))

            embeddings = self.vectorizer.get_embeddings(descriptions) if descriptions else []
            return identifiers, labels, descriptions, embeddings
        except Exception as e:
            logger.error(f"Failed to fetch OLS data for page {page}: {str(e)}")
            raise

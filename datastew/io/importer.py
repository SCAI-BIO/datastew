import logging
from collections.abc import Sequence
from pathlib import Path

from datastew.io.bulk import BulkImportResult, import_bulk
from datastew.io.source import DataDictionarySource
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import Concept, Mapping, Terminology

logger = logging.getLogger(__name__)


class Importer:
    def __init__(self, repository: PostgreSQLRepository) -> None:
        """Initializes the Import with a database repository connection.

        :param repository: The configured PostgreSQLRepository instance.
        """
        self.repository = repository
        self.vectorizer = repository.vectorizer

    def import_data_dictionary(
        self, data_dictionary: DataDictionarySource, terminology_name: str, short_name: str
    ) -> None:
        """Imports a data dictionary, generating concepts and embeddings, and stores them in the database.

        :param data_dictionary: The parsed DataDictionarySource object containing variables and descriptions.
        :param terminology_name: The full name to assign to the new Terminology.
        :param short_name: The short name or prefix to assign to the new Terminology.
        :raises RuntimeError: If the import process fails at any stage.
        """
        try:
            self.repository.store([Terminology(name=terminology_name, short_name=short_name)])
            terminology = self.repository.get_terminology_by_name(terminology_name)

            df = data_dictionary.to_dataframe()
            descriptions = df["description"].tolist()
            vectorizer_name = self.vectorizer.model_name
            variable_to_embedding = data_dictionary.get_embeddings(self.vectorizer)

            concepts = []
            concepts.extend(
                Concept(
                    terminology_id=terminology.id,
                    pref_label=variable,
                    concept_identifier=f"{terminology_name}:{variable}",
                )
                for variable in variable_to_embedding
            )
            self.repository.store(concepts)

            concept_identifiers = [c.concept_identifier for c in concepts]
            saved_concepts = (
                self.repository.session.query(Concept.id, Concept.concept_identifier)
                .filter(Concept.concept_identifier.in_(concept_identifiers), Concept.terminology_id == terminology.id)
                .all()
            )
            concept_map = {identifier: c_id for c_id, identifier in saved_concepts}

            mappings = []
            for variable, description in zip(variable_to_embedding.keys(), descriptions, strict=False):
                concept_id_str = f"{terminology_name}:{variable}"
                mappings.append(
                    Mapping(
                        concept_id=concept_map[concept_id_str],
                        text=description,
                        embedding=variable_to_embedding[variable],
                        vectorizer=vectorizer_name,
                    )
                )
            self.repository.store(mappings)

        except Exception as e:
            logger.exception("Failed to import data dictionary.")
            raise RuntimeError(f"Failed to import data dictionary source: {e}") from e

    def import_from_jsonl(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        batch_size: int = 1000,
        generate_embeddings: bool = False,
        embedding_batch_size: int = 256,
    ) -> BulkImportResult:
        """Import terminologies, concepts, and mappings from JSONL files using PostgreSQL COPY.

        Uses predefined terminology and concept IDs to avoid database ID resolution.
        The JSONL records contain terminology metadata, concept definitions, and
        associated mappings. Intended for importing into fresh databases.

        :param source: JSONL file path, glob pattern, or sequence of paths and patterns.
        :param batch_size: Number of concepts per database transaction, defaults to 1000.
        :param generate_embeddings: Whether to generate missing embeddings, defaults to False.
        :param embedding_batch_size: Number of texts per embedding request, defaults to 256.
        :raises ValueError: If import parameters or JSONL records are invalid.
        :raises FileNotFoundError: If an input file cannot be found.
        :raises RuntimeError: If embedding generation fails or the database driver is unsupported.
        :return: BulkImportResult containing the number of imported records and generated embeddings.
        """
        return import_bulk(
            repository=self.repository,
            source=source,
            batch_size=batch_size,
            generate_embeddings=generate_embeddings,
            embedding_batch_size=embedding_batch_size,
        )

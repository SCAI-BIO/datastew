import json
import logging
import os
from typing import Any, Callable, Literal

from sqlalchemy import text

from datastew.process.parsing import DataDictionarySource
from datastew.repository.model import Concept, Mapping, Terminology
from datastew.repository.postgresql import PostgreSQLRepository

logger = logging.getLogger(__name__)


class PostgreSQLImporter:
    def __init__(self, repository: PostgreSQLRepository):
        self.repository = repository
        self.engine = repository.engine
        self.vectorizer = repository.vectorizer

    def import_data_dictionary(self, data_dictionary: DataDictionarySource, terminology_name: str, short_name: str):
        """Imports a data dictionary, generating concepts and embeddings, and stores them in the database."""
        try:
            self.repository.store([Terminology(name=terminology_name, short_name=short_name)])
            terminology = self.repository.get_terminology_by_name(terminology_name)

            df = data_dictionary.to_dataframe()
            descriptions = df["description"].tolist()
            vectorizer_name = self.vectorizer.model_name
            variable_to_embedding = data_dictionary.get_embeddings(self.vectorizer)

            concepts = []
            for variable in variable_to_embedding.keys():
                concepts.append(
                    Concept(
                        terminology_id=terminology.id,
                        pref_label=variable,
                        concept_identifier=f"{terminology_name}:{variable}",
                    )
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
            for variable, description in zip(variable_to_embedding.keys(), descriptions):
                concept_id_str = f"{terminology_name}:{variable}"
                mappings.append(
                    Mapping(
                        concept_id=concept_map[concept_id_str],
                        text=description,
                        embedding=variable_to_embedding[variable],
                        sentence_embedder=vectorizer_name,
                    )
                )
            self.repository.store(mappings)

        except Exception as e:
            logger.exception("Failed to import data dictionary.")
            raise RuntimeError(f"Failed to import data dictionary source: {e}")

    def import_from_jsonl(
        self, jsonl_path: str, object_type: Literal["terminology", "concept", "mapping"], chunk_size: int = 50000
    ):
        """Imports data from a JSONL file and stores it in the database via staging tables."""
        if not os.path.exists(jsonl_path):
            raise RuntimeError(f"File not found: {jsonl_path}")

        if object_type == "terminology":
            self._import_terminology_staging(jsonl_path, chunk_size)
        elif object_type == "concept":
            self._import_concept_staging(jsonl_path, chunk_size)
        elif object_type == "mapping":
            self._import_mapping_staging(jsonl_path, chunk_size)
        else:
            raise ValueError(f"Unsupported object_type: {object_type}")

    def _load_and_insert_staging(
        self,
        conn,
        jsonl_path: str,
        chunk_size: int,
        object_type: Literal["concept", "mapping"],
        required_keys: list[str],
        insert_query: str,
        row_processor: Callable[[dict[str, Any]], dict[str, Any]],
    ):
        buffer = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                data = json.loads(line)
                self._validate_required_fields(data, required_keys, object_type)

                buffer.append(row_processor(data))

                if len(buffer) >= chunk_size:
                    conn.execute(text(insert_query), buffer)
                    buffer.clear()

            if buffer:
                conn.execute(text(insert_query), buffer)

    def _import_terminology_staging(self, jsonl_path: str, chunk_size: int):
        buffer = []
        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                data = json.loads(line)
                self._validate_required_fields(data, ["name", "short_name"], "terminology")
                buffer.append(Terminology(name=data["name"], short_name=data["short_name"]))

                if len(buffer) >= chunk_size:
                    self.repository.store(buffer)
                    buffer.clear()
            if buffer:
                self.repository.store(buffer)

    def _import_concept_staging(self, jsonl_path: str, chunk_size: int):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE UNLOGGED TABLE IF NOT EXISTS staging_concept (
                        terminology_short_name TEXT,
                        pref_label TEXT,
                        concept_identifier TEXT)
                    """
                )
            )
            conn.execute(text("TRUNCATE staging_concept"))

            def process_concept(data: dict[str, Any]) -> dict[str, Any]:
                return {
                    "terminology_short_name": data["terminology_short_name"],
                    "pref_label": data["pref_label"],
                    "concept_identifier": data["concept_identifier"],
                }

            insert_query = """
                INSERT INTO staging_concept (terminology_short_name, pref_label, concept_identifier)
                VALUES (:terminology_short_name, :pref_label, :concept_identifier)
            """

            self._load_and_insert_staging(
                conn=conn,
                jsonl_path=jsonl_path,
                chunk_size=chunk_size,
                object_type="concept",
                required_keys=["terminology_short_name", "pref_label", "concept_identifier"],
                insert_query=insert_query,
                row_processor=process_concept,
            )

            conn.execute(
                text(
                    """
                    INSERT INTO concept (terminology_id, pref_label, concept_identifier)
                    SELECT t.id, s.pref_label, s.concept_identifier
                    FROM staging_concept s
                    JOIN terminology t ON s.terminology_short_name = t.short_name
                    ON CONFLICT (terminology_id, concept_identifier)
                    DO UPDATE SET pref_label = EXCLUDED.pref_label
                    """
                )
            )
            conn.execute(text("DROP TABLE staging_concept"))

    def _import_mapping_staging(self, jsonl_path: str, chunk_size: int):
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE UNLOGGED TABLE IF NOT EXISTS staging_mapping (
                        concept_identifier TEXT,
                        text TEXT,
                        embedding vector(768),
                        sentence_embedder TEXT
                    )
                    """
                )
            )
            conn.execute(text("TRUNCATE staging_mapping"))

            def process_mapping(data: dict[str, Any]) -> dict[str, Any]:
                embedding = data.get("embedding")
                embedder = data.get("sentence_embedder")

                if (embedding is None or embedder is None) and self.vectorizer:
                    embedding = self.vectorizer.get_embedding(data["text"])
                    embedder = self.vectorizer.model_name

                embedding_str = str(embedding) if embedding else None

                return {
                    "concept_identifier": data["concept_identifier"],
                    "text": data["text"],
                    "embedding": embedding_str,
                    "sentence_embedder": embedder,
                }

            insert_query = """
                INSERT INTO staging_mapping (concept_identifier, text, embedding, sentence_embedder)
                VALUES (:concept_identifier, :text, CAST(:embedding AS vector(768)), :sentence_embedder)
            """

            self._load_and_insert_staging(
                conn=conn,
                jsonl_path=jsonl_path,
                chunk_size=chunk_size,
                object_type="mapping",
                required_keys=["concept_identifier", "text"],
                insert_query=insert_query,
                row_processor=process_mapping,
            )

            conn.execute(
                text(
                    """
                    INSERT INTO mapping (concept_id, text, embedding, sentence_embedder)
                    SELECT c.id, s.text, s.embedding, s.sentence_embedder
                    FROM staging_mapping s
                    JOIN concept c ON s.concept_identifier = c.concept_identifier
                    ON CONFLICT (concept_id, sentence_embedder, text)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                    """
                )
            )
            conn.execute(text("DROP TABLE staging_mapping"))

    def _validate_required_fields(
        self, data: dict[str, Any], required_keys: list[str], object_type: Literal["terminology", "concept", "mapping"]
    ):
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required field '{key}' for {object_type}")

import json
import logging
import os
from typing import Any, Callable, Literal

from sqlalchemy import text

from datastew.io.source import DataDictionarySource
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import Concept, Mapping, Terminology

logger = logging.getLogger(__name__)


class Importer:
    def __init__(self, repository: PostgreSQLRepository):
        """Initializes the Import with a database repository connection.

        :param repository: The configured PostgreSQLRepository instance.
        """
        self.repository = repository
        self.engine = repository.engine
        self.vectorizer = repository.vectorizer

    def import_data_dictionary(self, data_dictionary: DataDictionarySource, terminology_name: str, short_name: str):
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
                        vectorizer=vectorizer_name,
                    )
                )
            self.repository.store(mappings)

        except Exception as e:
            logger.exception("Failed to import data dictionary.")
            raise RuntimeError(f"Failed to import data dictionary source: {e}")

    def import_from_jsonl(
        self,
        jsonl_path: str,
        object_type: Literal["terminology", "concept", "mapping"],
        chunk_size: int = 50000,
        generate_embeddings: bool = False,
    ):
        """Imports data from a JSONL file and stores it in the database via staging tables.

        :param jsonl_path: The file path to the JSONL data.
        :param object_type: The type of data being imported ('terminology', 'concept', or 'mapping').
        :param chunk_size: The number of rows to process before committing to the database, defaults to 50000.
        :param generate_embeddings: If True, calculates embeddings for mappings on the fly, defaults to False.
        :raises RuntimeError: If the specified JSONL file does not exist.
        :raises ValueError: If an unsupported object_type is provided.
        """
        if not os.path.exists(jsonl_path):
            raise RuntimeError(f"File not found: {jsonl_path}")

        if object_type == "terminology":
            self._import_terminology_staging(jsonl_path, chunk_size)
        elif object_type == "concept":
            self._import_concept_staging(jsonl_path, chunk_size)
        elif object_type == "mapping":
            self._import_mapping_staging(jsonl_path, chunk_size, generate_embeddings)
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
        """Reads a JSONL file in chunks, processes each row, and executes a bulk insert query into a staging table.

        :param conn: The SQLAlchemy connection object.
        :param jsonl_path: The path to the JSONL file.
        :param chunk_size: The number of rows per batch to insert.
        :param object_type: The type of object being processed (used for error logging).
        :param required_keys: A list of dictionary keys that must be present in every JSONL row.
        :param insert_query: The SQL query used to insert data into the staging table.
        :param row_processor: A callable function to transform the raw JSON row into a database-ready format.
        """
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
        """Parses terminology data from a JSONL file and inserts it directly into the terminology table in chunks.

        :param jsonl_path: The path to the JSONL file containing terminology definitions.
        :param chunk_size: The number of terminologies to insert per transaction batch.
        """
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
        """Imports concept data from a JSONL file by first writing to a temporary
        unlogged staging table, then upserting into the main concept table resolving
        terminology foreign keys.

        :param jsonl_path: The path to the JSONL file containing concept definitions.
        :param chunk_size: The batch size for reading and inserting into the staging table.
        """
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

    def _import_mapping_staging(self, jsonl_path: str, chunk_size: int = 2048, generate_embeddings: bool = False):
        """Loads mappings from a JSONL file, optionally generates embeddings via LLM
        batch inference, and inserts them into the final table using a high-performance
        unlogged staging table.

        :param jsonl_path: The path to the JSONL file containing mapping text.
        :param chunk_size: Batch size for staging operations. Capped at 2048 if generating embeddings.
        :param generate_embeddings: Boolean flag indicating if embeddings should be computed on the fly.
        :raises ValueError: If embeddings are missing from data while generate_embeddings is False.
        """
        # Cap chunk size for LLM limits only if generating embeddings
        effective_chunk_size = min(chunk_size, 2048)

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE UNLOGGED TABLE IF NOT EXISTS staging_mapping (
                        concept_identifier TEXT,
                        text TEXT,
                        embedding vector(768),
                        vectorizer TEXT
                    )
                    """
                )
            )
            conn.execute(text("TRUNCATE staging_mapping"))

            buffer = []
            with open(jsonl_path, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    self._validate_required_fields(data, ["concept_identifier", "text"], "mapping")

                    if not generate_embeddings and ("embedding" not in data or "vectorizer" not in data):
                        raise ValueError(
                            "generate_embeddings is False, but 'embedding' or 'vectorizer' "
                            f"is missing in row: {data.get('concept_identifier')}"
                        )

                    buffer.append(data)

                    if len(buffer) >= effective_chunk_size:
                        self._embed_and_insert_mapping_chunk(conn, buffer, generate_embeddings)
                        buffer.clear()

                if buffer:
                    self._embed_and_insert_mapping_chunk(conn, buffer, generate_embeddings)

            # Resolve FKs and Upsert to Final Table
            conn.execute(
                text(
                    """
                    INSERT INTO mapping (concept_id, text, embedding, vectorizer)
                    SELECT c.id, s.text, s.embedding, s.vectorizer
                    FROM staging_mapping s
                    JOIN concept c ON s.concept_identifier = c.concept_identifier
                    ON CONFLICT (concept_id, vectorizer, text)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                    """
                )
            )
            conn.execute(text("DROP TABLE staging_mapping"))

    def _embed_and_insert_mapping_chunk(self, conn, buffer: list[dict[str, Any]], generate_embeddings: bool):
        """Executes LLM batch inference if flagged, formats the data, and executes the SQL insert into staging.

        :param conn: The SQLAlchemy connection object.
        :param buffer: A list of dictionary objects representing the rows to insert.
        :param generate_embeddings: Flag indicating whether to fetch embeddings from the vectorizer.
        :raises ValueError: If generate_embeddings is True but the repository lacks a configured vectorizer.
        :raises RuntimeError: If the vectorizer returns an inconsistent number of embeddings.
        """
        if generate_embeddings:
            texts = [item["text"] for item in buffer]
            embeddings = self.vectorizer.get_embeddings(texts)
            vectorizer_name = self.vectorizer.model_name

            if len(embeddings) != len(buffer):
                raise RuntimeError(f"LLM returned {len(embeddings)} embeddings for {len(buffer)} texts.")

            for item, emb in zip(buffer, embeddings):
                item["embedding"] = emb
                item["vectorizer"] = vectorizer_name

        insert_data = []
        for item in buffer:
            insert_data.append(
                {
                    "concept_identifier": item["concept_identifier"],
                    "text": item["text"],
                    "embedding": str(item["embedding"]),
                    "vectorizer": item["vectorizer"],
                }
            )

        conn.execute(
            text(
                """
                INSERT INTO staging_mapping (concept_identifier, text, embedding, vectorizer)
                VALUES (:concept_identifier, :text, CAST(:embedding AS vector(768)), :vectorizer)
                """
            ),
            insert_data,
        )

    def _validate_required_fields(
        self, data: dict[str, Any], required_keys: list[str], object_type: Literal["terminology", "concept", "mapping"]
    ):
        """Checks that a dictionary contains all necessary keys for a given object type.

        :param data: The parsed JSON dictionary row.
        :param required_keys: A list of keys that must exist in the dictionary.
        :param object_type: The string representing the object type (used for error formatting).
        :raises ValueError: If any key in required_keys is missing from the data dictionary.
        """
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required field '{key}' for {object_type}")

import csv
import gzip
import json
import math
from collections.abc import Iterator, Sequence
from itertools import groupby, islice
from pathlib import Path
from typing import TextIO

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from tqdm import tqdm

from datastew.embedding import Vectorizer
from datastew.io.bulk import ConceptObject, ImportObject, MappingObject, TerminologyObject
from datastew.repository import PostgreSQLRepository
from datastew.repository.model import EMBEDDING_DIMENSIONS, Concept, Mapping, Terminology


class JsonlAdapter:
    """Export self-contained terminology, concept, and mapping records to JSONL."""

    def __init__(self, dest_dir: str | Path, buffer_size: int = 1000) -> None:
        """Initialize the exporter and create its destination directory.

        :param dest_dir: Directory in which JSONL shards will be written.
        :param buffer_size: Number of concepts processed per batch, defaults to 1000.
        :raises ValueError: If buffer_size is not a positive integer.
        :return: None.
        """
        self._validate_positive_integer(buffer_size, "buffer_size")
        self.dest_dir = Path(dest_dir)
        self._buffer_size = buffer_size
        self.dest_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_positive_integer(value: int, field: str) -> None:
        """Validate a strictly positive integer configuration value.

        :param value: Integer value to validate.
        :param field: Name of the configuration parameter.
        :raises ValueError: If the value is not a positive integer.
        :return: None.
        """
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer.")

    @classmethod
    def _validate_id(cls, value: int, field: str) -> None:
        """Validate a predefined PostgreSQL INTEGER primary or foreign key.

        :param value: Database ID to validate.
        :param field: Name of the ID parameter.
        :raises ValueError: If the ID is outside the positive 32-bit integer range.
        :return: None.
        """
        cls._validate_positive_integer(value, field)
        max_postgres_integer = 2**31 - 1
        if value > max_postgres_integer:
            raise ValueError(f"{field} must not exceed {max_postgres_integer}.")

    @staticmethod
    def _read_ohdsi_rows(src: Path) -> Iterator[tuple[str, str]]:
        """Stream source identifiers and labels from an OHDSI CONCEPT.csv file.

        :param src: Path to the tab-separated OHDSI CONCEPT.csv file.
        :raises ValueError: If required columns or record identifiers and labels are missing.
        :raises csv.Error: If the CSV contains malformed records or oversized fields.
        :yield: The original OHDSI concept ID and preferred label.
        """
        previous_limit = csv.field_size_limit()
        csv.field_size_limit(2**31 - 1)

        try:
            with src.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")

                if reader.fieldnames is None or not {"concept_id", "concept_name"}.issubset(reader.fieldnames):
                    raise ValueError("OHDSI CONCEPT.csv requires 'concept_id' and 'concept_name' columns.")

                for row in reader:
                    original_id = row.get("concept_id")
                    label = row.get("concept_name")

                    if original_id is None or not original_id.strip():
                        raise ValueError(f"Missing OHDSI concept_id at line {reader.line_num}.")

                    if label is None or not label.strip():
                        raise ValueError(f"Missing OHDSI concept_name at line {reader.line_num}.")

                    yield original_id.strip(), label.strip()
        finally:
            csv.field_size_limit(previous_limit)

    @staticmethod
    def _batch_rows(rows: Iterator[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
        """Read OHDSI source rows into bounded batches.

        :param rows: Iterator of original concept identifiers and labels.
        :param size: Maximum number of rows in each batch.
        :yield: A nonempty batch of original OHDSI concept IDs and preferred labels.
        """
        while batch := list(islice(rows, size)):
            yield batch

    @staticmethod
    def _normalize_embedding(vector: Sequence[float]) -> list[float]:
        """Convert a vector to finite floats matching the PostgreSQL embedding dimension.

        :param vector: Numeric embedding components.
        :raises ValueError: If the embedding has an invalid dimension or contains nonfinite values.
        :return: Embedding components converted to Python floats.
        """
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Embedding must contain {EMBEDDING_DIMENSIONS} dimensions.")
        values = [float(value) for value in vector]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding must contain only finite values.")
        return values

    def _batch_embeddings(
        self, rows: list[tuple[str, str]], vectorizer: Vectorizer | None
    ) -> list[list[float] | None]:
        """Generate embeddings for an OHDSI batch, or defer them to the importer.

        :param rows: Batch of OHDSI concept identifiers and labels.
        :param vectorizer: Embedding model, or None to defer generation.
        :raises RuntimeError: If the model produces an unexpected number of embeddings.
        :raises ValueError: If an embedding has invalid dimensions or components.
        :return: Embeddings aligned with the input rows, or None for each deferred embedding.
        """
        if vectorizer is None:
            return [None] * len(rows)
        vectors = vectorizer.get_embeddings([label for _, label in rows])
        if len(vectors) != len(rows):
            raise RuntimeError(f"Vectorizer returned {len(vectors)} embeddings for {len(rows)} concepts.")
        return [self._normalize_embedding(vector) for vector in vectors]

    def _iter_ohdsi_records(
        self, src: Path, terminology_id: int, first_concept_id: int, vectorizer: Vectorizer | None
    ) -> Iterator[ImportObject]:
        """Convert OHDSI concepts into self-contained JSONL import records.

        :param src: Path to the source OHDSI CONCEPT.csv file.
        :param terminology_id: Preassigned ID of the OHDSI terminology.
        :param first_concept_id: First sequential concept database ID.
        :param vectorizer: Embedding model, or None to defer embedding generation.
        :raises ValueError: If an assigned concept ID exceeds PostgreSQL INTEGER capacity.
        :yield: A self-contained record with terminology, concept, and mappings.
        """
        terminology: TerminologyObject = {
            "id": terminology_id,
            "name": "Observational Health Data Sciences and Informatics",
            "short_name": "OHDSI",
        }
        next_id = first_concept_id
        batch_size = min(self._buffer_size, 2048)
        for rows in self._batch_rows(self._read_ohdsi_rows(src), batch_size):
            embeddings = self._batch_embeddings(rows, vectorizer)
            for (original_id, label), embedding in zip(rows, embeddings, strict=True):
                self._validate_id(next_id, "concept.id")
                concept: ConceptObject = {
                    "id": next_id,
                    "concept_identifier": f"OHDSI:{original_id}",
                    "pref_label": label,
                }
                mapping: MappingObject = {
                    "text": label,
                    "embedding": embedding,
                    "vectorizer": vectorizer.model_name if vectorizer is not None else None,
                }
                yield {"terminology": terminology, "concept": concept, "mappings": [mapping]}
                next_id += 1

    @classmethod
    def _repository_mapping(cls, mapping: Mapping) -> MappingObject:
        """Convert one stored mapping into a serializable JSONL mapping object.

        :param mapping: Stored mapping including its embedding and model name.
        :raises ValueError: If the mapping has no embedding or the vector is invalid.
        :return: Mapping object containing its text, vector, and model name.
        """
        if mapping.embedding is None:
            raise ValueError(f"Mapping.id={mapping.id} has no embedding.")
        return {
            "text": mapping.text,
            "embedding": cls._normalize_embedding(mapping.embedding),
            "vectorizer": mapping.vectorizer,
        }

    def _iter_repository_records(self, repository: PostgreSQLRepository) -> Iterator[ImportObject]:
        """Stream concepts and their associated terminology and mappings from the database.

        :param repository: Repository containing concepts, terminologies, and mappings.
        :raises ValueError: If a stored mapping has an invalid or missing embedding.
        :yield: A self-contained, import-compatible concept record.
        """
        stmt = (
            select(Concept)
            .options(selectinload(Concept.terminology), selectinload(Concept.mappings))
            .order_by(Concept.id)
            .execution_options(yield_per=self._buffer_size)
        )
        for concept in tqdm(repository.session.scalars(stmt), desc="Exporting concepts"):
            terminology = concept.terminology
            term_obj: TerminologyObject = {
                "id": terminology.id,
                "name": terminology.name,
                "short_name": terminology.short_name,
            }
            concept_obj: ConceptObject = {
                "id": concept.id,
                "concept_identifier": concept.concept_identifier,
                "pref_label": concept.pref_label,
            }
            mappings = [self._repository_mapping(mapping) for mapping in concept.mappings]
            yield {"terminology": term_obj, "concept": concept_obj, "mappings": mappings}

    @staticmethod
    def _open_shard(path: Path, compress: bool) -> TextIO:
        """Create a new JSONL shard, optionally compressed with gzip.

        :param path: Output path to create.
        :param compress: Whether the shard should use gzip compression.
        :raises FileExistsError: If the target file already exists.
        :return: Writable UTF-8 text stream for the new shard.
        """
        if compress:
            return gzip.open(path, "xt", encoding="utf-8")
        return path.open("x", encoding="utf-8")

    def _write_shards(
        self, records: Iterator[ImportObject], prefix: str, records_per_file: int, compress: bool
    ) -> list[Path]:
        """Write a record stream to new, optionally compressed JSONL shards.

        Newly created shards are deleted if generation fails. Existing files are
        never overwritten. Records are streamed without materializing whole shards.

        :param records: Stream of self-contained concept-centric records.
        :param prefix: Filename prefix distinguishing export sources.
        :param records_per_file: Maximum concepts written per JSONL shard.
        :param compress: Whether to write gzip-compressed shards.
        :raises ValueError: If the input stream contains no concepts.
        :raises FileExistsError: If an output shard already exists.
        :return: Paths to the created JSONL shards, in output order.
        """
        self._validate_positive_integer(records_per_file, "records_per_file")
        paths: list[Path] = []
        suffix = ".jsonl.gz" if compress else ".jsonl"

        try:
            grouped_records = groupby(enumerate(records), key=lambda item: item[0] // records_per_file)
            for shard_index, shard_records in grouped_records:
                path = self.dest_dir / f"{prefix}_{shard_index + 1:05d}{suffix}"
                with self._open_shard(path, compress) as handle:
                    paths.append(path)
                    for _, record in shard_records:
                        serialized = json.dumps(record, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
                        handle.write(serialized + "\n")

            if not paths:
                raise ValueError("The source contains no concepts to export.")
        except Exception:
            for path in paths:
                path.unlink(missing_ok=True)
            raise

        return paths

    def from_ohdsi(
        self,
        src: str | Path,
        vectorizer: Vectorizer | None = None,
        include_vectors: bool = False,
        *,
        terminology_id: int = 1,
        first_concept_id: int = 1,
        records_per_file: int = 500,
        compress: bool = False,
    ) -> list[Path]:
        """Convert OHDSI CONCEPT.csv into self-contained JSONL for fresh-database import.

        Every concept gets a sequential database ID, while its original OHDSI ID
        is preserved in ``concept_identifier``. Terminology metadata is repeated
        in each record. Missing embeddings may be generated during import.

        :param src: Path to the tab-separated OHDSI CONCEPT.csv source file.
        :param vectorizer: Model for precomputed embeddings, or None for the default model.
        :param include_vectors: Whether to precompute embeddings, defaults to False.
        :param terminology_id: Preassigned ID for the OHDSI terminology, defaults to 1.
        :param first_concept_id: First sequential concept ID, defaults to 1.
        :param records_per_file: Maximum concepts per JSONL shard, defaults to 50000.
        :param compress: Whether to gzip-compress JSONL shards, defaults to False.
        :raises FileNotFoundError: If the OHDSI source file does not exist.
        :raises ValueError: If the source or configuration contains invalid data.
        :raises FileExistsError: If a destination shard already exists.
        :return: Paths to the generated JSONL shards in import order.
        """
        source_path = Path(src)
        if not source_path.is_file():
            raise FileNotFoundError(f"OHDSI concept file '{source_path}' does not exist or is not a file.")
        self._validate_id(terminology_id, "terminology_id")
        self._validate_id(first_concept_id, "first_concept_id")
        self._validate_positive_integer(records_per_file, "records_per_file")

        model = (vectorizer or Vectorizer()) if include_vectors else None
        records = self._iter_ohdsi_records(source_path, terminology_id, first_concept_id, model)
        return self._write_shards(records, "ohdsi", records_per_file, compress)

    def from_repository(
        self, repository: PostgreSQLRepository, *, records_per_file: int = 50_000, compress: bool = False
    ) -> list[Path]:
        """Export a populated repository as import-compatible, concept-centric JSONL.

        Terminology and concept primary keys are preserved. Mapping primary keys
        are not stored in the current JSONL schema and are reassigned on import.
        Terminologies without concepts cannot be represented and are rejected.

        :param repository: Repository containing terminologies, concepts, and mappings.
        :param records_per_file: Maximum concepts per JSONL shard, defaults to 50000.
        :param compress: Whether to gzip-compress JSONL shards, defaults to False.
        :raises ValueError: If a terminology has no concepts or the repository is empty.
        :raises FileExistsError: If a destination shard already exists.
        :return: Paths to the generated JSONL shards in import order.
        """
        self._validate_positive_integer(records_per_file, "records_per_file")
        empty_terminology = repository.session.scalar(
            select(Terminology.id).where(~Terminology.concepts.any()).limit(1)
        )
        if empty_terminology is not None:
            raise ValueError(f"Terminology.id={empty_terminology} has no concepts and cannot be exported.")

        records = self._iter_repository_records(repository)
        return self._write_shards(records, "repository", records_per_file, compress)

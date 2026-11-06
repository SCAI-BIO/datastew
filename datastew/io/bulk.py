import glob
import gzip
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

from psycopg import Connection
from sqlalchemy import text

from datastew.repository import PostgreSQLRepository
from datastew.repository.model import EMBEDDING_DIMENSIONS


@dataclass(frozen=True)
class BulkImportResult:
    """Summarize a successfully completed bulk import.

    :param files: Number of JSONL files processed.
    :param concepts: Number of concepts inserted.
    :param mappings: Number of mappings inserted.
    :param generated_embeddings: Number of embeddings generated during the import.
    :param terminologies: Number of distinct terminologies inserted.
    """

    files: int
    concepts: int
    mappings: int
    generated_embeddings: int
    terminologies: int = 0


class TerminologyObject(TypedDict):
    """Describe a terminology in a self-contained JSONL record.

    :param id: Preassigned database primary key of the terminology.
    :param name: Full terminology name.
    :param short_name: Unique abbreviated terminology name.
    """

    id: int
    name: str
    short_name: str


class ConceptObject(TypedDict):
    """Describe a concept in a self-contained JSONL record.

    :param id: Preassigned database primary key of the concept.
    :param concept_identifier: Identifier assigned by the source terminology.
    :param pref_label: Preferred human-readable label of the concept.
    """

    id: int
    concept_identifier: str
    pref_label: str


class MappingObject(TypedDict):
    """Describe a mapping in the canonical JSONL format.

    :param text: Text associated with the concept.
    :param embedding: Embedding vector, or None when generation is required.
    :param vectorizer: Embedding model name, or None when generation is required.
    """

    text: str
    embedding: list[float] | None
    vectorizer: str | None


class ImportObject(TypedDict):
    """Describe a self-contained concept-centric JSONL record.

    :param terminology: Terminology definition, including its predefined ID.
    :param concept: Concept definition, including its predefined ID.
    :param mappings: List of mappings associated with the concept.
    """

    terminology: TerminologyObject
    concept: ConceptObject
    mappings: list[MappingObject]


class ConceptRow(ConceptObject):
    """Describe a validated concept row for PostgreSQL COPY.

    :param terminology_id: Foreign key of the terminology assigned to the concept.
    """

    terminology_id: int


class MappingRow(TypedDict):
    """Describe a validated mapping row for PostgreSQL COPY.

    :param concept_id: Foreign key of the concept associated with the mapping.
    :param text: Text used to create the embedding.
    :param embedding: Validated embedding or None until embedding generation.
    :param vectorizer: Embedding model name or None until embedding generation.
    """

    concept_id: int
    text: str
    embedding: list[float] | None
    vectorizer: str | None


def _positive_id(value: object, field: str) -> int:
    """Validate an explicit PostgreSQL integer primary or foreign key.

    :param value: Value to validate, potentially obtained from untrusted JSON.
    :param field: Field name to include in validation errors.
    :raises ValueError: If the value is not a positive 32-bit PostgreSQL integer.
    :return: The validated integer ID.
    """
    max_postgres_integer = 2**31 - 1
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= max_postgres_integer:
        raise ValueError(f"{field} must be a positive PostgreSQL INTEGER (1..{max_postgres_integer}).")
    return value


def _source_files(source: str | Path | Sequence[str | Path]) -> list[Path]:
    """Resolve source paths and glob patterns into JSONL files.

    :param source: JSONL file, glob pattern, or sequence of paths and patterns.
    :raises FileNotFoundError: If a path or pattern matches no files.
    :raises ValueError: If a matched path is not a file or the source is empty.
    :return: Input files in source order, sorted within each glob pattern.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    files: list[Path] = []
    for item in sources:
        matches = sorted(glob.glob(str(item)))
        if not matches:
            raise FileNotFoundError(f"No JSONL files found for {item!s}")
        for matched in matches:
            path = Path(matched)
            if not path.is_file():
                raise ValueError(f"Not a file: {path}")
            files.append(path)
    if not files:
        raise ValueError("At least one JSONL source is required.")
    return files


def _read_records(files: Sequence[Path]) -> Iterator[tuple[Path, int, dict[str, object]]]:
    """Stream unvalidated JSON objects from JSONL and gzip-compressed JSONL files.

    :param files: Source paths to read sequentially.
    :raises ValueError: If a nonblank line is not a valid JSON object.
    :yield: The file path, line number, and decoded JSON object for each nonblank line.
    """
    for path in files:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record: object = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Record at {path}:{line_number} must be a JSON object.")
                # JSON object keys are strings; values still require validation.
                yield path, line_number, cast(dict[str, object], record)


def _parse_terminology(record: dict[str, object]) -> TerminologyObject:
    """Validate terminology metadata included in a JSONL record.

    :param record: Unvalidated JSONL record with a nested terminology object.
    :raises ValueError: If the terminology ID, name, or short name is invalid.
    :return: Validated terminology attributes for PostgreSQL COPY.
    """
    terminology = record.get("terminology")
    if not isinstance(terminology, dict):
        raise ValueError("'terminology' must be an object.")

    terminology_id = _positive_id(terminology.get("id"), "terminology.id")
    name = terminology.get("name")
    short_name = terminology.get("short_name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("terminology.name must be a nonempty string.")
    if not isinstance(short_name, str) or not short_name.strip():
        raise ValueError("terminology.short_name must be a nonempty string.")

    return {"id": terminology_id, "name": name, "short_name": short_name}


def _parse_concept(record: dict[str, object], terminology_id: int) -> ConceptRow:
    """Validate a concept and prepare its PostgreSQL row representation.

    :param record: Unvalidated JSONL record containing a concept object.
    :param terminology_id: Foreign key taken from the record's terminology object.
    :raises ValueError: If the concept or a required field is invalid.
    :return: Normalized concept row containing its ID and terminology foreign key.
    """
    concept = record.get("concept")
    if not isinstance(concept, dict):
        raise ValueError("'concept' must be an object.")

    concept_id = _positive_id(concept.get("id"), "concept.id")
    concept_identifier = concept.get("concept_identifier")
    pref_label = concept.get("pref_label")

    if not isinstance(concept_identifier, str) or not concept_identifier.strip():
        raise ValueError("concept.concept_identifier must be a nonempty string.")
    if not isinstance(pref_label, str) or not pref_label.strip():
        raise ValueError("concept.pref_label must be a nonempty string.")

    return {
        "id": concept_id,
        "terminology_id": terminology_id,
        "concept_identifier": concept_identifier,
        "pref_label": pref_label,
    }


def _parse_embedding(value: object, index: int) -> list[float] | None:
    """Validate a supplied embedding and convert its elements to floats.

    :param value: Unvalidated embedding value from a mapping object.
    :param index: Mapping position used for error messages.
    :raises ValueError: If the embedding has the wrong shape or nonfinite/nonnumeric values.
    :return: Validated embedding vector, or None if the value is absent.
    """
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"mappings[{index}].embedding must contain {EMBEDDING_DIMENSIONS} numbers.")

    embedding: list[float] = []
    for element in value:
        if isinstance(element, bool) or not isinstance(element, (int, float)) or not math.isfinite(element):
            raise ValueError(f"mappings[{index}].embedding must contain only finite numbers.")
        embedding.append(float(element))
    return embedding


def _parse_mapping(mapping: object, concept_id: int, index: int) -> MappingRow:
    """Validate a mapping and prepare its PostgreSQL row representation.

    :param mapping: Unvalidated mapping object from a JSONL record.
    :param concept_id: Preassigned database ID of the associated concept.
    :param index: Position in the concept's mappings list, used in errors.
    :raises ValueError: If the mapping, text, vectorizer, or embedding is invalid.
    :return: Normalized mapping row referencing the concept's ID.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"mappings[{index}] must be an object.")

    content = mapping.get("text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"mappings[{index}].text must be a nonempty string.")

    embedding = _parse_embedding(mapping.get("embedding"), index)
    model = mapping.get("vectorizer")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError(f"mappings[{index}].vectorizer must be a nonempty string or null.")

    return {"concept_id": concept_id, "text": content, "embedding": embedding, "vectorizer": model}


def _parse_mappings(record: dict[str, object], concept_id: int) -> list[MappingRow]:
    """Extract and validate the mapping or mappings belonging to a concept.

    :param record: Unvalidated JSONL record with either mapping or mappings.
    :param concept_id: Preassigned database ID of the parent concept.
    :raises ValueError: If both mapping keys are given or the mapping structure is invalid.
    :return: Normalized mapping rows, possibly an empty list.
    """
    if "mapping" in record and "mappings" in record:
        raise ValueError("Use either 'mapping' or 'mappings', not both.")

    raw_mappings: list[object]
    if "mapping" in record:
        single = record["mapping"]
        if not isinstance(single, dict):
            raise ValueError("'mapping' must be an object.")
        raw_mappings = [single]
    else:
        collection = record.get("mappings")
        if not isinstance(collection, list):
            raise ValueError("'mappings' must be a list (an empty list is allowed).")
        raw_mappings = list(collection)

    return [_parse_mapping(mapping, concept_id, index) for index, mapping in enumerate(raw_mappings)]


def _parse_record(record: dict[str, object]) -> tuple[TerminologyObject, ConceptRow, list[MappingRow]]:
    """Validate one self-contained terminology, concept, and mappings JSONL record.

    :param record: Unvalidated JSONL record containing terminology, concept, and mappings.
    :raises ValueError: If terminology, concept, or mapping fields are invalid.
    :return: Validated terminology row, concept row, and mapping rows.
    """
    terminology = _parse_terminology(record)
    concept = _parse_concept(record, terminology["id"])
    mappings = _parse_mappings(record, concept["id"])
    return terminology, concept, mappings


def _normalize_generated_embedding(vector: Sequence[float]) -> list[float]:
    """Validate an embedding returned by the configured vectorizer.

    :param vector: Sequence of numeric embedding components.
    :raises ValueError: If the vector has an invalid dimension or contains nonfinite values.
    :return: Validated embedding components converted to floats.
    """
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Generated embeddings must have {EMBEDDING_DIMENSIONS} dimensions.")
    values = [float(element) for element in vector]
    if not all(math.isfinite(element) for element in values):
        raise ValueError("Generated embeddings must contain only finite numbers.")
    return values


def _prepare_embeddings(
    repository: PostgreSQLRepository,
    mappings: list[MappingRow],
    generate_embeddings: bool,
    embedding_batch_size: int,
) -> int:
    """Generate missing embeddings and update mapping rows in place.

    :param repository: Repository providing the configured embedding model.
    :param mappings: Mapping rows to validate and modify in place.
    :param generate_embeddings: Whether missing embeddings may be generated.
    :param embedding_batch_size: Maximum texts sent to the model per request.
    :raises ValueError: If embedding data is incomplete or generated vectors have invalid dimensions.
    :raises RuntimeError: If the model returns an unexpected number of embeddings.
    :return: Number of embeddings generated for this batch.
    """
    missing: list[MappingRow] = []
    for mapping in mappings:
        if mapping["embedding"] is None:
            if not generate_embeddings:
                raise ValueError(
                    f"Missing embedding for concept_id={mapping['concept_id']}; "
                    "set generate_embeddings=True to compute it."
                )
            if mapping["vectorizer"] is not None and mapping["vectorizer"] != repository.vectorizer.model_name:
                raise ValueError(
                    "Mapping vectorizer does not match the configured model used to generate its embedding."
                )
            missing.append(mapping)
        elif mapping["vectorizer"] is None:
            raise ValueError(f"An existing embedding for concept_id={mapping['concept_id']} needs a vectorizer name.")

    for start in range(0, len(missing), embedding_batch_size):
        chunk = missing[start : start + embedding_batch_size]
        vectors = repository.vectorizer.get_embeddings([mapping["text"] for mapping in chunk])
        if len(vectors) != len(chunk):
            raise RuntimeError(f"Vectorizer returned {len(vectors)} embeddings for {len(chunk)} texts.")
        for mapping, vector in zip(chunk, vectors, strict=True):
            mapping["embedding"] = _normalize_generated_embedding(vector)
            mapping["vectorizer"] = repository.vectorizer.model_name
    return len(missing)


def _copy_batch(
    repository: PostgreSQLRepository,
    terminologies: list[TerminologyObject],
    concepts: list[ConceptRow],
    mappings: list[MappingRow],
) -> None:
    """COPY new terminologies, concepts, and mappings through synchronous psycopg3.

    The SQLAlchemy session owns the transaction. Terminologies are copied before
    their concepts, and concepts are copied before their mappings.

    :param repository: Repository whose SQLAlchemy session owns the transaction.
    :param terminologies: New terminology rows not inserted by earlier batches.
    :param concepts: Validated concept rows with predefined primary keys.
    :param mappings: Mapping rows referencing the corresponding concept IDs.
    :raises RuntimeError: If the database does not expose a synchronous psycopg3 connection.
    :return: None. The caller handles the transaction commit or rollback.
    """
    connection = repository.session.connection().connection.driver_connection
    if not isinstance(connection, Connection):
        raise RuntimeError("Bulk import requires a synchronous psycopg3 connection.")

    with connection.cursor() as cursor:
        if terminologies:
            with cursor.copy("COPY terminology (id, name, short_name) FROM STDIN") as copy:
                for terminology in terminologies:
                    copy.write_row((terminology["id"], terminology["name"], terminology["short_name"]))

        with cursor.copy("COPY concept (id, terminology_id, concept_identifier, pref_label) FROM STDIN") as copy:
            for concept in concepts:
                copy.write_row(
                    (concept["id"], concept["terminology_id"], concept["concept_identifier"], concept["pref_label"])
                )

        if mappings:
            with cursor.copy("COPY mapping (concept_id, text, embedding, vectorizer) FROM STDIN") as copy:
                for mapping in mappings:
                    vector = json.dumps(mapping["embedding"], separators=(",", ":"), allow_nan=False)
                    copy.write_row((mapping["concept_id"], mapping["text"], vector, mapping["vectorizer"]))


def _validate_bulk_options(batch_size: int, embedding_batch_size: int) -> None:
    """Validate bulk-import batch sizes before writing to the database.

    :param batch_size: Maximum number of concepts per COPY transaction.
    :param embedding_batch_size: Maximum number of texts in each embedding request.
    :raises ValueError: If either batch size is invalid.
    :return: None.
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if (
        not isinstance(embedding_batch_size, int)
        or isinstance(embedding_batch_size, bool)
        or not 1 <= embedding_batch_size <= 2048
    ):
        raise ValueError("embedding_batch_size must be between 1 and 2048.")


def _register_terminology(
    terminology: TerminologyObject,
    by_id: dict[int, TerminologyObject],
    by_name: dict[str, int],
    by_short_name: dict[str, int],
) -> bool:
    """Register terminology metadata and detect inconsistent repeated definitions.

    :param terminology: Validated terminology metadata from a JSONL record.
    :param by_id: Previously observed terminologies, keyed by their IDs.
    :param by_name: Previously observed terminology names, mapped to their IDs.
    :param by_short_name: Previously observed short names, mapped to their IDs.
    :raises ValueError: If an ID is redefined or a name is associated with a different ID.
    :return: True for a new terminology, or False for a consistent repeat.
    """
    terminology_id = terminology["id"]
    existing = by_id.get(terminology_id)
    if existing is not None:
        if existing != terminology:
            raise ValueError(f"Conflicting metadata for terminology.id={terminology_id}.")
        return False

    if terminology["name"] in by_name:
        raise ValueError(f"Repeated terminology.name={terminology['name']!r} with a different ID.")
    if terminology["short_name"] in by_short_name:
        raise ValueError(f"Repeated terminology.short_name={terminology['short_name']!r} with a different ID.")

    by_id[terminology_id] = terminology
    by_name[terminology["name"]] = terminology_id
    by_short_name[terminology["short_name"]] = terminology_id
    return True


def _iter_bulk_batches(
    files: Sequence[Path], batch_size: int
) -> Iterator[tuple[list[TerminologyObject], list[ConceptRow], list[MappingRow]]]:
    """Yield COPY-ready batches while inserting each distinct terminology only once.

    Terminology definitions are tracked across all input files. Concept IDs are
    checked within each batch; PostgreSQL constraints catch conflicts across batches.

    :param files: JSONL files to stream in source order.
    :param batch_size: Maximum number of concepts yielded in each batch.
    :raises ValueError: If records are invalid or a terminology definition changes.
    :yield: New terminology rows, concept rows, and mapping rows for each batch.
    """
    terminologies: list[TerminologyObject] = []
    concepts: list[ConceptRow] = []
    mappings: list[MappingRow] = []
    seen_concept_ids: set[int] = set()
    terminology_by_id: dict[int, TerminologyObject] = {}
    terminology_by_name: dict[str, int] = {}
    terminology_by_short_name: dict[str, int] = {}

    for path, line_number, record in _read_records(files):
        try:
            terminology, concept, concept_mappings = _parse_record(record)
            if concept["id"] in seen_concept_ids:
                raise ValueError(f"Repeated concept.id={concept['id']} within the current batch.")
            if _register_terminology(terminology, terminology_by_id, terminology_by_name, terminology_by_short_name):
                terminologies.append(terminology)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc

        seen_concept_ids.add(concept["id"])
        concepts.append(concept)
        mappings.extend(concept_mappings)
        if len(concepts) >= batch_size:
            yield terminologies, concepts, mappings
            terminologies, concepts, mappings, seen_concept_ids = [], [], [], set()

    if concepts:
        yield terminologies, concepts, mappings


def _commit_bulk_batch(
    repository: PostgreSQLRepository,
    terminologies: list[TerminologyObject],
    concepts: list[ConceptRow],
    mappings: list[MappingRow],
    generate_embeddings: bool,
    embedding_batch_size: int,
) -> int:
    """Generate missing embeddings, COPY all three tables, and commit one batch.

    :param repository: Repository providing the vectorizer and active SQLAlchemy session.
    :param terminologies: New terminologies first encountered in this batch.
    :param concepts: Validated concepts for this batch.
    :param mappings: Mapping rows updated with generated embeddings when needed.
    :param generate_embeddings: Whether to generate missing embeddings.
    :param embedding_batch_size: Maximum texts embedded per model request.
    :raises ValueError: If required embedding data is missing or invalid.
    :raises RuntimeError: If the vectorizer or psycopg3 driver is incompatible.
    :return: Number of embeddings generated for this committed batch.
    """
    try:
        generated = _prepare_embeddings(repository, mappings, generate_embeddings, embedding_batch_size)
        _copy_batch(repository, terminologies, concepts, mappings)
        repository.session.commit()
    except Exception:
        repository.session.rollback()
        raise
    return generated


def _synchronize_bulk_sequences(repository: PostgreSQLRepository) -> None:
    """Advance terminology and concept sequences after inserting predefined IDs.

    :param repository: Repository providing the active SQLAlchemy session.
    :raises SQLAlchemyError: If a sequence update fails.
    :return: None. A failure rolls back the session.
    """
    session = repository.session
    try:
        for table in ("terminology", "concept"):
            session.execute(
                text(
                    f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), "
                    f"COALESCE((SELECT MAX(id) FROM {table}), 1), "
                    f"EXISTS (SELECT 1 FROM {table}))"
                )
            )
        session.commit()
    except Exception:
        session.rollback()
        raise


def import_bulk(
    repository: PostgreSQLRepository,
    source: str | Path | Sequence[str | Path],
    *,
    batch_size: int = 1000,
    generate_embeddings: bool = False,
    embedding_batch_size: int = 256,
) -> BulkImportResult:
    """Import self-contained terminology, concept, and mapping records via PostgreSQL COPY.

    Designed for fresh databases with no conflicting IDs or names. Each JSONL line
    has sibling ``terminology``, ``concept``, and ``mappings`` (or ``mapping``)
    objects. The importer inserts each distinct terminology once and commits
    bounded batches independently; earlier batches remain after a later failure.

    :param repository: Configured PostgreSQL repository with an active session.
    :param source: Source JSONL file, glob pattern, or sequence of such paths.
    :param batch_size: Number of concepts per database batch, defaults to 1000.
    :param generate_embeddings: Whether missing embeddings should be generated, defaults to False.
    :param embedding_batch_size: Maximum texts per embedding request, defaults to 256.
    :raises ValueError: If import options, terminology metadata, concepts, or mappings are invalid.
    :raises FileNotFoundError: If any source path or pattern matches no files.
    :raises RuntimeError: If embedding generation or driver initialization fails.
    :return: Counts of files, terminologies, concepts, mappings, and generated embeddings.
    """
    _validate_bulk_options(batch_size, embedding_batch_size)
    files = _source_files(source)
    total_terminologies = 0
    total_concepts = 0
    total_mappings = 0
    generated_embeddings = 0

    for terminologies, concepts, mappings in _iter_bulk_batches(files, batch_size):
        generated_embeddings += _commit_bulk_batch(
            repository, terminologies, concepts, mappings, generate_embeddings, embedding_batch_size
        )
        total_terminologies += len(terminologies)
        total_concepts += len(concepts)
        total_mappings += len(mappings)

    _synchronize_bulk_sequences(repository)
    return BulkImportResult(
        files=len(files),
        concepts=total_concepts,
        mappings=total_mappings,
        generated_embeddings=generated_embeddings,
        terminologies=total_terminologies,
    )

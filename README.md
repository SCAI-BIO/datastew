# datastew

<p align="left"><a href="https://doi.org/10.5281/zenodo.16871713"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.16871713-blue.svg" alt="DOI"></a>&nbsp;<a href="https://github.com/SCAI-BIO/datastew/actions/workflows/tests.yml"><img src="https://github.com/SCAI-BIO/datastew/actions/workflows/tests.yml/badge.svg" alt="tests"></a>&nbsp;<a href="https://codecov.io/gh/SCAI-BIO/datastew"><img src="https://codecov.io/gh/SCAI-BIO/datastew/branch/main/graph/badge.svg" alt="codecov"></a>&nbsp;<a href="https://pypi.org/project/datastew/"><img src="https://img.shields.io/pypi/v/datastew" alt="version"></a>&nbsp;<a href="https://pepy.tech/projects/datastew"><img src="https://static.pepy.tech/personalized-badge/datastew?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=BLUE&left_text=downloads" alt="PyPI Downloads"></a></p>

Datastew is a python library for intelligent data harmonization using Large Language Model (LLM) vector embeddings.

## Installation

```bash
pip install datastew
```

## Usage

### Harmonizing excel/csv resources

You can directly import common data models, terminology sources or data dictionaries for harmonization directly from a
csv, tsv or excel file. An example how to match two separate variable descriptions is shown in
[datastew/scripts/mapping_excel_example.py](datastew/scripts/mapping_excel_example.py):

```python
from datastew.io.source import DataDictionarySource
from datastew.harmonization import map_dictionary_to_dictionary

# Variable and description refer to the corresponding column names in your excel sheet
source = DataDictionarySource("source.xlxs", variable_field="var", description_field="desc")
target = DataDictionarySource("target.xlxs", variable_field="var", description_field="desc")

df = map_dictionary_to_dictionary(source, target)
df.to_excel("result.xlxs")
```

The resulting file contains the pairwise variable mapping based on the closest similarity for all possible matches
as well as a similarity measure per row.

Per default this will use the local MPNet model, which may not yield the optimal performance. If you got an OpenAI API
key it is possible to use their embedding API instead. To use your key, create a Vectorizer model and pass it to the
function:

```python
from datastew.embedding import Vectorizer
from datastew.harmonization import map_dictionary_to_dictionary

vectorizer = Vectorizer("text-embedding-ada-002", key="your_api_key")
df = map_dictionary_to_dictionary(source, target, vectorizer=vectorizer)
```

---

### Creating and using stored mappings

Datastew uses a PostgreSQL backend with the pgvector extension to store and query embeddings. This allows for persistent terminology management and high-performance semantic search.

1.  Initialize the repository and embedding model:

    First, set up your database engine and ensure the schema is initialized. Datastew uses the psycopg (v3) driver for synchronous communication.

    ```python
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from datastew.embedding import Vectorizer
    from datastew.repository import PostgreSQLRepository

    # 1. Define connection string (note the +psycopg driver)
    DB_URL = "postgresql+psycopg://user:password@localhost:5432/testdb"
    engine = create_engine(DB_URL)

    # 2. Initialize the database schema and pgvector extension (Run once)
    PostgreSQLRepository.setup_database(engine)

    # 3. Create a session factory
    SessionLocal = sessionmaker(bind=engine, autoflush=False)
    ```

2.  Populate the Repository

    Use a session context to add terminologies, concepts, and mappings. Note: You must call session.commit() to persist changes before they become searchable.

    ```python
    vectorizer = Vectorizer()

    with SessionLocal() as session: # Inject the session into the repository
        repository = PostgreSQLRepository(session=session, vectorizer=vectorizer)

        # Add a terminology
        terminology = repository.add_terminology(name="snomed CT", short_name="SNOMED")

        # Create a concept
        text1 = "Diabetes mellitus (disorder)"
        concept1 = repository.add_concept(
            terminology_id=terminology.id,
            pref_label=text1,
            concept_identifier="SNOMED:11893007"
        )

        # Add a mapping (this generates the embedding and stores it)
        repository.add_mapping(concept_id=concept1.id, text=text1)

        # Persist the data
        session.commit()
    ```

3.  Retrieve Closest Mappings

    Query the database by generating an embedding for a new phrase and comparing it against stored records.

    ```python
    with SessionLocal() as session:
        repository = PostgreSQLRepository(session=session, vectorizer=vectorizer)

        query_text = "Sugar sickness"
        embedding = vectorizer.get_embedding(query_text)

        # Retrieve top 2 matches with similarity scores
        results = repository.get_closest_mappings(embedding, similarities=True, limit=2)

        for r in results:
            # Returns a MappingResult object containing the Mapping and a similarity float
            print(f"{r.mapping.concept.pref_label} | Similarity: {r.similarity:.4f}")
    ```

output:

```python
snomed CT > Concept ID: 11893007 : Diabetes mellitus (disorder) | Diabetes mellitus (disorder) | Similarity: 0.4735338091850281
snomed CT > Concept ID: 73211009 : Hypertension (disorder) | Hypertension (disorder) | Similarity: 0.2003161907196045
```

You can also import data from file sources (csv, tsv, xlsx) or from a public API like OLS. An example script to
download & compute embeddings for SNOMED from ebi OLS can be found in
[datastew/scripts/ols_snomed_retrieval.py](datastew/scripts/ols_snomed_retrieval.py).

---

### Embedding visualization

You can visualize the embedding space of multiple data dictionary sources with t-SNE plots utilizing different
language models. An example how to generate a t-sne plot is shown in
[datastew/scripts/tsne_visualization.py](datastew/scripts/tsne_visualization.py):

```python
from datastew.embedding import Vectorizer
from datastew.io.source import DataDictionarySource
from datastew.visualisation import plot_embeddings

# Variable and description refer to the corresponding column names in your excel sheet
data_dictionary_source_1 = DataDictionarySource("source1.xlsx", variable_field="var", description_field="desc")
data_dictionary_source_2 = DataDictionarySource("source2.xlsx", variable_field="var", description_field="desc")

vectorizer = Vectorizer()
plot_embeddings([data_dictionary_source_1, data_dictionary_source_2], vectorizer=vectorizer)
```

![t-SNE plot](./docs/tsne_plot.png)

# Citation

If you use this work in your research, please cite as:

```bibtex
@article{Salimi_Evaluating_language_model_2025,
  author  = {Salimi, Yasamin and Adams, Tim and Ay, Mehmet Can and
             Balabin, Helena and Jacobs, Marc and
             Hofmann-Apitius, Martin},
  title   = {Evaluating language model embeddings for Parkinson's disease cohort harmonization using a novel manually curated variable mapping schema},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.1038/s41598-025-06447-2}
}
```

**Reference**

Salimi Y, Adams T, Ay MC, Balabin H, Jacobs M, Hofmann-Apitius M. *Evaluating language model embeddings for Parkinson's disease cohort harmonization using a novel manually curated variable mapping schema*. **Scientific Reports**. 2025. https://doi.org/10.1038/s41598-025-06447-2

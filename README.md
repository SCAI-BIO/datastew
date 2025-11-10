# datastew

[![DOI](https://zenodo.org/badge/822570156.svg)](https://doi.org/10.5281/zenodo.16871713) ![tests](https://github.com/SCAI-BIO/datastew/actions/workflows/tests.yml/badge.svg) ![GitHub Release](https://img.shields.io/github/v/release/SCAI-BIO/datastew)

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
from datastew.process.parsing import DataDictionarySource
from datastew.process.mapping import map_dictionary_to_dictionary

# Variable and description refer to the corresponding column names in your excel sheet
source = DataDictionarySource("source.xlxs", variable_field="var", description_field="desc")
target = DataDictionarySource("target.xlxs", variable_field="var", description_field="desc")

df = map_dictionary_to_dictionary(source, target)
df.to_excel("result.xlxs")
```

The resulting file contains the pairwise variable mapping based on the closest similarity for all possible matches
as well as a similarity measure per row.

Per default this will use the local MiniLM model, which may not yield the optimal performance. If you got an OpenAI API
key it is possible to use their embedding API instead. To use your key, create a Vectorizer model and pass it to the
function:

```python
from datastew.embedding import Vectorizer
from datastew.process.mapping import map_dictionary_to_dictionary

vectorizer = Vectorizer("text-embedding-ada-002", key="your_api_key")
df = map_dictionary_to_dictionary(source, target, vectorizer=vectorizer)
```

---

### Creating and using stored mappings

A simple example how to initialize an in memory database and compute a similarity mapping is shown in
[datastew/scripts/mapping_db_example.py](datastew/scripts/mapping_db_example.py):

1.  Initialize the repository and embedding model:

    ```python
    from datastew.embedding import Vectorizer
    from datastew.repository import PostgreSQLRepository
    from datastew.repository.model import Concept, Mapping, MappingResult, Terminology

    POSTGRES_USER = "user"
    POSTGRES_PASSWORD = "password"
    POSTGRES_HOST = "localhost"
    POSTGRES_PORT = "5432"
    POSTGRES_DB = "testdb"

    connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    # You can use an OpenAI model if you have an API key:
    # vectorizer = Vectorizer("text-embedding-3-small", key="your_openai_api_key")
    vectorizer = Vectorizer()
    repository = PostgreSQLRepository(connection_string, vectorizer=vectorizer)
    ```

2.  Create a baseline of data to map to in the initialized repository. Text gets attached to any unique concept of an
    existing or custom vocabulary or terminology namespace in the form of a mapping object containing the text, embedding,
    and the name of sentence embedder used. Multiple Mapping objects with textually different but semantically equal
    descriptions can point to the same Concept.

    ```python
    terminology = Terminology("snomed CT", "SNOMED")

    text1 = "Diabetes mellitus (disorder)"
    concept1 = Concept(terminology, text1, "Concept ID: 11893007")
    mapping1 = Mapping(concept1, text1, vectorizer.get_embedding(text1), vectorizer.model_name)

    text2 = "Hypertension (disorder)"
    concept2 = Concept(terminology, text2, "Concept ID: 73211009")
    mapping2 = Mapping(concept2, text2, vectorizer.get_embedding(text2), vectorizer.model_name)

    repository.store_all([terminology, concept1, mapping1, concept2, mapping2])
    ```

3.  Retrieve the closest mappings and their similarities for a given text:

    ```python
    query_text = "Sugar sickness"  # semantically similar to "Diabetes mellitus (disorder)"
    embedding = vectorizer.get_embedding(query_text)
    results = repository.get_closest_mappings(embedding, similarities=True, limit=2)

    print(f'Query: "{query_text}"\n')
    for r in results:
        # If similarities=True, repo returns MappingResult; else Mapping.
        if isinstance(r, MappingResult):
            print(r)
        else:
            print(str(r))

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
from datastew.process.parsing import DataDictionarySource
from datastew.visualisation import plot_embeddings

# Variable and description refer to the corresponding column names in your excel sheet
data_dictionary_source_1 = DataDictionarySource("source1.xlsx", variable_field="var", description_field="desc")
data_dictionary_source_2 = DataDictionarySource("source2.xlsx", variable_field="var", description_field="desc")

vectorizer = Vectorizer()
plot_embeddings([data_dictionary_source_1, data_dictionary_source_2], vectorizer=vectorizer)
```

![t-SNE plot](./docs/tsne_plot.png)

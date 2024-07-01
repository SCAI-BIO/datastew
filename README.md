# datastew

![tests](https://github.com/SCAI-BIO/datastew/actions/workflows/tests.yml/badge.svg) ![GitHub Release](https://img.shields.io/github/v/release/SCAI-BIO/datastew)

Datastew is a python library for intelligent data harmonization using Large Language Model (LLM) vector embeddings. 
## Installation

```bash
pip install datastew
```

## Usage

### Harmonizing excel/csv resources

You can directly import common data models, terminology sources or data dictionaries for harmonization directly from a
csv, tsv or excel file. An example how to match two seperate variable descriptions is shown in
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

Per default this will use the local MPNet model, which may not yield the optimal performance. If you got an OpenAI API 
key it is possible to use their embedding API instead. To use your key, create an OpenAIAdapter model and pass it to the 
function:

```python
from datastew.embedding import GPT4Adapter

embedding_model = GPT4Adapter(key="your_api_key")
df = map_dictionary_to_dictionary(source, target, embedding_model=embedding_model)
```

### Creating and using stored mappings

A simple example how to initialize an in memory database and compute a similarity mapping is shown in 
[datastew/scripts/mapping_db_example.py](datastew/scripts/mapping_db_example.py):

```python
from datastew.repository.sqllite import SQLLiteRepository
from datastew.repository.model import Terminology, Concept, Mapping
from datastew.embedding import MPNetAdapter

# omit mode to create a permanent db file instead
repository = SQLLiteRepository(mode="memory")
embedding_model = MPNetAdapter()

terminology = Terminology("snomed CT", "SNOMED")

text1 = "Diabetes mellitus (disorder)"
concept1 = Concept(terminology, text1, "Concept ID: 11893007")
mapping1 = Mapping(concept1, text1, embedding_model.get_embedding(text1))

text2 = "Hypertension (disorder)"
concept2 = Concept(terminology, text2, "Concept ID: 73211009")
mapping2 = Mapping(concept2, text2, embedding_model.get_embedding(text2))

repository.store_all([terminology, concept1, mapping1, concept2, mapping2])

text_to_map = "Sugar sickness"
embedding = embedding_model.get_embedding(text_to_map)
mappings, similarities = repository.get_closest_mappings(embedding, limit=2)
for mapping, similarity in zip(mappings, similarities):
    print(f"Similarity: {similarity} -> {mapping}")
```

output:

```plaintext
Similarity: 0.47353370635583486 -> Concept ID: 11893007 : Diabetes mellitus (disorder) | Diabetes mellitus (disorder)
Similarity: 0.20031612264852067 -> Concept ID: 73211009 : Hypertension (disorder) | Hypertension (disorder)
```

You can also import data from file sources (csv, tsv, xlsx) or from a public API like OLS. An example script to
download & compute embeddings for SNOMED from ebi OLS can be found in 
[datastew/scripts/ols_snomed_retrieval.py](datastew/scripts/ols_snomed_retrieval.py).

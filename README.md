# INDEX – the Intelligent Data Steward Toolbox

![example workflow](https://github.com/SCAI-BIO/index/actions/workflows/tests.yml/badge.svg) ![GitHub Release](https://img.shields.io/github/v/release/SCAI-BIO/index)

INDEX is an intelligent data steward toolbox that leverages Large Language Model embeddings for automated Data-Harmonization. 

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Introduction

INDEX uses vector embeddings from variable descriptions to suggest mappings for datasets based on their semantic 
similarity. Mappings are stored with their vector representations in a knowledge base, where they can be used for 
subsequent harmonisation tasks, potentially improving the following suggestions with each iteration. Models for 
the computation as well as databases for storage are meant to be configurable and extendable to adapt the tool for
specific use-cases.

## Installation

### Using pip

```bash
pip install datastew
```

### From source

Clone the repository:

```bash
git clone https://github.com/SCAI-BIO/index
cd index
```

Install python requirements:

```bash
pip install -r requirements.txt
```

### Starting the Backend locally

You can access the backend functionalities by accessing the provided REST API.

Run the Backend API on port 5000:

```bash
uvicorn datastew.api.routes:app --reload --port 5000
```

### Run the Backend via Docker

The API can also be run via docker.

You can either build the docker container locally or download the latest build from the index GitHub package registry. 


```bash
docker build . -t ghcr.io/scai-bio/datastew/backend:latest
```

```bash
docker pull ghcr.io/scai-bio/datastew/backend:latest
```

After build/download you will be able to start the container and access the INDEX API per default on [localhost:8000](http://localhost:8000):

```bash
docker run  -p 8000:80 ghcr.io/datastew/scai-bio/backend:latest
```
## Usage

### Python

#### Creating and using stored mappings

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

#### Harmonizing excel/csv resources

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

## Configuration

### Description Embeddings

You can configure INDEX to use either a local language model or call OPenAPIs embedding API. While using the OpenAI API
is significantly faster, you will need to provide an API key that is linked to your OpenAI account. 

Currently, the following local models are implemented:
* [Sentence Transformer (MPNet)](https://huggingface.co/docs/transformers/model_doc/mpnet)

The API will default to use a local embedding model. You can adjust the model loaded on start up in the configurations.

### Database

INDEX will by default store mappings in a file based db file in the [index/db](datastew/db) dir. For testing purposes
the initial SQLLite file based db contains a few of mappings to concepts in SNOMED CT. All available database adapter 
implementations can be found in [index/repository](datastew/repository).

To exchange the DB implementation, load your custom DB adapter or pre-saved file-based DB file on application startup
[here](https://github.com/SCAI-BIO/index/blob/923601677fd62d50c3748b7f11666420e82df609/index/api/routes.py#L14). 
The same can be done for any other embedding model.

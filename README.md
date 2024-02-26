# INDEX – the Intelligent Data Steward Toolbox

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
uvicorn index.api.routes:app --reload --port 5000
```

### Run the Backend via Docker

The API can also be run via docker.

You can either build the docker container locally or download the latest build from the index GitHub package registry. 


```bash
docker build . -t ghcr.io/scai-bio/backend:latest
```

```bash
docker pull ghcr.io/scai-bio/backend:latest
```

After build/download you will be able to start the container and access the INDEX API per default on [localhost:8000](http://localhost:8000):

```bash
docker run  -p 8000:80 ghcr.io/scai-bio/backend:latest
```
## Usage

### Python

A simple example how to initialize an in memory database and compute a similarity mapping is shown in 
[index/scripts/mapping_db_example.py](index/scripts/mapping_db_example.py):

```python
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
[index/scripts/ols_snomed_retrieval.py](index/scripts/ols_snomed_retrieval.py).

## Configuration

### Description Embeddings

You can configure INDEX to use either a local language model or call OPenAPIs embedding API. While using the OpenAI API
is significantly faster, you will need to provide an API key that is linked to your OpenAI account. 

Currently, the following local models are implemented:
* [Sentence Transformer (MPNet)](https://huggingface.co/docs/transformers/model_doc/mpnet)

The API will default to use a local embedding model. You can adjust the model loaded on start up in the configurations.

### Database

INDEX will by default store mappings in a file based db file in the 
[following directory](https://github.com/SCAI-BIO/index/tree/main/index/db). All available database adapter 
implementations are available in [index/repository](index/repository).

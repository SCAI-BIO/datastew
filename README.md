# INDEX – the Intelligent Data Steward Toolbox

INDEX is an intelligent data steward toolbox that leverages Large Language Model embeddings for automated Data-Harmonization. 

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)

## Introduction

INDEX uses vector embeddings from variable descriptions to suggest mappings for datasets, improving data indexing and retrieval. Confirmed mappings are stored with their vector representations in a knowledge base for fast search and retrieval, enhancing data management and analysis. New mappings can be added iteratively to improve suggestions for future harmonization tasks.

## Installation
Clone the repository:

```bash
git clone https://github.com/SCAI-BIO/index
cd index
```

### Starting the Backend locally

Install python requirements:

```bash
pip install -r requirements.txt
```


Run the Backend API on port 5000:

```bash
uvicorn index.api.routes:app --reload --port 5000
```

### Run the Backend via Docker

You can either build the docker container locally or downlaod the latest build from the index github package registry. 


```bash
docker build . -t ghcr.io/scai-bio/backend:latest
```

```bash
docker pull ghcr.io/scai-bio/backend:latest
```

After build/download you will be able to start the container and access the IDNEX API per default on [localhost:8000](http://localhost:8000):

```bash
docker run  -p 8000:80 ghcr.io/scai-bio/backend:latest
```

## Configuration

### Description Embeddings

You can configure INDEX to use either a local language model or call OPenAPIs embedding API. While using the OpenAI API
is significantly faster, you will need to provide an API key that is linked to your OpenAI account. 

Currently, the following local models are implemented:
* [MPNet](https://huggingface.co/docs/transformers/model_doc/mpnet)

The API will default to use a local embedding model. You can adjust the model loaded on start up in the configurations.

### Database

INDEX will be default store mappings in a file based db file in the [following directory](https://github.com/SCAI-BIO/index/tree/main/index/db).

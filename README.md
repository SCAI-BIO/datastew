# INDEX – the Intelligent Data Steward Toolbox

INDEX is an intelligent data steward toolbox that leverages Large Language Model embeddings for automated Data-Harmonization. 

## Table of Contents
- [Introduction](##ntroduction)
- [Installation & Usage](#installation)
- [Configuration](#configuration)

## Introduction

INDEX relies on vector embeddings calculated based on variable descriptions to generate mapping suggestions for any 
dataset, enabling efficient and accurate data indexing and retrieval. Confirmed mappings are stored alongside their 
vectorized representations in a knowledge base, facilitating rapid search and retrieval operations, ultimately enhancing 
data management and analysis capabilities. New mappings may be added to the knowledge base in an iterative procedure,
allowing for improved mapping suggestions in subsequent harmonization tasks.

## Installation & Usage

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
uvicorn main:app --reload --port 5000
```

### Run the Backend via Docker

Download the latest docker build:

```bash
docker pull ghcr.io/scai-bio/backend:latest
```

## Configuration

### Description Embeddings

You can configure INDEX to use either a local language model or call OPenAPIs embedding API. While using the OpenAI API
is significantly faster, you will need to provide an API key that is linked to your OpenAI account. 

Currently, the following local models are implemented:
* [MPNet](https://huggingface.co/docs/transformers/model_doc/mpnet)
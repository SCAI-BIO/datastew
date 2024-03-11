import json
import logging
from typing import Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from index.db.model import Terminology, Concept, Mapping
from index.repository.sqllite import SQLLiteRepository
from index.embedding import MPNetAdapter

app = FastAPI(
    title="INDEX",
    description="Intelligent data steward toolbox using Large Language Model embeddings "
                "for automated Data-Harmonization .",
    version="0.0.1",
    terms_of_service="https://www.scai.fraunhofer.de/",
    contact={
        "name": "Dr. Marc Jacobs",
        "email": "marc.jacobs@scai.fraunhofer.de",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.info")
repository = SQLLiteRepository()
embedding_model = MPNetAdapter()


@app.get("/", include_in_schema=False)
def swagger_redirect():
    return RedirectResponse(url='/docs')


@app.get("/version", tags=["info"])
def get_current_version():
    return app.version


@app.put("/terminologies/{id}", tags=["terminologies"])
async def create_or_update_terminology(id: str, name: str):
    try:
        terminology = Terminology(name=name, id=id)
        repository.store(terminology)
        return {"message": f"Terminology {id} created or updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create or update terminology: {str(e)}")


@app.put("/concepts/{id}", tags=["concepts"])
async def create_or_update_concept(id: str, terminology_id: str, name: str):
    try:
        terminology = repository.session.query(Terminology).filter(Terminology.id == terminology_id).first()
        if not terminology:
            raise HTTPException(status_code=404, detail=f"Terminology with id {terminology_id} not found")

        concept = Concept(terminology=terminology, name=name, id=id)
        repository.store(concept)
        return {"message": f"Concept {id} created or updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create or update concept: {str(e)}")


@app.put("/mappings/", tags=["mappings"])
async def create_or_update_mapping(concept_id: str, text: str):
    try:
        concept = repository.session.query(Concept).filter(Concept.id == concept_id).first()
        if not concept:
            raise HTTPException(status_code=404, detail=f"Concept with id {concept_id} not found")
        embedding = embedding_model.get_embedding(text)
        mapping = Mapping(concept=concept, text=text, embedding=embedding)
        repository.store(mapping)
        return {"message": f"Mapping created or updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create or update mapping: {str(e)}")


@app.post("/mappings", tags=["mappings"])
async def get_closest_mappings_for_text(text: str):
    embedding = embedding_model.get_embedding(text).tolist()
    print(embedding)
    closest_mappings, similarities = repository.get_closest_mappings(embedding)
    response_data = []
    for mapping, similarity in zip(closest_mappings, similarities):
        concept = mapping.concept
        terminology = concept.terminology
        response_data.append({
            "concept": {
                "id": concept.id,
                "name": concept.name,
                "terminology": {
                    "id": terminology.id,
                    "name": terminology.name
                }
            },
            "text": mapping.text,
            "similarity": similarity
        })
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)

import logging

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

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


@app.post("/mappings", tags=["mappings"])
async def get_closest_mappings_for_text(text: str):
    embedding = embedding_model.get_embedding(text)
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

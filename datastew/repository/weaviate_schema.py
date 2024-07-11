terminology_schema = {
    "class": "Terminology",
    "description": "A terminology entry",
    "properties": [
        {
            "name": "name",
            "dataType": ["string"]
        }
    ]
}

concept_schema = {
    "class": "Concept",
    "description": "A concept entry",
    "properties": [
        {
            "name": "conceptID",
            "dataType": ["string"]
        },
        {
            "name": "prefLabel",
            "dataType": ["string"]
        },
        {
            "name": "hasTerminology",
            "dataType": ["Terminology"]
        }
    ]
}

mapping_schema = {
    "class": "Mapping",
    "description": "A mapping entry",
    "properties": [
        {
            "name": "text",
            "dataType": ["string"]
        },
        {
            "name": "vector",
            "dataType": ["number[]"]
        },
        {
            "name": "hasConcept",
            "dataType": ["Concept"]
        }
    ]
}
import pandas as pd


class Terminology:

    def __int__(self, identifier: str, name: str):
        self.identifier = identifier
        self.name = name


class Concept:

    def __init__(self, identifier: str, terminology: Terminology):
        self.identifier = identifier
        self.terminology = terminology


class Embedding:

    def __init__(self, embedding: [float], source: str):
        self.embedding = embedding
        self.source = source

    def to_dataframe(self):
        return pd.DataFrame(self.embedding, columns=[self.source])


class Variable:

    def __init__(
        self, name: str, description: str, source: str, embedding: Embedding = None
    ):
        self.name = name
        self.description = description
        self.source = source
        self.embedding = embedding


class Mapping:

    def __init__(self, concept: Concept, variable: Variable, source: str):
        self.concept = concept
        self.variable = variable
        self.source = source

    def __eq__(self, other):
        return (
            self.concept.identifier == other.concept.identifier
            and self.variable.name == other.variable.name
        )

    def __hash__(self):
        return hash((self.concept.identifier, self.variable.name))

    def __str__(self):
        return f"{self.variable.name} ({self.variable.description}) -> {self.concept.identifier}"

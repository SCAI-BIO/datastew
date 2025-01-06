import pandas as pd
import numpy as np

from datastew.embedding import EmbeddingModel
from datastew.process.parsing import MappingSource, DataDictionarySource, EmbeddingSource


class _Terminology:

    def __int__(self, identifier: str, name: str):
        self.identifier = identifier
        self.name = name


class _Concept:

    def __init__(self, identifier: str, terminology: _Terminology):
        self.identifier = identifier
        self.terminology = terminology


class _Embedding:

    def __init__(self, embedding: [float], source: str):
        self.embedding = embedding
        self.source = source

    def to_dataframe(self):
        return pd.DataFrame(self.embedding, columns=[self.source])


class _Variable:

    def __init__(self, name: str, description: str, source: str, embedding: _Embedding = None):
        self.name = name
        self.description = description
        self.source = source
        self.embedding = embedding


class _Mapping:

    def __init__(self, concept: _Concept, variable: _Variable, source: str):
        self.concept = concept
        self.variable = variable
        self.source = source

    def __eq__(self, other):
        return self.concept.identifier == other.concept.identifier and self.variable.name == other.variable.pref_label

    def __hash__(self):
        return hash((self.concept.identifier, self.variable.name))

    def __str__(self):
        return f"{self.variable.name} ({self.variable.description}) -> {self.concept.identifier}"


class _MappingTable:

    def __init__(self, mapping_source: MappingSource,
                 data_dictionary_source: DataDictionarySource = None,
                 embedding_source: EmbeddingSource = None,
                 terminology: _Terminology = None):
        self.mapping_source: MappingSource = mapping_source
        self.data_dictionary_source: DataDictionarySource = data_dictionary_source
        self.embedding_source: EmbeddingSource = embedding_source
        self.terminology = terminology
        self.joined_mapping_table: pd.DataFrame = self.mapping_source.to_dataframe()
        if self.data_dictionary_source is not None:
            self.add_descriptions(data_dictionary_source)
        if self.embedding_source is not None:
            self.add_embeddings(embedding_source)

    def set_dictionary_source(self, data_dictionary_source: DataDictionarySource):
        self.data_dictionary_source = data_dictionary_source

    def set_terminology(self, terminology: _Terminology):
        self.terminology = terminology

    def set_mapping_source(self, mapping_source: MappingSource):
        self.mapping_source = mapping_source

    def add_descriptions(self, data_dictionary_source: DataDictionarySource):
        self.data_dictionary_source = data_dictionary_source
        data_dictionary_df = data_dictionary_source.to_dataframe()
        # FIXME: Join results in duplicate entries
        self.joined_mapping_table = pd.merge(self.joined_mapping_table, data_dictionary_df,
                                             left_on="variable",
                                             right_on="variable",
                                             how="left").drop_duplicates()

    def add_embeddings(self, embedding_source: EmbeddingSource):
        self.embedding_source = embedding_source
        # FIXME: Join results in duplicate entries
        self.joined_mapping_table = pd.merge(self.joined_mapping_table, embedding_source.to_dataframe(),
                                             left_on="description",
                                             right_on="description")

    def get_embeddings(self):
        if "embedding" not in self.joined_mapping_table.columns:
            raise ValueError("No embeddings found in mapping table.")
        if "description" not in self.joined_mapping_table.columns:
            raise ValueError("No descriptions found in mapping table.")
        else:
            return self.joined_mapping_table["embedding"].apply(np.array)

    def get_embeddings_numpy(self):
        return np.array(self.joined_mapping_table["embedding"].dropna().tolist())

    def save_embeddings(self, output_path: str):
        self.get_embeddings().to_csv(output_path, index=False)
        self.embedding_source = EmbeddingSource(output_path)

    def compute_embeddings(self, model: EmbeddingModel):
        descriptions = self.joined_mapping_table["description"].dropna().unique().tolist()
        embeddings = model.get_embeddings(descriptions)
        embedding_df = pd.DataFrame({"description": descriptions, "embedding": embeddings})
        self.joined_mapping_table = pd.merge(self.joined_mapping_table, embedding_df,
                                             left_on="description",
                                             right_on="description",
                                             how="left")

    def export_embeddings(self, output_path: str):
        descriptions = self.joined_mapping_table["description"].dropna().unique().tolist()
        embedding_df = pd.DataFrame({"description": descriptions, "embedding": self.joined_mapping_table["embedding"]})
        embedding_df.to_csv(output_path)

    def import_embeddings(self, input_path: str):
        embeddings = pd.read_csv(input_path)
        self.joined_mapping_table["embedding"] = embeddings["embedding"]

    def get_mapping_table(self) -> pd.DataFrame:
        return self.joined_mapping_table

    def get_mappings(self) -> [_Mapping]:
        mappings = []
        for index, row in self.joined_mapping_table.iterrows():
            concept_id = row["identifier"]
            variable_name = row["variable"]
            if self.data_dictionary_source is not None:
                description = row["description"]
            else:
                description = None
            if not pd.isna(concept_id) and not pd.isna(variable_name):
                concept = _Concept(concept_id, self.terminology)
                variable = _Variable(variable_name, description,
                                     self.data_dictionary_source.file_path
                                    if self.data_dictionary_source is not None else None)
                mapping = _Mapping(concept, variable, self.mapping_source.file_path)
                mappings.append(mapping)
        # remove duplicates
        return list(dict.fromkeys(mappings))

    def to_mapping_dto(self) -> [_Mapping]:
        mappings = []
        for index, row in self.joined_mapping_table.iterrows():
            concept_id = row["identifier"]
            variable_name = row["variable"]
            if self.data_dictionary_source is not None:
                description = row["description"]
            else:
                description = None
            if not pd.isna(concept_id) and not pd.isna(variable_name):
                concept = _Concept(concept_id, self.terminology)
                variable = _Variable(variable_name, description,
                                     self.data_dictionary_source.file_path
                                    if self.data_dictionary_source is not None else None)
                mapping = _Mapping(concept, variable, self.mapping_source.file_path)
                mappings.append(mapping)
        # remove duplicates
        return list(dict.fromkeys(mappings))


def parse_float_array(s):
    return [float(x) for x in s.strip("[]").split(",")]

from abc import ABC
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from datastew.embedding import EmbeddingModel, MPNetAdapter


class Source(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def to_dataframe(self) -> pd.DataFrame:
        if self.file_path.endswith(".csv"):
            return pd.read_csv(self.file_path)
        # back to general encodings
        elif self.file_path.endswith(".tsv"):
            return pd.read_csv(self.file_path, sep="\t")
        elif self.file_path.endswith(".xlsx"):
            with pd.ExcelFile(self.file_path) as xls:
                dfs = [pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names]
            for df in dfs:
                # Replace control sequences in string columns / headers & remove trailing whitespaces
                df.columns = df.columns.str.replace("\r", "", regex=True).str.strip()
                string_columns = df.select_dtypes(include=["object"]).columns
                df[string_columns] = df[string_columns].apply(lambda x: x.str.replace("\r", "").str.strip(), axis=1)
                combined_df = pd.concat(dfs, ignore_index=True)
            return combined_df
        else:
            raise ValueError("Unsupported file extension")


class MappingSource(Source):
    """
    Contains curated mapping of variable -> concept identifier
    """

    def __init__(self, file_path: str, variable_field: str, identifier_field: str):
        self.variable_field = variable_field
        self.identifier_field = identifier_field
        self.file_path = file_path

    def to_dataframe(self) -> pd.DataFrame:
        df = super().to_dataframe()
        # sanity check
        if self.variable_field not in df.columns:
            raise ValueError(f"Variable field {self.variable_field} not found in {self.file_path}")
        if self.identifier_field not in df.columns:
            raise ValueError(f"Identifier field {self.identifier_field} not found in {self.file_path}")
        df = df[[self.variable_field, self.identifier_field]]
        df = df.rename(columns={self.variable_field: "variable", self.identifier_field: "identifier"})
        df.dropna(subset=["variable", "identifier"], inplace=True)
        return df


class DataDictionarySource(Source):

    def __init__(self, file_path: str, variable_field: str, description_field: str):
        """
        Initialize the DataDictionarySource with the path to the data dictionary file
        and the fields that represent the variables and their descriptions.

        :param file_path: Path to the data dictionary file.
        :param variable_field: The column that contains the variable names.
        :param description_field: The column that contains the variable descriptions.
        """
        self.file_path: str = file_path
        self.variable_field: str = variable_field
        self.description_field: str = description_field

    def to_dataframe(self, dropna: bool = True) -> pd.DataFrame:
        """
        Load the data dictionary file into a pandas DataFrame, select the variable and 
        description fields, and ensure they exist. Optionally remove rows with missing 
        variables or descriptions based on the 'dropna' parameter.

        :param dropna: If True, rows with missing 'variable' or 'description' values are 
                       dropped. Defaults to True.
        :return: A DataFrame containing two columns:
                 - 'variable': The variable names from the data dictionary.
                 - 'description': The descriptions corresponding to each variable.
        :raises ValueError: If either the variable field or the description field is not 
                            found in the data dictionary file.
        """
        df = super().to_dataframe()
        # sanity check
        if self.variable_field not in df.columns:
            raise ValueError(f"Variable field {self.variable_field} not found in {self.file_path}")
        if self.description_field not in df.columns:
            raise ValueError(f"Description field {self.description_field} not found in {self.file_path}")
        df = df[[self.variable_field, self.description_field]]
        df = df.rename(columns={self.variable_field: "variable", self.description_field: "description"})
        if dropna:
            df.dropna(subset=["variable", "description"], inplace=True)
        return df
    
    def get_embeddings(self, embedding_model: Optional[EmbeddingModel] = None) -> Dict[str, Sequence[float]]:
        """
        Compute embedding vectors for each description in the data dictionary. The 
        resulting vectors are mapped to their respective variables and returned as a 
        dictionary.

        :param embedding_model: The embedding model used to compute embeddings for the descriptions. Defaults to None.
        :return: A dictionary where each key is a variable name and the value is the  embedding vector for the
            corresponding description.
        """
        # Compute vectors for all descriptions
        df: pd.DataFrame = self.to_dataframe()
        descriptions: list[str] = df["description"].tolist()
        if embedding_model is None:
            embedding_model = MPNetAdapter()
        embeddings = embedding_model.get_embeddings(descriptions)
        # variable identify descriptions -> variable to embedding
        variable_to_embedding: Dict[str, Sequence[float]] = dict(zip(df["variable"], embeddings))
        return variable_to_embedding
        

class EmbeddingSource:
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.description_field = "description"
        self.embedding_field = "embedding"

    def to_dataframe(self):
        return pd.read_csv(self.source_path)

    def to_numpy(self):
        # TODO: this should be default
        df = self.to_dataframe()
        return np.array([parse_float_array(s) for s in df["embedding"].tolist()])

    def export(self, dst_path: str):
        self.to_dataframe().to_csv(dst_path)


def parse_float_array(s):
    return [float(x) for x in s.strip("[]").split(",")]


class ConceptSource:
    """
    identifier -> description
    """

    pass
    pass

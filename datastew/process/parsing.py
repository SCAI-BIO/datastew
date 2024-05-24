from abc import ABC

import pandas as pd
import numpy as np


class Source(ABC):
    def __int__(self, file_path: str):
        self.file_path = file_path

    def to_dataframe(self) -> pd.DataFrame:
        if self.file_path.endswith(".csv"):
            return pd.read_csv(self.file_path)
        # back to general encodings
        elif self.file_path.endswith(".tsv"):
            return pd.read_csv(self.file_path, sep="\t")
        elif self.file_path.endswith(".xlsx"):
            xls = pd.ExcelFile(self.file_path)
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
    """
    Contains mapping of variable -> description
    """

    def __init__(self, file_path: str, variable_field: str, description_field: str):
        self.file_path = file_path
        self.variable_field = variable_field
        self.description_field = description_field

    def to_dataframe(self) -> pd.DataFrame:
        df = super().to_dataframe()
        # sanity check
        if self.variable_field not in df.columns:
            raise ValueError(f"Variable field {self.variable_field} not found in {self.file_path}")
        if self.description_field not in df.columns:
            raise ValueError(f"Description field {self.description_field} not found in {self.file_path}")
        df = df[[self.variable_field, self.description_field]]
        df = df.rename(columns={self.variable_field: "variable", self.description_field: "description"})
        df.dropna(subset=["variable", "description"], inplace=True)
        return df


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

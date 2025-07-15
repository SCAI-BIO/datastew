from abc import ABC, abstractmethod
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from datastew.embedding import Vectorizer


class Source(ABC):
    """
    Abstract base class for all data sources. Handles file loading and basic preprocessing.
    """

    def __init__(self, file_path: str):
        """
        :param file_path: Path to the input file (.csv, .tsv, or .xlsx)
        """
        self.file_path = file_path

    @property
    @abstractmethod
    def required_fields(self) -> Dict[str, str]:
        """Returns a mapping from original column names to standardized names required for downstream processing."""
        pass

    def to_dataframe(self, dropna: bool = True) -> pd.DataFrame:
        """Loads the file, renames required columns, and optionally drops rows with missing values.

        :param dropna: If True, drop rows with missing values in any required field, defaults to True
        :return: Cleaned DataFrame with standardized column names.
        """
        raw_df = self._load_dataframe()
        return self._select_and_rename_columns(raw_df, self.required_fields, dropna)

    def _load_dataframe(self) -> pd.DataFrame:
        """Loads the raw data file based on its extension.

        :raises ValueError: If file extension is unsupported.
        :return: A pandas DataFrame with raw content.
        """
        if self.file_path.endswith(".csv"):
            return pd.read_csv(self.file_path)
        elif self.file_path.endswith(".tsv"):
            return pd.read_csv(self.file_path, sep="\t")
        elif self.file_path.endswith(".xlsx"):
            with pd.ExcelFile(self.file_path) as xls:
                dfs = [pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names]
            return self._clean_excel_sheets(dfs)
        else:
            raise ValueError(f"Unsupported file extension: {self.file_path}")

    def _select_and_rename_columns(
        self, df: pd.DataFrame, required: Dict[str, str], dropna: bool = True
    ) -> pd.DataFrame:
        """Selects and renames specified columns in the DataFrame.

        :param df: The raw DataFrame.
        :param required: Mapping of original column names to new standardized names.
        :param dropna: If True, drops rows with missing values in required columns, defaults to True
        :raises ValueError: If any required field is not found in the DataFrame.
        :return: A DataFrame with selected and renamed columns.
        """
        for original in required.keys():
            if original not in df.columns:
                raise ValueError(f"Field '{original}' not found in {self.file_path}")
        df = df[list(required.keys())].rename(columns=required)
        if dropna:
            df.dropna(subset=list(required.values()), inplace=True)
        return df

    def _clean_excel_sheets(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Cleans up Excel sheet DataFrames by stripping whitespace and carriage returns from strings and headers.

        :param dfs: A list of DataFrames, one per sheet.
        :return: A single concatenated and cleaned DataFrame.
        """
        for df in dfs:
            # Replace control sequences in string columns / headers & remove trailing whitespaces
            df.columns = df.columns.str.replace("\r", "", regex=True).str.strip()
            string_columns = df.select_dtypes(include=["object"]).columns
            df[string_columns] = df[string_columns].apply(lambda x: x.str.replace("\r", "").str.strip(), axis=1)
        return pd.concat(dfs, ignore_index=True)


class MappingSource(Source):
    """
    Contains curated mapping of variable -> concept identifier
    """

    def __init__(self, file_path: str, variable_field: str, identifier_field: str):
        """
        :param file_path: Path to the mapping file.
        :param variable_field: Column name containing variable names.
        :param identifier_field: Column name containing concept identifiers.
        """
        super().__init__(file_path)
        self.variable_field = variable_field
        self.identifier_field = identifier_field

    @property
    def required_fields(self) -> Dict[str, str]:
        return {self.variable_field: "variable", self.identifier_field: "identifier"}


class DataDictionarySource(Source):
    """
    Source class for loading variable descriptions from a data dictionary file.
    """

    def __init__(self, file_path: str, variable_field: str, description_field: str):
        """
        :param file_path: Path to the data dictionary file.
        :param variable_field: The column that contains the variable names.
        :param description_field: The column that contains the variable descriptions.
        """
        super().__init__(file_path)
        self.variable_field: str = variable_field
        self.description_field: str = description_field

    @property
    def required_fields(self) -> Dict[str, str]:
        return {self.variable_field: "variable", self.description_field: "description"}

    def get_embeddings(self, vectorizer: Vectorizer = Vectorizer()) -> Dict[str, Sequence[float]]:
        """Computes embedding vectors for each variable's description.

        :param vectorizer: Vectorizer instance used to compute embeddings, defaults to Vectorizer().
        :return: Dictionary mapping each variable to its corresponding embedding vector.
        """
        df = self.to_dataframe()
        descriptions: list[str] = df["description"].tolist()
        embeddings = vectorizer.get_embeddings(descriptions)
        return dict(zip(df["variable"], embeddings))


class EmbeddingSource(Source):
    """
    Source class for precomputed description -> embedding mappings.
    """

    def __init__(self, file_path: str, description_field: str, embedding_field: str):
        """
        :param file_path: Path to the file containing embeddings.
        :param description_field: Column name containing descriptions.
        :param embedding_field: Column name containing embedding vectors (as strings).
        """
        super().__init__(file_path)
        self.description_field = description_field
        self.embedding_field = embedding_field

    @property
    def required_fields(self) -> Dict[str, str]:
        return {self.description_field: "description", self.embedding_field: "embedding"}

    def to_numpy(self) -> np.ndarray:
        """Converts the embeddings column into a NumPy array.

        :return: A NumPy array where each row is an embedding vector.
        """
        # TODO: this should be default
        df = self.to_dataframe()
        return np.array([self._parse_float_array(s) for s in df["embedding"].tolist()])

    def export(self, dst_path: str):
        """Exports the internal DataFrame to a CSV file.

        :param dst_path: Destination path for the CSV file.
        """
        self.to_dataframe().to_csv(dst_path)

    def _parse_float_array(self, s: str) -> Sequence[float]:
        """Parses a stringified float array (e.g., "[0.1, 0.2, 0.3]") into a Python list.

        :param s: String representation of a float array.
        :raises ValueError: If parsing fails due to formatting issues.
        :return: Parsed list of floats.
        """
        try:
            return [float(x) for x in s.strip("[]").split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(f"Failed to parse embedding from string: '{s}'") from e


class ConceptSource(Source):
    """
    Source class for concept dictionaries mapping identifiers to descriptions.
    """

    def __init__(self, file_path: str, identifier_field: str, description_field: str):
        """
        :param file_path: Path to the concept dictionary file.
        :param identifier_field: Column name for concept identifiers.
        :param description_field: Column name for human-readable descriptions.
        """
        super().__init__(file_path)
        self.identifier_field = identifier_field
        self.description_field = description_field

    @property
    def required_fields(self) -> Dict[str, str]:
        return {self.identifier_field: "identifier", self.description_field: "description"}

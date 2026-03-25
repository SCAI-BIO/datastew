from abc import ABC, abstractmethod
from typing import Sequence

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
    def required_fields(self) -> dict[str, str]:
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
        """Loads the raw data file based on its extension, filtering for required columns early.

        :raises ValueError: If file extension is unsupported.
        :return: A pandas DataFrame with raw content.
        """
        file_ext = self.file_path.lower()
        required_cols = list(self.required_fields.keys())

        def usecols_func(col: str) -> bool:
            return col in required_cols

        if file_ext.endswith(".csv"):
            return pd.read_csv(self.file_path, usecols=usecols_func)
        elif file_ext.endswith(".tsv"):
            return pd.read_csv(self.file_path, sep="\t", usecols=usecols_func)
        elif file_ext.endswith(".xlsx"):
            with pd.ExcelFile(self.file_path) as xls:
                dfs = [
                    pd.read_excel(xls, sheet_name=sheet_name, usecols=usecols_func) for sheet_name in xls.sheet_names
                ]
            return self._clean_excel_sheets(dfs)
        else:
            raise ValueError(f"Unsupported file extension: {self.file_path}")

    def _select_and_rename_columns(
        self, df: pd.DataFrame, required: dict[str, str], dropna: bool = True
    ) -> pd.DataFrame:
        """Selects and renames specified columns in the DataFrame.

        :param df: The raw DataFrame.
        :param required: Mapping of original column names to new standardized names.
        :param dropna: If True, drops rows with missing values in required columns, defaults to True.
        :raises ValueError: If any required field is not found in the DataFrame.
        :return: A DataFrame with selected and renamed columns.
        """
        missing_cols = [col for col in required.keys() if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Required fields '{missing_cols}' not found in {self.file_path}")

        df = df[list(required.keys())].rename(columns=required)
        if dropna:
            df.dropna(subset=list(required.values()), inplace=True)

        return df

    def _clean_excel_sheets(self, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """Cleans up Excel sheet DataFrames by stripping whitespace and carriage returns.

        :param dfs: A list of DataFrames, one per sheet.
        :return: A single concatenated and cleaned DataFrame.
        """
        cleaned_dfs = []
        for df in dfs:
            # Clean headers
            df.columns = df.columns.str.replace("\r", "", regex=False).str.strip()

            # Vectorized cleaning for string columns
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].astype(str).str.replace("\r", "", regex=False).str.strip()

            cleaned_dfs.append(df)

        return pd.concat(cleaned_dfs, ignore_index=True)


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
    def required_fields(self) -> dict[str, str]:
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
    def required_fields(self) -> dict[str, str]:
        return {self.variable_field: "variable", self.description_field: "description"}

    def get_embeddings(self, vectorizer: Vectorizer = Vectorizer()) -> dict[str, Sequence[float]]:
        """Computes embedding vectors for each variable's description.

        :param vectorizer: Vectorizer instance used to compute embeddings, defaults to Vectorizer().
        :return: Dictionary mapping each variable to its corresponding embedding vector.
        """
        df = self.to_dataframe()
        descriptions = df["description"].tolist()
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
    def required_fields(self) -> dict[str, str]:
        return {self.description_field: "description", self.embedding_field: "embedding"}

    def to_numpy(self) -> np.ndarray:
        """Converts the embeddings column into a NumPy array.

        :return: A NumPy array where each row is an embedding vector.
        """
        df = self.to_dataframe()
        return np.array([self._parse_float_array(s) for s in df["embedding"]])

    def export(self, dst_path: str):
        """Exports the internal DataFrame to a CSV file.

        :param dst_path: Destination path for the CSV file.
        """
        self.to_dataframe().to_csv(dst_path, index=False)

    def _parse_float_array(self, s: str) -> Sequence[float]:
        """Parses a stringified float array (e.g., "[0.1, 0.2, 0.3]") into a Python list.

        :param s: String representation of a float array.
        :raises ValueError: If parsing fails due to formatting issues.
        :return: Parsed list of floats.
        """
        try:
            return np.fromstring(s.strip("[]"), sep=",").tolist()
        except Exception as e:
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
    def required_fields(self) -> dict[str, str]:
        return {self.identifier_field: "identifier", self.description_field: "description"}

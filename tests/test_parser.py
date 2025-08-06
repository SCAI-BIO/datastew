import os
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from datastew.process.parsing import DataDictionarySource, EmbeddingSource, MappingSource


class TestParsing(TestCase):
    TEST_DIR = os.path.dirname(os.path.realpath(__file__))

    def setUp(self):
        self.data_dict_path = os.path.join(self.TEST_DIR, "resources", "test_data_dict.csv")
        self.data_dict_excel_path = os.path.join(self.TEST_DIR, "resources", "test_data_dict.xlsx")
        self.mapping_path = os.path.join(self.TEST_DIR, "resources", "test_mapping.xlsx")
        self.embedding_path = os.path.join(self.TEST_DIR, "resources", "test_embedding.csv")

    def test_parse_mapping(self):
        mapping_source = MappingSource(self.mapping_path, "VAR_1", "ID_1")
        df = mapping_source.to_dataframe()
        self.assertIn("variable", df.columns)
        self.assertIn("identifier", df.columns)

    def test_parse_data_dict(self):
        data_dict_source = DataDictionarySource(self.data_dict_path, "VAR_1", "DESC")
        df = data_dict_source.to_dataframe()
        self.assertIn("variable", df.columns)
        self.assertIn("description", df.columns)

    def test_parse_data_dict_wrong_column_name(self):
        data_dict_source = DataDictionarySource(self.data_dict_path, "NOT_EXIST", "ID")
        with self.assertRaises(ValueError):
            data_dict_source.to_dataframe()

    def test_parse_data_dict_excel(self):
        data_dict_source = DataDictionarySource(self.data_dict_excel_path, "VAR_1", "DESC")
        df = data_dict_source.to_dataframe()
        self.assertIn("variable", df.columns)
        self.assertIn("description", df.columns)

    @patch("datastew.process.parsing.Vectorizer")
    def test_get_embeddings(self, mock_vectorizer_class):
        mock_vectorizer = Mock()
        mock_vectorizer.get_embeddings.return_value = [[0.1 * 5]] * 11
        mock_vectorizer_class.return_value = mock_vectorizer

        source = DataDictionarySource(self.data_dict_path, "VAR_1", "DESC")
        embeddings = source.get_embeddings(vectorizer=mock_vectorizer)
        self.assertEqual(len(embeddings), 11)
        self.assertIn("Q_8", embeddings)

    def test_embedding_source_to_numpy(self):
        source = EmbeddingSource(self.embedding_path, "DESC", "EMB")
        arr = source.to_numpy()
        np.testing.assert_array_equal(arr, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_embedding_source_export(self):
        try:
            dst_path = self.embedding_path.replace(".csv", "_out.csv")
            source = EmbeddingSource(self.embedding_path, "DESC", "EMB")
            source.export(dst_path)
            exported_df = pd.read_csv(dst_path)
            self.assertIn("embedding", exported_df.columns)
        finally:
            if os.path.exists(dst_path):
                os.remove(dst_path)

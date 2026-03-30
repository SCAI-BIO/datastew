import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from datastew.io.source import DataDictionarySource
from datastew.visualisation import (
    _get_off_diag_mean,
    bar_chart_average_acc_two_distributions,
    enrichment_plot,
    get_plot_for_current_database_state,
    plot_embeddings,
)


class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
        cls.dict_path_1 = os.path.join(cls.TEST_DIR_PATH, "resources", "data_dict1.csv")
        cls.dict_path_2 = os.path.join(cls.TEST_DIR_PATH, "resources", "data_dict2.csv")

    def test_off_diag_calculation(self):
        """Verify the mathematical correctness of off-diagnol averaging."""
        df = pd.DataFrame({"A": [1.0, 0.4], "B": [0.2, 1.0]})
        result = _get_off_diag_mean(df)
        self.assertAlmostEqual(result, 0.3)
        self.assertIsInstance(result, float)

    def test_enrichment_plot_mismatched_lengths(self):
        """Ensure ValueError is raised when input lists have different lengths."""
        with self.assertRaises(ValueError):
            enrichment_plot([0.5], [0.5, 0.6], [0.5], "Should Fail")

    def test_bar_chart_non_square_matrix(self):
        """Ensure ValueError is raised if matrices are not square (needed for diagonal masking)."""
        df_rect = pd.DataFrame(np.random.rand(3, 2))
        df_sq = pd.DataFrame(np.random.rand(3, 3))
        with self.assertRaises(ValueError):
            bar_chart_average_acc_two_distributions(df_rect, df_sq, df_sq, df_sq, df_sq, df_sq, "Title", "L1", "L2")

    @patch("matplotlib.pyplot.show")
    def test_enrichment_plot(self, mock_show):
        """Verify the function completes and calls the plot display."""
        acc = [0.4, 0.6, 0.8]
        enrichment_plot(acc, acc, acc, "Test Title", save_plot=False)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_bar_chart_execution(self, mock_show):
        """Verify bar chart logic executes and displays."""
        labels = ["M1", "M2"]
        df = pd.DataFrame({"M1": [1, 0.5], "M2": [0.5, 1]}, index=labels).T
        bar_chart_average_acc_two_distributions(df, df, df, df, df, df, "Title", "A", "B")
        mock_show.assert_called_once()

    @patch("plotly.graph_objects.Figure.show")
    def test_plot_data_dict(self, mock_plotly_show):
        """Verify t-SNE logic and Plotly integration."""
        if os.path.exists(self.dict_path_1) and os.path.exists(self.dict_path_2):
            source1 = DataDictionarySource(self.dict_path_1, "VAR", "DESC")
            source2 = DataDictionarySource(self.dict_path_2, "VAR", "DESC")
            plot_embeddings([source1, source2], perplexity=1)
            mock_plotly_show.assert_called()
        else:
            self.skipTest("Test resource files missing; skipping embedding plot test.")

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    @patch("os.makedirs")
    def test_enrichment_plot_save(self, mock_makedirs, mock_savefig, mock_show):
        """Cover directory creation and plot saving."""
        acc = [0.4, 0.6, 0.8]
        enrichment_plot(acc, acc, acc, "Test Title", save_plot=True, save_dir="/tmp/mock_dir")
        mock_makedirs.assert_called_once_with("/tmp/mock_dir", exist_ok=True)
        mock_savefig.assert_called_once()

    def test_bar_chart_dimension_mismatch(self):
        """Cover dimension equality validation."""
        df_2x2 = pd.DataFrame(np.random.rand(2, 2))
        df_3x3 = pd.DataFrame(np.random.rand(3, 3))
        with self.assertRaisesRegex(ValueError, "same dimensions"):
            bar_chart_average_acc_two_distributions(
                df_2x2, df_3x3, df_2x2, df_2x2, df_2x2, df_2x2, "Title", "L1", "L2"
            )

    def test_bar_chart_label_mismatch(self):
        """Cover row/column label matching validation."""
        df1 = pd.DataFrame(np.random.rand(2, 2), index=["A", "B"], columns=["A", "B"])
        df2 = pd.DataFrame(np.random.rand(2, 2), index=["C", "D"], columns=["C", "D"])
        with self.assertRaisesRegex(ValueError, "Row and column labels must match"):
            bar_chart_average_acc_two_distributions(df1, df2, df1, df1, df1, df1, "Title", "L1", "L2")

    def test_get_plot_empty_database(self):
        """Cover early return for empty database state."""
        mock_repo = MagicMock()
        mock_repo.get_mappings.return_value.items = []
        result = get_plot_for_current_database_state(mock_repo)
        self.assertEqual(result, "<b>No entries found in the database.</b>")

    @patch("plotly.graph_objects.Figure.to_html")
    def test_get_plot_valid_data_html(self, mock_to_html):
        """Cover database plot HTML generation."""
        mock_to_html.return_value = "<html>mock</html>"
        mock_repo = MagicMock()
        mock_repo.get_mappings.return_value.items = [
            MagicMock(embedding=[float(i), float(i + 1), float(i + 2)]) for i in range(4)
        ]
        result = get_plot_for_current_database_state(mock_repo, return_type="html")
        self.assertEqual(result, "<html>mock</html>")
        mock_to_html.assert_called_once()

    @patch("plotly.graph_objects.Figure.to_json")
    def test_get_plot_valid_data_json(self, mock_to_json):
        """Cover database plot JSON generation."""
        mock_to_json.return_value = '{"mock": "json"}'
        mock_repo = MagicMock()
        mock_repo.get_mappings.return_value.items = [
            MagicMock(embedding=[float(i), float(i + 1), float(i + 2)]) for i in range(4)
        ]
        result = get_plot_for_current_database_state(mock_repo, return_type="json")
        self.assertEqual(result, '{"mock": "json"}')
        mock_to_json.assert_called_once()

    @patch("datastew.visualisation._get_safe_perplexity")
    def test_get_plot_too_few_entries(self, mock_safe_perp):
        """Cover fallback return when n_samples <= actual_perplexity."""
        mock_safe_perp.return_value = 100
        mock_repo = MagicMock()
        mock_mapping = MagicMock()
        mock_mapping.embedding = [0.1, 0.2]
        mock_repo.get_mappings.return_value.items = [mock_mapping]
        result = get_plot_for_current_database_state(mock_repo)
        self.assertEqual(result, "<b>Too few database entries to visualize</b>")

    @patch("builtins.print")
    def test_plot_embeddings_insufficient_data(self, mock_print):
        """Cover insufficient data terminal output in plot_embeddings."""
        mock_dict = MagicMock()
        mock_dict.name = "Test Source"
        mock_dict.get_embeddings.return_value = {"id1": [0.1, 0.2]}
        mock_dict.to_dataframe.return_value = pd.DataFrame({"description": ["test desc"]})
        plot_embeddings([mock_dict], perplexity=5)
        mock_print.assert_called_once_with("Insufficient data points (1) to visualize with perplexity 5.")


if __name__ == "__main__":
    unittest.main()

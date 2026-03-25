import os
import unittest
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from datastew.io.source import DataDictionarySource
from datastew.visualisation import (
    _get_off_diag_mean,
    bar_chart_average_acc_two_distributions,
    enrichment_plot,
    plot_embeddings,
)


class TestVisualization(TestCase):
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


if __name__ == "__main__":
    unittest.main()

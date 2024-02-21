import os
from unittest import TestCase

import numpy as np
import pandas as pd

from index.evaluation import evaluate
from index.mapping import MappingTable
from index.parsing import MappingSource, DataDictionarySource
from index.visualisation import scatter_plot_two_distributions, enrichment_plot, scatter_plot_all_cohorts, \
    bar_chart_average_acc_two_distributions


class Test(TestCase):
    TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    mapping_source = MappingSource(os.path.join(TEST_DIR_PATH, "resources", "test_mapping.xlsx"), "VAR_1", "ID_1")
    data_dictionary_source = DataDictionarySource(os.path.join(TEST_DIR_PATH, "resources", "test_data_dict.csv"), "VAR_1", "DESC")

    embeddings1 = [
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
        [7.7, 8.8, 9.9],
        np.nan,
        [13.4, 14.5, 15.6],
        [16.7, 17.8, 18.9],
        [19.1, 20.2, 21.3],
        [22.4, 23.5, 24.6],
        [25.7, 26.8, 27.9],
        [28.1, 29.2, 30.3],
        [31.4, 32.5, 33.6],
    ]

    embeddings2 = [
        [2.1, 3.2, 4.3],
        [5.4, 6.5, 7.6],
        [8.7, 9.8, 10.9],
        [11.1, 12.2, 13.3],
        [14.4, 15.5, 16.6],
        [17.7, 18.8, 19.9],
        [20.1, 21.2, 22.3],
        [23.4, 24.5, 25.6],
        [26.7, 27.8, 28.9],
        np.nan,
        [32.4, 33.5, 34.6],
    ]

    embeddings3 = [
        [3.1, 4.2, 5.3],
        np.nan,
        [9.7, 10.8, 11.9],
        [12.1, 13.2, 14.3],
        [15.4, 16.5, 17.6],
        [18.7, 19.8, 20.9],
        [21.1, 22.2, 23.3],
        [24.4, 25.5, 26.6],
        [27.7, 28.8, 29.9],
        [30.1, 31.2, 32.3],
        [33.4, 34.5, 35.6],
    ]

    embeddings4 = [
        np.nan,
        [7.4, 8.5, 9.6],
        [10.7, 11.8, 12.9],
        [13.1, 14.2, 15.3],
        [16.4, 17.5, 18.6],
        [19.7, 20.8, 21.9],
        [22.1, 23.2, 24.3],
        [25.4, 26.5, 27.6],
        [28.7, 29.8, 30.9],
        [31.1, 32.2, 33.3],
        np.nan,
    ]

    def test_scatter_plot_two_distributions(self):
        mapping_table1 = MappingTable(self.mapping_source)
        mapping_table1.add_descriptions(self.data_dictionary_source)
        mapping_table2 = MappingTable(self.mapping_source)
        mapping_table2.add_descriptions(self.data_dictionary_source)
        mapping_table3 = MappingTable(self.mapping_source)
        mapping_table3.add_descriptions(self.data_dictionary_source)
        mapping_table4 = MappingTable(self.mapping_source)
        mapping_table4.add_descriptions(self.data_dictionary_source)

        mapping_table1.joined_mapping_table["embedding"] = self.embeddings1
        mapping_table2.joined_mapping_table["embedding"] = self.embeddings2
        mapping_table3.joined_mapping_table["embedding"] = self.embeddings3
        mapping_table4.joined_mapping_table["embedding"] = self.embeddings4
        scatter_plot_two_distributions([mapping_table1, mapping_table2], [mapping_table3, mapping_table4], "A", "B",
                                       store_html=False)

    def test_scatter_plot_all_cohorts(self):
        mapping_table1 = MappingTable(self.mapping_source)
        mapping_table1.add_descriptions(self.data_dictionary_source)
        mapping_table2 = MappingTable(self.mapping_source)
        mapping_table2.add_descriptions(self.data_dictionary_source)
        mapping_table3 = MappingTable(self.mapping_source)
        mapping_table3.add_descriptions(self.data_dictionary_source)
        mapping_table4 = MappingTable(self.mapping_source)
        mapping_table4.add_descriptions(self.data_dictionary_source)

        mapping_table1.joined_mapping_table["embedding"] = self.embeddings1
        mapping_table2.joined_mapping_table["embedding"] = self.embeddings2
        mapping_table3.joined_mapping_table["embedding"] = self.embeddings3
        mapping_table4.joined_mapping_table["embedding"] = self.embeddings4
        scatter_plot_all_cohorts([mapping_table1, mapping_table2], [mapping_table3, mapping_table4],
                                 ["A1", "A2"], ["B1", "B2"], store_html=False)

    def test_enrichment_plot(self):
        acc_gpt = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
        acc_mpnet = [0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
        acc_fuzzy = [0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
        title = "Test"
        enrichment_plot(acc_gpt, acc_mpnet, acc_fuzzy, title, save_plot=False)

    def test_bar_chart_average_acc_two_distributions(self):
        labels = ["M1", "M2", "M3"]
        fuzzy_1 = pd.DataFrame({"M1": [1, 0.2, 0.23], "M2": [0.3, 1, 0.16], "M3": [0.27, 0.22, 1]}, index=labels).T
        fuzzy_2 = pd.DataFrame({"M1": [1, 0.19, 0.21], "M2": [0.29, 1, 0.18], "M3": [0.29, 0.21, 1]}, index=labels).T
        gpt_1 = pd.DataFrame({"M1": [1, 0.9, 0.78], "M2": [0.8, 1, 0.78], "M3": [0.82, 0.89, 1]}, index=labels).T
        gpt_2 = pd.DataFrame({"M1": [1, 0.88, 0.78], "M2": [0.79, 1, 0.78], "M3": [0.81, 0.85, 1]}, index=labels).T
        mpnet_1 = pd.DataFrame({"M1": [1, 0.8, 0.7], "M2": [0.7, 0.9, 0.68], "M3": [0.72, 0.79, 0.9]}, index=labels).T
        mpnet_2 = pd.DataFrame({"M1": [0.9, 0.78, 0.68], "M2": [0.69, 0.9, 0.68], "M3": [0.71, 0.75, 0.9]}, index=labels).T

        bar_chart_average_acc_two_distributions(fuzzy_1, gpt_1, mpnet_1, fuzzy_2, gpt_2, mpnet_2, "title", "AD", "PD")

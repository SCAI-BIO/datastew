import os
from unittest import TestCase

import numpy as np

from datastew.evaluation import match_closest_descriptions, MatchingMethod, enrichment_analysis, score_mappings
from datastew.mapping import MappingTable
from datastew.process.parsing import MappingSource, DataDictionarySource


class Test(TestCase):

    TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    mapping_source = MappingSource(os.path.join(TEST_DIR_PATH, "resources", 'test_mapping.xlsx'), "VAR_1", "ID_1")
    data_dictionary_source = DataDictionarySource(os.path.join(TEST_DIR_PATH, "resources", 'test_data_dict.csv'),
                                                  "VAR_1", "DESC")

    def test_match_closest_descriptions_embeddings(self):
        mapping_table1 = MappingTable(self.mapping_source, self.data_dictionary_source)
        mapping_table2 = MappingTable(self.mapping_source, self.data_dictionary_source)
        # make the second mapping table shorter to test the case where there are more descriptions in the first
        mapping_table2.joined_mapping_table = mapping_table2.joined_mapping_table.iloc[:-2]
        embeddings1 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], np.nan, np.nan, [8, 8], [9, 9], np.nan]
        embeddings2 = [[0, 0], np.nan, [9, 9], [3, 3], [7, 7], [5.1, 5.1], [5, 5], [4, 4], np.nan]
        mapping_table1.joined_mapping_table['embedding'] = embeddings1
        mapping_table2.joined_mapping_table['embedding'] = embeddings2
        result = match_closest_descriptions(mapping_table1, mapping_table2)
        self.assertEqual(3, result["correct"].sum())

    def test_score_mappings(self):
        mapping_table1 = MappingTable(self.mapping_source, self.data_dictionary_source)
        mapping_table2 = MappingTable(self.mapping_source, self.data_dictionary_source)
        # make the second mapping table shorter to test the case where there are more descriptions in the first
        mapping_table2.joined_mapping_table = mapping_table2.joined_mapping_table.iloc[:-2]
        embeddings1 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], np.nan, np.nan, [8, 8], [9, 9], np.nan]
        embeddings2 = [[0, 0], np.nan, [9, 9], [3, 3], [7, 7], [5.1, 5.1], [5, 5], [4, 4], np.nan]
        mapping_table1.joined_mapping_table['embedding'] = embeddings1
        mapping_table2.joined_mapping_table['embedding'] = embeddings2
        # 2 should be correct out of a total of 4 valid mappings (possible matches, no nan)
        result = match_closest_descriptions(mapping_table1, mapping_table2)
        acc = score_mappings(result)
        self.assertEqual(3/5, acc)

    def test_match_closest_description_fuzzy(self):
        mapping_table1 = MappingTable(self.mapping_source, self.data_dictionary_source)
        mapping_table2 = MappingTable(self.mapping_source, self.data_dictionary_source)
        # make the second mapping table shorter to test the case where there are more descriptions in the first
        mapping_table2.joined_mapping_table = mapping_table2.joined_mapping_table.iloc[:-2]
        result = match_closest_descriptions(mapping_table1, mapping_table2,
                                            matching_method=MatchingMethod.FUZZY_STRING_MATCHING)
        self.assertEqual(7, result["correct"].sum())

    def test_enrichment_analysis_embeddings(self):
        mapping_table1 = MappingTable(self.mapping_source, self.data_dictionary_source)
        mapping_table2 = MappingTable(self.mapping_source, self.data_dictionary_source)
        # make the second mapping table shorter to test the case where there are more descriptions in the first
        mapping_table2.joined_mapping_table = mapping_table2.joined_mapping_table.iloc[:-2]
        embeddings1 = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], np.nan, np.nan, [8, 8], [9, 9], np.nan]
        embeddings2 = [[0, 0], np.nan, [9, 9], [3, 3], [7, 7], [5.1, 5.1], [5, 5], [4, 4], np.nan]
        mapping_table1.joined_mapping_table['embedding'] = embeddings1
        mapping_table2.joined_mapping_table['embedding'] = embeddings2
        result = enrichment_analysis(mapping_table1, mapping_table2, 5)
        self.assertListEqual([3 / 5, 3 / 5, 4 / 5, 4 / 5, 1], result.tolist())

    def test_enrichment_analysis_fuzzy(self):
        mapping_table1 = MappingTable(self.mapping_source, self.data_dictionary_source)
        mapping_table2 = MappingTable(self.mapping_source, self.data_dictionary_source)
        # make the second mapping table shorter to test the case where there are more descriptions in the first
        mapping_table2.joined_mapping_table = mapping_table2.joined_mapping_table.iloc[:-2]
        result = enrichment_analysis(mapping_table1, mapping_table2, 5,
                                     matching_method=MatchingMethod.FUZZY_STRING_MATCHING)
        self.assertListEqual([1, 1, 1, 1, 1], result.tolist())

import os
import unittest
from datastew.embedding import MPNetAdapter, TextEmbedding
import numpy as np

from datastew.process.mapping import map_dictionary_to_dictionary
from datastew.process.parsing import DataDictionarySource


class TestEmbedding(unittest.TestCase):

    TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    data_dictionary_source = DataDictionarySource(os.path.join(TEST_DIR_PATH, "resources", 'test_data_dict.csv'),
                                                  "VAR_1", "DESC")

    def test_map_dictionary_to_dictionary(self):
        df = map_dictionary_to_dictionary(self.data_dictionary_source, self.data_dictionary_source)
        print(df)
import os
from unittest import TestCase

from index.mapping import MappingTable
from index.process.parsing import MappingSource, DataDictionarySource


class Test(TestCase):
    TEST_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    mapping_source = MappingSource(os.path.join(TEST_DIR_PATH, "resources", 'test_mapping.xlsx'), "VAR_1", "ID_1")
    data_dictionary_source = DataDictionarySource(os.path.join(TEST_DIR_PATH, "resources", 'test_data_dict.csv'),
                                                  "VAR_1", "DESC")

    def test_parse(self):
        mapping_table = MappingTable(self.mapping_source, self.data_dictionary_source)
        mappings = mapping_table.get_mappings()
        self.assertEqual(11, len(mappings))
        self.assertEqual("brain", mappings[0].variable.description)

    def test_parse_no_data_dict(self):
        mapping_table = MappingTable(self.mapping_source)
        mappings = mapping_table.get_mappings()
        self.assertEqual(11, len(mappings))
        self.assertEqual(None, mappings[0].variable.description)

    def test_parse_add_description_later(self):
        mapping_table = MappingTable(self.mapping_source)
        mapping_table.add_descriptions(self.data_dictionary_source)
        mappings = mapping_table.get_mappings()
        self.assertEqual(11, len(mappings))
        self.assertEqual("brain", mappings[0].variable.description)

    def test_parse_data_dict_excel(self):
        mapping_table = MappingTable(self.mapping_source)
        data_dictionary_source = DataDictionarySource(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources", 'test_data_dict.xlsx'),
            "VAR_1", "DESC")
        mapping_table.add_descriptions(data_dictionary_source)
        mappings = mapping_table.get_mappings()
        self.assertEqual(11, len(mappings))

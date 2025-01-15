import os
from unittest import TestCase

from datastew import Terminology, Mapping, Concept, MPNetAdapter
from datastew.process.json_adapter import WeaviateJsonConverter
from datastew.repository import WeaviateRepository


class TestWeaviateRepositoryExport(TestCase):

    @classmethod
    def setUp(cls) -> None:

        cls.repository = WeaviateRepository()
        terminology = Terminology("snomed CT", "SNOMED")

        embedding_model = MPNetAdapter()

        text1 = "Diabetes mellitus (disorder)"
        concept1 = Concept(terminology, text1, "Concept ID: 11893007")
        mapping1 = Mapping(concept1, text1, embedding_model.get_embedding(text1), embedding_model.get_model_name())

        cls.repository.store_all([terminology, concept1, mapping1])

    def test_json_export(self):
        converter = WeaviateJsonConverter(dest_path="test_export.json")
        converter.from_repository(self.repository)
        # assert that the file was created
        self.assertTrue(converter.output_file_path)
        # assert that the file is not empty
        with open(converter.output_file_path, 'r') as file:
            self.assertTrue(file.read())
        # assert that the file contains the expected data
        with open(converter.output_file_path, 'r') as file:
            data = file.read()
            self.assertIn("Diabetes mellitus (disorder)", data)
            self.assertIn("Concept ID: 11893007", data)
            self.assertIn("snomed CT", data)
        # delete the generated export file from disk
        os.remove(converter.output_file_path)
        # close the repository
        self.repository.close()
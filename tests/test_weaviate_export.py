import os
import shutil
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

    @classmethod
    def tearDownClass(cls) -> None:
        cls.repository.close()
        shutil.rmtree(os.path.join(os.getcwd(), "db"))

    def test_json_export(self):
        converter = WeaviateJsonConverter(dest_dir="test_export")
        converter.from_repository(self.repository)
        # assert that the dest dir
        self.assertTrue(converter.dest_dir)
        # assert that the files is not empty
        with open(converter.dest_dir + "/terminology.json", 'r') as file:
            self.assertTrue(file.read())
        with open(converter.dest_dir + "/concept.json", 'r') as file:
            self.assertTrue(file.read())
        with open(converter.dest_dir + "/mapping.json", 'r') as file:
            self.assertTrue(file.read())
        # assert that the file contains the expected data
        with open(converter.dest_dir + "/terminology.json", 'r') as file:
            self.assertIn("snomed CT", file.read())
        with open(converter.dest_dir + "/concept.json", 'r') as file:
            self.assertIn("Diabetes mellitus (disorder)", file.read())
        with open(converter.dest_dir + "/mapping.json", 'r') as file:
            self.assertIn("Diabetes mellitus (disorder)", file.read())
        # remove the created dir and files
        os.remove(converter.dest_dir + "/terminology.json")
        os.remove(converter.dest_dir + "/concept.json")
        os.remove(converter.dest_dir + "/mapping.json")
        # remove the created dir
        os.rmdir(converter.dest_dir)
        # close the db connection
        self.repository.close()
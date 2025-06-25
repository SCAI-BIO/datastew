from datastew.process.jsonl_adapter import WeaviateJsonlConverter
from datastew.repository.weaviate import WeaviateRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestWeaviateRepository(BaseRepositoryTestSetup):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repository = WeaviateRepository(vectorizer=cls.vectorizer1)
        cls.jsonl_converter = WeaviateJsonlConverter(dest_dir="test_export")

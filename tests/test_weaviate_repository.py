from datastew.repository.weaviate import WeaviateRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestWeaviateRepository(BaseRepositoryTestSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repository = WeaviateRepository(vectorizer=cls.vectorizer1)

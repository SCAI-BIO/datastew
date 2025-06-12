from datastew.repository.weaviate import WeaviateRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestWeaviateRepository(BaseRepositoryTestSetup):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._repository_instance = WeaviateRepository(vectorizer=cls.vectorizer1)

    def setUp(self):
        self.repository = self._repository_instance
        super().setUp()

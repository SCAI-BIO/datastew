import os

from datastew.repository.postgresql import PostgreSQLRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestPostgreSQLRepository(BaseRepositoryTestSetup):

    POSTGRES_TEST_URL = os.getenv("TEST_POSTGRES_URI", "postgresql://testuser:testpass@localhost/testdb")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repo_args = (cls.POSTGRES_TEST_URL, cls.vectorizer1)
        cls._repository_instance = PostgreSQLRepository(*cls.repo_args)

    def setUp(self):
        self.repository = self._repository_instance
        super().setUp()

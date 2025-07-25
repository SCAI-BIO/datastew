import os

from datastew.process.jsonl_adapter import SQLJsonlConverter
from datastew.repository.postgresql import PostgreSQLRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestPostgreSQLRepository(BaseRepositoryTestSetup):
    __test__ = True
    POSTGRES_TEST_URL = os.getenv("TEST_POSTGRES_URI", "postgresql://testuser:testpass@localhost/testdb")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repo_args = (cls.POSTGRES_TEST_URL, cls.vectorizer1)
        cls.repository = PostgreSQLRepository(*cls.repo_args)
        cls.jsonl_converter = SQLJsonlConverter(dest_dir="test_export")

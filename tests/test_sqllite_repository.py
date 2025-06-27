from datastew.process.jsonl_adapter import SQLJsonlConverter
from datastew.repository.sqllite import SQLLiteRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestSQLLiteRepository(BaseRepositoryTestSetup):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repo_args = ("disk", "sqlite_db", cls.vectorizer1)
        cls.repository = SQLLiteRepository(*cls.repo_args)
        cls.jsonl_converter = SQLJsonlConverter(dest_dir="test_export")

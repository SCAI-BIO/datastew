from datastew.repository.sqllite import SQLLiteRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestSQLLiteRepository(BaseRepositoryTestSetup):
    __test__ = True

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repo_args = ("disk", "sqlite_db", cls.vectorizer1)
        cls.repository = SQLLiteRepository(*cls.repo_args)

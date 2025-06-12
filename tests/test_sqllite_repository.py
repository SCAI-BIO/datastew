from datastew.repository.sqllite import SQLLiteRepository
from tests.base_repository_test_setup import BaseRepositoryTestSetup


class TestSQLLiteRepository(BaseRepositoryTestSetup):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.repo_args = ("disk", "sqlite_db", cls.vectorizer1)
        cls._repository_instance = SQLLiteRepository(*cls.repo_args)

    def setUp(self):
        self.repository = self._repository_instance
        super().setUp()

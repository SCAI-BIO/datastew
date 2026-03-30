import os
import unittest
from unittest.mock import MagicMock, mock_open, patch

import requests

from datastew.integrations.ols.client import OlsClient


class TestOlsClient(unittest.TestCase):
    def setUp(self):
        """Setup mock vectorizer and default client parameters."""
        self.mock_vectorizer = MagicMock()
        self.mock_vectorizer.model_name = "mock-vectorizer-v1"
        self.mock_vectorizer.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

        self.ontology_id = "test_onto"
        self.client = OlsClient(vectorizer=self.mock_vectorizer, ontology_id=self.ontology_id, page_size=2)

    @patch("requests.get")
    def test_initialize_metadata_success(self, mock_get):
        """Tests successful retrieval of ontology metadata and page count."""
        mock_config_resp = MagicMock()
        mock_config_resp.json.return_value = {"config": {"title": "Test Ontology", "preferredPrefix": "TEST"}}
        mock_terms_resp = MagicMock()
        mock_terms_resp.json.return_value = {"page": {"totalPages": 5}}
        mock_get.side_effect = [mock_config_resp, mock_terms_resp]
        self.client._initialize_metadata()
        self.assertEqual(self.client._ontology_name, "Test Ontology")
        self.assertEqual(self.client._ontology_short_name, "TEST")
        self.assertEqual(self.client._num_pages, 5)

    @patch("requests.get")
    def test_initialize_metadata_failure(self, mock_get):
        """Tests that HTTP errors during initialization raise exceptions."""
        mock_get.side_effect = requests.exceptions.HTTPError("API Error")

        with self.assertRaises(requests.exceptions.HTTPError):
            self.client._initialize_metadata()

    @patch("requests.get")
    def test_fetch_page_data(self, mock_get):
        """Tests parsing logic for labels, single strings, lists, and missing descriptions."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "_embedded": {
                "terms": [
                    {"obo_id": "ID:1", "label": "Term 1", "description": ["List Desc"]},
                    {"obo_id": "ID:2", "label": "Term 2", "description": "String Desc"},
                    {"obo_id": "ID:3", "label": "Term 3"},
                    {"label": "No ID"},
                ]
            }
        }
        mock_get.return_value = mock_resp
        self.client._num_pages = 1
        self.mock_vectorizer.get_embeddings.return_value = [[0.1], [0.2], [0.3]]
        idents, labels, descs, embeddings = self.client._fetch_page_data(0)
        self.assertEqual(idents, ["ID:1", "ID:2", "ID:3"])
        self.assertEqual(labels, ["Term 1", "Term 2", "Term 3"])
        self.assertEqual(descs, ["List Desc", "String Desc", "Term 3"])
        self.assertEqual(len(embeddings), 3)

    @patch.object(OlsClient, "_initialize_metadata")
    @patch.object(OlsClient, "_fetch_page_data")
    def test_process_to_repository_success(self, mock_fetch, mock_init):
        """Tests the database processing loop, including commits and flushes."""
        self.client._ontology_name = "Test Ontology"
        self.client._ontology_short_name = "TEST"
        self.client._num_pages = 1
        mock_fetch.return_value = (["ID:1"], ["Term 1"], ["Desc 1"], [[0.1, 0.2]])
        mock_repo = MagicMock()
        mock_db_term = MagicMock()
        mock_db_term.id = 10
        mock_repo.get_terminology_by_name.return_value = mock_db_term
        mock_query = mock_repo.session.query.return_value
        mock_filter = mock_query.filter.return_value
        mock_filter.all.return_value = [(100, "ID:1")]
        self.client.process_to_repository(mock_repo)
        mock_init.assert_called_once()
        self.assertEqual(mock_repo.store.call_count, 3)
        self.assertTrue(mock_repo.session.flush.called)
        self.assertTrue(mock_repo.session.commit.called)

    @patch.object(OlsClient, "_initialize_metadata")
    @patch.object(OlsClient, "_fetch_page_data")
    def test_process_to_repository_rollback_on_error(self, mock_fetch, mock_init):
        """Tests that database errors trigger a rollback"""
        self.client._ontology_name = "Test"
        self.client._ontology_short_name = "TST"
        self.client._num_pages = 1
        mock_fetch.return_value = (["ID:1"], ["Term 1"], ["Desc 1"], [[0.1]])
        mock_repo = MagicMock()
        mock_db_term = MagicMock()
        mock_repo.get_terminology_by_name.return_value = mock_db_term
        mock_repo.store.side_effect = [None, Exception("DB Error")]
        with self.assertRaisesRegex(Exception, "DB Error"):
            self.client.process_to_repository(mock_repo)
        mock_init.assert_called_once()
        self.assertTrue(mock_repo.session.rollback.called)

    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(OlsClient, "_initialize_metadata")
    @patch.object(OlsClient, "_fetch_page_data")
    def test_process_to_json(self, mock_fetch, mock_init, mock_file, mock_makedirs):
        """Tests JSON export functionality and file structure creation"""
        self.client._ontology_name = "Test Ontology"
        self.client._ontology_short_name = "TEST"
        self.client._num_pages = 1
        mock_fetch.return_value = (["ID:1"], ["Term 1"], ["Desc 1"], [[0.1, 0.2]])
        dest_path = "/fake/out"
        self.client.process_to_json(dest_path)
        mock_init.assert_called_once()
        mock_makedirs.assert_called_with(dest_path, exist_ok=True)
        self.assertEqual(mock_file.call_count, 3)
        mock_file.assert_any_call(os.path.join(dest_path, "terminology.json"), "w", encoding="utf-8")
        mock_file.assert_any_call(os.path.join(dest_path, "concepts_0.json"), "w", encoding="utf-8")
        mock_file.assert_any_call(os.path.join(dest_path, "mappings_0.json"), "w", encoding="utf-8")

    def test_missing_metadata_raises_error(self):
        """Tests that processing fails if metadata is not properly initialized."""
        mock_repo = MagicMock()
        self.client._num_pages = None
        with patch.object(OlsClient, "_initialize_metadata"):
            with self.assertRaisesRegex(RuntimeError, "Ontology metadata failed to initialize properly"):
                self.client.process_to_repository(mock_repo)

            with self.assertRaisesRegex(RuntimeError, "Ontology metadata failed to initialize properly"):
                self.client.process_to_json("/fake/path")


if __name__ == "__main__":
    unittest.main()

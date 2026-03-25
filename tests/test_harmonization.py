import unittest
from unittest.mock import MagicMock

import pandas as pd

from datastew.harmonization import map_dictionary_to_dictionary


class TestHarmonization(unittest.TestCase):
    def setUp(self):
        self.mock_source = MagicMock()
        self.mock_source.to_dataframe.return_value = pd.DataFrame(
            {"variable": ["S1", "S2"], "description": ["Source 1", "Source 2"]}
        )

        self.mock_target = MagicMock()
        self.mock_target.to_dataframe.return_value = pd.DataFrame(
            {"variable": ["T1", "T2", "T3"], "description": ["Target 1", "Target 2", "Target 3"]}
        )

        self.mock_vectorizer = MagicMock()
        self.mock_vectorizer.get_embeddings.side_effect = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0]],
        ]

        self.expected_columns = [
            "Source Variable",
            "Target Variable",
            "Source Description",
            "Target Description",
            "Similarity",
        ]

    def test_map_dictionary_limit_one(self):
        """Test the logic branch where limit == 1."""
        df = map_dictionary_to_dictionary(self.mock_source, self.mock_target, vectorizer=self.mock_vectorizer, limit=1)

        # Verify output structure
        self.assertEqual(list(df.columns), self.expected_columns)
        self.assertEqual(len(df), 2)

        # Verify logic
        self.assertEqual(df.iloc[0]["Target Variable"], "T1")
        self.assertAlmostEqual(df.iloc[0]["Similarity"], 1.0, places=4)
        self.assertEqual(df.iloc[1]["Target Variable"], "T3")
        self.assertAlmostEqual(df.iloc[1]["Similarity"], 1.0, places=4)

    def test_map_dictionary_limit_multiple(self):
        """Test the logic branch where limit > 1."""
        # Reset side_effect because get_embeddings will be called again
        self.mock_vectorizer.get_embeddings.side_effect = [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0]],
        ]

        df = map_dictionary_to_dictionary(self.mock_source, self.mock_target, vectorizer=self.mock_vectorizer, limit=2)

        # Verify output structure
        self.assertEqual(list(df.columns), self.expected_columns)
        self.assertEqual(len(df), 4)

        # Verify logic for S1
        s1_matches = df[df["Source Variable"] == "S1"]
        self.assertEqual(s1_matches.iloc[0]["Target Variable"], "T1")
        self.assertEqual(s1_matches.iloc[1]["Target Variable"], "T2")

        # Verify logic for S2
        s2_matches = df[df["Source Variable"] == "S2"]
        self.assertEqual(s2_matches.iloc[0]["Target Variable"], "T3")
        self.assertEqual(s2_matches.iloc[1]["Target Variable"], "T2")

    def test_map_dictionary_limit_exceeds_target(self):
        """Test that a ValueError is raised if limit exceeds target variables."""
        with self.assertRaisesRegex(ValueError, "cannot be greater than the number of target variables"):
            map_dictionary_to_dictionary(self.mock_source, self.mock_target, vectorizer=self.mock_vectorizer, limit=5)


if __name__ == "__main__":
    unittest.main()

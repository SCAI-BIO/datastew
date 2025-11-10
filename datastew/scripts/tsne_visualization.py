"""
Example: Visualize the semantic similarity between two data dictionaries

This script demonstrates how to:
1. Load two data dictionaries from Excel files.
2. Generate text embeddings for each variable description.
3. Visualize their semantic relationships in a shared 2D embedding space.

The resulting plot highlights how conceptually similar variables
from both data dictionaries cluster together.

Before running this example:
- Ensure both Excel files contain columns for variable names and descriptions.
- The column names should match the ones you specify in `variable_field`
  and `description_field`.
"""

from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.visualisation import plot_embeddings

# --------------------------------------------------------------------
# 1) Load data dictionaries
# --------------------------------------------------------------------
# Replace these filenames and column names with your own.
data_dictionary_source_1 = DataDictionarySource("source1.xlsx", variable_field="var", description_field="desc")
data_dictionary_source_2 = DataDictionarySource("source2.xlsx", variable_field="var", description_field="desc")

# --------------------------------------------------------------------
# 2) Initialize the embedding model
# --------------------------------------------------------------------
# You can also specify a model name or API key if desired:
# vectorizer = Vectorizer("text-embedding-3-small", key="your_openai_api_key")
vectorizer = Vectorizer()

# --------------------------------------------------------------------
# 3) Generate embeddings and visualize
# --------------------------------------------------------------------
# The resulting visualization displays variables from both sources
# positioned based on semantic similarity.
plot_embeddings([data_dictionary_source_1, data_dictionary_source_2], vectorizer=vectorizer)

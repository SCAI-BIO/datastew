import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from index.embedding import EmbeddingModel, MPNetAdapter
from index.process.parsing import DataDictionarySource


def map_dictionary_to_dictionary(source: DataDictionarySource,
                                 target: DataDictionarySource,
                                 embedding_model: EmbeddingModel = MPNetAdapter()) -> pd.DataFrame:
    # Load data
    df_source = source.to_dataframe()
    df_target = target.to_dataframe()

    # Compute embeddings
    embeddings_source = embedding_model.get_embeddings(df_source["description"].tolist())
    embeddings_target = embedding_model.get_embeddings(df_target["description"].tolist())

    # Compute cosine similarities
    similarities = cosine_similarity(embeddings_source, embeddings_target)

    # Find closest matches
    max_similarities = np.max(similarities, axis=1)
    closest_match_indices = np.argmax(similarities, axis=1)

    # Create DataFrame for closest matches
    result_df = pd.DataFrame({
        'Source Variable': df_source["variable"],
        'Target Variable': df_target.iloc[closest_match_indices]["variable"].values,
        'Similarity': max_similarities
    })

    return result_df

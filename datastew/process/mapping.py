import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from datastew.embedding import EmbeddingModel, MPNetAdapter
from datastew.process.parsing import DataDictionarySource


def map_dictionary_to_dictionary(source: DataDictionarySource,
                                 target: DataDictionarySource,
                                 embedding_model: EmbeddingModel = MPNetAdapter(),
                                 limit: int = 1) -> pd.DataFrame:
    """
    Map variables from a source data dictionary to the closest matching variables in a target data dictionary
    based on the similarity of their descriptions.

    :param source: The source data dictionary containing variables and their descriptions.
    :param target: The target data dictionary containing variables and their descriptions to be matched against.
    :param embedding_model: The model used to convert descriptions into embeddings for similarity comparison.
                            Defaults to MPNetAdapter().
    :param limit: The number of closest matches to retrieve for each source variable. Defaults to 1.
    :return: A DataFrame containing the closest matches with the following columns:
             - 'Source Variable': The variable names from the source data dictionary.
             - 'Target Variable': The closest matching variable names from the target data dictionary.
             - 'Source Description': The descriptions of the variables from the source data dictionary.
             - 'Target Description': The descriptions of the closest matching variables from the target data dictionary.
             - 'Similarity': The cosine similarity score between the source and target variable descriptions.
    """
    # Load data
    df_source = source.to_dataframe()
    df_target = target.to_dataframe()

    # Compute embeddings
    embeddings_source = embedding_model.get_embeddings(df_source["description"].tolist())
    embeddings_target = embedding_model.get_embeddings(df_target["description"].tolist())

    # Compute cosine similarities
    similarities = cosine_similarity(embeddings_source, embeddings_target)

    if limit == 1:
        # Find the closest matches
        max_similarities = np.max(similarities, axis=1)
        closest_match_indices = np.argmax(similarities, axis=1)

        # Create DataFrame for closest matches
        result_df = pd.DataFrame({
            "Source Variable": df_source["variable"],
            "Target Variable": df_target.iloc[closest_match_indices]["variable"].values,
            "Source Description": df_source["description"],
            "Target Description": df_target.iloc[closest_match_indices]["description"].values,
            "Similarity": max_similarities
        })

    else:
        if limit > len(df_target):
            ValueError(f"The limit {limit} cannot be greater than the number of target variables {len(df_target)}.")

        # Get the indices of the top "limit" matches for each source variable
        top_matches_indices = np.argsort(similarities, axis=1)[:, -limit:][:, ::-1]

        # Flatten indices for easier DataFrame construction
        flat_indices = top_matches_indices.flatten()
        source_repeated = np.repeat(df_source.index, limit)

        # Create DataFrame for closest matches
        result_df = pd.DataFrame({
            "Source Variable": df_source.iloc[source_repeated]["variable"].values,
            "Target Variable": df_target.iloc[flat_indices]["variable"].values,
            "Source Description": df_source.iloc[source_repeated]["description"].values,
            "Target Description": df_target.iloc[flat_indices]["description"].values,
            "Similarity": np.take_along_axis(similarities, top_matches_indices, axis=1).flatten()
        })

    return result_df

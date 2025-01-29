from enum import Enum

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_distances
from thefuzz import fuzz, process

from datastew._mapping import _MappingTable


class MatchingMethod(Enum):
    EUCLIDEAN_EMBEDDING_DISTANCE = 1,
    FUZZY_STRING_MATCHING = 2,
    COSINE_EMBEDDING_DISTANCE = 3


def enrichment_analysis(source_table: _MappingTable, target_table: _MappingTable, max_cumulative_match_rank: int = 10,
                        matching_method=MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE) -> np.ndarray:
    """
    Calculate accuracy for the n the closest matches for two mapping tables

    :param source_table: the table containing the source descriptions which should be matched
    :param target_table: the table containing the target descriptions to which the source descriptions should be matched
    :param matching_method: How the matching should be performed - either based on vector embeddings or fuzzy string
    matching
    :param max_cumulative_match_rank: The n the closest matches that should be taken into consideration
    :return: a dataframe containing the matches
    """
    # index n will correspond to correctly match within the n the closest variables
    correct_matches = np.zeros(max_cumulative_match_rank)
    # not every variable can be matched
    max_matches = 0
    # clean up source and target table (missing embeddings, descriptions etc.)
    source_table.joined_mapping_table.drop_duplicates(subset=["variable"], keep="first", inplace=True)
    source_table.joined_mapping_table.dropna(subset=["description"], inplace=True)
    target_table.joined_mapping_table.dropna(subset=["description"], inplace=True)
    if (matching_method == MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE or matching_method == MatchingMethod.COSINE_EMBEDDING_DISTANCE):
        source_table.joined_mapping_table.dropna(subset=["embedding"], inplace=True)
        target_table.joined_mapping_table.dropna(subset=["embedding"], inplace=True)
    # re-index to account for dropped rows
    target_table.joined_mapping_table = target_table.joined_mapping_table.reset_index(drop=True)
    for idx, source_table_row in source_table.joined_mapping_table.iterrows():
        correct_target_index = target_table.joined_mapping_table[
            target_table.joined_mapping_table["identifier"] == source_table_row["identifier"]].index
        if len(correct_target_index) == 0:
            # can not be matched -> skip
            continue
        # match is possible
        max_matches += 1
        # compute distances to all possible matches
        distances = []
        for idy, target_table_row in target_table.joined_mapping_table.iterrows():
            if matching_method == MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE:
                source_table_embedding = source_table_row["embedding"]
                target_table_embedding = target_table_row["embedding"]
                distances.append(np.linalg.norm(np.array(source_table_embedding) - np.array(target_table_embedding)))
            elif matching_method == MatchingMethod.COSINE_EMBEDDING_DISTANCE:
                source_table_embedding = np.array(source_table_row["embedding"])
                target_table_embedding = np.array(target_table_row["embedding"])
                distances.append(distance.cosine(source_table_embedding, target_table_embedding))
            elif matching_method == MatchingMethod.FUZZY_STRING_MATCHING:
                source_table_description = source_table_row["description"]
                target_table_description = target_table_row["description"]
                distances.append(100 - fuzz.ratio(source_table_description, target_table_description))
            else:
                raise NotImplementedError("Specified matching method is not implemented!")
        min_distance_indices = np.argsort(np.array(distances))[:max_cumulative_match_rank]
        for n in range(max_cumulative_match_rank):
            # (due to upper level concepts) there may be more than one correct mapping
            if any(element in min_distance_indices[: n + 1] for element in correct_target_index):
                correct_matches[n] += 1
    return (correct_matches / max_matches).round(2)


def match_closest_descriptions(source_table: _MappingTable, target_table: _MappingTable,
                               matching_method=MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE) -> pd.DataFrame:
    """
    Match descriptions from source table to target table based on the biggest similarity

    :param source_table: the table containing the source descriptions which should be matched
    :param target_table: the table containing the target descriptions to which the source descriptions should be matched
    :param matching_method: How the matching should be performed - either based on vector embeddings or fuzzy string
    matching
    :return: a dataframe containing the matches
    """
    # sometimes the same concept gets mapped against multiple concepts in CDM, resulting in artifacts in the results
    # -> drop duplicates, only keep first
    source_table.joined_mapping_table.drop_duplicates(subset=["variable"], keep="first", inplace=True)
    # remove rows from source and target that do not contain either a description (in general) or embedding (for gpt)
    source_table.joined_mapping_table.dropna(subset=["description"], inplace=True)
    target_table.joined_mapping_table.dropna(subset=["description"], inplace=True)
    if (matching_method == MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE or matching_method == MatchingMethod.COSINE_EMBEDDING_DISTANCE):
        source_table.joined_mapping_table.dropna(subset=["embedding"], inplace=True)
        target_table.joined_mapping_table.dropna(subset=["embedding"], inplace=True)
    # method -> compute distance based on embeddings
    if (matching_method == MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE or matching_method == MatchingMethod.COSINE_EMBEDDING_DISTANCE):
        if ("embedding" not in source_table.joined_mapping_table.columns or "embedding" not in target_table.joined_mapping_table.columns):
            raise ValueError("Mapping tables must contain an 'embedding' column")
    # re-index to account for dropped rows
    target_table.joined_mapping_table = target_table.joined_mapping_table.reset_index(drop=True)
    # METHOD: Euclidean Distance based on embeddings
    if matching_method == MatchingMethod.EUCLIDEAN_EMBEDDING_DISTANCE:
        if ("embedding" not in source_table.joined_mapping_table.columns or "embedding" not in target_table.joined_mapping_table.columns):
            raise ValueError("Mapping tables must contain an 'embedding' column")
        source_embeddings = source_table.get_embeddings_numpy()
        target_embeddings = target_table.get_embeddings_numpy()
        distance_matrix = np.linalg.norm(source_embeddings[:, np.newaxis] - target_embeddings, axis=-1)
        closest_indices = np.argmin(distance_matrix, axis=1)
        distances = np.min(distance_matrix, axis=1)
        matched_target_descriptions = target_table.joined_mapping_table.loc[closest_indices, "description"].tolist()
    # METHOD: Cosine Distance based on embeddings
    elif matching_method == MatchingMethod.COSINE_EMBEDDING_DISTANCE:
        if ("embedding" not in source_table.joined_mapping_table.columns or "embedding" not in target_table.joined_mapping_table.columns):
            raise ValueError("Mapping tables must contain an 'embedding' column")
        source_embeddings = source_table.get_embeddings_numpy()
        target_embeddings = target_table.get_embeddings_numpy()
        distance_matrix = cosine_distances(source_embeddings, target_embeddings)
        closest_indices = np.argmin(distance_matrix, axis=1)
        distances = np.min(distance_matrix, axis=1)
        matched_target_descriptions = target_table.joined_mapping_table.loc[closest_indices, "description"].tolist()
    # METHOD: Fuzzy String Matching based on Levenstein Distance
    elif matching_method == MatchingMethod.FUZZY_STRING_MATCHING:
        if ("description" not in source_table.joined_mapping_table.columns or "description" not in target_table.joined_mapping_table.columns):
            raise ValueError("Mapping tables must contain an 'description' column")
        source_descriptions = source_table.joined_mapping_table["description"].to_numpy()
        target_descriptions = target_table.joined_mapping_table["description"].to_numpy()
        target_descriptions_dict = {idx: el for idx, el in enumerate(target_descriptions)}
        closest_indices = []
        distances = []
        matched_target_descriptions = []
        for source_description in source_descriptions:
            matched_target_description, distance, target_idx = process.extractOne(source_description,
                                                                                  target_descriptions_dict)
            closest_indices.append(target_idx)
            matched_target_descriptions.append(matched_target_description)
            # it is not a distance but a score [0,100] in this case -> take inverse (+1 to avoid division by 0)
            distances.append(1 / (101 - distance))
    # NOT IMPLEMENTED -> raise error
    else:
        raise ValueError("Specified Matching method is not implemented!")
    source_concept_label = source_table.joined_mapping_table["identifier"]
    target_concept_label = target_table.joined_mapping_table.loc[closest_indices, "identifier"].tolist()
    source_variable = source_table.joined_mapping_table["variable"]
    target_variable = target_table.joined_mapping_table.loc[closest_indices, "variable"].tolist()
    correct = source_concept_label == target_concept_label
    ground_truth_target_descriptions = get_ground_truth_target_descriptions(source_table.joined_mapping_table,
                                                                            target_table.joined_mapping_table)
    source_descriptions = source_table.joined_mapping_table["description"]
    result = pd.DataFrame(
        {
            "correct": correct,
            "source_variable": source_variable,
            "target_variable": target_variable,
            "source_concept_label": source_concept_label,
            "target_concept_label": target_concept_label,
            "source_description": source_descriptions,
            "matched_target_description": matched_target_descriptions,
            "ground_truth_target_description": ground_truth_target_descriptions,
            "distance": distances,
        }
    )
    return result


def get_ground_truth_target_descriptions(source_table: pd.DataFrame, target_table: pd.DataFrame) -> np.ndarray[str]:
    """
    Get the ground truth target descriptions based on the matched identifiers

    :param source_table: The source table containing the identifiers
    :param target_table: The target table containing the identifiers and descriptions
    :return: An ordered numpy array containing the ground truth target descriptions
    """
    # TODO: This is a very slow implementation, but it works for now
    descriptions = []
    for source_id in source_table["identifier"]:
        try:
            target_description = target_table.loc[target_table["identifier"] == source_id, "description"].iloc[0]
            descriptions.append(target_description)
        except IndexError:
            descriptions.append(None)
    return np.array(descriptions)


def score_mappings(matches: pd.DataFrame) -> float:
    """
    Evaluate the matches based on the accuracy

    :param matches: the matches to be evaluated
    :return: the accuracy
    """
    # ignore matches where there is no possible match for the source description
    matches = matches[matches["ground_truth_target_description"].notnull()]
    # TODO: investigate this
    matches = matches[matches["target_concept_label"].notnull()]
    accuracy = matches["correct"].sum() / len(matches)
    return accuracy


def evaluate(datasets, labels, store_results=False, model="gpt", results_root_dir="resources/results/pd"):
    data = {}
    for idx, source in enumerate(datasets):
        acc = []
        for idy, target in enumerate(datasets):
            if model == "gpt":
                map = match_closest_descriptions(source, target)
            elif model == "mpnet":
                map = match_closest_descriptions(source,target, matching_method=MatchingMethod.COSINE_EMBEDDING_DISTANCE)
            elif model == "fuzzy":
                map = match_closest_descriptions(source, target, matching_method=MatchingMethod.FUZZY_STRING_MATCHING)
            else:
                raise NotImplementedError("Specified model is not implemented!")
            if store_results:
                map.to_excel(results_root_dir + f"/{model}_" + f"{labels[idx]}_to_{labels[idy]}.xlsx")
            acc.append(round(score_mappings(map), 2))
        data[labels[idx]] = acc
    # transpose to have from -> to | row -> column like in the paper
    model_output = pd.DataFrame(data, index=labels).T
    return model_output

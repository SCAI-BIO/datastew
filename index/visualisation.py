from enum import Enum

import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go

from index.conf import COLORS_AD, COLORS_PD
from index.mapping import MappingTable


class PlotSide(Enum):
    LEFT = 1,
    RIGHT = 2,
    BOTH = 3


def size_array_to_boundaries(array: np.array):
    for i in range(1, len(array)):
        array[i] += array[i - 1]
    return array


def get_cohort_specific_color_code(cohort_name: str):
    if cohort_name.lower() in COLORS_AD:
        return COLORS_AD[cohort_name.lower()]
    elif cohort_name.lower() in COLORS_PD:
        return COLORS_PD[cohort_name.lower()]
    else:
        print(f'No color code found for cohort {cohort_name}')
        return None


def enrichment_plot(acc_gpt, acc_fuzzy, title, save_plot=False, save_dir="resources/results/plots"):
    if len(acc_gpt) != len(acc_fuzzy):
        raise ValueError("acc_gpt and acc_fuzzy should be of the same length!")
    data = {"Maximum Considered Rank": list(range(1, len(acc_gpt) + 1)), "GPT": acc_gpt,
            "Fuzzy": acc_fuzzy}
    df = pd.DataFrame(data)
    sns.set(style="whitegrid")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="GPT", label="GPT")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="Fuzzy", label="Fuzzy String Matching")
    sns.set(style="whitegrid")
    plt.xlabel("Maximum Considered Rank")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(acc_gpt) + 1), labels=range(1, len(acc_gpt) + 1))
    plt.yticks([i / 10 for i in range(11)])
    plt.gca().set_yticklabels([f'{i:.1f}' for i in plt.gca().get_yticks()])
    plt.title(title)
    plt.legend()
    if save_plot:
        plt.savefig(save_dir + "/" + title)
    plt.show()


def concat_embeddings(tables1: [MappingTable], tables2: [MappingTable]):
    # remove entries that do not contain an embedding -> have no corresponding vector
    tables1_cleaned = [copy.deepcopy(table) for table in tables1]
    tables2_cleaned = [copy.deepcopy(table) for table in tables2]
    for table1, table2 in zip(tables1_cleaned, tables2_cleaned):
        table1.joined_mapping_table.dropna(subset=['embedding', 'description'], inplace=True)
        table2.joined_mapping_table.dropna(subset=['embedding', 'description'], inplace=True)
    vectors_tables1 = np.concatenate([table.get_embeddings_numpy() for table in tables1_cleaned])
    vectors_tables2 = np.concatenate([table.get_embeddings_numpy() for table in tables2_cleaned])
    descriptions_table1 = np.concatenate([table.joined_mapping_table["description"] for table in tables1_cleaned])
    descriptions_table2 = np.concatenate([table.joined_mapping_table["description"] for table in tables2_cleaned])
    boundaries1 = np.array([table.joined_mapping_table["embedding"].index.size for table in tables1_cleaned])
    boundaries2 = np.array([table.joined_mapping_table["embedding"].index.size for table in tables2_cleaned])
    vectors_concatenated = np.concatenate([vectors_tables1, vectors_tables2])
    descriptions_concatenated = np.concatenate([descriptions_table1, descriptions_table2])
    boundaries_concatenated = size_array_to_boundaries(np.concatenate([boundaries1, boundaries2]))
    return vectors_concatenated, descriptions_concatenated, boundaries_concatenated


def scatter_plot_two_distributions(tables1: [MappingTable], tables2: [MappingTable], label1: str, label2: str,
                                   store_html: bool = True,
                                   store_destination: str = "resources/results/plots/ad_vs_pd.html"):
    vectors_tables1 = np.concatenate([table.get_embeddings_numpy() for table in tables1])
    vectors_tables2 = np.concatenate([table.get_embeddings_numpy() for table in tables2])
    # remove entries that do not contain an embedding -> have no corresponding vector
    [table.joined_mapping_table.dropna(subset=['embedding'], inplace=True) for table in tables1]
    [table.joined_mapping_table.dropna(subset=['embedding'], inplace=True) for table in tables2]
    # get descriptions as interactive labels
    labels_table1 = np.concatenate([table.joined_mapping_table["description"] for table in tables1])
    labels_table2 = np.concatenate([table.joined_mapping_table["description"] for table in tables2])
    # boundary for concatenated vector
    class_boundary = len(vectors_tables1)
    vectors_concatenated = np.concatenate([vectors_tables1, vectors_tables2])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(vectors_concatenated)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tsne_result[:class_boundary, 0], y=tsne_result[:class_boundary, 1],
                             mode="markers", name=label1, text=labels_table1))
    fig.add_trace(go.Scatter(x=tsne_result[class_boundary:, 0], y=tsne_result[class_boundary:, 1],
                             mode="markers", name=label2, text=labels_table2))
    fig.show()
    if store_html:
        fig.write_html(store_destination)


def scatter_plot_all_cohorts(tables1: [MappingTable], tables2: [MappingTable], labels1: [str], labels2: [str],
                             plot_side: PlotSide = PlotSide.BOTH, store_html: bool = True,
                             store_base_dir: str = "resources/results/plots"):
    if not len(tables1) == len(labels1) or not len(tables2) == len(labels2):
        raise ValueError("Length of corresponding tables and labels must be equal!")
    tables_boundary = len(tables1)
    vectors, descriptions, boundaries = concat_embeddings(tables1, tables2)
    tsne = TSNE(n_components=2, perplexity=(30 if len(vectors) > 30 else len(vectors) - 1), random_state=42)
    tsne_result = tsne.fit_transform(vectors)
    fig = go.Figure()
    # first cohort is from 0 to x
    boundaries = np.insert(boundaries, 0, 0)
    for idx in range(len(tables1)):
        if labels1[idx]:
            fig.add_trace(go.Scatter(x=tsne_result[boundaries[idx]:boundaries[idx + 1], 0],
                                     y=tsne_result[boundaries[idx]:boundaries[idx + 1], 1],
                                     mode="markers", name=labels1[idx],
                                     text=descriptions[boundaries[idx]:boundaries[idx + 1]],
                                     line=dict(color=get_cohort_specific_color_code(labels1[idx]))))
    for idy in range(len(tables1), len(boundaries) - 1):
        fig.add_trace(go.Scatter(x=tsne_result[boundaries[idy]:boundaries[idy + 1], 0],
                                 y=tsne_result[boundaries[idy]:boundaries[idy + 1], 1],
                                 mode="markers", name=labels2[idy - len(tables1)],
                                 text=descriptions[boundaries[idy]:boundaries[idy + 1]],
                                 line=dict(color=get_cohort_specific_color_code(labels2[idy - len(tables1)]))))
    if store_html:
        fig.write_html(store_base_dir + "/tsne_all_cohorts.html")
    fig.show()

import copy
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Optional, List

from datastew.process.parsing import DataDictionarySource
from datastew.embedding import EmbeddingModel, MPNetAdapter
from datastew.conf import COLORS_AD, COLORS_PD
from datastew._mapping import _MappingTable
from datastew.repository.base import BaseRepository


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
        print(f"No color code found for cohort {cohort_name}")
        return None


def enrichment_plot(acc_gpt, acc_mpnet, acc_fuzzy, title, save_plot=False, save_dir="resources/results/plots"):
    if not (len(acc_gpt) == len(acc_fuzzy) == len(acc_mpnet)):
        raise ValueError("acc_gpt, acc_mpnet and acc_fuzzy should be of the same length!")
    data = {"Maximum Considered Rank": list(range(1, len(acc_gpt) + 1)), "GPT": acc_gpt,
            "MPNet": acc_mpnet, "Fuzzy": acc_fuzzy}
    df = pd.DataFrame(data)
    sns.set(style="whitegrid")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="GPT", label="GPT")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="MPNet", label="MPNet")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="Fuzzy", label="Fuzzy String Matching")
    sns.set(style="whitegrid")
    plt.xlabel("Maximum Considered Rank")
    plt.ylabel("Accuracy")
    plt.xticks(range(1, len(acc_gpt) + 1), labels=range(1, len(acc_gpt) + 1))
    plt.yticks([i / 10 for i in range(11)])
    plt.gca().set_yticklabels([f"{i:.1f}" for i in plt.gca().get_yticks()])
    plt.title(title)
    plt.legend()
    if save_plot:
        plt.savefig(save_dir + "/" + title)
    plt.show()


def concat_embeddings(tables1: [_MappingTable], tables2: [_MappingTable]):
    # remove entries that do not contain an embedding -> have no corresponding vector
    tables1_cleaned = [copy.deepcopy(table) for table in tables1]
    tables2_cleaned = [copy.deepcopy(table) for table in tables2]
    for table1, table2 in zip(tables1_cleaned, tables2_cleaned):
        table1.joined_mapping_table.dropna(subset=["embedding", "description"], inplace=True)
        table2.joined_mapping_table.dropna(subset=["embedding", "description"], inplace=True)
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


def bar_chart_average_acc_two_distributions(dist1_fuzzy: pd.DataFrame, dist1_gpt: pd.DataFrame,
                                            dist1_mpnet: pd.DataFrame, dist2_fuzzy: pd.DataFrame,
                                            dist2_gpt: pd.DataFrame, dist2_mpnet: pd.DataFrame,
                                            title: str, label1: str, label2: str):
    if not all(dist.shape == fuzzy.shape == mpnet.shape for dist, mpnet, fuzzy in
               [(dist1_gpt, dist1_mpnet, dist1_fuzzy), (dist2_gpt, dist2_mpnet, dist2_fuzzy)]):
        raise ValueError("Each pair of dist and fuzzy DataFrames must have the same dimensions")
    if not all(dist.shape[0] == dist.shape[1] for dist in [dist1_fuzzy, dist2_fuzzy]):
        raise ValueError("Each dist DataFrame must be square")
    if not all(dist.index.equals(fuzzy.index) and dist.columns.equals(fuzzy.columns) for dist, fuzzy in
               [(dist1_fuzzy, dist1_gpt), (dist2_fuzzy, dist2_gpt)]):
        raise ValueError("All row and column labels within each pair of dist and fuzzy DataFrames must be equal")
    # average value without the diagonal, since diagonal contains matching of the same pair
    avg_acc_fuzzy1 = np.mean(dist1_fuzzy.values[~np.eye(dist1_fuzzy.shape[0], dtype=bool)])
    avg_acc_fuzzy2 = np.mean(dist2_fuzzy.values[~np.eye(dist2_fuzzy.shape[0], dtype=bool)])
    avg_acc_gpt1 = np.mean(dist1_gpt.values[~np.eye(dist1_gpt.shape[0], dtype=bool)])
    avg_acc_gpt2 = np.mean(dist2_gpt.values[~np.eye(dist2_gpt.shape[0], dtype=bool)])
    avg_acc_mpnet1 = np.mean(dist1_mpnet.values[~np.eye(dist1_mpnet.shape[0], dtype=bool)])
    avg_acc_mpnet2 = np.mean(dist2_mpnet.values[~np.eye(dist2_mpnet.shape[0], dtype=bool)])
    data = {"Fuzzy String Matching": [avg_acc_fuzzy1, avg_acc_fuzzy2], "GPT Embeddings": [avg_acc_gpt1, avg_acc_gpt2],
            "MPNet Embeddings": [avg_acc_mpnet1, avg_acc_mpnet2]}
    df = pd.DataFrame(data, index=[label1, label2])
    print(df)
    df_melted = df.reset_index().melt(id_vars="index", var_name="Method", value_name="Accuracy")
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    sns.barplot(x="index", y="Accuracy", hue="Method", data=df_melted)
    plt.xlabel("")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    plt.show()


def scatter_plot_two_distributions(tables1: [_MappingTable], tables2: [_MappingTable], label1: str, label2: str,
                                   store_html: bool = True, legend_font_size: int = 16,
                                   store_destination: str = "resources/results/plots/ad_vs_pd.html"):
    vectors_tables1 = np.concatenate([table.get_embeddings_numpy() for table in tables1])
    vectors_tables2 = np.concatenate([table.get_embeddings_numpy() for table in tables2])
    # remove entries that do not contain an embedding -> have no corresponding vector
    [table.joined_mapping_table.dropna(subset=["embedding"], inplace=True) for table in tables1]
    [table.joined_mapping_table.dropna(subset=["embedding"], inplace=True) for table in tables2]
    # get descriptions as interactive labels
    labels_table1 = np.concatenate([table.joined_mapping_table["description"] for table in tables1])
    labels_table2 = np.concatenate([table.joined_mapping_table["description"] for table in tables2])
    # boundary for concatenated vector
    class_boundary = len(vectors_tables1)
    vectors_concatenated = np.concatenate([vectors_tables1, vectors_tables2])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(vectors_concatenated)
    fig = go.Figure()
    # bigger legend size
    fig.update_layout(legend=dict(font=dict(size=legend_font_size)))
    fig.add_trace(go.Scatter(x=tsne_result[:class_boundary, 0], y=tsne_result[:class_boundary, 1],
                             mode="markers", name=label1, text=labels_table1))
    fig.add_trace(go.Scatter(x=tsne_result[class_boundary:, 0], y=tsne_result[class_boundary:, 1],
                             mode="markers", name=label2, text=labels_table2))
    fig.show()
    if store_html:
        fig.write_html(store_destination)


def scatter_plot_all_cohorts(tables1: [_MappingTable], tables2: [_MappingTable], labels1: [str], labels2: [str],
                             plot_side: PlotSide = PlotSide.BOTH, store_html: bool = True,
                             legend_font_size: int = 16, store_base_dir: str = "resources/results/plots"):
    if not len(tables1) == len(labels1) or not len(tables2) == len(labels2):
        raise ValueError("Length of corresponding tables and labels must be equal!")
    tables_boundary = len(tables1)
    vectors, descriptions, boundaries = concat_embeddings(tables1, tables2)
    tsne = TSNE(n_components=2, perplexity=(30 if len(vectors) > 30 else len(vectors) - 1), random_state=42)
    tsne_result = tsne.fit_transform(vectors)
    # more distinct colors
    color_scale = px.colors.qualitative.Set3
    fig = go.Figure()
    # bigger legend size
    fig.update_layout(legend=dict(font=dict(size=legend_font_size)))
    # first cohort is from 0 to x
    boundaries = np.insert(boundaries, 0, 0)
    for idx in range(len(tables1)):
        if labels1[idx]:
            fig.add_trace(go.Scatter(x=tsne_result[boundaries[idx]: boundaries[idx + 1], 0],
                                     y=tsne_result[boundaries[idx]: boundaries[idx + 1], 1],
                                     mode="markers", name=labels1[idx],
                                     text=descriptions[boundaries[idx]: boundaries[idx + 1]],
                                     # line=dict(color=get_cohort_specific_color_code(labels1[idx]))
                                     ))
    for idy in range(len(tables1), len(boundaries) - 1):
        fig.add_trace(go.Scatter(x=tsne_result[boundaries[idy]: boundaries[idy + 1], 0],
                                 y=tsne_result[boundaries[idy]: boundaries[idy + 1], 1],
                                 mode="markers",
                                 name=labels2[idy - len(tables1)],
                                 text=descriptions[boundaries[idy]: boundaries[idy + 1]],
                                 # line=dict(color=get_cohort_specific_color_code(labels2[idy - len(tables1)]))
                                 ))
    if store_html:
        fig.write_html(store_base_dir + "/tsne_all_cohorts.html")
    fig.show()


def get_plot_for_current_database_state(repository: BaseRepository, terminology: Optional[str] = None,
                                        perplexity: int = 5, return_type="html") -> str:
    if not terminology:
        mappings = repository.get_mappings()
    else:
        mappings = repository.get_mappings(terminology_name=terminology)
    # Extract embeddings
    embeddings = np.array([mapping.embedding for mapping in mappings])
    # Increase perplexity up to 30 if applicable
    if embeddings.shape[0] > 30:
        perplexity = 30
    if embeddings.shape[0] > perplexity:
        # Compute t-SNE embeddings
        tsne_embeddings = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings)
        # Create Plotly scatter plot
        scatter_plot = go.Scatter(
            x=tsne_embeddings[:, 0],
            y=tsne_embeddings[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color="blue",
                opacity=0.5
            ),
            text=[str(mapping) for mapping in mappings],
            hoverinfo="text"
        )
        layout = go.Layout(
            title="t-SNE Embeddings of Database Mappings",
            xaxis=dict(title="t-SNE Component 1"),
            yaxis=dict(title="t-SNE Component 2"),
        )
        fig = go.Figure(data=[scatter_plot], layout=layout)
        if return_type == "html":
            plot = fig.to_html(full_html=False)
        elif return_type == "json":
            plot = fig.to_json()
        else:
            raise ValueError(f'Return type {return_type} is not viable. Use either "html" or "json".')
    else:
        plot = "<b>Too few database entries to visualize</b>"
    return plot


def plot_embeddings(data_dictionaries: List[DataDictionarySource], embedding_model: Optional[EmbeddingModel] = None,
                    perplexity: int = 5):
    """
    Plots a t-SNE representation of embeddings from multiple data dictionaries and displays the plot.

    :param data_dictionaries: A list of DataDictionarySource objects to extract embeddings from.
    :param embedding_model: The embedding model used to compute embeddings. Defaults to MPNetAdapter.
    :param perplexity: The perplexity for the t-SNE algorithm. Higher values give more global structure.
    """
    if embedding_model is None:
        embedding_model = MPNetAdapter()
    all_embeddings = []
    all_texts = []
    all_colors = []
    plotly_colors = px.colors.qualitative.Plotly
    for idx, dictionary in enumerate(data_dictionaries):
        embeddings_dict = dictionary.get_embeddings(embedding_model=embedding_model)
        embeddings = list(embeddings_dict.values())
        texts = dictionary.to_dataframe()['description']
        color = plotly_colors[idx % len(plotly_colors)]
        all_embeddings.extend(embeddings)
        all_texts.extend(texts)
        all_colors.extend([color] * len(embeddings))
    embeddings_array = np.array(all_embeddings)
    # Adjust perplexity if there are enough points
    if embeddings_array.shape[0] > 30:
        perplexity = min(perplexity, 30)
    if embeddings_array.shape[0] > perplexity:
        # Compute t-SNE embeddings
        tsne_embeddings = TSNE(n_components=2, perplexity=perplexity).fit_transform(embeddings_array)
        # Create Plotly scatter plot
        scatter_plot = go.Scatter(
            x=tsne_embeddings[:, 0],
            y=tsne_embeddings[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=all_colors,  # Use the assigned colors from Plotly palette
                opacity=0.7
            ),
            text=all_texts,
            hoverinfo="text"
        )
        layout = go.Layout(
            title="t-SNE Embeddings of Data Dictionaries",
            xaxis=dict(title="t-SNE Component 1"),
            yaxis=dict(title="t-SNE Component 2"),
        )
        fig = go.Figure(data=[scatter_plot], layout=layout)
        # Display the plot
        fig.show()
    else:
        print("Too few data dictionary entries to visualize. Adjust param 'perplexity' to a value less then the number "
              "of data points.")

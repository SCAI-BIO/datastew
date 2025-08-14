from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.manifold import TSNE

from datastew.embedding import Vectorizer
from datastew.process.parsing import DataDictionarySource
from datastew.repository.base import BaseRepository


def enrichment_plot(acc_gpt, acc_mpnet, acc_fuzzy, title, save_plot=False, save_dir="resources/results/plots"):
    if not (len(acc_gpt) == len(acc_fuzzy) == len(acc_mpnet)):
        raise ValueError("acc_gpt, acc_mpnet and acc_fuzzy should be of the same length!")
    data = {
        "Maximum Considered Rank": list(range(1, len(acc_gpt) + 1)),
        "GPT": acc_gpt,
        "MPNet": acc_mpnet,
        "Fuzzy": acc_fuzzy,
    }
    df = pd.DataFrame(data)
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="GPT", label="GPT")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="MPNet", label="MPNet")
    sns.lineplot(data=df, x="Maximum Considered Rank", y="Fuzzy", label="Fuzzy String Matching")
    sns.set_theme(style="whitegrid")
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


def bar_chart_average_acc_two_distributions(
    dist1_fuzzy: pd.DataFrame,
    dist1_gpt: pd.DataFrame,
    dist1_mpnet: pd.DataFrame,
    dist2_fuzzy: pd.DataFrame,
    dist2_gpt: pd.DataFrame,
    dist2_mpnet: pd.DataFrame,
    title: str,
    label1: str,
    label2: str,
):
    if not all(
        dist.shape == fuzzy.shape == mpnet.shape
        for dist, mpnet, fuzzy in [(dist1_gpt, dist1_mpnet, dist1_fuzzy), (dist2_gpt, dist2_mpnet, dist2_fuzzy)]
    ):
        raise ValueError("Each pair of dist and fuzzy DataFrames must have the same dimensions")
    if not all(dist.shape[0] == dist.shape[1] for dist in [dist1_fuzzy, dist2_fuzzy]):
        raise ValueError("Each dist DataFrame must be square")
    if not all(
        dist.index.equals(fuzzy.index) and dist.columns.equals(fuzzy.columns)
        for dist, fuzzy in [(dist1_fuzzy, dist1_gpt), (dist2_fuzzy, dist2_gpt)]
    ):
        raise ValueError("All row and column labels within each pair of dist and fuzzy DataFrames must be equal")
    # average value without the diagonal, since diagonal contains matching of the same pair
    avg_acc_fuzzy1 = np.mean(dist1_fuzzy.values[~np.eye(dist1_fuzzy.shape[0], dtype=bool)])
    avg_acc_fuzzy2 = np.mean(dist2_fuzzy.values[~np.eye(dist2_fuzzy.shape[0], dtype=bool)])
    avg_acc_gpt1 = np.mean(dist1_gpt.values[~np.eye(dist1_gpt.shape[0], dtype=bool)])
    avg_acc_gpt2 = np.mean(dist2_gpt.values[~np.eye(dist2_gpt.shape[0], dtype=bool)])
    avg_acc_mpnet1 = np.mean(dist1_mpnet.values[~np.eye(dist1_mpnet.shape[0], dtype=bool)])
    avg_acc_mpnet2 = np.mean(dist2_mpnet.values[~np.eye(dist2_mpnet.shape[0], dtype=bool)])
    data = {
        "Fuzzy String Matching": [avg_acc_fuzzy1, avg_acc_fuzzy2],
        "GPT Embeddings": [avg_acc_gpt1, avg_acc_gpt2],
        "MPNet Embeddings": [avg_acc_mpnet1, avg_acc_mpnet2],
    }
    df = pd.DataFrame(data, index=[label1, label2])
    print(df)
    df_melted = df.reset_index().melt(id_vars="index", var_name="Method", value_name="Accuracy")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x="index", y="Accuracy", hue="Method", data=df_melted)
    plt.xlabel("")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    plt.show()


def get_plot_for_current_database_state(
    repository: BaseRepository,
    terminology: Optional[str] = None,
    sentence_embedder: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
    perplexity: int = 5,
    return_type: str = "html",
) -> str:
    mappings = repository.get_mappings(
        terminology_name=terminology, sentence_embedder=sentence_embedder, limit=limit, offset=offset
    ).items
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
            marker=dict(size=8, color="blue", opacity=0.5),
            text=[str(mapping) for mapping in mappings],
            hoverinfo="text",
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


def plot_embeddings(
    data_dictionaries: List[DataDictionarySource], vectorizer: Vectorizer = Vectorizer(), perplexity: int = 5
):
    """
    Plots a t-SNE representation of embeddings from multiple data dictionaries and displays the plot.

    :param data_dictionaries: A list of DataDictionarySource objects to extract embeddings from.
    :param embedding_model: The embedding model used to compute embeddings. Defaults to MPNetAdapter.
    :param perplexity: The perplexity for the t-SNE algorithm. Higher values give more global structure.
    """
    all_embeddings = []
    all_texts = []
    all_colors = []
    plotly_colors = px.colors.qualitative.Plotly
    for idx, dictionary in enumerate(data_dictionaries):
        embeddings_dict = dictionary.get_embeddings(vectorizer)
        embeddings = list(embeddings_dict.values())
        texts = dictionary.to_dataframe()["description"]
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
            marker=dict(size=8, color=all_colors, opacity=0.7),  # Use the assigned colors from Plotly palette
            text=all_texts,
            hoverinfo="text",
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
        print(
            "Too few data dictionary entries to visualize. Adjust param 'perplexity' to a value less then the number "
            "of data points."
        )

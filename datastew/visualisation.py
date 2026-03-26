import os
from typing import Literal, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.manifold import TSNE

from datastew.embedding import Vectorizer
from datastew.io.source import DataDictionarySource
from datastew.repository import PostgreSQLRepository


def enrichment_plot(
    acc_gpt: list[float],
    acc_mpnet: list[float],
    acc_fuzzy: list[float],
    title: str,
    save_plot: bool = False,
    save_dir: str = "resources/results/plots",
):
    """Generate and display a line plot comparing the accuracy of GPT, MPNet, and Fuzzy
    matching accross different ranks.

    :param acc_gpt: List of accuracy scores for GPT embeddings.
    :param acc_mpnet: List of accuracy scores for MPNet embeddings.
    :param acc_fuzzy: List of accuracy scores for Fuzzy string matching.
    :param title: The title of the generated plot.
    :param save_plot: Boolean flag to save the plot to a file, defaults to False.
    :param save_dir: Directory path where the plot image will be saved, defaults to "resources/results/plots".
    :raises ValueError: If the input accuracy lists are not of equal length.
    """
    if not (len(acc_gpt) == len(acc_fuzzy) == len(acc_mpnet)):
        raise ValueError("acc_gpt, acc_mpnet and acc_fuzzy should be of the same length!")

    ranks = list(range(1, len(acc_gpt) + 1))
    df_wide = pd.DataFrame(
        {"Maximum Considered Rank": ranks, "GPT": acc_gpt, "MPNET": acc_mpnet, "Fuzzy String Matching": acc_fuzzy}
    )

    # Melt for automatic Seaborn legend and color mapping
    df_long = df_wide.melt(id_vars="Maximum Considered Rank", var_name="Method", value_name="Accuracy")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df_long, x="Maximum Considered Rank", y="Accuracy", hue="Method")
    plt.title(title)
    plt.xlabel("Maximum Considered Rank")
    plt.ylabel("Accuracy")
    plt.xticks(ranks, labels=[str(r) for r in ranks])
    plt.yticks(np.linspace(0, 1, 11))
    if save_plot:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(save_path)
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
    """Compare the average accuracy of three matching methods across two distinct
    distributions using a grouped bar chart.

    :param dist1_fuzzy: Square DataFrame of fuzzy matching scores for the first distribution.
    :param dist1_gpt: Square DataFrame of GPT matching scores for the first distribution.
    :param dist1_mpnet: Square DataFrame of MPNet matching scores for the first distribution.
    :param dist2_fuzzy: Square DataFrame of fuzzy matching scores for the second distribution.
    :param dist2_gpt: Square DataFrame of GPT matching scores for the second distribution.
    :param dist2_mpnet: Square DataFrame of MPNet matching scores for the second distribution.
    :param title: The title of the generated bar chart.
    :param label1: Label for the first distribution (e.g., 'Source A').
    :param label2: Label for the second distribution (e.g., 'Source B').
    :raises ValueError: If DataFrames in a set have mismatched dimensions.
    :raises ValueError: If DataFrames in a set are not squiare.
    :raises ValueError: If DataFrames in a set have inconsistent indices/columns.
    """
    sets = [(dist1_gpt, dist1_mpnet, dist1_fuzzy), (dist2_gpt, dist2_mpnet, dist2_fuzzy)]
    for gpt, mpnet, fuzzy in sets:
        if not (gpt.shape == mpnet.shape == fuzzy.shape):
            raise ValueError("DataFrames within a distribution set must have the same dimensions")
        if gpt.shape[0] != gpt.shape[1]:
            raise ValueError("DataFrames must be square to mask the diagonal correctly")
        if not (gpt.index.equals(fuzzy.index) and gpt.columns.equals(fuzzy.columns)):
            raise ValueError("Row and column labels must match across DataFrames in the same set")
    data = {
        "Fuzzy String Matching": [_get_off_diag_mean(dist1_fuzzy), _get_off_diag_mean(dist2_fuzzy)],
        "GPT Embeddings": [_get_off_diag_mean(dist1_gpt), _get_off_diag_mean(dist2_gpt)],
        "MPNet Embeddings": [_get_off_diag_mean(dist1_mpnet), _get_off_diag_mean(dist2_mpnet)],
    }
    df = pd.DataFrame(data, index=[label1, label2])
    df_melted = df.reset_index().melt(id_vars="index", var_name="Method", value_name="Accuracy")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x="index", y="Accuracy", hue="Method", data=df_melted)
    plt.xlabel("")
    plt.ylabel("Average Accuracy")
    plt.title(title)
    plt.show()


def get_plot_for_current_database_state(
    repository: PostgreSQLRepository,
    terminology: Optional[str] = None,
    vectorizer: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
    perplexity: int = 5,
    return_type: Literal["html", "json"] = "html",
) -> str:
    """Retrieve mappings from a database and generate an interactive t-SNE scatter plot
    representing the state of the database.

    :param repository: The PostgreSQLRepository instance to query.
    :param terminology: Optional filter for a specific terminology name, defaults to None.
    :param vectorizer: Optional filter for a specific vectorizer model name, defaults to None.
    :param limit: Maximum number of entries to retrieve, defaults to 1000.
    :param offset: Pagination offset for the database query, defaults to 0.
    :param perplexity: The perplexity value for the t-SNE algorithm, defaults to 5.
    :param return_type: Format of the returned plot data, defaults to "html".
    :return: A string containing either HTML or JSON representation of the Plotly
             figure, or a bolded HTML warning message if data is insufficient.
    """
    mappings = repository.get_mappings(
        terminology_name=terminology, vectorizer=vectorizer, limit=limit, offset=offset
    ).items

    if not mappings:
        return "<b>No entries found in the database.</b>"

    # Extract embeddings
    embeddings = np.array([mapping.embedding for mapping in mappings])
    n_samples = embeddings.shape[0]

    # Safe perplexity calculation
    actual_perplexity = _get_safe_perplexity(n_samples, 30 if n_samples > 30 else perplexity)

    if n_samples > actual_perplexity:
        tsne_embeddings = TSNE(
            n_components=2, perplexity=actual_perplexity, init="pca", learning_rate="auto"
        ).fit_transform(embeddings)

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=tsne_embeddings[:, 0],
                    y=tsne_embeddings[:, 1],
                    mode="markers",
                    marker=dict(size=8, color="blue", opacity=0.5),
                    text=[str(mapping) for mapping in mappings],
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title="t-SNE Embeddings of Database Mappings",
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
        )

        if return_type == "html":
            return cast(str, fig.to_html(full_html=False))
        elif return_type == "json":
            return cast(str, fig.to_json())
    return "<b>Too few database entries to visualize</b>"


def plot_embeddings(
    data_dictionaries: list[DataDictionarySource], vectorizer: Vectorizer = Vectorizer(), perplexity: int = 5
):
    """Generate and display an interactive t-SNE scatter plot for embeddings extracted
    from multiple data dictionary sources.

    :param data_dictionaries: A list of DataDictionarySource objects to extract embeddings from.
    :param vectorizer: The model used to compute embeddings. Defaults to Vectorizer().
    :param perplexity: The perplexity for the t-SNE algorithm, defaults to 5.
    """
    all_embeddings = []
    all_texts = []
    all_source_names = []

    for idx, dictionary in enumerate(data_dictionaries):
        source_name = getattr(dictionary, "name", f"Source {idx+1}")
        embeddings_dict = dictionary.get_embeddings(vectorizer)
        all_embeddings.extend(list(embeddings_dict.values()))
        all_texts.extend(dictionary.to_dataframe()["description"].tolist())
        all_source_names.extend([source_name] * len(embeddings_dict))
    embeddings_array = np.array(all_embeddings)
    n_samples = embeddings_array.shape[0]
    safe_perplexity = _get_safe_perplexity(n_samples, perplexity)

    if n_samples > safe_perplexity and n_samples > 1:
        tsne_results = TSNE(
            n_components=2, perplexity=safe_perplexity, init="pca", learning_rate="auto"
        ).fit_transform(embeddings_array)

        fig = px.scatter(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            color=all_source_names,
            hover_name=all_texts,
            title="t-SNE Embeddings of Data Dictionaries",
            labels={"x": "t-SNE Component 1", "y": "t-SNE Component 2", "color": "Source"},
        )
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.show()
    else:
        print(f"Insufficient data points ({n_samples}) to visualize with perplexity {perplexity}.")


def _get_safe_perplexity(n_samples: int, requested_perplexity: int) -> int:
    """Helper to ensure perplexity is always < n_samples and >= 1."""
    if n_samples <= 1:
        return 0
    return max(1, min(requested_perplexity, n_samples - 1))


def _get_off_diag_mean(df: pd.DataFrame) -> float:
    return float(np.mean(df.values[~np.eye(df.shape[0], dtype=bool)]))

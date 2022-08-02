from distutils.ccompiler import new_compiler
import numpy as np
import pathlib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import rgb2hex
import seaborn as sns
import pandas as pd
import umap

np.random.seed(0)


def get_features_data(load_path: pathlib.Path) -> pd.DataFrame:
    """get features data from csv at load path
    Args:
        load_path (pathlib.Path): path to training data csv
    Returns:
        pd.DataFrame: training dataframe
    """
    # read dataset into pandas dataframe
    features_data = pd.read_csv(load_path, index_col=0)

    # remove training data with ADCCM class as this class was not used for classification in original paper
    features_data = features_data[
        features_data["Mitocheck_Phenotypic_Class"] != "ADCCM"
    ]

    # replace shape1 and shape3 labels with their correct respective classes
    features_data = features_data.replace("Shape1", "Binuclear")
    features_data = features_data.replace("Shape3", "Polylobed")

    return features_data


def show_1D_umap(features_dataframe: pd.DataFrame, metadata_series: pd.Series, save_path: str = None):
    """show 1D umap with features, colored by metadata categories

    Args:
        features_dataframe (pd.DataFrame): features to compress with umap
        metadata_series (pd.Series): metadata to color umap
        save_path (str, optional): save path for umap embeddings, should end in .tsv. If none embeddings will not be saved Defaults to None.
    """
    # create umap object for dimension reduction
    reducer = umap.UMAP(random_state=0, n_components=1)

    # get feature values as numpy array
    feature_data = features_dataframe.values

    # Fit UMAP and extract latent var 1
    embedding = pd.DataFrame(reducer.fit_transform(feature_data), columns=["UMAP1"])

    # create random y distribution to space out points
    y_distribution = np.random.rand(feature_data.shape[0])
    embedding["y_distribution"] = y_distribution.tolist()

    # add phenotypic class to embeddings
    embedding[metadata_series.name] = metadata_series.tolist()

    plt.figure(figsize=(15, 12))

    # Produce scatterplot with umap data, using phenotypic classses to color
    sns_plot = sns.scatterplot(
        palette="rainbow",
        x="UMAP1",
        y="y_distribution",
        data=embedding,
        hue=embedding[metadata_series.name].tolist(),
        alpha=0.5,
        linewidth=0,
    )
    # Adjust legend
    sns_plot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # Label axes, title
    sns_plot.set_xlabel("UMAP 1")
    sns_plot.set_ylabel("Random Distribution")
    sns_plot.set_title("1 Dimensional UMAP")

    # save embedding
    if not save_path == None:
        embedding.to_csv(save_path, sep="\t", index=False)


def show_2D_umap(features_dataframe: pd.DataFrame, metadata_series: pd.Series, save_path=None):
    """show 2D umap with features, colored by metadata categories

    Args:
        features_dataframe (pd.DataFrame): features to compress with umap
        metadata_series (pd.Series): metadata to color umap
        save_path (str, optional): save path for umap embeddings, should end in .tsv. If none embeddings will not be saved Defaults to None.
    """
    # create umap object for dimension reduction
    reducer = umap.UMAP(random_state=0, n_components=2)

    # get feature values as numpy array
    feature_data = features_dataframe.values

    # Fit UMAP and extract latent vars 1-2
    embedding = pd.DataFrame(
        reducer.fit_transform(feature_data), columns=["UMAP1", "UMAP2"]
    )

    # add phenotypic class to embeddings
    embedding[metadata_series.name] = metadata_series.tolist()

    plt.figure(figsize=(15, 12))

    # Produce scatterplot with umap data, using phenotypic classses to color
    sns_plot = sns.scatterplot(
        palette="rainbow",
        x="UMAP1",
        y="UMAP2",
        data=embedding,
        hue=embedding[metadata_series.name].tolist(),
        alpha=0.5,
        linewidth=0,
    )
    # Adjust legend
    sns_plot.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # Label axes, title
    sns_plot.set_xlabel("UMAP 1")
    sns_plot.set_ylabel("UMAP 2")
    sns_plot.set_title("2 Dimensional UMAP")

    # save embedding
    if not save_path == None:
        embedding.to_csv(save_path, sep="\t", index=False)


def show_3D_umap(features_dataframe: pd.DataFrame, metadata_series: pd.Series, save_path=None):
    """show 3D umap with features, colored by metadata categories

    Args:
        features_dataframe (pd.DataFrame): features to compress with umap
        metadata_series (pd.Series): metadata to color umap
        save_path (str, optional): save path for umap embeddings, should end in .tsv. If none embeddings will not be saved Defaults to None.
    """
    # create umap object for dimension reduction
    reducer = umap.UMAP(random_state=0, n_components=3)

    # get feature values as numpy array
    feature_data = features_dataframe.values

    # Fit UMAP and extract latent vars 1-3
    embedding = pd.DataFrame(
        reducer.fit_transform(feature_data), columns=["UMAP1", "UMAP2", "UMAP3"]
    )

    # add phenotypic class to embeddings
    embedding[metadata_series.name] = metadata_series.tolist()

    fig = plt.figure(figsize=(17, 17))
    ax = fig.gca(projection="3d")
    cmap = sns.color_palette(
        "rainbow", embedding[metadata_series.name].nunique()
    )
    legend_elements = []

    # add each phenotypic class to 3d graph and legend
    for index, metadata_class in enumerate(
        embedding[metadata_series.name].unique().tolist()
    ):
        class_embedding = embedding.loc[
            embedding[metadata_series.name] == metadata_class
        ]
        x = class_embedding["UMAP1"]
        y = class_embedding["UMAP2"]
        z = class_embedding["UMAP3"]
        ax.scatter(x, y, z, c=rgb2hex(cmap[index]), marker="o", alpha=0.5)
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=metadata_class,
                markerfacecolor=rgb2hex(cmap[index]),
                markersize=10,
            )
        )

    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
    # Label axes, title
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    ax.set_title("3 Dimensional UMAP")

    plt.show()

    # save embedding
    if not save_path == None:
        embedding.to_csv(save_path, sep="\t", index=False)

"""
utilities for evaluating logistic regression models on training and testing datasets
"""

from typing import Tuple, Literal
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns

from train_utils import get_X_y_data


# set numpy seed to make random operations reproduceable
np.random.seed(0)


def class_PR_curves(
    single_cell_data: pd.DataFrame, log_reg_model: LogisticRegression, feature_type: str
) -> Tuple[Figure, pd.DataFrame]:
    """
    save precision recall curves for each class to the save directory
    also, return the precision/recall data for each class in tidy long format

    Parameters
    ----------
    single_cell_data : pd.DataFrame
        single cell dataframe with correct cell metadata and feature data
    log_reg_model : sklearn classifier
        clasifier to get precision recall curves for
    feature_type : str
        which feature type is being evaluated (CP, DP, CP_and_DP)

    Returns
    -------
    matplotlib.figure.Figure
        figure with compilation of all class PR curves

    pd.DataFrame
        dataframe with precision/recall data for each class in tidy long format
    """

    phenotypic_classes = log_reg_model.classes_
    
    X, y = get_X_y_data(single_cell_data, feature_type)

    # binarize labels for precision recall curve function
    y_binarized = label_binarize(y, classes=phenotypic_classes)
    # predict class probabilities for feature data
    y_score = log_reg_model.predict_proba(X)

    # data from PR curves will be stored in tidy data format (eventually pandas dataframe)
    PR_data = []

    fig, axs = plt.subplots(3, 5)
    fig.set_size_inches(15, 9)
    ax_x = 0
    ax_y = 0
    for i in range(phenotypic_classes.shape[0]):
        precision, recall, threshold = precision_recall_curve(
            y_binarized[:, i], y_score[:, i]
        )
        # last values in precision/recall curve don't correspond to cell dataset
        threshold = np.append(threshold, None)
        PR_data.append(
            pd.DataFrame(
                {
                    "Phenotypic_Class": phenotypic_classes[i],
                    "PR_Threshold": threshold,
                    "Precision": precision,
                    "Recall": recall,
                }
            )
        )

        axs[ax_x, ax_y].plot(recall, precision, lw=2, label=phenotypic_classes[i])
        axs[ax_x, ax_y].set_title(phenotypic_classes[i])
        axs[ax_x, ax_y].set(xlabel="Recall", ylabel="Precision")

        # increase row coordinate counter (this marks which subplot to plot on in vertical direction)
        ax_x += 1
        # if row coordinate counter is at maximum (3 rows of subplots)
        if ax_x == 3:
            # set row coordinate counter to 0
            ax_x = 0
            # increase column coordinate counter (this marks which subplot to plot on in horizontal direction)
            ax_y += 1

    # only label outer plots
    for ax in axs.flat:
        ax.label_outer()

    PR_data = pd.concat(PR_data, axis=0).reset_index(drop=True)
    return fig, PR_data


def get_SCM_model_data(
    given_single_cell_data: pd.DataFrame, phenotypic_class: str, evaluation_type: str
) -> pd.DataFrame:
    """
    convert single cell data with metadata and features to usable single class model data
    rename phenotypic classes that are not the desired class to "Not {phenotypic_class}"
    if evaluation type is training, downsample negative samples to get 50/50 positive/negative split

    Parameters
    ----------
    given_single_cell_data : pd.DataFrame
        single cell data with metadata and features that has all phenotypic classes
    phenotypic_class : str
        desired phenotypic class
    evaluation_type : str
        type of dataset to evaluate with (train or test)

    Returns
    -------
    pd.DataFrame
        single cell data usable for single class models/evaluation
    """

    # create deep copy so original dataframe is not affected
    single_cell_data = given_single_cell_data.copy(deep=True)

    # rename false labels to "{positive label} Negative"
    single_cell_data.loc[
        single_cell_data["Mitocheck_Phenotypic_Class"] != phenotypic_class,
        "Mitocheck_Phenotypic_Class",
    ] = f"{phenotypic_class} Negative"

    # because we downsampled negative labels (to offset large label imbalance) in 2.train_model,
    # it is necessary to get the subset of training data that was used to actually train this specific model
    if evaluation_type == "train":
        # first, get indexes of all positive labels (labels that are the desired phenotypic class)
        positive_label_indexes = (
            single_cell_data.loc[
                single_cell_data["Mitocheck_Phenotypic_Class"] == phenotypic_class
            ]
        ).index
        # next, get the same number of negative labels (labels that are not the desired phenotypic class)
        negative_label_indexes = (
            (
                single_cell_data.loc[
                    single_cell_data["Mitocheck_Phenotypic_Class"] != phenotypic_class
                ]
            )
            .sample(positive_label_indexes.shape[0], random_state=0)
            .index
        )
        # the new class training data are the two subsets found above
        # this new class training data will have equal numbers of positive and negative labels
        # this removes the drastic class imbalances
        single_cell_data = single_cell_data.loc[
            positive_label_indexes.union(negative_label_indexes)
        ]

    return single_cell_data


def class_PR_curves_SCM(
    single_cell_data: pd.DataFrame,
    single_class_model: LogisticRegression,
    fig: Figure,
    axs: np.ndarray,
    phenotypic_class_index: int,
    data_split_colors: dict,
    model_type: Literal["final", "shuffled_baseline"],
    feature_type: Literal["CP", "DP", "CP_and_DP"],
    evaluation_type: Literal["train", "test"],
    phenotypic_class: str,
) -> pd.DataFrame:
    """
    add PR curves to fig, axs for the single class model using feature_type (CP, DP, CP_and_DP), evaluation_type (test or train), and phenotypic_class
    also, return PR curve data

    Parameters
    ----------
    single_cell_data : pd.DataFrame
        single cell data with multi-class labels (all phenotypic classes), metadata, and feature data
    single_class_model : LogisticRegression
        single class model to create PR data for
    fig : Figure
        matplotlib figure to add PR curve plot to
    axs : np.ndarray
        axes for matplotlib figure
    phenotypic_class_index : int
        index of phenotypic class in all phenotypic classes (used to determine location of PR curve in figure)
    data_split_colors : dict
        dictionary with information of which colors to use for which model, feature, evaluation type when plotting
    model_type : str
        type of model (final or shuffled baseline)
    feature_type : str
        feature type model uses (CP, DP, or CP_and_DP)
    evaluation_type : str
        type of data being used for evaluation (test or train)
    phenotypic_class : str
        phenotypic class of single cell model being evaluated
    Returns
    -------
    pd.DataFrame
        data for PR curves for single cell model
    """

    # keep track of PR data for later analysis
    PR_data = []

    # rename negative labels and downsample negative labels if we are evaluating on training data
    single_cell_data = get_SCM_model_data(
        single_cell_data, phenotypic_class, evaluation_type
    )

    model_classes = single_class_model.classes_
    X, y = get_X_y_data(single_cell_data, feature_type)

    # predict class probabilities for feature data
    y_probas = single_class_model.predict_proba(X)

    # figure out where to plot PR curves
    # ax_x is x axis coordinate for positive label PR curve, ax_y is y axis coordinate for positive label PR curve
    # negative label PR curves go below positive label (so ax_x+1)
    ax_x = (
        int(phenotypic_class_index / 5) * 2
    )  # multiply by 2 because every positive label has a negative label below it
    ax_y = phenotypic_class_index % 5  # modulus gives us y coordinate

    color_key = f"{feature_type} ({evaluation_type})"
    plot_color = data_split_colors[color_key]

    for index, model_class in enumerate(model_classes):
        # get precision, recall, threshold values for the model class (positive or negative label)
        precision, recall, thresholds = precision_recall_curve(
            y, y_probas[:, index], pos_label=model_class
        )
        # last values in precision/recall curve don't correspond to cell dataset
        thresholds = np.append(thresholds, None)

        # plot PR data
        axs[ax_x + index, ax_y].plot(
            recall, precision, label=color_key, color=plot_color
        )
        # set title and axis labels for subplot
        axs[ax_x + index, ax_y].set_title(model_class)
        axs[ax_x + index, ax_y].set(xlabel="Recall", ylabel="Precision")

        # add pr data to compiled dataframe
        PR_data.append(
            pd.DataFrame(
                {
                    "Model_Class": model_class,
                    "PR_Threshold": thresholds,
                    "Precision": precision,
                    "Recall": recall,
                    "data_split": evaluation_type,
                    "shuffled": "shuffled" in model_type,
                    "feature_type": feature_type,
                }
            )
        )

    # only label outer plots
    for ax in axs.flat:
        ax.label_outer()

    # add legend to figure with all subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    # compile PR data
    # some thresholds are None because last PR value doesn't correspond to cell dataset (these values are always P=1, R=0), remove these rows from PR data
    PR_data = pd.concat(PR_data, axis=0)

    return PR_data


def model_confusion_matrix(
    log_reg_model: LogisticRegression,
    dataset: pd.DataFrame,
    feature_type: str,
    ax: Axes = None,
) -> Tuple[pd.DataFrame, Axes]:
    """
    display confusion matrix for logistic regression model on dataset

    Parameters
    ----------
    log_reg_model : LogisticRegression
        logistic regression model to evaluate
    dataset : pd.DataFrame
        dataset to evaluate model on
    feature_type : str
        which feature type is being evaluated (CP, DP, CP_and_DP)
    ax : Axes, optional
        Axes object to plot confusion matrix on, by default None

    Returns
    -------
    pd.DataFrame
        confusion matrix of model evaluated on dataset
    matplotlib.axes.Axes
        Axes object with confusion matrix display
    """
    
    X, y = get_X_y_data(dataset, feature_type)

    # get predictions from model
    y_pred = log_reg_model.predict(X)

    # create confusion matrix
    conf_mat = confusion_matrix(y, y_pred, labels=log_reg_model.classes_)
    conf_mat = pd.DataFrame(
        conf_mat, columns=log_reg_model.classes_, index=log_reg_model.classes_
    )

    # create confusion matrix figure on ax that is given or make new ax
    if ax is None:
        ax = sns.heatmap(
            data=conf_mat,
            annot=True,
            fmt=".0f",
            cmap="viridis",
            square=True,
            cbar=False,
        )
    else:
        sns.heatmap(
            data=conf_mat,
            annot=True,
            fmt=".0f",
            cmap="viridis",
            square=True,
            cbar=False,
            ax=ax,
        )

    return conf_mat, ax


def model_F1_score(
    log_reg_model: LogisticRegression,
    dataset: pd.DataFrame,
    feature_type: str,
    ax: Axes = None,
) -> Tuple[pd.DataFrame, Axes]:
    """
    get model F1 score for given dataset and create bar graph with class/weighted F1 scores
    also return axes with bar graph of F1 scores

    Parameters
    ----------
    log_reg_model : LogisticRegression
        model to evaluate
    dataset : pd.DataFrame
        dataset with features and true phenotypic class labels to evaluate model with
    feature_type : str
        which feature type is being evaluated (CP, DP, CP_and_DP)
    ax : Axes, optional
        Axes object to plot F1 scores bar graph on, by default None

    Returns
    -------
    pd.DataFrame
        dataframe with phenotpic class and weighted F1 scores
    matplotlib.axes.Axes
        Axes object with F1 scores bar graph
    """
    
    X, y = get_X_y_data(dataset, feature_type)

    # get predictions from model
    y_pred = log_reg_model.predict(X)

    # display precision vs phenotypic class bar chart
    scores = f1_score(
        y, y_pred, average=None, labels=log_reg_model.classes_, zero_division=0
    )
    weighted_score = f1_score(
        y, y_pred, average="weighted", labels=log_reg_model.classes_, zero_division=0
    )
    scores = pd.DataFrame(scores).T
    scores.columns = log_reg_model.classes_
    scores["Weighted"] = weighted_score

    # create bar graph figure on ax that is given or make new ax
    if ax is None:
        ax = sns.barplot(data=scores)
    else:
        sns.barplot(data=scores, ax=ax)

    # try to rotate x axis labels for better fitting
    plt.xticks(rotation=90)

    return scores, ax

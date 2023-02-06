"""
utilities for evaluating logistic regression models on training and testing datasets
"""

from typing import Tuple
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from train_utils import get_X_y_data


# set numpy seed to make random operations reproduceable
np.random.seed(0)


def class_PR_curves(
    single_cell_data: pd.DataFrame, log_reg_model: LogisticRegression
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

    Returns
    -------
    matplotlib.figure.Figure
        figure with compilation of all class PR curves

    pd.DataFrame
        dataframe with precision/recall data for each class in tidy long format
    """

    phenotypic_classes = log_reg_model.classes_
    X, y = get_X_y_data(single_cell_data)

    # binarize labels for precision recall curve function
    y_binarized = label_binarize(y, classes=phenotypic_classes)
    # predict class probabilities for feature data
    y_score = log_reg_model.predict_proba(X)

    # data from PR curves will be stored in tidy data format (eventually pandas dataframe)
    PR_data = []
    # which thresholds the precision/recalls correspond to
    PR_threshold = np.arange(single_cell_data.shape[0])
    # last values in precision/recall curve don't correspond to cell dataset
    PR_threshold = np.append(PR_threshold, None)

    fig, axs = plt.subplots(4, 4)
    fig.set_size_inches(15, 15)
    ax_x = 0
    ax_y = 0
    for i in range(phenotypic_classes.shape[0]):
        precision, recall, _ = precision_recall_curve(y_binarized[:, i], y_score[:, i])
        PR_data.append(
            pd.DataFrame(
                {
                    "Phenotypic_Class": phenotypic_classes[i],
                    "PR_Threshold": PR_threshold,
                    "Precision": precision,
                    "Recall": recall,
                }
            )
        )

        axs[ax_x, ax_y].plot(recall, precision, lw=2, label=phenotypic_classes[i])
        axs[ax_x, ax_y].set_title(phenotypic_classes[i])
        axs[ax_x, ax_y].set(xlabel="Recall", ylabel="Precision")

        ax_x += 1
        if ax_x == 4:
            ax_x = 0
            ax_y += 1

    # only label outer plots
    for ax in axs.flat:
        ax.label_outer()

    PR_data = pd.concat(PR_data, axis=0).reset_index(drop=True)
    return fig, PR_data


def model_cm(
    log_reg_model: LogisticRegression, dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    display confusion matrix for logistic regression model on dataset

    Parameters
    ----------
    log_reg_model : LogisticRegression
        logistic regression model to evaluate
    dataset : pd.DataFrame
        dataset to evaluate model on

    Returns
    -------
    pd.DataFrame
        confusion matrix of model evaluated on dataset
    """

    # get features and labels dataframes
    X, y = get_X_y_data(dataset)

    # get predictions from model
    y_pred = log_reg_model.predict(X)

    # create confusion matrix
    conf_mat = confusion_matrix(y, y_pred, labels=log_reg_model.classes_)
    conf_mat = pd.DataFrame(conf_mat, columns=log_reg_model.classes_, index=log_reg_model.classes_)

    # display confusion matrix
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(data=conf_mat, annot=True, fmt=".0f", cmap="viridis", square=True)
    ax = plt.xlabel("Predicted Label")
    ax = plt.ylabel("True Label")
    ax = plt.title("Phenotypic Class Predicitions")
    plt.show()

    return conf_mat

def model_score(log_reg_model: LogisticRegression, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    get model F1 score for given dataset and create bar graph with class/weighted F1 scores

    Parameters
    ----------
    log_reg_model : LogisticRegression
        model to evaluate
    dataset : pd.DataFrame
        dataset with features and true phenotypic class labels to evaluate model with

    Returns
    -------
    pd.DataFrame
        dataframe with phenotpic class and weighted F1 scores
    """

    # get features and labels dataframes
    X, y = get_X_y_data(dataset)

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

    plt.figure(figsize=(20, 8))
    plt.xlabel("Phenotypic Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Phenotpyic Class")
    plt.xticks(rotation=90)
    ax = sns.barplot(data=scores)
    
    plt.show()
    
    return scores

def evaluate_model_cm(
    log_reg_model: LogisticRegression, dataset: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    display confusion matrix for logistic regression model on dataset

    Parameters
    ----------
    log_reg_model : LogisticRegression
        logistic regression model to evaluate
    dataset : pd.DataFrame
        dataset to evaluate model on

    Returns
    -------
    np.ndarray
        true labels
    np.ndarray
        predicted labels
    """

    # get features and labels dataframes
    X, y = get_X_y_data(dataset)

    # get predictions from model
    y_pred = log_reg_model.predict(X)

    # create confusion matrix
    conf_mat = confusion_matrix(y, y_pred, labels=log_reg_model.classes_)
    conf_mat = pd.DataFrame(conf_mat)
    conf_mat.columns = log_reg_model.classes_
    conf_mat.index = log_reg_model.classes_

    # display confusion matrix
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(data=conf_mat, annot=True, fmt=".0f", cmap="viridis", square=True)
    ax = plt.xlabel("Predicted Label")
    ax = plt.ylabel("True Label")
    ax = plt.title("Phenotypic Class Predicitions")

    return y, y_pred


def evaluate_model_score(log_reg_model: LogisticRegression, dataset: pd.DataFrame):
    """display bar graph for model with scoring metric on each class
    Args:
        log_reg_model (LogisticRegression): logisitc regression model to evaluate
        dataset (pd.DataFrame): dataset to evaluate model on
    """

    # get features and labels dataframes
    X, y = get_X_y_data(dataset)

    # get predictions from model
    y_pred = log_reg_model.predict(X)

    # display precision vs phenotypic class bar chart
    scores = f1_score(
        y, y_pred, average=None, labels=log_reg_model.classes_, zero_division=0
    )
    scores = pd.DataFrame(scores).T
    scores.columns = log_reg_model.classes_

    sns.set(rc={"figure.figsize": (20, 8)})
    plt.xlabel("Phenotypic Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Phenotpyic Class")
    plt.xticks(rotation=90)
    ax = sns.barplot(data=scores)

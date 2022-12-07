"""
utilities for evaluating logistic regression models on training and testing datasets
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

from train_utils import get_X_y_data

# set numpy seed to make random operations reproduceable
np.random.seed(0)

def evaluate_model_cm(
    log_reg_model: LogisticRegression, dataset: pd.DataFrame
):
    """display confusion matrix for logistic regression model on dataset
    Args:
        log_reg_model (LogisticRegression): logisitc regression model to evaluate
        dataset (pd.DataFrame): dataset to evaluate model on
    Returns:
        np.ndarray, np.ndarray: true, predicted labels
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

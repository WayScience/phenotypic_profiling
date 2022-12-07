"""
utilities for training logistic regression models on MitoCheck single-cell dataset
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# set numpy seed to make random operations reproduceable
np.random.seed(0)

def get_dataset(
    features_dataframe: pd.DataFrame, data_split_indexes: pd.DataFrame, label: str
) -> pd.DataFrame:
    """get testing data from features dataframe and the data split indexes
    Args:
        features_dataframe (pd.DataFrame): dataframe with all features data
        data_split_indexes (pd.DataFrame): dataframe with split indexes
        label (str): label to get data for (train, test, holdout)
    Returns:
        pd.DataFrame: _description_
    """
    indexes = data_split_indexes.loc[data_split_indexes["label"] == label]
    indexes = indexes["index"]
    data = features_dataframe.loc[indexes]

    return data


def get_X_y_data(training_data: pd.DataFrame):
    """generate X (features) and y (labels) dataframes from training data
    Args:
        training_data (pd.DataFrame): training dataframe
    Returns:
        pd.DataFrame, pd.DataFrame: X, y dataframes
    """

    # all features from DeepProfiler have "efficientnet" in their column name
    morphology_features = [
        col for col in training_data.columns.tolist() if "efficientnet" in col
    ]

    # extract features
    X = training_data.loc[:, morphology_features].values

    # extract phenotypic class label
    y = training_data.loc[:, ["Mitocheck_Phenotypic_Class"]].values
    # make Y data
    y = np.ravel(y)

    # shuffle data because as it comes from MitoCheck same labels tend to be in grou
    X, y = shuffle(X, y, random_state=0)

    return X, y

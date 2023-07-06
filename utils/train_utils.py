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
    indexes = indexes["labeled_data_index"]
    data = features_dataframe.loc[indexes]

    return data


def get_X_y_data(labeled_data: pd.DataFrame, dataset: str = "CP_and_DP"):
    """generate X (features) and y (labels) dataframes from training data
    Args:
        labeled_data (pd.DataFrame):
            training dataframe
        dataset : str, optional
            which dataset columns to get feature data for
            can be "CP" or "DP" or by default "CP_and_DP"
    Returns:
        pd.DataFrame, pd.DataFrame: X, y dataframes
    """
    all_cols = labeled_data.columns.tolist()

    # get DP,CP, or both features from all columns depending on desired dataset
    if dataset == "CP":
        feature_cols = [col for col in all_cols if "CP__" in col]
    elif dataset == "DP":
        feature_cols = [col for col in all_cols if "DP__" in col]
    elif dataset == "CP_and_DP":
        feature_cols = [col for col in all_cols if "P__" in col]

    # extract features
    X = labeled_data.loc[:, feature_cols].values

    # extract phenotypic class label
    y = labeled_data.loc[:, ["Mitocheck_Phenotypic_Class"]].values
    # make Y data
    y = np.ravel(y)

    # shuffle data because as it comes from MitoCheck same labels tend to be in grou
    X, y = shuffle(X, y, random_state=0)

    return X, y

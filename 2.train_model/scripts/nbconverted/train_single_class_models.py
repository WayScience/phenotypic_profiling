#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings

import pandas as pd
import numpy as np
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.utils import shuffle, parallel_backend
from sklearn.exceptions import ConvergenceWarning
from joblib import dump

import sys

sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_dataset, get_X_y_data


# In[2]:


# set numpy seed to make random operations reproduceable
np.random.seed(0)

# load training data from indexes and features dataframe
data_split_path = pathlib.Path(f"../1.split_data/indexes/data_split_indexes.tsv")
features_dataframe_path = pathlib.Path("../0.download_data/data/labeled_data.csv.gz")

# dataframe with only the labeled data we want (exclude certain phenotypic classes)
features_dataframe = get_features_data(features_dataframe_path)
data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)

# get training data from labeled data
training_data = get_dataset(features_dataframe, data_split_indexes, "train")
training_data


# In[3]:


# specify model types, feature types, and phenotypic classes
model_types = ["final", "shuffled_baseline"]
feature_types = ["CP", "DP", "CP_and_DP"]
phenotypic_classes = training_data["Mitocheck_Phenotypic_Class"].unique()

# create stratified data sets for k-fold cross validation
straified_k_folds = StratifiedKFold(n_splits=10, shuffle=False)

# create logistic regression model with following parameters
log_reg_model = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    max_iter=100,
    n_jobs=-1,
    random_state=0,
    class_weight="balanced"
)

# specify parameters to tune for
parameters = {"C": np.logspace(-3, 3, 7), "l1_ratio": np.linspace(0, 1, 11)}
print(f"Parameters being tested during grid search: {parameters}\n")

# create grid search with cross validation with hypertuning params
grid_search_cv = GridSearchCV(
    log_reg_model, parameters, cv=straified_k_folds, n_jobs=-1, scoring="f1_weighted"
)

# train model on each combination of model type, feature type, and phenotypic class
for model_type, feature_type, phenotypic_class in itertools.product(model_types, feature_types, phenotypic_classes):
    # create results directory
    results_dir = pathlib.Path(
        f"models/single_class_models/{phenotypic_class}_models/"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # get number of labels for this specific phenotypic_class
    phenotypic_class_counts = (
        training_data.loc[
            training_data["Mitocheck_Phenotypic_Class"] == phenotypic_class
        ]
    ).shape[0]
    print(
        f"Training {model_type} model on {feature_type} features for {phenotypic_class} phenotypic class with {phenotypic_class_counts} positive labels..."
    )

    # create deep copy of training data so we can make modifications without affecting original training data
    class_training_data = training_data.copy(deep=True)
    # convert labels that are not phenotypic class to 0 (negative)
    class_training_data.loc[
        class_training_data["Mitocheck_Phenotypic_Class"] != phenotypic_class,
        "Mitocheck_Phenotypic_Class",
    ] = f"Not {phenotypic_class}"
    
    # because the label balance is so great for some classes (ex: 50 positive labels to 2400 negative labels),
    # it is nessary to undersample negative labels
    # the following code completes the undersampling
    # first, get indexes of all positive labels (labels that are the desired phenotypic class) 
    positive_label_indexes = (
        training_data.loc[
            training_data["Mitocheck_Phenotypic_Class"] == phenotypic_class
        ]
    ).index
    # next, get the same number of negative labels (labels that are not the desired phenotypic class) 
    negative_label_indexes = (
        training_data.loc[
            training_data["Mitocheck_Phenotypic_Class"] != phenotypic_class
        ]
    ).sample(phenotypic_class_counts, random_state=0).index
    # the new class training data are the two subsets found above
    # this new class training data will have equal numbers of positive and negative labels
    # this removes the drastic class imbalances
    class_training_data = class_training_data.loc[positive_label_indexes.union(negative_label_indexes)]

    # get X (features) and y (labels) data
    X, y = get_X_y_data(class_training_data, feature_type)
    print(f"X has shape {X.shape}, y has shape {y.shape}")
    
    # shuffle columns of X (features) dataframe independently to create shuffled baseline
    if model_type == "shuffled_baseline":
        for column in X.T:
            np.random.shuffle(column)

    # fit grid search cv to X and y data
    # capture convergence warning from sklearn
    # this warning does not affect the model but takes up lots of space in the output
    with parallel_backend("multiprocessing"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            grid_search_cv = grid_search_cv.fit(X, y)

    # print info for best estimator
    print(f"Best parameters: {grid_search_cv.best_params_}")
    print(f"Score of best estimator: {grid_search_cv.best_score_}\n")

    # save final estimator
    dump(
        grid_search_cv.best_estimator_,
        f"{results_dir}/{model_type}__{feature_type}.joblib",
    )


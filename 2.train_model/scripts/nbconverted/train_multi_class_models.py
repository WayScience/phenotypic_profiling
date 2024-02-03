#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pathlib
import warnings
import itertools

import pandas as pd
import numpy as np

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


warnings.simplefilter("ignore", category=ConvergenceWarning)


# ### Specify load paths, dataset types, feature types, model types

# In[3]:


# set numpy seed to make random operations reproduceable
np.random.seed(0)

# create results directory
results_dir = pathlib.Path("models/multi_class_models/")
results_dir.mkdir(parents=True, exist_ok=True)

# load training data from indexes and features dataframe
data_split_dir = pathlib.Path(f"../1.split_data/indexes/data_split_indexes.tsv")
labeled_data_dir = pathlib.Path("../0.download_data/data/labeled_data.csv.gz")

# specify dataset type, model types, feature, and balance types
model_types = ["final", "shuffled_baseline"]
feature_types = ["CP", "DP", "CP_and_DP", "CP_zernike_only", "CP_areashape_only"]
balance_types = ["balanced", "unbalanced"]
dataset_types = ["ic", "no_ic"]


# ## Specify parameters to tune for

# In[4]:


parameters = {"C": np.logspace(-3, 3, 7), "l1_ratio": np.linspace(0, 1, 11)}
print(f"Parameters being tested during grid search: {parameters}\n")


# ## Find best parameters, train models

# In[5]:


for model_type, feature_type, balance_type, dataset_type in itertools.product(
    model_types, feature_types, balance_types, dataset_types
):
    # print what combination of types we are training model for
    print(
        f"Training model for: \nModel Type: {model_type} \nFeature Type: {feature_type} \nBalance Type: {balance_type} \nDataset Type: {dataset_type}"
    )
    
    # load training data from indexes and features dataframe
    data_split_path = pathlib.Path(f"../1.split_data/indexes/data_split_indexes__{dataset_type}.tsv")
    features_dataframe_path = pathlib.Path(f"../0.download_data/data/labeled_data__{dataset_type}.csv.gz")
    
    # dataframe with only the labeled data we want (exclude certain phenotypic classes)
    features_dataframe = get_features_data(features_dataframe_path)
    data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)

    # get training data from labeled data
    training_data = get_dataset(features_dataframe, data_split_indexes, "train")
    
    # get X,y data from training data
    X, y = get_X_y_data(
                training_data,
                feature_type,
            )
    print(f"X has shape {X.shape}, y has shape {y.shape}")
    
    # shuffle columns of X (features) dataframe independently to create shuffled baseline
    if model_type == "shuffled_baseline":
        for column in X.T:
            np.random.shuffle(column)
            
    # fit grid search cv to X and y data
    with parallel_backend("multiprocessing"):
        
        # create stratified data sets for k-fold cross validation
        straified_k_folds = StratifiedKFold(n_splits=10, shuffle=False)
        
        # create logistic regression model with following parameters
        log_reg_model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            class_weight= "balanced" if balance_type == "balanced" else None,
            max_iter=100,
            n_jobs=-1,
            random_state=0
        )
        
        # create grid search with cross validation with hypertuning params
        grid_search_cv = GridSearchCV(
            log_reg_model,
            parameters,
            cv=straified_k_folds,
            n_jobs=-1,
            scoring="f1_weighted",
        )
        grid_search_cv = grid_search_cv.fit(X, y)

    # print info for best estimator
    print(f"Best parameters: {grid_search_cv.best_params_}")
    print(f"Score of best estimator: {grid_search_cv.best_score_}\n")

    # save final estimator
    save_path = pathlib.Path(f"{results_dir}/{model_type}__{feature_type}__{balance_type}__{dataset_type}.joblib")
    dump(
        grid_search_cv.best_estimator_,
        save_path,
    )
    
    print(f"Saved trained model to {save_path}\n")


#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pathlib
import warnings

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


# ### Specify results directory, load training data

# In[3]:


# set numpy seed to make random operations reproduceable
np.random.seed(0)

# create results directory
results_dir = pathlib.Path("models/multi_class_models/")
results_dir.mkdir(parents=True, exist_ok=True)

# load training data from indexes and features dataframe
data_split_path = pathlib.Path(f"../1.split_data/indexes/data_split_indexes.tsv")
features_dataframe_path = pathlib.Path("../0.download_data/data/labeled_data.csv.gz")

# dataframe with only the labeled data we want (exclude certain phenotypic classes)
features_dataframe = get_features_data(features_dataframe_path)
data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)

# get training data from labeled data
training_data = get_dataset(features_dataframe, data_split_indexes, "train")
training_data


# ### Train model on each combination of model type and feature type

# In[4]:


# specify model types and feature types
model_types = ["final", "shuffled_baseline"]
feature_types = ["CP", "DP", "CP_and_DP", "CP_zernike_only", "CP_areashape_only"]
balanced_types = ["balanced", "unbalanced"]

# specify parameters to tune for
parameters = {"C": np.logspace(-3, 3, 7), "l1_ratio": np.linspace(0, 1, 11)}
print(f"Parameters being tested during grid search: {parameters}\n")

# train model on each combination of model type, feature type, and balance type
for balance in balanced_types:
    # Set sklearn class_weight parameter for balanced or unbalanced models
    if balance == "balanced":
        balance_model = "balanced"
    else:
        balance_model = None
        
    for model_type in model_types:
        for feature_type in feature_types:
            
            if feature_type == "CP_zernike_only":
                zernike_only = True
                dataset = "CP"
            else:
                zernike_only = False
                dataset = feature_type
                
            if feature_type == "CP_areashape_only":
                area_shape_only = True
                dataset = "CP"
            else:
                area_shape_only = False
    
            print(f"Training {model_type} {balance} model on {feature_type} features with zernike only {zernike_only} or area features only {area_shape_only}...")
            
            X, y = get_X_y_data(
                training_data,
                dataset,
                zernike_only,
                area_shape_only
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
                    class_weight=balance_model,
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
            dump(
                grid_search_cv.best_estimator_,
                f"{results_dir}/{model_type}__{feature_type}__{balance}.joblib",
            )


#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries
# 

# In[1]:


import sys
import pathlib
import itertools

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from joblib import load

sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_dataset, get_X_y_data
from evaluate_utils import get_SCM_model_data


# ## Set Data Load Paths
# 

# In[2]:


# load features data from indexes and features dataframe
data_split_dir_path = pathlib.Path("../1.split_data/indexes/")
features_dataframe_dir_path = pathlib.Path("../0.download_data/data/")


# ### Get Each Model Predictions on Each Dataset (multi class models)
# 

# In[3]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/multi_class_models")

# use a list to keep track of scores in tidy long format for each model and dataset combination
compiled_predictions = []

# iterate through each model (final model, shuffled baseline model, etc)
# sorted so final models are shown before shuffled_baseline
for model_path in sorted(models_dir.iterdir()):
    model = load(model_path)
    # determine model/feature type/balance/dataset type from model file name
    model_components = model_path.name.split("__")
    model_type = model_components[0]
    feature_type = model_components[1]
    balance_type = model_components[2]
    dataset_type = model_components[3].replace(".joblib", "")
    
    # load features data from indexes and features dataframe
    data_split_path = pathlib.Path(
        f"{data_split_dir_path}/data_split_indexes__{dataset_type}.tsv"
    )
    data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)
    features_dataframe_path = pathlib.Path(
        f"{features_dataframe_dir_path}/labeled_data__{dataset_type}.csv.gz"
    )
    features_dataframe = get_features_data(features_dataframe_path)

    # iterate through label datasets (labels correspond to train, test, etc)
    # with nested for loops, we test each model on each dataset(corresponding to a label)
    for label in data_split_indexes["label"].unique():
        print(
            f"Getting predictions for model with types {model_type}, {balance_type}, {feature_type}, {dataset_type} on {label} dataset"
        )

        # load dataset (train, test, etc)
        data = get_dataset(features_dataframe, data_split_indexes, label)

        # get features and labels dataframes
        X, y = get_X_y_data(data, feature_type)

        # get predictions from model
        y_pred = model.predict(X)

        # create dataframe with dataset index of cell being predicted,
        # predicted phenotypic class,
        # true phenotypic class,
        # and which dataset/models are involved in prediction
        predictions_df = pd.DataFrame(
            {
                "Cell_UUID": data["Cell_UUID"],
                "Phenotypic_Class_Predicted": y,
                "Phenotypic_Class_True": y_pred,
                "data_split": label,
                "shuffled": "shuffled" in model_type,
                "feature_type": feature_type,
                "balance_type": balance_type,
                "dataset_type": dataset_type,
            }
        )

        compiled_predictions.append(predictions_df)


# ### Compile and Save Predictions (multi class models)
# 

# In[4]:


# compile predictions and reset index of dataframe
compiled_predictions = pd.concat(compiled_predictions).reset_index(drop=True)

# specify save path
compiled_predictions_save_path = pathlib.Path("predictions/compiled_predictions.tsv")
compiled_predictions_save_path.parent.mkdir(parents=True, exist_ok=True)

# save data as tsv
compiled_predictions.to_csv(compiled_predictions_save_path, sep="\t")

# preview compiled predictions
compiled_predictions


# ### Get Each Model Predictions on Each Dataset (single class models)
# 

# In[5]:


# load features data from indexes and features dataframe
# single class models only trained on ic data
data_split_path = pathlib.Path("../1.split_data/indexes/data_split_indexes__ic.tsv")
data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)
features_dataframe_path = pathlib.Path(
    "../0.download_data/data/labeled_data__ic.csv.gz"
)
features_dataframe = get_features_data(features_dataframe_path)


# In[6]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/single_class_models")

# use a list to keep track of scores in tidy long format for each model and dataset combination
compiled_predictions = []

# define combinations to test over
model_types = [
    "final",
    "shuffled_baseline",
]  # only perform LOIO with hyper params from final models so skip shuffled_baseline models
feature_types = ["CP", "DP", "CP_and_DP"]
evaluation_types = ["train", "test"]
phenotypic_classes = features_dataframe["Mitocheck_Phenotypic_Class"].unique()

# iterate through each combination of feature_types, evaluation_types, phenotypic_classes
for model_type, feature_type, phenotypic_class, evaluation_type in itertools.product(
    model_types, feature_types, phenotypic_classes, evaluation_types
):
    # load single class model for this combination of model type, feature type, and phenotypic class
    single_class_model_path = pathlib.Path(
        f"{models_dir}/{phenotypic_class}_models/{model_type}__{feature_type}.joblib"
    )
    model = load(single_class_model_path)

    print(
        f"Getting predictions for {phenotypic_class} model: {model_type}, trained with features: {feature_type}, on dataset: {evaluation_type}"
    )

    # load dataset (train, test, etc)
    data = get_SCM_model_data(features_dataframe, phenotypic_class, evaluation_type)

    # get features and labels dataframe
    X, y = get_X_y_data(data, feature_type)

    # get predictions from model
    y_pred = model.predict(X)

    # create dataframe with dataset index of cell being predicted,
    # predicted phenotypic class,
    # true phenotypic class,
    # and which dataset/models are involved in prediction
    predictions_df = pd.DataFrame(
        {
            "Cell_UUID": data["Cell_UUID"],
            "Phenotypic_Class_Predicted": y,
            "Phenotypic_Class_True": y_pred,
            "data_split": evaluation_type,
            "shuffled": "shuffled" in model_type,
            "feature_type": feature_type,
        }
    )

    compiled_predictions.append(predictions_df)


# ### Compile and Save Predictions (single class models)
# 

# In[7]:


# compile predictions and reset index of dataframe
compiled_predictions = pd.concat(compiled_predictions).reset_index(drop=True)

# specify save path
compiled_predictions_save_path = pathlib.Path(
    "predictions/compiled_SCM_predictions.tsv"
)
compiled_predictions_save_path.parent.mkdir(parents=True, exist_ok=True)

# save data as tsv
compiled_predictions.to_csv(compiled_predictions_save_path, sep="\t")

# preview compiled predictions
compiled_predictions


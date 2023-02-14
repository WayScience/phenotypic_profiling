#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import pathlib

from sklearn.metrics import f1_score
from joblib import load

import sys
sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_dataset, get_X_y_data


# ### Load necessary data

# In[2]:


# load features data from indexes and features dataframe
data_split_path = pathlib.Path("../1.split_data/indexes/data_split_indexes.tsv")
data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)
features_dataframe_path = pathlib.Path("../0.download_data/data/training_data.csv.gz")
features_dataframe = get_features_data(features_dataframe_path)


# ### Get Each Model Predictions on Each Dataset

# In[3]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/")

# use a list to keep track of scores in tidy long format for each model and dataset combination
compiled_predictions = []

# iterate through each model (final model, shuffled baseline model, etc)
for model_path in models_dir.iterdir():
    model = load(model_path)
    model_name = model_path.name.replace("log_reg_", "").replace(".joblib", "")

    # iterate through label datasets (labels correspond to train, test, etc)
    # with nested for loops, we test each model on each dataset(corresponding to a label)
    for label in data_split_indexes["label"].unique():
        print(f"Getting predictions for {model_name} on dataset {label}")

        # get indexes of each cell being predicted for the dataset the cell is from
        dataset_indexes = data_split_indexes.loc[data_split_indexes["label"] == label][
            "index"
        ]

        # load dataset (train, test, etc)
        data = get_dataset(features_dataframe, data_split_indexes, label)

        # get features and labels dataframes
        X, y = get_X_y_data(data)

        # get predictions from model
        y_pred = model.predict(X)

        # create dataframe with dataset index of cell being predicted, 
        # predicted phenotypic class, 
        # true phenotypic class, 
        # and which dataset/models are involved in prediction
        predictions_df = pd.DataFrame(
            {
                "Dataset_Index": dataset_indexes,
                "Phenotypic_Class_Predicted": y,
                "Phenotypic_Class_True": y_pred,
                "data_split": label,
                "shuffled": "shuffled" in model_name,
            }
        )

        compiled_predictions.append(predictions_df)


# ### Compile and Save Predictions

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


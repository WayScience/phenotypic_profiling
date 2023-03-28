#!/usr/bin/env python
# coding: utf-8

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
from joblib import dump, load

import sys

sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_dataset, get_X_y_data


# In[2]:


# load labeled data
labeled_data_path = pathlib.Path("../0.download_data/data/labeled_data.csv.gz")
labeled_data = get_features_data(labeled_data_path)

# preview labeled data
print(labeled_data.shape)
labeled_data.head(5)


# In[3]:


# see number of images to 
num_images = labeled_data["Metadata_DNA"].unique().shape[0]
print(f"There are {num_images} images to perform LOIO evaluation on per model.")


# In[4]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/")

# use a list to keep track of LOIO probabilities in tidy long format for each model combination
compiled_LOIO_wide_data = []

count = 0

# iterate through each model (final model, shuffled baseline model, etc)
# sorted so final models are loaded before shuffled_baseline
for model_path in sorted(models_dir.iterdir()):
    
    # only perform LOIO with hyper params from final models
    if "shuffled" in model_path.name:
        continue

    model = load(model_path)
    # determine feature type from model file name
    feature_type = model_path.name.split("__")[1].replace(".joblib", "")

    print(
        f"Performing LOIO for feature type {feature_type} with parameters C: {model.C}, l1_ratio: {model.l1_ratio}"
    )

    # iterate through image paths
    for image_path in labeled_data["Metadata_DNA"].unique():
        # get training and testing cells from image path
        # every cell from the image path is for testing, the rest are for training
        train_cells = labeled_data.loc[labeled_data["Metadata_DNA"] != image_path]
        test_cells = labeled_data.loc[labeled_data["Metadata_DNA"] == image_path]

        # get X, y from training and testing cells
        X_train, y_train = get_X_y_data(train_cells, feature_type)
        X_test, y_test = get_X_y_data(test_cells, feature_type)

        # capture convergence warning from sklearn
        # this warning does not affect the model but takes up lots of space in the output
        with parallel_backend("multiprocessing"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=ConvergenceWarning, module="sklearn"
                )

                # fit a logisitc regression model on the training X, y
                LOIO_model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=100,
                    n_jobs=-1,
                    random_state=0,
                    C=model.C,
                    l1_ratio=model.l1_ratio,
                ).fit(X_train, y_train)

        # create metadata dataframe for test cells with model parameters
        metadata_dataframe = pd.concat(
            [
                test_cells["Cell_UUID"],
                test_cells["Metadata_DNA"],
                test_cells["Mitocheck_Phenotypic_Class"],
            ],
            axis=1,
        ).reset_index(drop=True)
        metadata_dataframe["Model_Feature_Type"] = feature_type
        metadata_dataframe["Model_C"] = model.C
        metadata_dataframe["Model_l1_ratio"] = model.l1_ratio

        # predict probabilities for test cells and make these probabilities into a dataframe
        probas = LOIO_model.predict_proba(X_test)
        probas_dataframe = pd.DataFrame(probas, columns=model.classes_)

        # combine metadata and probabilities dataframes for test cells to create wide data
        test_cells_wide_data = pd.concat([metadata_dataframe, probas_dataframe], axis=1)
        
        # add tidy long data to compiled data
        compiled_LOIO_wide_data.append(test_cells_wide_data)
        
        # DELETE THE REST OF THE LINES IN THIS NOTEBOOK BEFORE FINAL RUN
        score = LOIO_model.score(X_test, y_test)
        print(
            f"Leaving out image: {image_path}; number of cells: {test_cells.shape[0]}, score: {score}"
        )

        count +=1
        if count%5==0:
            break


# In[5]:


# compile list of wide data into one dataframe
compiled_LOIO_wide_data = pd.concat(compiled_LOIO_wide_data).reset_index(drop=True)

# convert wide data to tidy long data and sort by Cell_UUID, Model_Feature_Type, and Model_Phenotypic_Class for pretty formatting
compiled_LOIO_tidy_long_data = (
    pd.melt(
        compiled_LOIO_wide_data,
        id_vars=metadata_dataframe.columns,
        value_vars=probas_dataframe.columns,
        var_name="Model_Phenotypic_Class",
        value_name="Predicted_Probability",
    )
    .sort_values(["Model_Feature_Type", "Cell_UUID", "Model_Phenotypic_Class"])
    .reset_index(drop=True)
)

# specify results directory
LOIO_probas_dir = pathlib.Path("evaluations/LOIO_probas/")
LOIO_probas_dir.mkdir(parents=True, exist_ok=True)

# define save path
compiled_LOIO_save_path = pathlib.Path(f"{LOIO_probas_dir}/compiled_LOIO_probabilites.tsv")

# save data as tsv
compiled_LOIO_tidy_long_data.to_csv(compiled_LOIO_save_path, sep="\t")

compiled_LOIO_tidy_long_data


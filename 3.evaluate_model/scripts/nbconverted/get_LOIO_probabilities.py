#!/usr/bin/env python
# coding: utf-8

# ### Load Libraries
# 

# In[1]:


import pathlib
import warnings
import sys
import itertools

import numpy as np
import pandas as pd
from joblib import load

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.utils import parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score

sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_X_y_data
from evaluate_utils import get_SCM_model_data


# ### Set Load Path
# 

# In[2]:


# load labeled data
labeled_data_dir_path = pathlib.Path("../0.download_data/data/")


# ### See number of cells for LOIO evaluation

# In[3]:


# load labeled data
labeled_data_path = pathlib.Path(f"{labeled_data_dir_path}/labeled_data__ic.csv.gz")
labeled_data = get_features_data(labeled_data_path)

# see number of images to evaluate on
num_images = labeled_data["Metadata_DNA"].unique().shape[0]
print(f"There are {num_images} images to perform LOIO evaluation on per model.")


# ### Get LOIO probabilities (multi class models)

# In[4]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/multi_class_models")

# Which models to perform LOIO
all_model_paths = [model for model in sorted(models_dir.iterdir())]
print(f"There are {len(all_model_paths)} models to run LOIO\n\nThey include:")
all_model_paths


# In[5]:


# use a list to keep track of LOIO probabilities in tidy long format for each model combination
compiled_LOIO_wide_data = []

# iterate through each model (final model, shuffled baseline model, etc)
# sorted so final models are shown before shuffled_baseline
for model_path in all_model_paths:
    model = load(model_path)
    # determine model/feature type/balance/dataset type from model file name
    model_components = model_path.name.split("__")
    model_type = model_components[0]
    feature_type = model_components[1]
    balance_type = model_components[2]
    # version of dataset used to train model (ic, no_ic)
    dataset_type = model_components[3].replace(".joblib", "")

    print(
        f"Performing LOIO for model with types {model_type}, {balance_type}, {feature_type}, {dataset_type}"
    )
    
    # load labeled data
    labeled_data_path = pathlib.Path(f"{labeled_data_dir_path}/labeled_data__{dataset_type}.csv.gz")
    labeled_data = get_features_data(labeled_data_path)

    # iterate through image paths
    for image_path in labeled_data["Metadata_DNA"].unique():
        print(f"Training on everything but: {image_path}")
        # get training and testing cells from image path
        # every cell from the image path is for testing, the rest are for training
        train_cells = labeled_data.loc[labeled_data["Metadata_DNA"] != image_path]
        test_cells = labeled_data.loc[labeled_data["Metadata_DNA"] == image_path]

        # get X, y from training and testing cells
        X_train, y_train = get_X_y_data(train_cells, feature_type)
        
        # shuffle columns of X (features) dataframe independently to create shuffled baseline
        if model_type == "shuffled_baseline":
            for column in X_train.T:
                np.random.shuffle(column)
                
        X_test, y_test = get_X_y_data(test_cells, feature_type)

        # capture convergence warning from sklearn
        # this warning does not affect the model but takes up lots of space in the output
        # this warning must be caught with parallel_backend because the logistic regression model uses parallel_backend
        # (n_jobs=-1 means use all processors)
        with parallel_backend("multiprocessing"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=ConvergenceWarning, module="sklearn"
                )

                # fit a logisitc regression model on the training X, y
                # Use the optimal model parameters as identified previously.
                # Note that we tried performing a full grid search once again,
                # but this did not impact performance (data not shown)
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
        metadata_dataframe["Model_type"] = model_type
        metadata_dataframe["Dataset_type"] = dataset_type
        metadata_dataframe["Balance_type"] = balance_type

        # predict probabilities for test cells and make these probabilities into a dataframe
        print(f"Evaluating: {image_path}")
        probas = LOIO_model.predict_proba(X_test)
        probas_dataframe = pd.DataFrame(probas, columns=model.classes_)

        # combine metadata and probabilities dataframes for test cells to create wide data
        test_cells_wide_data = pd.concat([metadata_dataframe, probas_dataframe], axis=1)

        # add tidy long data to compiled data
        compiled_LOIO_wide_data.append(test_cells_wide_data)


# ### Format and save LOIO probabilities (multi class models)

# In[6]:


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
compiled_LOIO_save_path = pathlib.Path(
    f"{LOIO_probas_dir}/compiled_LOIO_probabilites_withshuffled_withnoic.tsv"
)

# save data as tsv
compiled_LOIO_tidy_long_data.to_csv(compiled_LOIO_save_path, sep="\t")

# preview tidy long data
compiled_LOIO_tidy_long_data


# ### Get LOIO probabilities (single class models)
# 

# In[7]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/single_class_models")

# use a list to keep track of LOIO probabilities in tidy long format for each model combination
compiled_LOIO_wide_data = []

# define combinations to test over
model_types = [
    "final"
]  # only perform LOIO with hyper params from final models so skip shuffled_baseline models
feature_types = ["CP", "DP", "CP_and_DP"]
phenotypic_classes = labeled_data["Mitocheck_Phenotypic_Class"].unique()

# iterate through each combination of feature_types, evaluation_types, phenotypic_classes
for model_type, feature_type, phenotypic_class in itertools.product(
    model_types, feature_types, phenotypic_classes
):
    single_class_model_path = pathlib.Path(
        f"{models_dir}/{phenotypic_class}_models/{model_type}__{feature_type}.joblib"
    )

    # load the model
    model = load(single_class_model_path)

    print(
        f"Performing LOIO on {phenotypic_class} model for feature type {feature_type} with parameters C: {model.C}, l1_ratio: {model.l1_ratio}"
    )

    # iterate through image paths
    for image_path in labeled_data["Metadata_DNA"].unique():
        # get training and testing cells from image path
        # every cell from the image path is for testing, the rest are for training
        train_cells = labeled_data.loc[labeled_data["Metadata_DNA"] != image_path]
        test_cells = labeled_data.loc[labeled_data["Metadata_DNA"] == image_path]

        # rename negative label and downsample over represented classes
        train_cells = get_SCM_model_data(train_cells, phenotypic_class, "train")
        test_cells = get_SCM_model_data(test_cells, phenotypic_class, "test")

        # get X, y from training and testing cells
        X_train, y_train = get_X_y_data(train_cells, feature_type)
        X_test, y_test = get_X_y_data(test_cells, feature_type)

        # capture convergence warning from sklearn
        # this warning does not affect the model but takes up lots of space in the output
        # this warning must be caught with parallel_backend because the logistic regression model uses parallel_backend
        # (n_jobs=-1 means use all processors)
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
        metadata_dataframe["Model_Phenotypic_Class"] = phenotypic_class

        # predict probabilities for test cells and make these probabilities into a dataframe
        probas = LOIO_model.predict_proba(X_test)
        probas_dataframe = pd.DataFrame(probas, columns=model.classes_)
        # make column names consistent for all single cell models (SCMs)
        # positive label corresponds to that SCM's phenotypic class, negative is all other labels
        probas_dataframe = probas_dataframe.rename(
            columns={
                phenotypic_class: "Positive_Label",
                f"Not {phenotypic_class}": "Negative_Label",
            }
        )

        # combine metadata and probabilities dataframes for test cells to create wide data
        test_cells_wide_data = pd.concat([metadata_dataframe, probas_dataframe], axis=1)

        # add tidy long data to compiled data
        compiled_LOIO_wide_data.append(test_cells_wide_data)


# ### Format and save LOIO probabilities (single class models)
# 

# In[8]:


# compile list of wide data into one dataframe
compiled_LOIO_wide_data = pd.concat(compiled_LOIO_wide_data).reset_index(drop=True)

# convert wide data to tidy long data and sort by Cell_UUID, Model_Feature_Type, and Model_Phenotypic_Class for pretty formatting
compiled_LOIO_tidy_long_data = (
    pd.melt(
        compiled_LOIO_wide_data,
        id_vars=metadata_dataframe.columns,
        value_vars=probas_dataframe.columns,
        var_name="Predicted_Label",
        value_name="Predicted_Probability",
    )
    .sort_values(["Model_Feature_Type", "Cell_UUID", "Model_Phenotypic_Class"])
    .reset_index(drop=True)
)

# specify results directory
LOIO_probas_dir = pathlib.Path("evaluations/LOIO_probas/")
LOIO_probas_dir.mkdir(parents=True, exist_ok=True)

# define save path
compiled_LOIO_save_path = pathlib.Path(
    f"{LOIO_probas_dir}/compiled_SCM_LOIO_probabilites.tsv"
)

# save data as tsv
compiled_LOIO_tidy_long_data.to_csv(compiled_LOIO_save_path, sep="\t")

# preview tidy long data
compiled_LOIO_tidy_long_data


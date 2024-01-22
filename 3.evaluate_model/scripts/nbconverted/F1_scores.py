#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries
# 

# In[1]:


import pathlib
from joblib import load
import itertools

import pandas as pd
from matplotlib import pyplot as plt

import sys

sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_dataset
from evaluate_utils import model_F1_score, get_SCM_model_data


# ## Set Data Load Paths
# 

# In[2]:


# load features data from indexes and features dataframe
data_split_dir_path = pathlib.Path("../1.split_data/indexes/")
features_dataframe_dir_path = pathlib.Path("../0.download_data/data/")


# ### Evaluate Each Model on Each Dataset (multi class models)
# 

# In[3]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/multi_class_models/")

# use a list to keep track of scores in tidy long format for each model and dataset combination
compiled_scores = []

# iterate through each model (final model, shuffled baseline model, etc)
# sorted so final models are shown before shuffled_baseline
for model_path in sorted(models_dir.iterdir()):
    model = load(model_path)
    # determine model/feature type/balance/dataset type from model file name
    model_components = model_path.name.split("__")
    model_type = model_components[0]
    feature_type = model_components[1]
    balance_type = model_components[2]
    # version of dataset used to train model (ic, no_ic)
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
            f"Evaluating model with types {model_type}, {balance_type}, {feature_type}, {dataset_type} on {label} dataset"
        )

        # load dataset (train, test, etc)
        data = get_dataset(features_dataframe, data_split_indexes, label)
        # find model F1 scores on dataset and create bar graph for them
        plt.figure(figsize=(15, 6))
        score, ax = model_F1_score(model, data, feature_type)
        ax.set_xlabel("Phenotypic Class")
        ax.set_ylabel("F1 Score")
        plt.show()

        # add score data to compiled dataframe in tidy format
        # transpose data and reset index to make dataframe resemble tidy long format
        score = score.T.reset_index()
        # change columns to their respective names
        score.columns = ["Phenotypic_Class", "F1_Score"]
        # add data split column to indicate which dataset scores are from (train, test, etc)
        score["data_split"] = label
        # add shuffled column to indicate if the model has been trained with shuffled data (random baseline) or not
        score["shuffled"] = "shuffled" in model_type
        # add feature type column to indicate which features model has been trained on/is using
        score["feature_type"] = feature_type
        # add balance type column
        score["balance_type"] = balance_type
        # add dataset type column
        score["dataset_type"] = dataset_type
        # add this score data to the tidy scores compiling list
        compiled_scores.append(score)


# ### Save scores from each evaluation (multi class models)
# 

# In[4]:


# compile list of tidy data into one dataframe
compiled_scores = pd.concat(compiled_scores).reset_index(drop=True)

# specify results directory
f1_scores_dir = pathlib.Path("evaluations/F1_scores/")
f1_scores_dir.mkdir(parents=True, exist_ok=True)

# define save path
compiled_scores_save_path = pathlib.Path(f"{f1_scores_dir}/compiled_F1_scores.tsv")

# save data as tsv
compiled_scores.to_csv(compiled_scores_save_path, sep="\t")

# preview tidy data
print(compiled_scores.shape)
compiled_scores.head()


# ### Evaluate Each Model on Each Dataset (single class models)
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
models_dir = pathlib.Path("../2.train_model/models/single_class_models/")

# use a list to keep track of scores in tidy long format for each model and dataset combination
compiled_scores = []

# define combinations to test over
model_types = ["final", "shuffled_baseline"]
feature_types = ["CP", "DP", "CP_and_DP"]
evaluation_types = ["train", "test"]
phenotypic_classes = features_dataframe["Mitocheck_Phenotypic_Class"].unique()

# iterate through each combination of feature_types, evaluation_types, phenotypic_classes
for model_type, feature_type, evaluation_type in itertools.product(
    model_types, feature_types, evaluation_types
):
    # create a figure that has 3x5 subplots for 15 phenotypic classes
    fig, axs = plt.subplots(3, 5)
    fig.set_size_inches(30, 18)
    # variables to keep track of figure subplot coordinates
    ax_x = 0
    ax_y = 0

    for phenotypic_class in phenotypic_classes:
        # load single class model for this combination of model type, feature type, and phenotypic class
        single_class_model_path = pathlib.Path(
            f"{models_dir}/{phenotypic_class}_models/{model_type}__{feature_type}.joblib"
        )
        single_class_model = load(single_class_model_path)

        # load dataset (train, test, etc)
        single_cell_data = get_dataset(
            features_dataframe, data_split_indexes, evaluation_type
        )
        # rename negative labels and downsample negative labels if we are evaluating on training data
        single_cell_data = get_SCM_model_data(
            single_cell_data, phenotypic_class, evaluation_type
        )

        # find model F1 scores on evaluation dataset
        score, _ = model_F1_score(
            single_class_model, single_cell_data, feature_type, axs[ax_x, ax_y]
        )

        # add score data to compiled dataframe in tidy format
        # transpose data and reset index to make dataframe resemble tidy long format
        score = score.T.reset_index()
        # change columns to their respective names
        score.columns = ["Phenotypic_Class", "F1_Score"]
        # add data split column to indicate which dataset scores are from (train, test, etc)
        score["data_split"] = evaluation_type
        # add shuffled column to indicate if the model has been trained with shuffled data (random baseline) or not
        score["shuffled"] = "shuffled" in model_type
        # add feature type column to indicate which features model has been trained on/is using
        score["feature_type"] = feature_type

        # add this score data to the tidy scores compiling list
        compiled_scores.append(score)

        # increase row coordinate counter (this marks which subplot to plot on in vertical direction)
        ax_x += 1
        # if row coordinate counter is at maximum (3 rows of subplots)
        if ax_x == 3:
            # set row coordinate counter to 0
            ax_x = 0
            # increase column coordinate counter (this marks which subplot to plot on in horizontal direction)
            ax_y += 1

    # add title and axes labels to figure
    fig.suptitle(
        f"Single Class Model F1 Scores for Combination: {model_type}, {feature_type}, {evaluation_type}"
    )
    fig.supxlabel("Phenotypic Class")
    fig.supylabel("F1 Score")
    plt.plot()


# ### Save scores from each evaluation (single class models)
# 

# In[7]:


# compile list of tidy data into one dataframe
compiled_scores = pd.concat(compiled_scores).reset_index(drop=True)

# specify results directory
f1_scores_dir = pathlib.Path("evaluations/F1_scores/")
f1_scores_dir.mkdir(parents=True, exist_ok=True)

# define save path
compiled_scores_save_path = pathlib.Path(f"{f1_scores_dir}/compiled_SCM_F1_scores.tsv")

# save data as tsv
compiled_scores.to_csv(compiled_scores_save_path, sep="\t")

# preview tidy data
print(compiled_scores.shape)
compiled_scores.head()


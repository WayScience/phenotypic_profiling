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
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

sys.path.append("../utils")
from split_utils import get_features_data


# ### Load Labeled Data
# 

# In[2]:


# load labeled data
labeled_data_path = pathlib.Path("../0.download_data/data/labeled_data.csv.gz")
labeled_data = get_features_data(labeled_data_path)
labeled_data


# ### Creat esingle class model interpretations
# 

# In[3]:


# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/single_class_models/")

# use a list to keep track of scores in tidy long format for each model and dataset combination
compiled_coefficients = []

# define combinations to test over
model_types = ["final", "shuffled_baseline"]
feature_types = ["CP", "DP", "CP_and_DP"]
phenotypic_classes = labeled_data["Mitocheck_Phenotypic_Class"].unique()

# iterate through each combination of feature_types, evaluation_types, phenotypic_classes
for model_type, feature_type in itertools.product(
    model_types, feature_types
):
    # create figures with 3x5 subplots for 15 phenotypic classes
    # heatmap figure/axs
    fig_hm, axs_hm = plt.subplots(3, 5)
    fig_hm.set_size_inches(15, 5)

    # variables to keep track of figure subplot coordinates
    ax_x = 0
    ax_y = 0

    for phenotypic_class in phenotypic_classes:
        # load single class model for this combination of model type, feature type, and phenotypic class
        single_class_model_path = pathlib.Path(
            f"{models_dir}/{phenotypic_class}_models/{model_type}__{feature_type}.joblib"
        )
        model = load(single_class_model_path)

        # get model coefficients and reshape them into a more useable format
        coefs = model.coef_
        coefs = pd.DataFrame(coefs).T

        # create tidy dataframe to keep track of coefficients in tidy long format
        tidy_data = coefs.copy(deep=True)
        tidy_data.columns = ["Coefficent_Value"]
        tidy_data["Phenotypic_Class"] = phenotypic_class
        tidy_data["shuffled"] = "shuffled" in model_type
        tidy_data["feature_type"] = feature_type

        # add feature names to coefficients dataframe
        # feature names depends on feature type
        all_cols = labeled_data.columns.tolist()
        # get DP,CP, or both features from all columns depending on desired dataset
        feature_cols = []
        if feature_type == "CP":
            feature_cols = [col for col in all_cols if "CP__" in col]
        elif feature_type == "DP":
            feature_cols = [col for col in all_cols if "DP__" in col]
        elif feature_type == "CP_and_DP":
            feature_cols = [col for col in all_cols if "P__" in col]
        tidy_data["Feature_Name"] = feature_cols

        # add tidy data to the compilation list
        compiled_coefficients.append(tidy_data)

        # add heatmap to figure
        sns.heatmap(data=coefs.T, ax=axs_hm[ax_x, ax_y])
        axs_hm[ax_x, ax_y].set_yticks([])
        axs_hm[ax_x, ax_y].set_ylabel(phenotypic_class)

        # increase row coordinate counter (this marks which subplot to plot on in vertical direction)
        ax_x += 1
        # if row coordinate counter is at maximum (3 rows of subplots)
        if ax_x == 3:
            # set row coordinate counter to 0
            ax_x = 0
            # increase column coordinate counter (this marks which subplot to plot on in horizontal direction)
            ax_y += 1

    # add title and axes labels to figure
    fig_hm.suptitle(
        f"Heatmaps for Combination: {model_type}, {feature_type}"
    )
    plt.plot()


# ### Save single class model coefficients

# In[4]:


# compile list of tidy data into one dataframe
compiled_coefficients = pd.concat(compiled_coefficients).reset_index(drop=True)

# specify results directory
coefficients_dir = pathlib.Path("coefficients/")
coefficients_dir.mkdir(parents=True, exist_ok=True)

# define save path
compiled_coefficients_save_path = pathlib.Path(f"{coefficients_dir}/compiled_SCM_coefficients.tsv")

# save data as tsv
compiled_coefficients.to_csv(compiled_coefficients_save_path, sep="\t")

# preview tidy data
compiled_coefficients


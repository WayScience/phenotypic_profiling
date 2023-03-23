#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries
# 

# In[1]:


import pandas as pd
import numpy as np
import pathlib

from joblib import load

import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.append("../utils")
from split_utils import get_features_data


# ### Load Labeled Data
# 

# In[2]:


# load labeled data
labeled_data_path = pathlib.Path("../0.download_data/data/labeled_data.csv.gz")
labeled_data = get_features_data(labeled_data_path)
labeled_data


# ### Save Coefficients and Create Interpretation Figures
# 

# In[3]:


# directory to save models to
coefs_save_directory = pathlib.Path(f"coefficients/")
coefs_save_directory.mkdir(parents=True, exist_ok=True)

# directory to load the models from
models_dir = pathlib.Path("../2.train_model/models/")

# use a list to keep track of scores in tidy long format for each model and dataset combination
compiled_class_PR_curves = []

# iterate through each model (final model, shuffled baseline model, etc)
# sorted so final models are shown before shuffled_baseline
for model_path in sorted(models_dir.iterdir()):
    # load model
    model = load(model_path)
    # determine model/feature type from model file name
    model_type = model_path.name.split("__")[0]
    feature_type = model_path.name.split("__")[1].replace(".joblib", "")

    print(f"Interpreting model: {model_type} \nTrained with features: {feature_type}")

    # get model coefficients and reshape them into a more useable format
    coefs = model.coef_
    coefs = pd.DataFrame(coefs).T
    coefs.columns = model.classes_
    print(f"Shape of model coefficients{coefs.shape}")

    # restructure/rename dataframe to tidy long format (see preview below)
    tidy_data = coefs.stack()
    tidy_data = pd.DataFrame(tidy_data).reset_index(level=[0, 1])
    tidy_data.columns = ["Feature_Name", "Phenotypic_Class", "ML_coefficient"]

    # add feature names to coefficients dataframe
    # feature names depends on feature type
    all_cols = labeled_data.columns.tolist()
    # get DP,CP, or both features from all columns depending on desired dataset
    if feature_type == "CP":
        feature_cols = [col for col in all_cols if "CP__" in col]
    elif feature_type == "DP":
        feature_cols = [col for col in all_cols if "DP__" in col]
    elif feature_type == "CP_and_DP":
        feature_cols = [col for col in all_cols if "P__" in col]
    # add repeats to feature cols equal to the number of classes
    feature_names = [col for col in feature_cols for i in range(len(model.classes_))]
    tidy_data["Feature_Name"] = feature_names

    # save tidy coefficients dataframe
    coefs_save_path = pathlib.Path(
        f"{coefs_save_directory}/{model_type}__{feature_type}__coefficients.tsv"
    )
    tidy_data.to_csv(coefs_save_path, sep="\t")

    # display heatmap of average coefs
    plt.figure(figsize=(20, 10))
    plt.title("Heatmap of Coefficients Matrix")
    ax = sns.heatmap(data=coefs.T)
    plt.show()

    # display clustered heatmap of coefficients
    ax = sns.clustermap(
        data=coefs.T, figsize=(20, 10), row_cluster=True, col_cluster=True
    )
    ax = ax.fig.suptitle("Clustered Heatmap of Coefficients Matrix")
    plt.show()

    # display density plot for coefficient values of each class
    sns.set(rc={"figure.figsize": (20, 8)})
    plt.xlim(-0.02, 0.1)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Density")
    plt.title("Density of Coefficient Values Per Phenotypic Class")
    ax = sns.kdeplot(data=coefs)
    plt.show()

    # display average coefficient value vs phenotypic class bar chart
    pheno_class_ordered = coefs.reindex(
        coefs.mean().sort_values(ascending=False).index, axis=1
    )
    sns.set(rc={"figure.figsize": (20, 8)})
    plt.xlabel("Phenotypic Class")
    plt.ylabel("Average Coefficient Value")
    plt.title("Coefficient vs Phenotypic Class")
    plt.xticks(rotation=90)
    ax = sns.barplot(data=pheno_class_ordered)
    plt.show()


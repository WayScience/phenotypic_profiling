#!/usr/bin/env python
# coding: utf-8

# ## Calculate Silhouette Scores
# 
# - per phenotype
# - per feature space

# In[1]:


import pathlib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

import sys

sys.path.append("../utils")
from split_utils import get_features_data


# In[2]:


np.random.seed(1234)

# For consistent Silhouette input space dimensionality
n_pca_components = 50


# In[3]:


eval_path = pathlib.Path("evaluations")

output_silhouette_results = pathlib.Path(
    eval_path, "silhouette_score_results.tsv"
)
output_silhouette_samples_results = pathlib.Path(
    eval_path, "silhouette_score_results_per_sample.tsv"
)


# In[4]:


# load x (features) and y (labels) dataframes
labeled_data_path = pathlib.Path("../0.download_data/data/labeled_data__ic.csv.gz")
labeled_data = get_features_data(labeled_data_path).reset_index(drop=True)

print(labeled_data.shape)
labeled_data.head(3)


# In[5]:


metadata_columns = [
    "Mitocheck_Phenotypic_Class",
    "Cell_UUID",
    "Location_Center_X",
    "Location_Center_Y",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Frame",
    "Metadata_Site",
    "Metadata_Plate_Map_Name",
    "Metadata_DNA",
    "Metadata_Gene",
    "Metadata_Gene_Replicate",
    "Metadata_Object_Outline",
]

feature_groups = [
    "CP",
    "DP",
    "CP_and_DP"
]

all_phenotypes = labeled_data.Mitocheck_Phenotypic_Class.unique().tolist()
all_phenotypes


# In[6]:


silhouette_results_df = []

for feature_group in feature_groups:
    # Compile dataset
    if feature_group == "CP_and_DP":
        input_data_to_silhouette = labeled_data.drop(metadata_columns, axis=1)
    else:
        input_data_to_silhouette = labeled_data.loc[:, labeled_data.columns.str.startswith(feature_group)]

    # Apply PCA to make sure consistent dimensions applied in calculation
    pca = PCA(n_components=n_pca_components)
    input_data_to_silhouette = pca.fit_transform(input_data_to_silhouette)
    
    for phenotype in all_phenotypes:

        focused_label = [x if x == phenotype else "other" for x in labeled_data.Mitocheck_Phenotypic_Class]

        # Calculate per phenotype average silhouette score
        silhouette_results = silhouette_score(
            input_data_to_silhouette,
            focused_label
        )

        silhouette_results_df.append([
            feature_group, phenotype, silhouette_results
        ])


# In[7]:


silhouette_results_df = pd.DataFrame(silhouette_results_df)

silhouette_results_df.columns = [
    "feature_space",
    "phenotype",
    "silhouette_score"
]

silhouette_results_df.to_csv(output_silhouette_results, sep="\t", index=False)

print(silhouette_results_df.shape)
silhouette_results_df.head()


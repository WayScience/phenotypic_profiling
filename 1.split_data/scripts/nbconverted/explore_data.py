#!/usr/bin/env python
# coding: utf-8

# ## Explore data
# 
# - Calculate pairwise correlations between single-cells

# In[1]:


import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_X_y_data


# In[2]:


def create_tidy_corr_matrix(data_array, labels):
    # Calculate the pairwise correlation matrix
    correlation_matrix = np.corrcoef(data_array, rowvar=True)
    
    # Convert the correlation matrix to a DataFrame for easier manipulation
    df_corr = pd.DataFrame(correlation_matrix)
    
    # Melt the correlation matrix
    melted_corr = df_corr.stack().reset_index()
    melted_corr.columns = ["Row_ID", "Pairwise_Row_ID", "Correlation"]
    
    # Filter out the lower triangle including diagonal
    melted_corr = melted_corr[melted_corr["Row_ID"] < melted_corr["Pairwise_Row_ID"]]
    
    # Add labels for the rows and columns
    melted_corr["Row_Label"] = melted_corr["Row_ID"].apply(lambda x: labels[x])
    melted_corr["Pairwise_Row_Label"] = melted_corr["Pairwise_Row_ID"].apply(lambda x: labels[x])
    
    # Reorder columns
    melted_corr = melted_corr[["Row_ID", "Pairwise_Row_ID", "Correlation", "Row_Label", "Pairwise_Row_Label"]]
    
    return melted_corr


# In[3]:


# Set constants
feature_spaces = ["CP", "DP", "CP_and_DP"]

output_dir = "data"
output_basename = pathlib.Path(output_dir, "pairwise_correlations")


# In[4]:


# load x (features) and y (labels) dataframes
labeled_data_path = pathlib.Path("../0.download_data/data/labeled_data__ic.csv.gz")
labeled_data = get_features_data(labeled_data_path)

print(labeled_data.shape)
labeled_data.head(3)


# In[5]:


for feature_space in feature_spaces:
    # Get specific feature sets
    cp_feature_df, cp_label_df = get_X_y_data(labeled_data, dataset=feature_space)

    # Calculate pairwise correlations between nuclei
    cp_tidy_corr_df = create_tidy_corr_matrix(cp_feature_df, cp_label_df)

    # Output to file
    output_file = f"{output_basename}_{feature_space}.tsv.gz"
    cp_tidy_corr_df.to_csv(output_file, sep="\t", index=False)


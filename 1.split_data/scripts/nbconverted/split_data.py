#!/usr/bin/env python
# coding: utf-8

# # Split feature data
# ## Create tsv file with indexes for held out data, training data, and testing data
# ### Import libraries

# In[1]:


import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../utils")
from split_utils import get_features_data


# ## Specify datasets, save path

# In[2]:


# datasets to split
datasets = ["ic", "no_ic"]

# make results dir for saving
results_dir = pathlib.Path("indexes/")
results_dir.mkdir(parents=True, exist_ok=True)


# ## Load, split, save data indexes

# In[3]:


for dataset in datasets:
    print(f"Splitting data for {dataset} dataset")
    
    # load x (features) and y (labels) dataframes
    labeled_data_path = pathlib.Path(f"../0.download_data/data/labeled_data__{dataset}.csv.gz")
    labeled_data = get_features_data(labeled_data_path)
    print(f"Dataset shape: {labeled_data.shape}")
    
    # ratio of data to be used for testing (ex 0.15 = 15%)
    test_ratio = 0.15

    # get indexes of training and testing data
    training_data, testing_data = train_test_split(
        labeled_data,
        test_size=test_ratio,
        stratify=labeled_data[["Mitocheck_Phenotypic_Class"]],
        random_state=1,
    )
    train_indexes = training_data.index.to_numpy()
    test_indexes = testing_data.index.to_numpy()

    print(f"Training data has shape: {training_data.shape}")
    print(f"Testing data has shape: {testing_data.shape}")
    
    # create pandas dataframe with all indexes and their respective labels, stratified by phenotypic class
    index_data = []
    for index in train_indexes:
        index_data.append({"labeled_data_index": index, "label": "train"})
    for index in test_indexes:
        index_data.append({"labeled_data_index": index, "label": "test"})

    # make index data a dataframe and sort it by labeled data index
    index_data = (
        pd.DataFrame(index_data)
        .sort_values(["labeled_data_index"])
    )
    
    # save indexes as tsv file
    index_data.to_csv(f"{results_dir}/data_split_indexes__{dataset}.tsv", sep="\t")
    print(f"Saved index data\n")


# ## Preview index data

# In[4]:


index_data


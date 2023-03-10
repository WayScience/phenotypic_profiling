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


# ### Load data and set holdout/test parameters

# In[2]:


# load x (features) and y (labels) dataframes
labeled_data_path = pathlib.Path("../0.download_data/data/labeled_data.csv.gz")
labeled_data = get_features_data(labeled_data_path)
print(labeled_data.shape)


# In[3]:


# ratio of data to be used for testing (ex 0.15 = 15%)
test_ratio = 0.15

# get indexes of training and testing data
training_data, testing_data = train_test_split(labeled_data, test_size=test_ratio, random_state=0)
train_indexes = training_data.index.to_numpy()
test_indexes = testing_data.index.to_numpy()

print(f"Training data has shape: {training_data.shape}")
print(f"Testing data has shape: {testing_data.shape}")


# In[4]:


# create pandas dataframe with all indexes and their respective labels
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

index_data


# ### Save indexes

# In[5]:


# make results dir for saving
results_dir = pathlib.Path("indexes/")
results_dir.mkdir(parents=True, exist_ok=True)
# save indexes as tsv file
index_data.to_csv(f"{results_dir}/data_split_indexes.tsv", sep="\t")


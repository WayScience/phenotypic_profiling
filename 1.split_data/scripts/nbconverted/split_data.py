#!/usr/bin/env python
# coding: utf-8

# # Split feature data
# ## Create tsv file with indexes for held out data, training data, and testing data
# ### Import libraries

# In[1]:


import pathlib
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

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


# ratio of data to be reserved for testing (ex 0.15 = 15%)
test_ratio = 0.15

# test_data is pandas dataframe with test split, stratified by Mitocheck_Phenotypic_Class
testing_data = labeled_data.groupby("Mitocheck_Phenotypic_Class", group_keys=False).apply(
    lambda x: x.sample(frac=test_ratio)
)
test_indexes = testing_data.index

# training data is labeled data - test data
training_data = labeled_data.drop(pd.Index(data=test_indexes))
train_indexes = np.array(training_data.index)

print(f"Training data has shape: {training_data.shape}")
print(f"Testing data has shape: {testing_data.shape}")


# In[4]:


# create pandas dataframe with all indexes and their respective labels
index_data = []
for index in test_indexes:
    index_data.append({"labeled_data_index": index, "label": "test"})
for index in train_indexes:
    index_data.append({"labeled_data_index": index, "label": "train"})
index_data = pd.DataFrame(index_data)
# put indexes into sorted order
index_data = index_data.sort_values(["labeled_data_index"])

index_data


# ### Save indexes

# In[5]:


# make results dir for saving
results_dir = pathlib.Path("indexes/")
results_dir.mkdir(parents=True, exist_ok=True)
# save indexes as tsv file
index_data.to_csv(f"{results_dir}/data_split_indexes.tsv", sep="\t")


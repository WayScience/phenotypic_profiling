#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pathlib
import pandas as pd


# ## Specify mitocheck_data types and paths
# 
# We use multiple types of data for model training/testing.

# In[2]:


labeled_data_hash__ic = "20369033f579dca1334cb2c58a1c6d532322f93e"
labeled_data_url__ic = f"https://raw.github.com/WayScience/mitocheck_data/{labeled_data_hash__ic}/3.normalize_data/normalized_data/training_data.csv.gz"

# ADD PATH TO DATA FROM GITHUB
# labeled_data_hash = ""
labeled_data_url__no_ic = f"/home/roshankern/Desktop/Github/mitocheck_data/3.normalize_data/normalized_data__no_ic/training_data.csv.gz"

labeled_data_paths = {
    "ic": labeled_data_url__ic,
    "no_ic": labeled_data_url__no_ic
}


# ### Load and save labeled data

# In[3]:


# make parent directory for labeled data
labeled_data_save_dir = pathlib.Path("data/")
labeled_data_save_dir.mkdir(parents=True, exist_ok=True)

for data_type in labeled_data_paths:
    # Load data
    labeled_data_load_path = labeled_data_paths[data_type]
    labeled_data = pd.read_csv(labeled_data_load_path, compression="gzip", index_col=0)
    
    # save data
    labeled_data_save_path = pathlib.Path(f"{labeled_data_save_dir}/labeled_data__{data_type}.csv.gz")
    labeled_data.to_csv(labeled_data_save_path, compression="gzip")


# ## Preview data

# In[4]:


labeled_data


# ### View counts of each class in the labeled dataset

# In[5]:


labeled_data["Mitocheck_Phenotypic_Class"].value_counts()


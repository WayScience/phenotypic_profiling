#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pandas as pd
import pathlib

import sys
sys.path.append("../utils")
from download_utils import combine_datasets


# ### Specify version of mitocheck_data to download from

# In[2]:


hash_2006 = "19bfa5b0959d6b7536f83e7bb85745ba3edf7ff9"
file_url_2006 = f"https://raw.github.com/WayScience/mitocheck_data/{hash_2006}/3.normalize_data/normalized_data/training_data.csv.gz"

hash_2015 = "3ebd0ca7c288f608e9b23987a8ddbabd5476bd8f"
file_url_2015 = f"https://raw.github.com/WayScience/mitocheck_data/{hash_2015}/3.normalize_data/normalized_data/training_data.csv.gz"


# ### Load/combine training data from github

# In[3]:


training_data_2006 = pd.read_csv(file_url_2006, compression="gzip", index_col=0)
# remove unnecessary mitocheck object ID as this ID is not present for data repeated in 2015 dataset
training_data_2006 = training_data_2006.drop(columns=["Mitocheck_Object_ID"])

training_data_2015 = pd.read_csv(file_url_2015, compression="gzip", index_col=0)


# In[4]:


training_data_2006


# In[5]:


training_data = combine_datasets(training_data_2006, training_data_2015)


# ### Preview dataset

# In[6]:


# remove undefinedCondensed class with very low representation
training_data = training_data[training_data["Mitocheck_Phenotypic_Class"] != "UndefinedCondensed"]
training_data = training_data.reset_index(drop=True)
training_data


# ### Save training data

# In[7]:


training_data_save_dir = pathlib.Path("data/")
training_data_save_dir.mkdir(parents=True, exist_ok=True)

training_data_save_path = pathlib.Path(f"{training_data_save_dir}/training_data.csv.gz")
training_data.to_csv(training_data_save_path, compression="gzip")


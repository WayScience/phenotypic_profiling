#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pandas as pd
import pathlib


# ### Specify version of mitocheck_data to download from

# In[2]:


hash = "19bfa5b0959d6b7536f83e7bb85745ba3edf7ff9"
file_url = f"https://raw.github.com/WayScience/mitocheck_data/{hash}/3.normalize_data/normalized_data/training_data.csv.gz"
print(file_url)


# ### Load training data from github

# In[3]:


training_data = pd.read_csv(file_url, compression="gzip", index_col=0)
training_data


# ### Save training data

# In[4]:


training_data_save_dir = pathlib.Path("data/")
training_data_save_dir.mkdir(parents=True, exist_ok=True)

training_data_save_path = pathlib.Path(f"{training_data_save_dir}/training_data.csv.gz")
training_data.to_csv(training_data_save_path, compression="gzip")


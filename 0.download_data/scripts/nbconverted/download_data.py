#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


import pathlib
import pandas as pd


# ### Specify version of mitocheck_data to download from

# In[2]:


labeled_data_hash = "e1f86cd007657f8247310b78df92891b22e51621"
labeled_data_url = f"https://raw.github.com/WayScience/mitocheck_data/{labeled_data_hash}/3.normalize_data/normalized_data/training_data.csv.gz"


# ### Load labeled data from github

# In[3]:


labeled_data = pd.read_csv(labeled_data_url, compression="gzip", index_col=0)
labeled_data


# ### View counts of each class in the labeled dataset

# In[4]:


labeled_data["Mitocheck_Phenotypic_Class"].value_counts()


# ### Save labeled data

# In[5]:


labeled_data_save_path = pathlib.Path("data/labeled_data.csv.gz")
labeled_data_save_path.parent.mkdir(parents=True, exist_ok=True)
labeled_data.to_csv(labeled_data_save_path, compression="gzip")


#!/usr/bin/env python
# coding: utf-8

# # Analyze all feature data
# 
# ### Import libraries

# In[1]:


import numpy as np
import pathlib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, rgb2hex
from pylab import cm
import seaborn as sns
import pandas as pd
import umap

from utils.analysisUtils import get_features_data, show_1D_umap, show_2D_umap, show_3D_umap 


# ### Initalize analysis

# In[2]:


# make random numpy operations consistent
np.random.seed(0)

# create results dir for saving results
results_dir = pathlib.Path("results/")
results_dir.mkdir(parents=True, exist_ok=True)


# ### Load dataframe

# In[3]:


# load features dataframe
features_dataframe_path = pathlib.Path("../1.format_data/data/training_data.csv.gz")
features_dataframe = get_features_data(features_dataframe_path)

# split metadata from features
metadata_dataframe = features_dataframe.iloc[:,:13]
features_dataframe = features_dataframe.iloc[:,13:]

features_dataframe


# ### Counts for all phenotypic classes

# In[4]:


metadata_dataframe["Mitocheck_Phenotypic_Class"].value_counts()


# ### Only keep certain phenoytpic classes for analysis

# In[5]:


classes_to_keep = [
    "Polylobed",
    "Binuclear",
    "Grape",
    "Prometaphase",
    "Interphase",
    "Artefact",
    "Apoptosis",
    "SmallIrregular",
    "MetaphaseAlignment",
    "Hole",
    "Metaphase",
    "Large",
    "Folded",
    "Elongated",
    "UndefinedCondensed",
]

features_dataframe = features_dataframe.loc[
    metadata_dataframe["Mitocheck_Phenotypic_Class"].isin(classes_to_keep)
]
metadata_dataframe = metadata_dataframe.loc[
    metadata_dataframe["Mitocheck_Phenotypic_Class"].isin(classes_to_keep)
]
features_dataframe.shape


# ### 1D UMAP

# In[6]:


phenotypic_classes = metadata_dataframe["Mitocheck_Phenotypic_Class"]
show_1D_umap(features_dataframe, phenotypic_classes, results_dir)


# ### 2D UMAP

# In[7]:


phenotypic_classes = metadata_dataframe["Mitocheck_Phenotypic_Class"]
show_2D_umap(features_dataframe, phenotypic_classes, results_dir)


# ### 3D UMAP

# In[8]:


phenotypic_classes = metadata_dataframe["Mitocheck_Phenotypic_Class"]
show_3D_umap(features_dataframe, phenotypic_classes, results_dir)


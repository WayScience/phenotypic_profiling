#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("../utils")
from validate_utils import create_classification_profiles


# ### Load Cell Health Profile Labels

# In[2]:


cell_health_hash = "30ea5de393eb9cfc10b575582aa9f0f857b44c59"
cell_health_labels_link = f"https://raw.github.com/broadinstitute/cell-health/{cell_health_hash}/1.generate-profiles/data/consensus/cell_health_median.tsv.gz"

cell_health_labels = pd.read_csv(cell_health_labels_link, compression="gzip", sep="\t")
cell_health_labels


# ### Create Classification Profiles

# In[3]:


cell_health_data_hash = "4ee08b16f4b5c5266309a295b0a1697f0b9540b8"
plate_classifications_dir_link = f"https://github.com/WayScience/cell-health-data/raw/{cell_health_data_hash}/4.classify-features/plate_classifications"
plate_names = ["SQ00014610", "SQ00014611", "SQ00014612", "SQ00014613", "SQ00014614", "SQ00014615", "SQ00014616", "SQ00014617", "SQ00014618"]

cell_line_plates = {
    "A549": ["SQ00014610", "SQ00014611", "SQ00014612"],
    "ES2": ["SQ00014613", "SQ00014614", "SQ00014615"],
    "HCC44": ["SQ00014616", "SQ00014617", "SQ00014618"],
}

classification_profiles = create_classification_profiles(plate_classifications_dir_link, cell_line_plates)
classification_profiles


# In[4]:


# combine cell health label profiles and classification profiles on perturbation and cell line
final_profile_dataframe = pd.merge(cell_health_labels, classification_profiles, on=["Metadata_pert_name", "Metadata_cell_line"])
final_profile_dataframe


# ### Find Correlation

# In[5]:


correlation_method = "pearson"

# combine cell health label profiles and classification profiles on perturbation and cell line
final_profile_dataframe = pd.merge(cell_health_labels, classification_profiles, on=["Metadata_pert_name", "Metadata_cell_line"])
# find correlation
corr = final_profile_dataframe.corr(method=correlation_method)
# convert correlation to diagram-friendly format
corr_graph = corr.iloc[70:, :70]


# ### Show Correlation with Clustermap

# In[6]:


sns.clustermap(corr_graph, 
            xticklabels=corr_graph.columns,
            yticklabels=corr_graph.index,
            cmap='RdBu_r',
            linewidth=0.5,
            figsize=(20,10))


# ### Save Correlation Data in Tidy Long Format

# In[7]:


corr_data_save_path = pathlib.Path(f"validations/cell_health_{correlation_method}_correlations.tsv")
corr_data_save_path.parents[0].mkdir(parents=True, exist_ok=True)

tidy_data = corr_graph.stack()
tidy_data = pd.DataFrame(tidy_data).reset_index(level=[0,1])
tidy_data.columns = ["Phenotypic_Class", "Cell_Health_Label", "Correlation"]
tidy_data.to_csv(corr_data_save_path, sep="\t")


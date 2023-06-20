#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries
# 

# In[1]:


import sys
import pathlib

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from ccc.coef import ccc
from scipy.spatial.distance import squareform

sys.path.append("../utils")
import validate_utils


# #### Set load/save paths

# In[2]:


# external path to be set
classification_profiles_save_dir = pathlib.Path(
    "/media/roshankern/63af2010-c376-459e-a56e-576b170133b6/data/cell-health-plate-classification-profiles"
)

MCM_classification_profiles_save_dir = pathlib.Path(
    f"{classification_profiles_save_dir}/multi_class_models/"
)
SCM_classification_profiles_save_dir = pathlib.Path(
    f"{classification_profiles_save_dir}/single_class_models/"
)

tidy_long_corrs_save_dir = pathlib.Path("validations")
tidy_long_corrs_save_dir.mkdir(exist_ok=True, parents=True)


# ### Load Cell Health Profile Labels
# 

# In[3]:


cell_health_hash = "30ea5de393eb9cfc10b575582aa9f0f857b44c59"
cell_health_labels_link = f"https://raw.github.com/broadinstitute/cell-health/{cell_health_hash}/1.generate-profiles/data/consensus/cell_health_median.tsv.gz"

cell_health_labels = pd.read_csv(cell_health_labels_link, compression="gzip", sep="\t")
cell_health_labels


# ### Derive classification profile and cell health label correlations (multi-class models)
# 

# In[4]:


print("Deriving multi-class model correlations...")
# list for compiling tidy long correlation data
compiled_tidy_long_corrs = []
for classification_profiles_path in MCM_classification_profiles_save_dir.iterdir():
    
    # get information about the current model
    model_type = classification_profiles_path.name.split("__")[0]
    feature_type = classification_profiles_path.name.split("__")[1]
    
    print(f"Deriving correlations for {model_type} model with {feature_type} features...")
    
    # load classification profiles
    classification_profiles = pd.read_csv(classification_profiles_path, sep="\t")
    
    # combine cell health label profiles and classification profiles on perturbation and cell line
    final_profile_dataframe = pd.merge(
        cell_health_labels,
        classification_profiles,
        on=["Metadata_pert_name", "Metadata_cell_line"],
    )
    
    # get tidy long correlations for this model's predictions
    model_tidy_long_corrs = validate_utils.get_tidy_long_corrs(final_profile_dataframe)
    
    # add model metadata
    model_tidy_long_corrs["model_type"] = model_type
    model_tidy_long_corrs["feature_type"] = feature_type
    
    # add correlations to compilation list
    compiled_tidy_long_corrs.append(model_tidy_long_corrs)
    

# compile and save tidy long data
compiled_tidy_long_corrs = pd.concat(compiled_tidy_long_corrs)
compiled_tidy_long_corrs.to_csv(f"{tidy_long_corrs_save_dir}/compiled_correlations__MCM.tsv", sep="\t")

# preview tidy data
compiled_tidy_long_corrs


# ### Derive classification profile and cell health label correlations (single-class models)

# In[5]:


print("Deriving single-class model correlations...")
# list for compiling tidy long correlation data
compiled_tidy_long_corrs = []
for phenotypic_class_path in SCM_classification_profiles_save_dir.iterdir():
    for classification_profiles_path in phenotypic_class_path.iterdir():
        
        # get information about the current model
        phenotypic_class = phenotypic_class_path.name
        model_type = classification_profiles_path.name.split("__")[0]
        feature_type = classification_profiles_path.name.split("__")[1]
        
        print(f"Deriving correlations for {model_type}, {phenotypic_class} model with {feature_type} features...")
        
        # load classification profiles
        classification_profiles = pd.read_csv(classification_profiles_path, sep="\t")
        
        # combine cell health label profiles and classification profiles on perturbation and cell line
        final_profile_dataframe = pd.merge(
            cell_health_labels,
            classification_profiles,
            on=["Metadata_pert_name", "Metadata_cell_line"],
        )
        
        # get tidy long correlations for this model's predictions
        model_tidy_long_corrs = validate_utils.get_tidy_long_corrs(final_profile_dataframe)
        
        # add model metadata
        model_tidy_long_corrs["model_type"] = model_type
        model_tidy_long_corrs["feature_type"] = feature_type
        
        # add correlations to compilation list
        compiled_tidy_long_corrs.append(model_tidy_long_corrs)
        
    
# compile and save tidy long data
compiled_tidy_long_corrs = pd.concat(compiled_tidy_long_corrs)
compiled_tidy_long_corrs.to_csv(f"{tidy_long_corrs_save_dir}/compiled_correlations__SCM.tsv", sep="\t")

# preview tidy data
compiled_tidy_long_corrs


#!/usr/bin/env python
# coding: utf-8

# ### Import libraries
# 

# In[1]:


import pathlib

import pandas as pd

sys.path.append("../utils")
import validate_utils


# ### Examine multi-class model correlations
# 

# In[2]:


tidy_long_corrs_save_dir = pathlib.Path("validations")
corrs_path = pathlib.Path(f"{tidy_long_corrs_save_dir}/compiled_correlations__MCM.tsv")

tidy_corr_data = pd.read_csv(corrs_path, sep="\t", index_col=0)
tidy_corr_data


# In[3]:


cell_line = "all"  # all, A549, ES2, HCC44
corr_type = "pearson"  # pearson, ccc
model_type = "final"  # final, shuffled baseline
feature_type = "CP_and_DP"  # CP, DP, CP_and_DP

validate_utils.get_corr_clustermap(
    tidy_corr_data, cell_line, corr_type, model_type, feature_type
)


# In[4]:


cell_line = "all"  # all, A549, ES2, HCC44
corr_type = "pearson"  # pearson, ccc
model_type = "shuffled_baseline"  # final, shuffled_baseline
feature_type = "CP_and_DP"  # CP, DP, CP_and_DP

validate_utils.get_corr_clustermap(
    tidy_corr_data, cell_line, corr_type, model_type, feature_type
)


# ### Examine single-class model correlations
# 

# In[5]:


corrs_path = pathlib.Path(f"{tidy_long_corrs_save_dir}/compiled_correlations__SCM.tsv")

tidy_corr_data = pd.read_csv(corrs_path, sep="\t", index_col=0)
tidy_corr_data


# In[6]:


cell_line = "all"  # all, A549, ES2, HCC44
corr_type = "pearson"  # pearson, ccc
model_type = "final"  # final, shuffled baseline
feature_type = "CP_and_DP"  # CP, DP, CP_and_DP

validate_utils.get_corr_clustermap(
    tidy_corr_data, cell_line, corr_type, model_type, feature_type
)


# In[7]:


cell_line = "all"  # all, A549, ES2, HCC44
corr_type = "pearson"  # pearson, ccc
model_type = "shuffled_baseline"  # final, shuffled_baseline
feature_type = "CP_and_DP"  # CP, DP, CP_and_DP

validate_utils.get_corr_clustermap(
    tidy_corr_data, cell_line, corr_type, model_type, feature_type
)


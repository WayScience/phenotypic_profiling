#!/usr/bin/env python
# coding: utf-8

# ## Process JUMP phenotypic profiles
# 
# We applied the AreaShape only class-balanced multiclass elastic net logistic regression model to all single-cell profiles in the JUMP dataset.
# 
# We then performed a series of KS tests to identify how different treatment distributions of all phenotype probabilities differed from controls.
# 
# See https://github.com/WayScience/JUMP-single-cell/tree/main/3.analyze_data#analyze-predicted-probabilities for complete details.
# 
# Here, we perform the following:
# 
# 1. Load in this data from the JUMP-single-cell repo
# 2. Summarize replicate KS test metrics (mean value) and align across cell types and time variables
# 3. Explore the top results per phenotype/treatment_type/model_type (Supplementary Table S1)
# 4. Convert it to wide format
# 
# This wide format represents a "phenotypic profile" which we can use similarly as an image-based morphology profile.

# In[1]:


import pathlib
from typing import List
import pandas as pd


# In[2]:


# Set file paths
# JUMP phenotype probabilities from AreaShape model
commit = "4225e427fd9da59159de69f53be65c31b4d4644a"

url = "https://github.com/WayScience/JUMP-single-cell/raw"
file = "3.analyze_data/class_balanced_well_log_reg_comparison_results/class_balanced_well_log_reg_areashape_model_comparisons.parquet"

jump_sc_pred_file = f"{url}/{commit}/{file}"

# Set constants
n_top_results_to_explore = 100


# In[3]:


# Set output files
output_dir = "jump_phenotype_profiles"

cell_type_time_comparison_file = pathlib.Path(output_dir, "jump_compare_cell_types_and_time_across_phenotypes.tsv.gz")
top_results_summary_file = pathlib.Path(output_dir, "jump_most_significant_phenotype_enrichment.tsv")
final_jump_phenotype_file = pathlib.Path(output_dir, "jump_phenotype_profiles.tsv.gz")
shuffled_jump_phenotype_file = pathlib.Path(output_dir, "jump_phenotype_profiles_shuffled.tsv.gz")


# ## Load and process data

# In[4]:


# Load KS test results and drop uninformative columns
jump_pred_df = (
    pd.read_parquet(jump_sc_pred_file)
    .drop(columns=["statistical_test", "comparison_metric"])
)

print(jump_pred_df.shape)
jump_pred_df.head()


# In[5]:


# Process data to match treatments and scores across cell types
jump_pred_compare_df = (
    jump_pred_df
    # Summarize replicate scores
    .groupby([
        "Cell_type",
        "Time",
        "treatment",
        "treatment_type",
        "Metadata_model_type",
        "phenotype"
    ])
    .agg({
        "comparison_metric_value": "mean",
        "p_value": "mean"
    })
    .reset_index()
    # Compare per treatment scores across cell types
    .pivot(
        index=[
            "treatment",
            "treatment_type",
            "Time",
            "phenotype",
            "Metadata_model_type"
        ],
        columns="Cell_type",
        values=[
            "comparison_metric_value",
            "p_value"
        ]
    )
    .reset_index()
)

# Clen up column names
jump_pred_compare_df.columns = jump_pred_compare_df.columns.map(lambda x: '_'.join(filter(None, x)))

# Output file
jump_pred_compare_df.to_csv(cell_type_time_comparison_file, sep="\t", index=False)

print(jump_pred_compare_df.shape)
jump_pred_compare_df.head()


# In[6]:


# Focus on the top results for downstream interpretation
jump_focused_top_results_df = (
    jump_pred_df
    .query("Metadata_model_type == 'final'")
    .groupby(["Metadata_model_type", "treatment_type", "Cell_type", "Time", "phenotype"])
    .apply(lambda x: x.nsmallest(n_top_results_to_explore, "p_value"))
    .reset_index(drop=True)
)

jump_focused_top_results_df.to_csv(top_results_summary_file, sep="\t", index=False)

print(jump_focused_top_results_df.shape)
jump_focused_top_results_df.head()


# ## Summarize data

# In[7]:


# How many unique plates?
jump_pred_df.Metadata_Plate.nunique()


# In[8]:


# How many different individual treatments?
jump_pred_df.query("Metadata_model_type == 'final'").treatment_type.value_counts()


# In[9]:


# How many unique treatments per treatment type?
jump_pred_df.groupby("treatment_type").treatment.nunique()


# In[10]:


# How many treatments with phenotype predictions?
jump_pred_df.query("Metadata_model_type == 'final'").phenotype.value_counts()


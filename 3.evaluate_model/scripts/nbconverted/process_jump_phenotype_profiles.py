#!/usr/bin/env python
# coding: utf-8

# ## Process JUMP phenotypic profiles
# 
# We applied the AreaShape only class-balanced multiclass elastic net logistic regression model to all single-cell profiles in the JUMP dataset.
# 
# We then performed a series of KS tests to identify how different treatment distributions of all phenotype probabilities differed from controls.
# 
# See https://github.com/WayScience/JUMP-single-cell for complete details.
# 
# Here, we perform the following:
# 
# 1. Load in this data from the JUMP-single-cell repo
# 2. Explore the top results per phenotype/treatment_type/model_type
# 3. Convert it to wide format
# 
# This wide format represents a "phenotypic profile" which we can use similarly as an image-based morphology profile.

# In[1]:


import pathlib
import pandas as pd


# In[2]:


# Set file paths
# 1) JUMP phenotype probabilities from AreaShape model
commit = "2c063b6dc48049201a57b060d18f97a5fc783488"

url = "https://github.com/WayScience/JUMP-single-cell/raw"
file = "3.analyze_data/class_balanced_well_log_reg_comparison_results/class_balanced_well_log_reg_areashape_model_comparisons.parquet"

jump_sc_pred_file = f"{url}/{commit}/{file}"

# 2) JUMP additional metadata needed to summarize/groupby results
jump_metadta_commit = "a18fd7719c05b638c731142b0d42a92c645e2b33"

jump_metadta_url = "https://github.com/jump-cellpainting/2023_Chandrasekaran_submitted/raw"
jump_metadta_file = "benchmark/output/experiment-metadata.tsv"

jump_metadata_full_file = f"{jump_metadta_url}/{jump_metadta_commit}/{jump_metadta_file}"

# Set constants
n_top_results_to_explore = 10


# In[3]:


# Set output files
output_dir = "jump_phenotype_profiles"

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


# Load JUMP metadata for JUMP-CP Pilot
# For an explanation of these metadata columns see: 
# https://github.com/jump-cellpainting/2023_Chandrasekaran_submitted/blob/9edd26d60524a62f993d4df40a5d8908206714f5/README.md#batch-and-plate-metadata
jump_metadata_df = (
    pd.read_csv(jump_metadata_full_file, sep="\t")
    .query("Batch == '2020_11_04_CPJUMP1'")
)

print(jump_metadata_df.shape)
jump_metadata_df.head()


# In[6]:


# Merge dataframes and retain only informative columns
jump_pred_df = (
    jump_pred_df
    .merge(
        jump_metadata_df,
        left_on="Metadata_Plate",
        right_on="Assay_Plate_Barcode"
    )
    .drop(columns=[
        "Batch",
        "Plate_Map_Name",
        "Perturbation",
        "Density",
        "Antibiotics",
        "Cell_line",
        "Time_delay",
        "Times_imaged",
        "Anomaly",
        "Number_of_images",
        "Assay_Plate_Barcode"
    ])
    .reset_index(drop=True)
)

print(jump_pred_df.shape)
jump_pred_df.head()


# In[7]:


# Focus on the top results for downstream interpretation
jump_focused_top_results_df = (
    jump_pred_df
    .groupby(["Metadata_model_type", "treatment_type", "Cell_type", "Time", "phenotype"])
    .apply(lambda x: x.nsmallest(n_top_results_to_explore, "p_value"))
    .reset_index(drop=True)
)

jump_focused_top_results_df.to_csv(top_results_summary_file, sep="\t", index=False)

print(jump_focused_top_results_df.shape)
jump_focused_top_results_df.head()


# ## Summarize data

# In[8]:


# How many unique plates?
jump_pred_df.Metadata_Plate.nunique()


# In[9]:


# How many different individual treatments?
jump_pred_df.query("Metadata_model_type == 'final'").treatment_type.value_counts()


# In[10]:


# How many unique treatments per treatment type?
jump_pred_df.groupby("treatment_type").treatment.nunique()


# In[11]:


# How many treatments with phenotype predictions?
jump_pred_df.query("Metadata_model_type == 'final'").phenotype.value_counts()


# ## Convert data to phenotypic profiles

# In[12]:


jump_wide_final_df = (
    jump_pred_df
    .query("Metadata_model_type == 'final'")
    .drop(columns=["p_value"])
    .pivot(index=[
        "Metadata_Plate",
        "treatment",
        "treatment_type",
        "Cell_type",
        "Time",
        "Metadata_Well",
        "cell_count"
    ], columns="phenotype", values="comparison_metric_value")
    .reset_index()
)

jump_wide_final_df.to_csv(final_jump_phenotype_file, sep="\t", index=False)

print(jump_wide_final_df.shape)
jump_wide_final_df.head()


# In[13]:


jump_wide_shuffled_df = (
    jump_pred_df
    .query("Metadata_model_type == 'shuffled'")
    .drop(columns=["p_value"])
    .pivot(index=[
        "Metadata_Plate",
        "treatment",
        "treatment_type",
        "Cell_type",
        "Time",
        "Metadata_Well",
        "cell_count"
    ], columns="phenotype", values="comparison_metric_value")
    .reset_index()
)

jump_wide_shuffled_df.to_csv(shuffled_jump_phenotype_file, sep="\t", index=False)

print(jump_wide_shuffled_df.shape)
jump_wide_shuffled_df.head()


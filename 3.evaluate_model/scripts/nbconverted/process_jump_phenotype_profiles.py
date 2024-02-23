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
# 3. Explore the top results per phenotype/treatment_type/model_type
# 4. Convert it to wide format
# 
# This wide format represents a "phenotypic profile" which we can use similarly as an image-based morphology profile.
# 
# We also fit a UMAP to this phenotypic profile for downstream visualization.

# In[1]:


import pathlib
from typing import List
import pandas as pd

import umap


# In[2]:


def umap_phenotype(
    phenotype_df: pd.DataFrame,
    feature_columns: List[str],
    metadata_columns: List[str],
    n_components: int,
    random_seed: int,
    model_type: str
) -> pd.DataFrame:
    """
    Fit a UMAP (Uniform Manifold Approximation and Projection) model on the provided phenotype profile and return a transformed DataFrame with metadata.

    Parameters:
    - phenotype_df (pd.DataFrame): DataFrame containing the phenotype profile with both feature and metadata columns.
    - feature_columns (List[str]): List of column names in phenotype_df that represent the features to be used for UMAP embedding.
    - metadata_columns (List[str]): List of column names in phenotype_df that represent metadata to be retained in the output.
    - n_components (int): Number of dimensions for the UMAP embedding.
    - random_seed (int): Random seed for reproducibility of the UMAP model.
    - model_type (str): Identifier for the model type, to be added as a column in the output DataFrame.

    Returns:
    - umap_embeddings_with_metadata_df (pd.DataFrame): DataFrame with UMAP embeddings and specified metadata columns, including an additional 'model_type' column.
    """
    
    # Initialize UMAP
    umap_fit = umap.UMAP(random_state=random_seed, n_components=n_components)
    
    # Fit UMAP and convert to pandas DataFrame
    embeddings = pd.DataFrame(
        umap_fit.fit_transform(phenotype_df.loc[:, feature_columns]),
        columns=[f"UMAP{x}" for x in range(0, n_components)],
    )
    
    # Combine with metadata
    umap_embeddings_with_metadata_df = pd.concat([phenotype_df.loc[:, metadata_columns], embeddings], axis=1).assign(model_type=model_type)
    return umap_embeddings_with_metadata_df


# In[3]:


# Set file paths
# JUMP phenotype probabilities from AreaShape model
commit = "4225e427fd9da59159de69f53be65c31b4d4644a"

url = "https://github.com/WayScience/JUMP-single-cell/raw"
file = "3.analyze_data/class_balanced_well_log_reg_comparison_results/class_balanced_well_log_reg_areashape_model_comparisons.parquet"

jump_sc_pred_file = f"{url}/{commit}/{file}"

# Set constants
n_top_results_to_explore = 100


# In[4]:


# Set output files
output_dir = "jump_phenotype_profiles"

cell_type_time_comparison_file = pathlib.Path(output_dir, "jump_compare_cell_types_and_time_across_phenotypes.tsv.gz")
top_results_summary_file = pathlib.Path(output_dir, "jump_most_significant_phenotype_enrichment.tsv")
final_jump_phenotype_file = pathlib.Path(output_dir, "jump_phenotype_profiles.tsv.gz")
shuffled_jump_phenotype_file = pathlib.Path(output_dir, "jump_phenotype_profiles_shuffled.tsv.gz")

jump_umap_file = pathlib.Path(output_dir, "jump_phenotype_profiling_umap.tsv.gz")


# ## Load and process data

# In[5]:


# Load KS test results and drop uninformative columns
jump_pred_df = (
    pd.read_parquet(jump_sc_pred_file)
    .drop(columns=["statistical_test", "comparison_metric"])
)

print(jump_pred_df.shape)
jump_pred_df.head()


# In[6]:


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


# In[7]:


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


metadata_columns = [
    "Metadata_Plate",
    "treatment",
    "treatment_type",
    "Cell_type",
    "Time",
    "Metadata_Well",
    "cell_count"
]


# In[13]:


jump_wide_final_df = (
    jump_pred_df
    .query("Metadata_model_type == 'final'")
    .drop(columns=["p_value"])
    .pivot(index=metadata_columns, columns="phenotype", values="comparison_metric_value")
    .reset_index()
)

jump_wide_final_df.to_csv(final_jump_phenotype_file, sep="\t", index=False)

print(jump_wide_final_df.shape)
jump_wide_final_df.head()


# In[14]:


jump_wide_shuffled_df = (
    jump_pred_df
    .query("Metadata_model_type == 'shuffled'")
    .drop(columns=["p_value"])
    .pivot(index=metadata_columns, columns="phenotype", values="comparison_metric_value")
    .reset_index()
)

jump_wide_shuffled_df.to_csv(shuffled_jump_phenotype_file, sep="\t", index=False)

print(jump_wide_shuffled_df.shape)
jump_wide_shuffled_df.head()


# ## Apply UMAP to phenotypic profiles

# In[15]:


umap_random_seed = 123
umap_n_components = 2

feature_columns = jump_wide_final_df.drop(columns=metadata_columns).columns.tolist()
print(len(feature_columns))


# In[16]:


umap_with_metadata_df = umap_phenotype(
    phenotype_df=jump_wide_final_df,
    feature_columns=feature_columns,
    metadata_columns=metadata_columns,
    n_components=umap_n_components,
    random_seed=umap_random_seed,
    model_type="final"
)


# In[17]:


umap_shuffled_with_metadata_df = umap_phenotype(
    phenotype_df=jump_wide_shuffled_df,
    feature_columns=feature_columns,
    metadata_columns=metadata_columns,
    n_components=umap_n_components,
    random_seed=umap_random_seed,
    model_type="shuffled"
)


# In[18]:


# Output file
umap_full_df = pd.concat([
    umap_with_metadata_df,
    umap_shuffled_with_metadata_df
], axis="rows")

umap_full_df.to_csv(jump_umap_file, sep="\t", index=False)

print(umap_full_df.shape)
umap_full_df.head()


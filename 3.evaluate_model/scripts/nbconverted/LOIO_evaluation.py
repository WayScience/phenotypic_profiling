#!/usr/bin/env python
# coding: utf-8

# ## Evaluate Leave One Image Out analysis

# In[1]:


import pathlib
import numpy as np
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, auc


# In[2]:


def compute_avg_rank_and_pvalue(grouped_df):
    ranks = []
    p_values = []

    for _, group in grouped_df.groupby("Cell_UUID"):
        # Sort predicted probabilities in descending order
        sorted_probs = group.sort_values(by="Predicted_Probability", ascending=False).reset_index(drop=True)
        
        # Get the rank of the true class
        rank = sorted_probs[sorted_probs["Mitocheck_Phenotypic_Class"] == sorted_probs["Model_Phenotypic_Class"]].index[0] + 1
        
        # Get the p-value (predicted probability) of the true class
        p_value = sorted_probs.loc[rank - 1, "Predicted_Probability"]
        
        ranks.append(rank)
        p_values.append(p_value)

    # Calculate average rank and p-value for the group
    avg_rank = sum(ranks) / len(ranks)
    avg_p_value = sum(p_values) / len(p_values)
    
    # Calculate IQR and min/max within IQR for ranks
    iqr_rank = np.percentile(ranks, 75) - np.percentile(ranks, 25)
    min_iqr_rank = np.percentile(ranks, 25)
    max_iqr_rank = np.percentile(ranks, 75)
    
    # Calculate IQR and min/max within IQR for p-values
    iqr_p_value = np.percentile(p_values, 75) - np.percentile(p_values, 25)
    min_iqr_p_value = np.percentile(p_values, 25)
    max_iqr_p_value = np.percentile(p_values, 75)
    
    # Count number of comparisons
    count = len(ranks)
    
    return avg_rank, avg_p_value, min_iqr_rank, max_iqr_rank, min_iqr_p_value, max_iqr_p_value, count


# In[3]:


# Set I/O
proba_dir = pathlib.Path("evaluations", "LOIO_probas")
loio_file = pathlib.Path(proba_dir, "compiled_LOIO_probabilities.tsv")

output_summary_file = pathlib.Path(proba_dir, "LOIO_summary_ranks.tsv.gz")
output_summary_phenotype_file = pathlib.Path(proba_dir, "LOIO_summary_ranks_perphenotype.tsv.gz")


# In[4]:


loio_df = pd.read_csv(loio_file, sep="\t", index_col=0)

print(loio_df.shape)
loio_df.head()


# In[5]:


phenotype_classes = loio_df.Mitocheck_Phenotypic_Class.unique().tolist()
phenotype_classes


# ## Get average ranks and p value of correct prediction
# 
# - Per Image
# - Per model type (final vs. shuffled)
# - Per illumination correction function (IC vs. No-IC)
# - Phenotype
# - Feature Space

# In[6]:


rank_groups = [
    "Metadata_DNA",
    "Model_type",
    "Dataset_type",
    "Mitocheck_Phenotypic_Class",
    "Model_Feature_Type",
    "Balance_type"
]

# Output data columns
output_data_columns = [
    "Average_Rank",
    "Average_P_Value",
    "Min_IQR_Rank",
    "Max_IQR_Rank", 
    "Min_IQR_P_Value",
    "Max_IQR_P_Value", 
    "Count"
]

avg_ranks = (
    loio_df.groupby(rank_groups)
    .apply(compute_avg_rank_and_pvalue)
    .reset_index()
)

avg_ranks.columns = rank_groups + ["Average_Scores"]

loio_scores_df = (
    pd.concat([
        avg_ranks.drop(columns="Average_Scores"),
        pd.DataFrame(
            avg_ranks.Average_Scores.tolist(),
            columns=output_data_columns
        )
    ], axis="columns")
)

loio_scores_df.to_csv(output_summary_file, index=False, compression="gzip", sep="\t")

print(loio_scores_df.shape)
loio_scores_df.head()


# ## Get average ranks and p value per phenotype
# 
# - Per model type (final vs. shuffled)
# - Per illumination correction function (IC vs. No-IC)
# - Per Phenotype
# - Per Feature Space
# 
# (i.e., not on a per-image basis)

# In[7]:


# Calculate average rank for each phenotype
rank_groups.remove("Metadata_DNA")  # Remove the per image to group on

avg_ranks = (
    loio_df.groupby(rank_groups)
    .apply(compute_avg_rank_and_pvalue)
    .reset_index()
)

avg_ranks.columns = rank_groups + ["Average_Scores"]

loio_scores_df = (
    pd.concat([
        avg_ranks.drop(columns="Average_Scores"),
        pd.DataFrame(
            avg_ranks.Average_Scores.tolist(),
            columns=output_data_columns
        )
    ], axis="columns")
)

loio_scores_df.to_csv(output_summary_phenotype_file, index=False, compression="gzip", sep="\t")

print(loio_scores_df.shape)
loio_scores_df.head()


# 1. Split Data

In this module, we split the training data into training and testing datasets.

Data is split into subsets in [split_data.ipynb](split_data.ipynb).
The testing dataset is determined by randomly sampling 15% (stratified by phenotypic class) of the single-cell dataset.
The training dataset is the subset remaining after the testing samples are removed.
Sample indexes associated with training and testing subsets are stored in [data_split_indexes.tsv](indexes/data_split_indexes.tsv).
Sample indexes are later used to load subsets from [training_data.csv.gz](../0.download_data/data/training_data.csv.gz).

## Step 1: Split Data

Use the commands below to create indexes for training and testing data subsets:

```sh
# Make sure you are located in 1.split_data
cd 1.split_data

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Split data
bash split_data.sh
```

# 1. Split Data

In this module, we split the training data into training and testing datasets.

Data is split into subsets in [split_data.ipynb](split_data.ipynb).
The testing dataset is determined by randomly sampling 15% (stratified by phenotypic class) of the single-cell dataset.
The training dataset is the subset remaining after the testing samples are removed.
We store sample indexes associated with training and testing subsets in [indexes/](indexes/), and we later use these sample indexes to load subsets from labeled data in [0.download_data/data/](../0.download_data/data/).

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

# 1. Split Data

In this module, we split the training data into training and testing datasets.

First, we split the data into training, test, and holdout subsets in [split_data.ipynb](split_data.ipynb).
The `get_representative_images()` function used to create the holdout dataset determines which images to holdout such that all phenotypic classes can be represented in these holdout images.
The test dataset is determined by taking a random number of samples (stratified by phenotypic class) from the dataset after the holdout images are removed.
The training dataset is the subset remaining after holdout/test samples are removed.
Sample indexes associated with training, test, and holdout subsets are stored in [data_split_indexes.tsv](indexes/data_split_indexes.tsv).
Sample indexes are later used to load subsets from [training_data.csv.gz](../0.download_data/data/training_data.csv.gz).

## Step 1: Split Data

Use the commands below to create indexes for training, testing, and holdout data subsets:

```sh
# Make sure you are located in 1.split_data
cd 1.split_data

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Split data
bash split_data.sh
```

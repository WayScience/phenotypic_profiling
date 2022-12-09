# Download Data

In this module, we present our method for downloading and combining nucleus morphology data.

### Download/Preprocess Data

Complete instructions for data download and preprocessing can be found at: https://github.com/WayScience/mitocheck_data

### Usage

In this repository, all training data is downloaded from a version controlled [mitocheck_data](https://github.com/WayScience/mitocheck_data).

An earlier (2006) and later (2015) dataset are both downloaded from `mitocheck_data` and combined by checking if any of the plate/well/frame/coordinates of the cells from the 2006 and 2015 datasets match.
If all of this information matches, this must be the same cell and is only added once to the final combined dataset.
This combination method avoids repeating cells in the combined dataset which could lead to biases in the final model.

The version of mitocheck_data used is specified by the hash corresponding to a current commit.
The current hashes being used are `19bfa5b0959d6b7536f83e7bb85745ba3edf7ff9` for the 2006 dataset and `3ebd0ca7c288f608e9b23987a8ddbabd5476bd8f` for the 2015 dataset.
These correspond to [mitocheck_data/19bfa5b](https://github.com/WayScience/mitocheck_data/tree/19bfa5b0959d6b7536f83e7bb85745ba3edf7ff9) and [mitocheck_data/3ebd0ca](https://github.com/WayScience/mitocheck_data/tree/3ebd0ca7c288f608e9b23987a8ddbabd5476bd8f) respectively.
The `hash` variable can be set in [download_data.ipynb](download_data.ipynb) to change which version of mitocheck_data is being accessed.

## Step 1: Download Data

Use the commands below to download labeled training dataset:

```sh
# Make sure you are located in 0.download_data
cd 0.download_data

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Download data
bash download_data.sh
```

# Phenotypic Profiling Model

## Data

Instructions for data download/preprocessing can be found at: https://github.com/WayScience/mitocheck_data.

This repository downloads training data from a specific version of [MitoCheck_data](https://github.com/WayScience/mitocheck_data).
For more information see [0.download_data/](0.download_data/).

## Setup

Perform the following steps to set up the `phenotypic_profiling` environment necessary for processing data in this repository.

### Step 1: Create Phenotypic Profiling Environment

```sh
# Run this command to create the conda environment for phenotypic profiling

conda env create -f phenotypic_profiling_env.yml
```

### Step 2: Activate Phenotypic Profiling Environment

```sh
# Run this command to activate the conda environment for phenotypic profiling

conda activate phenotypic_profiling
```

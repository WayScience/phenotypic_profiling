# Phenotypic Profiling Model

## Data

Instructions for data download/preprocessing can be found at: https://github.com/WayScience/mitocheck_data.

This repository downloads training data from a specific version of [MitoCheck_data](https://github.com/WayScience/mitocheck_data).
For more information see [0.download_data/](0.download_data/).

## Repository Structure:

This repository is structured as follows:

| Order | Module | Description |
| :---- | :----- | :---------- |
| [0.download_data](0.download_data/) | Download training data | Download labeled single-cell dataset from [mitocheck_data](https://github.com/WayScience/mitocheck_data) |
| [1.split_data](1.split_data/) | Create data subsets | Create training, testing, and holdout data subsets |
| [2.train_model](2.train_model/) | Train model | Train ML model on training data subset |
| [3.evaluate_model](3.evaluate_model/) | Evaluate model | Evaluate ML model on all data subsets |
| [4.interpret_model](4.interpret_model/) | Interpret model | Interpret ML model |

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

# Phenotypic Profiling Model

## Repository Structure:

This repository is structured as follows:

| Order | Module | Description |
| :---- | :----- | :---------- |
| [0.download_data](0.download_data/) | Download training data | Download labeled single-cell dataset from [mitocheck_data](https://github.com/WayScience/mitocheck_data) |
| [1.split_data](1.split_data/) | Create data subsets | Create training and testing data subsets |
| [2.train_model](2.train_model/) | Train model | Train ML models on training data subset and shuffled baseline training dataset |
| [3.evaluate_model](3.evaluate_model/) | Evaluate model | Evaluate ML models on all data subsets |
| [4.interpret_model](4.interpret_model/) | Interpret model | Interpret ML models |
| [5.validate_model](5.validate_model/) | Validate model | Validate ML models |

## Data

Instructions for data download/preprocessing can be found at: https://github.com/WayScience/mitocheck_data.

This repository downloads training data from a specific version of [MitoCheck_data](https://github.com/WayScience/mitocheck_data).
For more information see [0.download_data/](0.download_data/).

## Machine Learning Model

We use [Scikit-learn (sklearn)](https://scikit-learn.org/) for data manipulation, model training, and model evaluation.
Scikit-learn is described in [Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html) as a machine learning library for Python.
Its ease of implementation in a pipeline make it ideal for our use case.

We consistently use the following parameters with many `sklearn` functions:

- `n_jobs=-1`: Use all CPU cores in parallel when completing a task.
- `random_state=0`: Use seed 0 when shuffling data or generating random numbers.
This allows "random" sklearn operations to have consist results.
We also use `np.random.seed(0)` to make "random" numpy operations have consistent results.

We use [seaborn](https://seaborn.pydata.org/) for data visualization. 
Seaborn is described in [Waskom, M.L., 2021](https://doi.org/10.21105/joss.03021) as a library for making statisical graphics in python.

All parts of the following pipeline are completed for a "final" model (from training data) and a "shuffled baseline" model (from shuffled training data).
This shuffled baseline model provides a suitable baseline comparison for the final model during evaluation.

**Note:** Throughout this repository, intermediate `.tsv` data are stored in tidy long format, a standardized data structure (see [Tidy Data](https://vita.had.co.nz/papers/tidy-data.pdf) by Hadley Wickham for more details).
This data structure make later analysis easier.

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

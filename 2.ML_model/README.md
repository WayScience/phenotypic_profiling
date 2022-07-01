# 2. Machine Learning Model

In this module, we train and validate a machine learning model for phenotypic classification of nuclei based on nuclei features.

## Step 1: Setup Download Environment

### Step 1a: Create Download Environment

```sh
# Run this command to create the conda environment for downloading data
conda env create -f 2.machine_learning_env.yml
```

### Step 1b: Activate Download Environment

```sh
# Run this command to activate the conda environment for downloading data
conda activate 2.ML_phenotypic_classification
```

## Step 2: Execute Training Data Preprocessing

```bash
# Run this script to preprocess training movies
bash 1.format_trainind_data.sh
```
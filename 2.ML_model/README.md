# 2. Machine Learning Model

In this module, we train and validate a machine learning model for phenotypic classification of nuclei based on nuclei features.

We use [Scikit-learn](https://scikit-learn.org/) for data manipulation, model training, and model evaluation. Scikit-learn was introduced in [Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html) as a machine learning library for Python. Its ease of implementation in a pipeline make it ideal for our use case.

### A. Data Preparation

Training data is read from [training_data.csv.gz](../1.format_data/data/training_data.csv.gz). We use [sklearn ]

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
bash 
```
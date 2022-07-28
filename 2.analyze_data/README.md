# 2. Analyze Features

In this module, we present our pipeline for analyzing features.

### Feature Analysis



## Step 1: Setup Feature Analysis Environment

### Step 1a: Create Feature Analysis Environment

```sh
# Run this command to create the conda environment for feature analysis
conda env create -f 2.analyze_data_env.yml
```

### Step 1b: Activate Feature Analysis Environment

```sh
# Run this command to activate the conda environment for feature analysis
conda activate 2.analyze_training_data
```

## Step 2: Normalize Single Cell Training Features

```bash
# Run this script to preprocess training features
bash 4.preprocess_training_features.sh
```

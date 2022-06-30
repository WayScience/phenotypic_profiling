# 1. Format Data

In this module, we use data from a specific version of [mitocheck_data](https://github.com/WayScience/mitocheck_data) to compile a [training dataframe file](data/training_data.csv.gz).

The [format training data script](format_training_data.ipynb) accesses the version of mitocheck_data to retrieve the preprocessed feature data, segmentation data, and trainingset.dat file.
With these data, the script is able to compile a dataframe with all labeled single-nuclei embeddings, their metadata, and their MitoCheck-assigned object ID/phenotypic class.

**Note**: The version of mitocheck_data used to compile training data can be changed by setting the `hash` variable in the [format training data script](format_training_data.ipynb) to the desired commit of mitocheck_data.

## Step 1: Setup Download Environment

### Step 1a: Create Download Environment

```sh
# Run this command to create the conda environment for downloading data
conda env create -f 1.format_env.yml
```

### Step 1b: Activate Download Environment

```sh
# Run this command to activate the conda environment for downloading data
conda activate 1.format_training_data
```

## Step 2: Execute Training Data Preprocessing

```bash
# Run this script to preprocess training movies
bash 1.format_trainind_data.sh
```
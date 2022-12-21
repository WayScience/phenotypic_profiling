# 5. Validate Module

In this module, we validate the final ML model.

### Validation Method 1

The final model from [2.train_model](../2.train_model/) is used to classify nuclei images from the [Cell Health Dataset](https://github.com/WayScience/cell-health-data).
The classification probabilies across CRISPR perturbation/cell line are then correlated to the Cell Health label for the the respective CRISPR perturbation/cell line.

The Cell Health dataset has cell painting images across 119 CRISPR guide perturbations (~2 per gene perturbation) and 3 cell lines.
More information regarding the generation of this dataset can be found at https://github.com/broadinstitute/cell-health.

In [Cell-Health-Data/4.classify-features](https://github.com/WayScience/cell-health-data/tree/master/4.classify-features), we use the final model to determine phenotypic class probabilities for each of the Cell Health cells.
These probabilities are averaged across CRISPR perturbation/cell line to create 357 *classifiction profiles* (119 CRISPR guides x 3 cell lines).

As part of [Predicting cell health phenotypes using image-based morphology profiling](https://www.molbiolcell.org/doi/10.1091/mbc.E20-12-0784), Way et al derived *Cell Health label profiles*.


These correlations are interpreted with the following diagrams:

- We use [seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to display the 
- We use [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) to display 

## Step 1: Interpret Model

Use the commands below to interpret the ML models:

```sh
# Make sure you are located in 4.interpret_model
cd 4.interpret_model

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Interpret model
bash interpret_model.sh
```

## Results

Each model's interpretations can be found in [interpret_model.ipynb](interpret_model.ipynb).
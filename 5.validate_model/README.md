# 5. Validate Model

In this module, we validate the final ML model.

### Validation Method 1

The models from [2.train_model](../2.train_model/) are used to classify nuclei images from the [Cell Health Dataset](https://github.com/WayScience/cell-health-data).
The classification probabilities across CRISPR guide/cell line are then correlated to the Cell Health label in [cell_health_correlations.ipynb](cell_health_correlations.ipynb) for the the respective CRISPR perturbation/cell line.

The Cell Health dataset has cell painting images across 119 CRISPR guide perturbations (~2 per gene perturbation) and 3 cell lines.
More information regarding the generation of this dataset can be found at https://github.com/broadinstitute/cell-health.

In [Cell-Health-Data/4.classify-features](https://github.com/WayScience/cell-health-data/tree/master/4.classify-features), we use the trained models to determine phenotypic class probabilities for each of the Cell Health cells.
These probabilities are averaged across CRISPR guide/cell line to create 357 *classifiction profiles* (119 CRISPR guides x 3 cell lines).

As part of [Predicting cell health phenotypes using image-based morphology profiling](https://www.molbiolcell.org/doi/10.1091/mbc.E20-12-0784), Way et al derived cell health indicators.
These indicators consist of 70 specific cell health phenotypes including proliferation, apoptosis, reactive oxygen species, DNA damage, and cell cycle stage.
These indicators are averaged across across CRISPR guide/cell line to create 357 [*Cell Health label profiles*](https://github.com/broadinstitute/cell-health/blob/master/1.generate-profiles/data/consensus/cell_health_median.tsv.gz).

We use [pandas.DataFrame.corr](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) to find the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between the *classifiction profiles* and the *Cell Health label profiles*.
The Pearson correlation coefficient measures the linear relationship between two datasets, with correlations of -1/+1 implying exact linear inverse/direct relationships respectively.

We also derive the [Clustermatch Correlation Coefficient (CCC)](https://github.com/greenelab/ccc) introduced in [Pividori et al, 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496326v1).
This is a not-only-linear coefficient based on machine learning models and gives an idea of how correlated the feature coefficients are (where 0 is no relationship and 1 is a perfect relationship).

These correlations are briefly interpreted in [view_cell_health_correlations.ipynb](view_cell_health_correlations.ipynb) with [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) to display the hierarchically-clustered correlation values.
Searborn clustermap groups similar correlations into clusters that are broadly similar to each other.

## Step 1: Define Folder Paths

Inside the notebook [cell_health_correlations.ipynb](cell_health_correlations.ipynb), the variable `classification_profiles_save_dir` needs to be set to specify where the classficiation profiles are saved.
We used an external harddrive and therefore needed to use specific paths.
The classification profiles are the output of [cell-health-data/4.classify-single-cell-phenotypes](https://github.com/roshankern/cell-health-data/tree/derive-classification-profiles/4.classify-single-cell-phenotypes).

## Step 2: Validate Model

Use the commands below to validate the final ML model:

```sh
# Make sure you are located in 5.validate_model
cd 5.validate_model

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Interpret model
bash validate_model.sh
```

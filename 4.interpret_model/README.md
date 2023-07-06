# 4. Interpret Model

In this module, we interpret the ML models.

After training the final and baseline models in [2.train_model](../2.train_model/), we load the coefficents of these models from [models/](../2.train_model/models).
These coefficients are interpreted with the following diagrams:

- We use [seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to display the coefficient values for each phenotypic class/feature.
- We use [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) to display a hierarchically-clustered heatmap of coefficient values for each phenotypic class/feature
- We use [seaborn.kedeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html) to display a density plot of coeffiecient values for each phenotypic class.
- We use [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html) to display a bar plot of average coeffiecient values per phenotypic class.

In [model_coefficient_correlations.ipynb](model_coefficient_correlations.ipynb), we compare the coefficients from the mutli-class and single-class models.
The coefficients matrix from multi-class models are of shape `(# phenotypic classes, # features)`, while the coefficients from single-class models are of shape `(1, # features)`.
Thus, we are able to compare the coefficient vectors for each phenotypic class per model.

We graph these coefficient vectors in a scatterplot where the coordinate pairs represent `(mutli-class model coefficient value, single-class model coefficient value)` for a particular feature.
For each of the coefficient vectors for the multi-class and single-class mdoels, we derive the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) with [numpy.coercoef](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html) to get an idea of how correlated these vectors are.
We also derive the [Clustermatch Correlation Coefficient (CCC)](https://github.com/greenelab/ccc) introduced in [Pividori et al, 2022](https://www.biorxiv.org/content/10.1101/2022.06.15.496326v1).
This is a not-only-linear coefficient based on machine learning models and gives an idea of how correlated the feature coefficients are (where 0 is no relationship and 1 is a perfect relationship).
The correlations for each pair of coefficient vectors are displayed above their scatterplots.

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

Each model's interpretations can be found in [interpret_model_coefficients.ipynb](interpret_model_coefficients.ipynb).

**Note:** Intermediate `.tsv` data are stored in tidy format, a standardized data structure (see [Tidy Data](https://vita.had.co.nz/papers/tidy-data.pdf) by Hadley Wickham for more details).
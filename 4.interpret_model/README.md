# 4. Interpret Model

In this module, we interpret the ML models.

After training the final and baseline models in [2.train_model](../2.train_model/), we load the coefficents of these models from [models/](../2.train_model/models).
These coefficients are interpreted with the following diagrams:

- We use [seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to display the coefficient values for each phenotypic class/feature.
- We use [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) to display a hierarchically-clustered heatmap of coefficient values for each phenotypic class/feature
- We use [seaborn.kedeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html) to display a density plot of coeffiecient values for each phenotypic class.
- We use [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html) to display a bar plot of average coeffiecient values per phenotypic class.

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

**Note:** Intermediate `.tsv` data are stored in tidy format, a standardized data structure (see [Tidy Data](https://vita.had.co.nz/papers/tidy-data.pdf) by Hadley Wickham for more details).
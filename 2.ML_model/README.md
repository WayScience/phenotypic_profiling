# 2. Machine Learning Model

In this module, we train and validate a machine learning model for phenotypic classification (mitotic stage) of nuclei based on nuclei features.

We use [Scikit-learn (sklearn)](https://scikit-learn.org/) for data manipulation, model training, and model evaluation.
Scikit-learn is described in [Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html) as a machine learning library for Python.
Its ease of implementation in a pipeline make it ideal for our use case.

We consistently use the following parameters with many sklearn functions:

- `n_jobs=-1`: Use all CPU cores in parallel when completing a task.
- `random_state=0`: Use seed 0 when shuffling data or generating random numbers.
This allows "random" operations to have consist results.

We use [seaborn](https://seaborn.pydata.org/) for data visualization. 
Seaborn is described in [Waskom, M.L., 2021](https://doi.org/10.21105/joss.03021) as a library for making statisical graphics in python.

### A. Data Preparation

Training data is loaded from [training_data.csv.gz](../1.format_data/data/training_data.csv.gz).

We use [sklearn.utils.shuffle](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html) to shuffle the training data in a consistent way.
This method is shuffling in the first dimension (samples).
This is necessary because the data as labeled from MitoCheck tends to have phenotypic classes in groups, which can introduce bias into the model.

We use [sklearn.model_selection.StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) to create stratified training/test data sets for cross validation.
The training and test data sets that are created have the same distribution of classes.
This ensures that each class is proportionally represented across all data sets.
We use `n_splits=10` to create 10 folds for cross-validation.

### B. Model Training

We use [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for our machine learning model.
We use the following parameters for our Logisic Regression model:

- `penalty='elasticnet'`: Use elasticnet as the penalty for our model.
Elastic-Net regularization is a combination of L1 and L2 regularization methods.
The mixing of these two methods is determined by the `l1_ratio` parameter which we optimize later.
- `solver='saga'`: We use the saga solver as this is the only solver that supports Elastic-Net regularization.
- `max_iter=100`: Set the maximum number of iterations for solver to converge. 100 iterations allows the solver to maximize performance without completing unnecessary iterations.

We use [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) to perform an exhaustive search for the following parameters:

- `l1_ratio`: Elastic-Net mixing parameter.
Used to combine L1 and L2 regularization methods.
- `C`: Inversely proportional to regularization strength.

We use [sklearn.model_selection.cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) and [sklearn.model_selection.cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) to train and evaluate a logistic regression model with the best parameters found with GridSearchCV.

### C. Model Evaluation

We compute the average score across estimators after running cross validation.

We use the predictions from `cross_val_predict()` to create a confusion matrix and precision vs class bar plot for the cross-validated model.

### D. Model Interpretation

After cross validation is complete, a final estimator is trained with the best parameters found during `GridSearchCV` and all of the training data.
The coefficient matrix from this final estimator is interpreted with the following visualizations:

- We use [seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to display the coefficient values for each phenotypic class/feature.
- We use [seaborn.clustermap](https://seaborn.pydata.org/generated/seaborn.clustermap.html) to display a hierarchically-clustered heatmap of coefficient values for each phenotypic class/feature
- We use [seaborn.kedeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html) to display a density plot of coeffiecient values for each phenotypic class.
- We use [seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html) to display a bar plot of average coeffiecient values per phenotypic class and feature.

### E. Baseline Comparison

After training, evaluating, and interpreting a most-accurate estimator, we perform a baseline comparison on shuffled data.
We create a baseline dataset by loading the training data in the same way as above, but then shuffling the `y `(labels) dataframe.
The train, evaluate, interpret pipeline is then rerun on this shuffled baseline dataset to derive a randomly shuffled baseline for comparison with our final estimator.

## Step 1: Setup Download Environment

### Step 1a: Create Download Environment

```sh
# Run this command to create the conda environment for machine learning
conda env create -f 2.machine_learning_env.yml
```

### Step 1b: Activate Download Environment

```sh
# Run this command to activate the conda environment for machine learning
conda activate 2.ML_phenotypic_classification
```

## Step 2: Execute Training Data Preprocessing

```bash
# Run this script to train, evaluate, and interpret DP model
bash 2.DP_model.sh
```
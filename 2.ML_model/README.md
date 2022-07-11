# 2. Machine Learning Model

In this module, we train and validate a machine learning model for phenotypic classification of nuclei based on nuclei features.

We use [Scikit-learn (sklearn)](https://scikit-learn.org/) for data manipulation, model training, and model evaluation.
Scikit-learn was introduced in [Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](http://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html) as a machine learning library for Python.
Its ease of implementation in a pipeline make it ideal for our use case.

We consistently use the following parameters with many sklearn functions:

- `n_jobs=-1`: Use all CPU cores in parallel when completing a task.
- `random_state=0`: Use seed 0 when shuffling data or generating random numbers.
This allows "random" operations to have consist results.

### A. Data Preparation

Training data is loaded from [training_data.csv.gz](../1.format_data/data/training_data.csv.gz).

We use [sklearn.utils.shuffle](https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html) to shuffle the training data in a consistent way.
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
We use these functions for cross validation scoring and estimation for each fold of data.

### C. Model Interpretation

We create a confusion matrix found in [DP_trained_model.ipynb](DP_trained_model.ipynb) using the predictions from `cross_val_predict()`.

Because cross validation produces multiple unique models (estimators), it is necessary to average metrics across the estimators to interpret model performance.
The coefficient matrices created during each fold of cross validation are averaged to create a single `average_coefs` matrix that we analyze.





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
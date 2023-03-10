# 2. Train Model

In this module, we train ML models to predict phenotypic class from cell features.

We train the models in [train_model.ipynb](1.train_model.ipynb).
We use [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for our machine learning models.
A final model 
We use the following parameters for our each Logisic Regression model:

- `penalty='elasticnet'`: Use elasticnet as the penalty for our model.
Elastic-Net regularization is a combination of L1 and L2 regularization methods.
The mixing of these two methods is determined by the `l1_ratio` parameter which we optimize later.
- `solver='saga'`: We use the saga solver as this is the only solver that supports Elastic-Net regularization.
- `max_iter=100`: Set the maximum number of iterations for solver to converge. 100 iterations allows the solver to maximize performance without completing unnecessary iterations.

We use [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) to perform an exhaustive search for the parameters below.
This searches for parameters that maximize the weighted F1 score of the model.
We optimize weighted F1 score because this metric measures model precision and recall aand accounts for label imbalance (see [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) for more details).

- `l1_ratio`: Elastic-Net mixing parameter.
Used to combine L1 and L2 regularization methods.
We search over the following parameters: `[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]`
- `C`: Inversely proportional to regularization strength.
We search over the following parameters: `[1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]`

A final model is trained with CP features, DP features, and merged (CP and DP) features.
A shuffled baseline model is trained for each of these feature types as well, with each column of the feature data being shuffled independently to create a shuffled baseline for comparison.
Each model is saved in [models/](models/).

## Step 1: Train Model

Use the commands below to train the ML model:

```sh
# Make sure you are located in 2.train_model
cd 2.train_model

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Train model
bash train_model.sh
```

## Results

The weighted F1 score of the best estimators for the grid searches can be found in [train_model.ipynb](train_model.ipynb).
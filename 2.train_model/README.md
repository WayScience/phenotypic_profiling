# 2. Train Model

In this module, we train ML models to predict phenotypic class from cell features.

In [train_multi_class_models.ipynb](train_multi_class_models.ipynb), we train models to predict the phenotypic class of the cell features from 15 possible classes (anaphase, metaphase, apoptosis, etc).
In [train_single_class_models.ipynb](train_single_class_models.ipynb), we train models to predict whether the cell features are from a particular phenotypic class or not.
This means a set of models are made to predict a "yes" or "no" for each of the 15 phenotypic classes used in the multi-class models. 

We use [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for our machine learning models.
We use the following parameters for our each Logisic Regression model:

- `penalty='elasticnet'`: Use elasticnet as the penalty for our model.
Elastic-Net regularization is a combination of L1 and L2 regularization methods.
The mixing of these two methods is determined by the `l1_ratio` parameter which we optimize later.
- `solver='saga'`: We use the saga solver as this is the only solver that supports Elastic-Net regularization.
- `max_iter=100`: Set the maximum number of iterations for solver to converge. 100 iterations allows the solver to maximize performance without completing unnecessary iterations.

We use [sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) to perform an exhaustive search for the parameters below.
This searches for parameters that maximize the weighted F1 score of the model.
We optimize weighted F1 score because this metric measures model precision and recall and accounts for label imbalance (see [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) for more details).

- `l1_ratio`: Elastic-Net mixing parameter.
Used to combine L1 and L2 regularization methods.
We search over the following parameters: `[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]`
- `C`: Inversely proportional to regularization strength.
We search over the following parameters: `[1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]`

We train models for each combination of the following model types, feature, balance, and dataset types:
- model_types: final, shuffled_baseline
    - Which version of features the model is trained on. For `shuffled_baseline`, each column of the feature data is shuffled independently to create a shuffled baseline for comparison.
- feature_types: CP, DP, CP_and_DP, CP_zernike_only, CP_areashape_only
    - Which features to use for trainining.
- balance_types: balanced, unbalanced
    - Whether or not to balance `class_weight` of each model when training.
- dataset_types: ic, no_ic
    - Which `mitocheck_data` dataset to use for feature training. We have datasets extracted with and without illumination correction.

The notebooks save each model in [models/](models/).

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
# 3. Evaluate Model

In this module, we evaluate the final and shuffled baseline ML models.

After training the final and baseline models in [2.train_model](../2.train_model/), we use these models to predict the labels of the training, testing, and holdout datasets.
These predictions are saved in [model_predictions.tsv](evaluations/model_predictions.tsv) and [shuffled_baseline_model_predictions.tsv](evaluations/shuffled_baseline_model_predictions.tsv) respectively.

We evaluate these 6 sets of predictions with a confusion matrix to see the true/false postives and negatives (see [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) for more details).

We also evaluate these 6 sets of predictions with [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) to determine the final/shuffled baseline model's predictive performance on each subset.
F1 score measures the models precision and recall performance for each phenotypic class.

## Step 1: Evaluate Model

Use the commands below to evaluate the ML models:

```sh
# Make sure you are located in 3.evaluate_model
cd 3.evaluate_model

# Activate phenotypic_profiling conda environment
conda activate phenotypic_profiling

# Evaluate model
bash evaluate_model.sh
```

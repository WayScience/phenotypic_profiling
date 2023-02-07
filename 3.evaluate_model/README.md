# 3. Evaluate Model

In this module, we evaluate the final and shuffled baseline ML models.

After training the final and baseline models in [2.train_model](../2.train_model/), we use these models to predict the labels of the training and testing datasets.
These predictions are saved in [model_predictions.tsv](evaluations/model_predictions.tsv) and [shuffled_baseline_model_predictions.tsv](evaluations/shuffled_baseline_model_predictions.tsv) respectively.

In [confusion_matrices.ipynb](confusion_matrices.ipynb), we evaluate these 4 sets of predictions with a confusion matrix to see the true/false positives and negatives (see [sklearn.metrics.confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) for more details).
The confusion matrix data are saved to [confusion_matrices](evaluations/confusion_matrices).

In [F1_scores.ipynb](F1_scores.ipynb), we evaluate each model (final, shuffled baseline) on each dataset (train, test, etc) to determine phenotypic and weighted f1 scores.
F1 score measures the models precision and recall performance for each phenotypic class (see [sklearn.metrics.f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) for more details).
The f1 score data are saved to [F1_scores](evaluations/F1_scores).

In [class_PR_curves.ipynb](class_PR_curves.ipynb), we use [sklearn.metrics.precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) to derive the precision-recall curves for final and shuffled baseline models on the training and testing data subsets.
The precision recall curves and their data are saved to [class_precision_recall_curves](evaluations/class_precision_recall_curves/).

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

## Results

Each model's evaluations can be found in [evaluate_model.ipynb](evaluate_model.ipynb).
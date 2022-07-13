#!/usr/bin/env python
# coding: utf-8

# # Train and validate ML model for phenotypic classification of nuclei
# 
# ### Use training data with DeepProfiler features from [training_data.csv.gz](../1.format_data/data/training_data.csv.gz)
# 
# ### Import libraries
# 

# In[1]:


import pandas as pd
import numpy as np
import pathlib
from typing import Tuple, Any, List, Union

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import seaborn as sns


# ### Helper functions for data loading, training, and validation
# 

# In[2]:


def get_training_data(load_path: pathlib.Path) -> pd.DataFrame:
    """get DP training data from csv at load path

    Args:
        load_path (pathlib.Path): path to training data csv

    Returns:
        pd.DataFrame: training dataframe
    """
    # read dataset into pandas dataframe
    training_data = pd.read_csv(load_path, index_col=0)

    # remove training data with ADCCM class as this class was not used for classification in original paper
    training_data = training_data[
        training_data["Mitocheck_Phenotypic_Class"] != "ADCCM"
    ]

    # replace shape1 and shape3 labels with their correct respective classes
    training_data = training_data.replace("Shape1", "Binuclear")
    training_data = training_data.replace("Shape3", "Polylobed")

    return training_data


def get_X_y_data(load_path: pathlib.Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """generate X (features) and y (labels) dataframes from training data

    Args:
        load_path (pathlib.Path): path to training data csv

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: X, y dataframes
    """
    training_data = get_training_data(load_path)

    # all features from DeepProfiler have "efficientnet" in their column name
    morphology_features = [
        col for col in training_data.columns.tolist() if "efficientnet" in col
    ]

    # extract features
    X = training_data.loc[:, morphology_features].values

    # extract phenotypic class label
    y = training_data.loc[:, ["Mitocheck_Phenotypic_Class"]].values
    # make Y data
    y = np.ravel(y)

    # shuffle data because as it comes from MitoCheck same labels tend to be in grou
    X, y = shuffle(X, y, random_state=0)

    return X, y


# In[3]:


# load x (features) and y (labels) dataframes
load_path = pathlib.Path("../1.format_data/data/training_data.csv.gz")
X, y = get_X_y_data(load_path)

print(X.shape)
print(y.shape)

# create stratified data sets for k-fold cross validation
straified_k_folds = StratifiedKFold(n_splits=10, shuffle=False)


# In[4]:


# create logistic regression model with following parameters
log_reg_model = LogisticRegression(
    penalty="elasticnet", solver="saga", max_iter=100, n_jobs=-1, random_state=0
)


# In[5]:


# hypertune parameters with GridSearchCV
parameters = {"C": np.logspace(-3, 3, 7), "l1_ratio": np.linspace(0, 1, 11)}
#parameters = {"C": [1.0], "l1_ratio": [0.8]}
print(f"Parameters being tested: {parameters}")
grid_search_cv = GridSearchCV(
    log_reg_model, parameters, cv=straified_k_folds, n_jobs=-1
)
grid_search_cv = grid_search_cv.fit(X, y)


# In[6]:


print(f"Best parameters: {grid_search_cv.best_params_}")
print(f"Score of best estimator: {grid_search_cv.best_score_}")
log_reg_model = LogisticRegression(
    C=grid_search_cv.best_params_["C"],
    l1_ratio=grid_search_cv.best_params_["l1_ratio"],
    penalty="elasticnet",
    solver="saga",
    max_iter=100,
    n_jobs=-1,
    random_state=0,
)


# In[7]:


# create confusion matrix for logistic regression model after cross validation
y_pred = cross_val_predict(log_reg_model, X, y, cv=straified_k_folds, n_jobs=-1)
conf_mat = confusion_matrix(y, y_pred, labels=grid_search_cv.best_estimator_.classes_)

# display confusion matrix
plt.clf()
plt.rcParams["figure.figsize"] = [15, 15]
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_mat, display_labels=grid_search_cv.best_estimator_.classes_
)
disp.plot()
plt.xticks(rotation=90)
plt.show()


# In[8]:


# display precision vs phenotypic class bar chart
precisions = precision_score(y, y_pred, average=None)
precisions = pd.DataFrame(precisions).T
precisions.columns = columns=grid_search_cv.best_estimator_.classes_

sns.set(rc={"figure.figsize": (20, 8)})
plt.xlabel("Phenotypic Class")
plt.ylabel("Precision")
plt.title("Precision vs Phenotpyic Class")
plt.xticks(rotation=90)
ax = sns.barplot(data=precisions)


# In[9]:


coefs = np.abs(grid_search_cv.best_estimator_.coef_)
coefs = pd.DataFrame(coefs).T
coefs.columns = grid_search_cv.best_estimator_.classes_

print(coefs.shape)
coefs.head()


# In[10]:


# display heatmap of average coefs
sns.set(rc={"figure.figsize": (20, 10)})
ax = sns.heatmap(data=coefs.T)


# In[11]:


# display clustered heatmap of coefficients
ax = sns.clustermap(
    data=coefs.T, figsize=(20, 10), row_cluster=True, col_cluster=True
)
ax.fig.suptitle("Clustered Heatmap of Coefficients Matrix")


# In[12]:


# display density plot for coefficient values of each class
sns.set(rc={"figure.figsize": (20, 8)})
plt.xlim(-0.02, 0.1)
plt.xlabel("Coefficient Value")
plt.ylabel("Density")
plt.title("Density of Coefficient Values Per Phenotpyic Class")
ax = sns.kdeplot(data=coefs)


# In[13]:


# display average coefficient value vs phenotypic class bar chart
pheno_class_ordered = coefs.reindex(
    coefs.mean().sort_values(ascending=False).index, axis=1
)
sns.set(rc={"figure.figsize": (20, 8)})
plt.xlabel("Phenotypic Class")
plt.ylabel("Average Coefficient Value")
plt.title("Coefficient vs Phenotpyic Class")
plt.xticks(rotation=90)
ax = sns.barplot(data=pheno_class_ordered)


# In[14]:


# display average coefficient value vs feature bar chart
feature_ordered = coefs.T.reindex(
    coefs.T.mean().sort_values(ascending=False).index, axis=1
)
sns.set(rc={"figure.figsize": (500, 8)})
plt.xlabel("Deep Learning Feature Number")
plt.ylabel("Average Coefficient Value")
plt.title("Coefficient vs Feature")
plt.xticks(rotation=90)
ax = sns.barplot(data=feature_ordered)


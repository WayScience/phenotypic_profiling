#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import pathlib

from sklearn.metrics import f1_score
from joblib import load

import sys
sys.path.append("../utils")
from split_utils import get_features_data
from train_utils import get_dataset, get_X_y_data
from evaluate_utils import evaluate_model_cm, evaluate_model_score


# ### Load necessary data

# In[2]:


# specify results directory
results_dir = pathlib.Path("evaluations/")
results_dir.mkdir(parents=True, exist_ok=True)

# load features data from indexes and features dataframe
data_split_path = pathlib.Path("../1.split_data/indexes/data_split_indexes.tsv")
data_split_indexes = pd.read_csv(data_split_path, sep="\t", index_col=0)
features_dataframe_path = pathlib.Path("../0.download_data/data/training_data.csv.gz")
features_dataframe = get_features_data(features_dataframe_path)


# ### Evaluate best model

# In[3]:


model_dir = pathlib.Path("../2.train_model/models/")
log_reg_model_path = pathlib.Path(f"{model_dir}/log_reg_model.joblib")
log_reg_model = load(log_reg_model_path)


# ### Evaluate with training data

# In[4]:


training_data = get_dataset(features_dataframe, data_split_indexes, "train")
training_data


# In[5]:


y_train, y_train_pred = evaluate_model_cm(log_reg_model, training_data)


# In[6]:


evaluate_model_score(log_reg_model, training_data)


# ### Evaluate with testing data

# In[7]:


testing_data = get_dataset(features_dataframe, data_split_indexes, "test")
testing_data


# In[8]:


y_test, y_test_pred = evaluate_model_cm(log_reg_model, testing_data)


# In[9]:


evaluate_model_score(log_reg_model, testing_data)


# ### Evaluate with holdout data

# In[10]:


holdout_data = get_dataset(features_dataframe, data_split_indexes, "holdout")
X_holdout, y_holdout = get_X_y_data(holdout_data)
holdout_data


# In[11]:


y_holdout, y_holdout_pred = evaluate_model_cm(log_reg_model, holdout_data)


# In[12]:


evaluate_model_score(log_reg_model, holdout_data)


# ### Save trained model predicitions

# In[13]:


predictions = []

predictions.append(y_train)
predictions.append(y_train_pred)

predictions.append(y_test)
predictions.append(y_test_pred)

predictions.append(y_holdout)
predictions.append(y_holdout_pred)

predictions = pd.DataFrame(predictions)
predictions.index = ["y_train", "y_train_pred", "y_test", "y_test_pred", "y_holdout", "y_holdout_pred"]
predictions.to_csv(f"{results_dir}/model_predictions.tsv", sep="\t")


# ### Evaluate shuffled baseline model

# In[14]:


shuffled_baseline_log_reg_model_path = pathlib.Path(f"{model_dir}/shuffled_baseline_log_reg_model.joblib")
shuffled_baseline_log_reg_model = load(shuffled_baseline_log_reg_model_path) 


# ### Evaluate with training data

# In[15]:


y_train, y_train_pred = evaluate_model_cm(shuffled_baseline_log_reg_model, training_data)


# In[16]:


evaluate_model_score(shuffled_baseline_log_reg_model, training_data)


# ### Evaluate with testing data

# In[17]:


y_test, y_test_pred = evaluate_model_cm(shuffled_baseline_log_reg_model, testing_data)


# In[18]:


evaluate_model_score(shuffled_baseline_log_reg_model, testing_data)


# ### Evaluate with holdout data

# In[19]:


y_holdout, y_holdout_pred = evaluate_model_cm(shuffled_baseline_log_reg_model, holdout_data)


# In[20]:


evaluate_model_score(shuffled_baseline_log_reg_model, holdout_data)


# ### Save trained model predicitions

# In[21]:


predictions = []

predictions.append(y_train)
predictions.append(y_train_pred)

predictions.append(y_test)
predictions.append(y_test_pred)

predictions.append(y_holdout)
predictions.append(y_holdout_pred)

predictions = pd.DataFrame(predictions)
predictions.index = ["y_train", "y_train_pred", "y_test", "y_test_pred", "y_holdout", "y_holdout_pred"]
predictions.to_csv(f"{results_dir}/shuffled_baseline_model_predictions.tsv", sep="\t")


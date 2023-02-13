#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import pathlib

from joblib import load

import matplotlib.pyplot as plt
import seaborn as sns


# ### Interpret best model

# In[2]:


model_dir = pathlib.Path("../2.train_model/models/")

log_reg_model_path = pathlib.Path(f"{model_dir}/log_reg_model.joblib")
log_reg_model = load(log_reg_model_path)


# ### Compile Coefficients Matrix

# In[3]:


coefs = np.abs(log_reg_model.coef_)
coefs = pd.DataFrame(coefs).T
coefs.columns = log_reg_model.classes_

print(coefs.shape)
coefs.head()


# ### Save Coefficients Matrix in Tidy Long Format

# In[4]:


coefs_save_path = pathlib.Path(f"coefficients/final_model_coefficients.tsv")
coefs_save_path.parent.mkdir(parents=True, exist_ok=True)

# restructure/rename dataframe to tidy long format (see preview below)
tidy_data = coefs.stack()
tidy_data = pd.DataFrame(tidy_data).reset_index(level=[0,1])
tidy_data.columns = ["Feature_Name", "Phenotypic_Class", "Value"]

# add efficientnet_ prefix to all feature names (DeepProfiler prefix for the model used to extract features)
tidy_data["Feature_Name"] = "efficientnet_" + tidy_data["Feature_Name"].astype(str)

tidy_data.to_csv(coefs_save_path, sep="\t")
tidy_data


# In[5]:


tidy_data


# ### Diagrams for interpreting coefficients

# In[6]:


# display heatmap of average coefs
plt.figure(figsize=(20, 10))
plt.title("Heatmap of Coefficients Matrix")
ax = sns.heatmap(data=coefs.T)


# In[7]:


# display clustered heatmap of coefficients
ax = sns.clustermap(data=coefs.T, figsize=(20, 10), row_cluster=True, col_cluster=True)
ax = ax.fig.suptitle("Clustered Heatmap of Coefficients Matrix")


# In[8]:


# display density plot for coefficient values of each class
sns.set(rc={"figure.figsize": (20, 8)})
plt.xlim(-0.02, 0.1)
plt.xlabel("Coefficient Value")
plt.ylabel("Density")
plt.title("Density of Coefficient Values Per Phenotpyic Class")
ax = sns.kdeplot(data=coefs)


# In[9]:


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


# In[10]:


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


# ### Interpret shuffled baseline model

# In[11]:


shuffled_baseline_log_reg_model_path = pathlib.Path(f"{model_dir}/shuffled_baseline_log_reg_model.joblib")
shuffled_baseline_log_reg_model = load(shuffled_baseline_log_reg_model_path)


# ### Save Coefficients Matrix in Tidy Long Format

# In[12]:


coefs_save_path = pathlib.Path(f"coefficients/shuffled_baseline_model_coefficients.tsv")

# restructure/rename dataframe to tidy long format (see preview below)
tidy_data = coefs.stack()
tidy_data = pd.DataFrame(tidy_data).reset_index(level=[0,1])
tidy_data.columns = ["Feature_Name", "Phenotypic_Class", "Value"]

# add efficientnet_ prefix to all feature names (DeepProfiler prefix for the model used to extract features)
tidy_data["Feature_Name"] = "efficientnet_" + tidy_data["Feature_Name"].astype(str)

tidy_data.to_csv(coefs_save_path, sep="\t")
tidy_data


# ### Compile Coefficients Matrix

# In[13]:


coefs = np.abs(shuffled_baseline_log_reg_model.coef_)
coefs = pd.DataFrame(coefs).T
coefs.columns = shuffled_baseline_log_reg_model.classes_

print(coefs.shape)
coefs.head()


# ### Diagrams for interpreting coefficients

# In[14]:


# display heatmap of average coefs
plt.figure(figsize=(20, 10))
plt.title("Heatmap of Coefficients Matrix")
ax = sns.heatmap(data=coefs.T)


# In[15]:


# display clustered heatmap of coefficients
ax = sns.clustermap(data=coefs.T, figsize=(20, 10), row_cluster=True, col_cluster=True)
ax = ax.fig.suptitle("Clustered Heatmap of Coefficients Matrix")


# In[16]:


# display density plot for coefficient values of each class
sns.set(rc={"figure.figsize": (20, 8)})
plt.xlim(-0.02, 0.1)
plt.xlabel("Coefficient Value")
plt.ylabel("Density")
plt.title("Density of Coefficient Values Per Phenotpyic Class")
ax = sns.kdeplot(data=coefs)


# In[17]:


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


# In[18]:


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

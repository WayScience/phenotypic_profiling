#!/usr/bin/env python
# coding: utf-8

# ## Apply tsne to each feature dataset
# 
# Input: Data representations
# Output: tsne embeddings for plotting

# In[1]:


import pathlib
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import sys

sys.path.append("../utils")
from split_utils import get_features_data


# In[2]:


np.random.seed(1234)


# In[3]:


output_file = pathlib.Path("evaluations", "tsne_embeddings.tsv.gz")


# In[4]:


# load x (features) and y (labels) dataframes
labeled_data_path = pathlib.Path("../0.download_data/data/labeled_data__ic.csv.gz")
labeled_data = get_features_data(labeled_data_path).reset_index(drop=True)

print(labeled_data.shape)
labeled_data.head(3)


# In[5]:


metadata_columns = [
    "Mitocheck_Phenotypic_Class",
    "Cell_UUID",
    "Location_Center_X",
    "Location_Center_Y",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Frame",
    "Metadata_Site",
    "Metadata_Plate_Map_Name",
    "Metadata_DNA",
    "Metadata_Gene",
    "Metadata_Gene_Replicate",
    "Metadata_Object_Outline",
]

feature_groups = [
    "CP",
    "DP",
    "CP_DP"
]


# ## Apply tSNE
# 
# We test different perplexities ranging from 2 to 300.
# 
# From scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html):
# 
# > The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significantly different results. The perplexity must be less than the number of samples.
# 
# We do not know what the appropriate value of perplexity is for our dataset, so we will test several.

# In[6]:


tsne_embedding_df = []

# Select a wide range of values. The initial paper suggests between 5-50.
# We want to see how this wide range impacts the groupings.
list_of_perplexities = [2, 10, 15, 30, 40, 60, 80, 100, 150, 300]

for perplexity in list_of_perplexities:
    for feature_group in feature_groups:
        # Compile dataset
        if feature_group == "CP_DP":
            input_data_to_tsne = labeled_data.drop(metadata_columns, axis=1)
        else:
            input_data_to_tsne = labeled_data.loc[:, labeled_data.columns.str.startswith(feature_group)]
    
        tsne_model = TSNE(
            n_components=2,
            learning_rate='auto',
            init='random',
            perplexity=perplexity
            )
    
        tsne_embedding = pd.DataFrame(
            tsne_model.fit_transform(input_data_to_tsne)
        )
    
        tsne_embedding.columns = ['tsne_x', 'tsne_y']
    
        tsne_embedding_df.append(
            pd.concat([
                labeled_data.loc[:, metadata_columns],
                tsne_embedding
                ], axis=1
                )
                .assign(
                    feature_group=feature_group,
                    perplexity=perplexity
                )
            )
    
tsne_embedding_df = pd.concat(tsne_embedding_df, axis=0).reset_index(drop=True)

print(tsne_embedding_df.shape)
tsne_embedding_df.head()


# In[7]:


# Output file
tsne_embedding_df.to_csv(output_file, sep="\t", index=False)


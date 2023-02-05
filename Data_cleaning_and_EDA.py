# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:01:41 2023

@author: acnzi
"""

import ETL as etl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


'''
2. Data Cleaning

Here, we shall try to clean the data by looking for null values, and find ways
to deal with the missing data
'''
#print(etl.tennis_updated2.info())
#print(etl.tennis_updated2.isnull().sum())

'''
No null values, therefore we can continue working on our data
'''

'''
3.Exploratory Data Analysis and Feature engineering

This stage is used to prepare for model building/analysis. Since we want this 
to be a classification algorithm, we want the Ordinal Variable to be arranged 
in an order such that different rankings have a particular numerical class


'''
#print(etl.tennis_updated2.Ranking.max())
#print(etl.tennis_updated2.Ranking.min())

'''
The values are shown to range from 3 to 1443. Therefore, we can make 20 divisions
for each ranking such that the new ranking feature shows values between 1 and 20
showing the players rank from 1 to 1450. We can do this by binning the data
'''

# Select the ranking integers from the tennis_updated2 table
data = etl.tennis_updated2['Ranking']

# Define the bin edges
bin_edges = np.arange(0, 1461, 73)

# Use the cut function to categorize the data into 20 ordinal categories
etl.tennis_updated2['Ranks'] = pd.cut(data, bin_edges, labels=False, right=False) + 1
#print(etl.tennis_updated2.Ranks)

'''

4. Feature engineering

Filter methods
We can see that 'Ranks' is a Categorical Variable and most of the other statistics 
we plan to use are quantitative variables. We can try and visualise the associations
between them visually to see which ones are associated to ranks

Afterwards, we can perform some filter methods which can help remove data which
is of little importance, rather data with no correlation or covariance
'''

X = etl.tennis_updated2.drop(columns=['Player', 'Ranking','Ranks','index','Ranks'])
y = etl.tennis_updated2.Ranks

#1.Remove the low variance features

from sklearn.feature_selection import VarianceThreshold
 
selector = VarianceThreshold(threshold=0)  # 0 is default
 
print(selector.fit_transform(X))

# Specify `indices=True` to get indices of selected features
print(selector.get_support(indices=True))

# Use indices to get the corresponding column names of selected features
num_cols = list(X.columns[selector.get_support(indices=True)])
 
print(num_cols)

X_num = X[num_cols]

'''
As we can see, we have removed features with a variance of less than 0. There
is no feature with a variance less than zero thus we can resume reducing the 
features

However, these features are very many(20) and some may contain important information
that I may not want to lose. For this reason, I have decided to use Principal 
Component Analysis to reduce the dimensionality of the data while still retaining
important information about the data. Moreover, I will use PCA to reduce the 
computational costs of running the ML model
'''

#Standardize the data
mean = X.mean(axis = 0)
sttd = X.std(axis=0)
data_standardized = (X-mean)/sttd

#Perform eigendecomposition
from sklearn.decomposition import PCA
pca = PCA()
components = pca.fit(data_standardized).components_
components = pd.DataFrame(components).transpose()
components.index = X.columns
print(components)

var_ratio = pca.explained_variance_ratio_
var_ratio = pd.DataFrame(var_ratio).transpose()
print(var_ratio)

#Project the data onto principal axes

#only keep 3 PC's
pca = PCA(n_components = 3)

#transform the data using the first 3 PC's
data_pcomp = pca.fit_transform(data_standardized)

#transform into a dataframe
data_pcomp = pd.DataFrame(data_pcomp)

#rename columns
data_pcomp.columns = ['PC1', 'PC2','PC3']

print(data_pcomp.head())




 

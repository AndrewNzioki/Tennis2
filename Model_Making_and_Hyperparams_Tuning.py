# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:12:09 2023
@author: acnzi
"""
import Data_cleaning_and_EDA as eda
import ETL as etl
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
Previously we had used Principal Component Analysis to reduce the data's 
dimenisionality. Now we can use a simple machine learning model(in this case
SVM(Support Vector Machine) to fit the data)
'''


#print(eda.data_pcomp.head())

X = eda.data_pcomp
y = etl.tennis_updated2.Ranks

#Train-Test Validation Split

X_train,X_test,y_train,y_test = train_test_split(X,y, train_size = 0.8, 
                                            test_size = 0.2,random_state= 6)

classifier = svm.SVC()
classifier.fit(X_train,y_train)
scores = classifier.score(X_test, y_test)
print("The score is " + str(scores))

'''
As we can see, the score is very low, which shows us that there was little to no
association between the variables as discussed earlier. Therefore, the features 
shown in the dataset are not good enough to predict the ranking of a player as it
will bring inconsistent results. For more verification, we can make a confusion
matrix to show us how much inconsistency lies between the training set and 
predicted results.
'''

y_true = y_test
y_pred = classifier.predict(X_test)

print(y_true)
print(y_pred)

y_array = np.array(y_true)
print(y_array)
print(confusion_matrix(y_true, y_array))

'''
From seeing the confusion matrix and the values of the arrays of the predicted
and true values, we can see that the model poorly predicts the ranking of 
the players based on the features and data provided

It seems the issue lies in finding the variance between the features and our 
target variable. We can visualize this variance using boxplots and by creating 
plotes for correlation between the features and target variables

'''

#First we rebin the data so that it is easily visualizable in the boxplot

data_copy = etl.tennis_updated2.copy()
data_copy_ranking = data_copy['Ranking']

bin_edges = np.arange(0, 1461, 292)

data_copy['Ranks'] = pd.cut(data_copy_ranking, bin_edges, labels=False, right=False) + 1
print(data_copy.Ranks)

sns.boxplot(data=data_copy, x='Ranks' ,y = 'FirstServe')
plt.show()

'''
As we can see from the boxplot on the first serve, there is a lot of overlap 
between the feaatures. Therefore, there is little to no variance between
the target variable and this particular feature

However, since we don't want too many graphs, we can visualize association by 
finding the correlation between all the features and the target variable
'''

corr_matrix = data_copy.corr()
 
# Isolate the column corresponding to `exam_score`
corr_target = corr_matrix[['Ranks']].drop(labels=['Ranks'])
 
sns.heatmap(corr_target, annot=True, fmt='.3', cmap='RdBu_r')
plt.show()

'''
As shown from the heatmap, the correlation between all the features and the 
target variable is on average less than 0.3 and -0.3, which shows the features
will be poor predictors of the target variable, subsequently shown by the score of the 
ML model made.
'''

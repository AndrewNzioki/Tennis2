# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:12:09 2023

@author: acnzi
"""
import Data_cleaning_and_EDA as eda
import ETL as etl
from sklearn.model_selection import train_test_split
from sklearn import svm

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
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 13:01:41 2023

@author: acnzi
"""

import ETL as etl

'''
2. Data Cleaning

Here, we shall try to clean the data by looking for null values, and find ways
to deal with the missing data
'''
print(etl.tennis_updated2.info())
print(etl.tennis_updated2.isnull().sum())

'''
No null values, therefore we can continue working on our data
'''

'''
3.Exploratory Data Analysis


'''
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 09:28:11 2023

@author: acnzi
"""

import pandas as pd
import logging
import sys


logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


"""
In this project, we want to predict the ranking tier of a player based on the 
variables we have been given. We have many features so let us start

we want this to be a classification algorithm, so our y values will need to
be in classes, therefore we will need to encode the ranking variables in a way
that shows these classifications


"""
'''
Step 1: Extract, load and Transform Data

'''

tennis = pd.read_csv("tennis_stats.csv")
#print(tennis.head())



#print info on the table
#print(tennis.info())

#there are no null values, so we can continue

'''
We now want to do a calculation which lists the stats of a player over a period  
of years and groups them such that each player has a ranking and we can see all
the players with their rankings
'''


#print(tennis['Player'].nunique())
#438 unique tennis players

tennis.sort_values(by='Player', ascending=True, inplace=True)
#print(tennis.head())

#print(tennis.Player.unique())

# group the dataframe by 'Player' and 'Ranking' columns
grouped = tennis.groupby(by=['Player', 'Ranking'])

# calculate the mean score for each group
tennis_averages = grouped[['FirstServe','FirstServePointsWon',
'FirstServeReturnPointsWon', 'SecondServePointsWon','SecondServeReturnPointsWon'
, 'BreakPointsConverted','BreakPointsSaved','ReturnGamesWon','ReturnPointsWon'
,'TotalPointsWon','TotalServicePointsWon']].mean()

# display the mean scores for each group
#print(tennis_averages)

#Do the same for the sums
tennis_sums = grouped[['Aces','BreakPointsFaced','BreakPointsOpportunities',
'DoubleFaults','ReturnGamesPlayed','ServiceGamesPlayed','Wins','Losses',
'Winnings']].sum()
#print(tennis_sums)

'''
We also know that this data is for players over a number of years, so we also 
count the number of years played
'''
tennis_count = grouped['Year'].count()
#print(tennis_count)

#Now we merge all the data
tennis_updated = pd.merge(tennis_averages, tennis_sums, 
                          on=['Player', 'Ranking'], how='inner').reset_index()
tennis_updated2 = pd.merge(tennis_updated, tennis_count, 
                          on=['Player', 'Ranking'], how='inner').reset_index()














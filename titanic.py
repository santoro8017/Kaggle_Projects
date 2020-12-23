# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 20:07:37 2020

@author: Matt Santoro

Base on titanic data provided by Kaggle, try to predict who will survive
"""

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import style
from sklearn import datasets, svm, tree, preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.ensemble as ske
from sklearn.svm import SVC

import random

train_df = pd.read_csv("titanic_train.csv")


#initial data exploration, will help understand the variables in the data set
train_df.info()
train_df.describe()
train_df.head()

#interesting data points, only 38% of the passengers survived
#some data con be better used by converting to numberic: sex and embarked
#several data fields are missing values: age and cabin

#how much data is missing from each of the fields
total = train_df.isnull().sum().sort_values(ascending=False)
percent1 = train_df.isnull().sum()/train_df.isnull().count() * 100
percent2 = (round(percent1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent2], axis=1, keys=['Total', '%'])
missing_data.head()

#the age and cabin information that's missing is probably too large to impute

#explore how the social factors may have effected survival
train_df.groupby('Pclass').mean()
classSexGroup = train_df.groupby(['Pclass', 'Sex']).mean()

#unsurprisingly, higher classes had higher survival rate as well as women and children

#it seems the size of the group you were traveling with had an effect on survival as well
sns.factorplot('Parch', 'Survived', data = train_df, aspect = 2.5)
sns.factorplot('SibSp', 'Survived', data = train_df, aspect = 2.5)

#data preparation
#some data just won't be relevant or has so many missing values we can't use it
train_df = train_df.drop(['Ticket','Cabin','Name'], axis = 1)

most_common = train_df['Embarked'].mode()
train_df['Embarked'].iloc[train_df[train_df['Embarked'].isnull()].index] = most_common[0]

train_df['Sex'].iloc[train_df[train_df['Sex']=='male'].index]=1
train_df['Sex'].iloc[train_df[train_df['Sex']=='female'].index]=2

train_df['Embarked'].iloc[train_df[train_df['Embarked']=='S'].index]=1
train_df['Embarked'].iloc[train_df[train_df['Embarked']=='C'].index]=2
train_df['Embarked'].iloc[train_df[train_df['Embarked']=='Q'].index]=3

#have to drop rows with missing ages, there are too many values missing to impute
train_df = train_df.dropna()

#let's try out a classifier now to see what happens
y_train = train_df['Survived']
x_train = train_df.drop(['Survived'], axis=1)

#Test a random forest classifier with cross validation
rf = RandomForestClassifier(n_estimators = 100)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, x_train, y_train, cv=10, scoring = "accuracy")
print("Scores: ", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

#average score of 80% isn't bad

#what factors are the most relevant?
rnd_clf = RandomForestClassifier(n_estimators = 100)
rnd_clf.fit(x_train,y_train)
importances = pd.DataFrame({'feature':x_train.columns, 'importance':np.round(rnd_clf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.plot.bar()

#let's see if we can improve performance with a voting classifier
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train_df, test_size=.2, random_state = 42)

x_train_set = train_set.drop(['Survived'],axis=1)
y_train_set = train_set['Survived']
x_test_set = test_set.drop(['Survived'],axis=1)
y_test_set = test_set['Survived']

rnd_clf = RandomForestClassifier(n_estimators = 100)
log_clf = LogisticRegression()
svm_clf = SVC()

voting_clf = VotingClassifier( estimators=[('lr', log_clf), ('rf',rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(x_train_set, y_train_set)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(x_train_set, y_train_set)
    y_pred = clf.predict(x_test_set)
    print(accuracy_score(y_test_set, y_pred))
    
#the voting classifier does slightly worse then the random forest
#lets look at a confusion matrix for some more insight into performance
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(rnd_clf, x_train, y_train, cv=3)
confusion_matrix(y_train, predictions)
confusion_matrix

##The first row of the confusion matrix is about the "not-survived-predicitons"
##366 passengers were correctly classified as "not survived" and 58 were incorrectly classified as "not survived"
##Second row is about the "survived-predictions"
##101 were incorrectly classifed as "survived" and 189 were correctly classified

##Create csv for submission to kaggle
# =============================================================================
test_df = pd.read_csv("titanic_test.csv")
test_df['Embarked'].iloc[test_df[test_df['Embarked'].isnull()].index] = most_common[0]
 
test_df = test_df.drop(['Ticket','Cabin','Name'], axis = 1)
test_df['Sex'].iloc[test_df[test_df['Sex']=='male'].index]=1
test_df['Sex'].iloc[test_df[test_df['Sex']=='female'].index]=2
 
test_df['Embarked'].iloc[test_df[test_df['Embarked']=='S'].index]=1
test_df['Embarked'].iloc[test_df[test_df['Embarked']=='C'].index]=2
test_df['Embarked'].iloc[test_df[test_df['Embarked']=='Q'].index]=3

#submission must have predictions for all rows so we are going to impute some values for missing ages
#
test_df['Age'].iloc[test_df[test_df['Age'].isnull()].index] = round(test_df['Age'].mean(skipna=True))
test_df['Fare'].iloc[test_df[test_df['Fare'].isnull()].index] = test_df['Fare'].mean(skipna=True)

pred = rnd_clf.predict(test_df)

output = pd.concat([test_df['PassengerId'], pd.DataFrame(pred,columns=['Survived'])], axis = 1 )
output.to_csv("C:\\Users\\steph\\Documents\\Matt\\Kaggle\\new_test.csv", index = False)

#submitted score was 76%
# =============================================================================

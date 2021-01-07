# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 20:06:10 2021

@author: 
"""

import numpy as np
import pandas as pd

#read file
df = pd.read_csv('..\\Kaggle\\Kaggle_tweets.train.csv')

#look at the file
print(df.head())

#i'm just going to focus on the text and target for the first go 
df.drop(['keyword', 'location'], axis=1, inplace=True)

#check for missing text fields
print(df.isnull().sum())

#no null text fields, now check for ones that might just be a space
blanks = []
for a,i,txt,tgt in df.itertuples():
    if type(txt)==str:
        if txt.isspace():
            blanks.append(i)
            
#no blanks either
print(len(blanks))

#let's look at the distribution
print(df['target'].value_counts())

# 0-not a disaster --> count was 4342
# 1-real disaster --> count was 3271

from sklearn.model_selection import train_test_split

#split training data to test for cross validation
X = df['text']
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.svm import LinearSVC

#setup a pipeline to vectorize tweets and use a Support Vector Classifier
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()),])

#train and predict
text_clf.fit(X_train, y_train)
predictions = text_clf.predict(X_test)

from sklearn import metrics

#check out the performance metrics, scored around 80%, not bad
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

#repeat to train on entire training data set
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC()),])

text_clf.fit(X, y)


#read file
df_test = pd.read_csv('..\\Kaggle\\Kaggle_tweets.test.csv')

#look at the file
print(df_test.head())

#i'm just going to focus on the text and target for the first go 
df_test.drop(['keyword', 'location'], axis=1, inplace=True)

#check for missing text fields
print(df_test.isnull().sum())

#no null text fields, now check for ones that might just be a space
blanks = []
for a,i,txt in df_test.itertuples():
    if type(txt)==str:
        if txt.isspace():
            blanks.append(i)

#no blanks either
print(len(blanks))            

#predict test data
predictions = text_clf.predict(df_test['text'])

#create submission 
df_test['target'] = pd.DataFrame(predictions)
df_test.drop(['text'], axis=1, inplace=True)
df_test.to_csv('tweet_Answer_submission.csv', index=False)

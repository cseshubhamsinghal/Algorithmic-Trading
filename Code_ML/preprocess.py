##_____________________________
## import the desired libraries

import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests
import time


##________________________________________
## processing data for machine learning

def process_data_for_labels(ticker):
    hm_days = 7                                                 # how many days (means for this much future days it will process)
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)    # read csv file which is joined based on adj column
    tickers = df.columns.values.tolist()                        # values of columns are put into the list named as tickers
    df.fillna(0, inplace=True)                                  # if anywhere value is not known, it is filled with 0

    for i in range(1, hm_days+1):                               # for the next seven days , values are calculated
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    #print(tickers)
    print(df)
    return tickers, df

process_data_for_labels('XOM')

##____________________________________________________________________________________________________
## creating target function (if it exceeds 2% then its a buy, if it falls 2% then its a sell else hold)

def buy_sell_hold(*args):                                       # setting the labels for each row i.e buy, sell, hold
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


##_____________________________________
## creating labels for machine learning
from collections import Counter

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))
#    print(df.head)
    vals = df['{}_target'.format(ticker)].values.tolist()       # target column values are fetched and put into list named as vals
    str_vals = [str(i) for i in vals]                           
#    print(vals)
    print('Data spread:',Counter(str_vals))                     # it will display the number of columns with 1,0,-1 labels


    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    #print(df.head())

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values                      # here all the 505 column values are seen as features and put into X for each row
    y = df['{}_target'.format(ticker)].values   # here the target column values are fetched and put into Y for each row

    #print (len(X[1]))
    #print(df_vals.head())

    return X,y,df

#extract_featuresets('XOM')


##__________________________________________
## importing the machine learning classifier

from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import metrics

from sklearn.linear_model import LogisticRegression, SGDClassifier
##_________________________________________
## training and testing the machine learning classifier

def do_ml(ticker):
    
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    
##    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.25)
    
    ## instantiate the classifier
    clf = neighbors.KNeighborsClassifier()
    #clf1 = LogisticRegression()
    

    ## fit the classifier over training set
    clf.fit(X_train, y_train)
    #clf1.fit(X_train,y_train)

##    save_classifier = open("pickled_algos/XOMtrainedclassifier.pickle","wb")
##    pickle.dump(clf, save_classifier)
##    save_classifier.close()
    
    ## testing the classifier over test set
    #y_predict = clf.predict(X_test)
    y_predict = clf.predict(X_test)
    print('predicted class counts:',Counter(y_predict))
    ##print('pred',predictions)

    ##print("model accuracy:", metrics.accuracy_score(y_test, y_predict))
    
    ##print()
    
    return

#do_ml('XOM')
##do_ml('AAPL')
##do_ml('ABT')

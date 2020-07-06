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

##_________________________________
## get tickers of s&p 500 companies

##def save_sp500_tickers():
##    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
##    soup = bs.BeautifulSoup(resp.text, 'lxml')
##    table = soup.find('table', {'class': 'wikitable sortable'})
##    tickers = []
##    for row in table.findAll('tr')[1:]:
##        ticker = row.findAll('td')[0].text
##        tickers.append(ticker)
##    with open("sp500tickers.pickle", "wb") as f:
##        pickle.dump(tickers, f)
##    return tickers

##save_sp500_tickers()

##_______________________________________
## get dataset of these s&p 500 companies

##def get_data_from_yahoo(reload_sp500=False):
##    
##    if reload_sp500:
##        tickers = save_sp500_tickers()
##    else:
##        with open("sp500tickers.pickle", "rb") as f:
##            tickers = pickle.load(f)
##    if not os.path.exists('stock_dfs'):
##        os.makedirs('stock_dfs')
##
##    start = pd.to_datetime(dt.datetime(2017, 1, 1))
##    end = pd.to_datetime(dt.datetime.now())
##
##    for ticker in tickers:
##        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
##            print(ticker)
##            df = web.DataReader(ticker, 'yahoo', start, end)
##            df.reset_index(inplace=True)
##            df.set_index("Date", inplace=True)
##            ##df = df.drop("Symbol", axis=1)
##            df.to_csv('stock_dfs/{}.csv'.format(ticker))
##        else:
##            print('Already have {}'.format(ticker))

##get_data_from_yahoo()

##_______________________________________
## compiling dataset of s&p 500 companies

##def compile_data():
##    with open("sp500tickers.pickle", "rb") as f:
##        tickers = pickle.load(f)
##
##    main_df = pd.DataFrame()        # main_df is defined as a pandas dataframe
##
##    for count, ticker in enumerate(tickers):
##        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
##        df.set_index('Date', inplace=True)
##
##        df.rename(columns={'Adj Close': ticker}, inplace=True)
##        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
##
##        if main_df.empty:
##            main_df = df
##        else:
##            main_df = main_df.join(df, how='outer')
##
##        if count % 10 == 0:
##            print(count)
##    print(main_df.head())
##    main_df.to_csv('sp500_joined_closes.csv')

##compile_data()

##____________________________________________________________________
## function to determine correlation of one company with other company
## and visualize the correlations on the heat map
    
def visualize_data():
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')
    data1 = df_corr.values              # this will give us numpy arrays of values in correlation tables
    fig1 = plt.figure()                 # build figure and axis
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)    # create heatmap
                                                        # cmap represents colormap
                                                        # here we have used RdY1Gn
                                                        # it is a color map that give us red for negative correlations
                                                        # green for positive correlations, and yellow for no correlations
    fig1.colorbar(heatmap1)
                                                        # set our x and y axis ticks so we know which companies are which
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)                            # tells the colormap that our range is going to be from -1 to positive 1
    plt.tight_layout()
    plt.show()


visualize_data()



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

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:        # create a pickle file of all the tickers dumped on the system from wikipedia
                                                        # name of the pickle file is sp500tickers.pickle
        pickle.dump(tickers, f)
    return tickers

##save_sp500_tickers()

##_______________________________________
## get dataset of these s&p 500 companies

def get_data_from_yahoo(reload_sp500=False):
    
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:        # all the tickers are put into the tickers variable 
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):                     # folder called stock_dfs is created
        os.makedirs('stock_dfs')

    start = pd.to_datetime(dt.datetime(2017, 1, 1))         # data is extracted from start to end date
    end = pd.to_datetime(dt.datetime.now())

    for ticker in tickers:                                  # for each ticker a new csv file is created , and data is put into that csv file.
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            print(ticker)
            df = web.DataReader(ticker, 'yahoo', start, end)    # data is extracted from yahoo finance
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)                  # date is set as index
            ##df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

##get_data_from_yahoo()

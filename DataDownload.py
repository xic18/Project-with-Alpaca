import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import alpaca
import pickle
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame


class DataDownload:

    def __init__(self,api_key,api_secret):
        # Set your API key and secret
        base_url = 'https://paper-api.alpaca.markets'  # Use paper trading base URL for testing
        # Initialize Alpaca API
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

        with open('./data/stockUniverse.pickle', 'rb') as f:
            self.stockUniverse = pickle.load(f)
        # self.stockUniverse=['ABBV', 'ACN', 'AEP', 'AIZ', 'ALLE', 'AMAT', 'AMP', 'AMZN', 'AVB',
        #          'AVY', 'AXP', 'B', 'BDX', 'BMY', 'BR', 'CARR', 'CDW', 'CE', 'CHTR',
        #          'CNC', 'CNP', 'COP', 'CTAS', 'CZR', 'DG', 'DPZ', 'DXC', 'FTV', 'GOOG',
        #          'GPC', 'HIG', 'HST', 'JPM', 'KR', 'META', 'OGN', 'PG', 'PLD', 'PPL',
        #          'PRU', 'PYPL', 'R', 'ROL', 'ROST', 'UNH', 'URI', 'V', 'VRSK', 'WRK',
        #          'XOM']

    def first_download(self,start_date,end_date,output_path='./data/raw_data/raw_data.csv'):
        data = self.api.get_bars(self.stockUniverse, TimeFrame.Day,start_date, end_date, adjustment='raw').df
        data['date'] = data.index
        data['date'] = data['date'].apply(lambda x: x.to_pydatetime().date())
        data = data.reset_index(drop=True)
        TradingDay = sorted(np.unique(data['date']))
        raw_data = pd.DataFrame()
        for ticker in self.stockUniverse:
            df = data[data['symbol'] == ticker].copy()
            df = df.set_index('date')
            for idx in TradingDay:
                if idx not in df.index:
                    df.loc[idx] = np.nan
            df = df.loc[TradingDay].sort_index().reset_index()
            df['symbol'] = ticker
            raw_data = pd.concat([raw_data, df])
        raw_data = raw_data.sort_values(['date', 'symbol'])
        raw_data.to_csv(output_path, index=False)
        return raw_data

    def update_to_today(self,last_date,today=pd.Timestamp('now').date(),delay=1,
                        history_path='./data/raw_data/raw_data.csv',output_path='./data/raw_data/raw_data.csv'):
        history_data = pd.read_csv(history_path)
        history_data['date']=pd.to_datetime(history_data['date']).dt.date
        if last_date!=max(history_data['date']):
            raise DataDownloadError


        # update
        data = self.api.get_bars(self.stockUniverse, TimeFrame.Day,last_date + datetime.timedelta(days=1),
                                       today - datetime.timedelta(days=delay), adjustment='raw').df
        data['date'] = data.index
        data['date'] = data['date'].apply(lambda x: x.to_pydatetime().date())
        data = data.reset_index(drop=True)
        TradingDay = sorted(np.unique(data['date']))
        raw_data = pd.DataFrame()
        for ticker in self.stockUniverse:
            df = data[data['symbol'] == ticker].copy()
            df = df.set_index('date')
            for idx in TradingDay:
                if idx not in df.index:
                    df.loc[idx] = np.nan
            df = df.loc[TradingDay].sort_index().reset_index()
            df['symbol'] = ticker
            raw_data = pd.concat([raw_data, df])
        raw_data = raw_data.sort_values(['date', 'symbol'])
        raw_data = pd.concat([history_data, raw_data])
        raw_data.to_csv(output_path, index=False)
        return raw_data


class DataDownloadError(Exception):
    def __init__(self):
        pass

    def __str__(self):
        return repr('wrong last_date')

if __name__=='__main__':
    DD=DataDownload(api_key='PKMKDK2DPCPTPSVGQZ6W',api_secret='Tm3JcjKbPHfxOS9eQZwdRfSVN5uA94LIeg7bgmvQ')
    data1=DD.first_download(start_date=datetime.date(2024, 2, 13)-datetime.timedelta(days=146),
                      end_date=datetime.date(2024, 2, 13)-datetime.timedelta(days=21))
    data2=DD.update_to_today(last_date=datetime.date(2024, 2, 13)-datetime.timedelta(days=21))

    DD.first_download(start_date=datetime.date(2024, 2, 13) - datetime.timedelta(days=146),
                      end_date=datetime.date(2024, 2, 13)- datetime.timedelta(days=1))




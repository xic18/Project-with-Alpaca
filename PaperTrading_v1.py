from DataDownload import DataDownload
from BackTest import Backtest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
import alpaca_trade_api as tradeapi
import time

warnings.filterwarnings("ignore", category=FutureWarning)

# Set your Alpaca API key and secret
API_KEY = 'PKMKDK2DPCPTPSVGQZ6W'
API_SECRET = 'Tm3JcjKbPHfxOS9eQZwdRfSVN5uA94LIeg7bgmvQ'
BASE_URL = 'https://paper-api.alpaca.markets'  # Use paper trading base URL for testing

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
account=api.get_account()

# 1.update data
DD=DataDownload(api_key='PKMKDK2DPCPTPSVGQZ6W',api_secret='Tm3JcjKbPHfxOS9eQZwdRfSVN5uA94LIeg7bgmvQ')
DD.first_download(start_date=datetime.date(2024, 2, 13) - datetime.timedelta(days=146),
                  end_date=datetime.date(2024, 2, 13)- datetime.timedelta(days=1))

# 2.generate portfolio from last day
# 每天收盘后更新data以及计算下一日的factor
today=pd.Timestamp('now').date()
BT=Backtest(start_day=today-datetime.timedelta(days=11),end_day=today)
universe=BT.universe
TradingDay=BT.TradingDay
data=BT.raw_data.copy()
data = data[(data['Date'] >= TradingDay[0]) & (data['Date'] <= TradingDay[-1])]
TradingDay=sorted(np.unique(data['Date']))
# 因子逻辑
data['logprice']=np.log(data['close'])
data['ret5']=data.groupby('Ticker')['logprice'].diff(5).shift(0)
data['factor']=data['ret5']
BT.factor_preprocess(data)
lastest_factor=data[data['Date']==TradingDay[-1]][['Ticker','factor']]
lastest_factor['value']=lastest_factor['factor']*int(float(account.effective_buying_power))


def find_position(x):
    try:
        info=api.get_position(x)
        return info['qty']*info['current_price']
    except:
        return 0

# 3.run following code in trading time every day
split_num = 3 # 拆单数量
for k in range(split_num):
    lastest_factor['current_position'] = lastest_factor['Ticker'].apply(lambda x: find_position(x))
    for i in range(len(lastest_factor)):
        ticker = lastest_factor['Ticker'].iloc[i]
        change_value = (lastest_factor['value'] - lastest_factor['current_position']).iloc[i]
        current_price = api.get_bars(ticker, '1Min', limit=1).df['close'].iloc[0]
        change_qty = int(change_value / current_price / (split_num - k))

        if change_qty > 0:
            api.submit_order(symbol=ticker, qty=change_qty, side='buy', type='market', time_in_force='gtc')
        elif change_qty < 0:
            api.submit_order(symbol=ticker, qty=-change_qty, side='sell', type='market', time_in_force='gtc')

    # break

print('successfully trade all stocks')
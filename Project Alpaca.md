# Project: Alpaca

Chen Xi 

UNI: cx2320

Github link: https://github.com/xic18/Project-with-Alpaca.git

## 1. Data Preparation and Storage

In this project we assume we can assign a given stock universe to trade. Therefore, we need to download market data from Alpaca API. The methods have been implemented and wrapped into class `DataDownload` , which supports `first_download` and `update_to_today`.

The first method takes start date and end date as parameters, return the market historic data for given universe (which is storage in pickle file: `stockUniverse.pickle`). Data will be restored into a csv file, as shown below.

![image-20240214005839056](C:\Users\xic\AppData\Roaming\Typora\typora-user-images\image-20240214005839056.png)

The second method will update data and store updated data into csv. For efficiency, only new data will be download, so there should be no overlap in dates.

```python
from DataDownload import DataDownload # import my toolkit

DD=DataDownload(api_key='PKMKDK2DPCPTPSVGQZ6W',api_secret='Tm3JcjKbPHfxOS9eQZwdRfSVN5uA94LIeg7bgmvQ')
# 1.first download
data1=DD.first_download(start_date=datetime.date(2024, 2, 13)-datetime.timedelta(days=146),
                      end_date=datetime.date(2024, 2, 13)-datetime.timedelta(days=21))
# 2.each time for updating, just download data after last date
data2=DD.update_to_today(last_date=datetime.date(2024, 2, 13)-datetime.timedelta(days=21))

# or direclty download data tile today
DD.first_download(start_date=datetime.date(2024, 2, 13) - datetime.timedelta(days=146),
                      end_date=datetime.date(2024, 2, 13)- datetime.timedelta(days=1))
```



## 2. Strategy Backtest

As known to all, momentum factors have been tested effective in historical data. Whereas these kinds of strategy became invalid in recent years. So my following strategy is a brief study for  short term momentum factor, to testify if we can still we can apply this strategy to achieve positive return.

Here we choose a stock universe with size 50 for convenience, and consider long-short strategy. And we evaluate some metrics like Sharpe and IC.

About toolkit:

Backtesting tools have been wrapped into a class named `Backtest`. This object provides several mode to test a factor. Like we can choose if considering transaction cost, if long only strategy or long-short strategy, what holding period we take.

```python
from BackTest import Backtest # import my toolkit

backtest=Backtest(holding_interval=(1,2),CostRate=0.00,mask_overlap=True)
# factor=backtest.gen_benchmark('equal_weight_long')
# factor=backtest.gen_benchmark('random_long')
factor=backtest.gen_benchmark('random_long_short')
backtest.quick_backtest_2(factor)
```

The following is an example of random weight long-short factor. Each day we consider a part of long stock with total weight 1, and a part of short stocks with total weight -1. The results include: annual return / annual sharpe / winning rate / ICIR.

Following graphs show the accumulative return for long/short, turnover rate, ticker number for long/short, cumulative IC for long/short    

![image-20240214011205107](C:\Users\xic\AppData\Roaming\Typora\typora-user-images\image-20240214011205107.png)

Apply this framework to study a simple momentum factor called RET5, which represents the accumulative return for individual stocks. There is a slightly positive return for a long-short portfolio, but not significant.

![image-20240214011824563](C:\Users\xic\AppData\Roaming\Typora\typora-user-images\image-20240214011824563.png)

## 3. Paper Trading

This strategy is a daily factor strategy. We get the trading information before market opening. And use such data to generate a target position today for each stock in our universe. To reduce impact cost, we split each order into 3 parts, each part takes equal volumes, to achieve a strategy similar to VWAP order.

So we run following code every day before market closing.

Firstly update data:

```python
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
```

Secondly, generate portfolio from last day data

```python
# 2.generate portfolio from last day
today=pd.Timestamp('now').date()
BT=Backtest(start_day=today-datetime.timedelta(days=11),end_day=today)
universe=BT.universe
TradingDay=BT.TradingDay
data=BT.raw_data.copy()
data = data[(data['Date'] >= TradingDay[0]) & (data['Date'] <= TradingDay[-1])]
TradingDay=sorted(np.unique(data['Date']))

data['logprice']=np.log(data['close'])
data['ret5']=data.groupby('Ticker')['logprice'].diff(5).shift(0)
data['factor']=data['ret5']
BT.factor_preprocess(data)
lastest_factor=data[data['Date']==TradingDay[-1]][['Ticker','factor']]
lastest_factor['value']=lastest_factor['factor']*int(float(account.effective_buying_power))
```

Thirdly, trade stocks in market to achieve target positions

```python
# 3.run following code in trading time every day
def find_position(x):
    try:
        info=api.get_position(x)
        return info['qty']*info['current_price']
    except:
        return 0

split_num = 3 
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

print('successfully trade all stocks')
```

